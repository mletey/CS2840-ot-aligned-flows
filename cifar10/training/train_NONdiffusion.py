# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.
# Authors: Kilian Fatras, Alexander Tong
import copy
import os
import glob

import torch
from absl import app, flags
from torchvision import datasets, transforms
from tqdm import trange
from utils_cifar import ema, generate_samples, infiniteloop

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

print("imported everything successfully",flush=True)

class DiffusionMatcher:
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def sample_location_and_conditional_flow(self, x0, x1):
        # x0 is noise (epsilon), x1 is data
        # We use the same schedule as VariancePreservingConditionalFlowMatcher (VP-CFM)
        # for fair comparison, which is a trigonometric interpolation.
        # xt = cos(pi*t/2) * x1 + sin(pi*t/2) * x0
        # Target is x0 (epsilon)
        
        t = torch.rand(x0.shape[0]).type_as(x0)
        # Pad t for broadcasting
        pad_t = t.view([-1] + [1] * (x1.ndim - 1))
        
        cos_t = torch.cos(torch.pi * pad_t / 2)
        sin_t = torch.sin(torch.pi * pad_t / 2)
        
        xt = cos_t * x1 + sin_t * x0
        target = x0 # In diffusion (epsilon-prediction), we predict the noise
        
        return t, xt, target

from utils_cifar import ema, generate_samples, infiniteloop, setup, DiffusionVelocityWrapper

# ---- NEW: wandb (optional) ----
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 400001, help="total training steps")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation / saving
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)

# ---- NEW: wandb flags ----
flags.DEFINE_bool("use_wandb", True, help="enable Weights & Biases logging")
flags.DEFINE_string("wandb_project", "cfm-cifar10", help="wandb project name")
flags.DEFINE_string("wandb_entity", None, help="wandb entity (team) name")
flags.DEFINE_string("wandb_run_name", None, help="wandb run name (optional)")
flags.DEFINE_integer("log_every", 1000, help="log metrics to wandb every N steps")

use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def _maybe_init_wandb(config_dict):
    if not FLAGS.use_wandb:
        return None
    if not _HAS_WANDB:
        print("[wandb] wandb not installed; set --use_wandb=false or `pip install wandb`.")
        return None
    run = wandb.init(
        project=FLAGS.wandb_project,
        entity=FLAGS.wandb_entity,
        name=FLAGS.wandb_run_name,
        config=config_dict,
    )
    return run


def _wandb_log_images_if_exist(savedir, step, tag_prefix="samples"):
    """Try to find images saved by `generate_samples` and log them to wandb."""
    if not (_HAS_WANDB and FLAGS.use_wandb):
        return
    # Common patterns used by sample utils; adjust if your utils save with other names
    patterns = [
        os.path.join(savedir, f"*{step}*ema*.png"),
        os.path.join(savedir, f"*{step}*normal*.png"),
        os.path.join(savedir, f"*{step}*.png"),
    ]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    # Deduplicate while preserving order
    seen, unique_paths = set(), []
    for p in paths:
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            unique_paths.append(p)
    if unique_paths:
        images = [wandb.Image(p, caption=os.path.basename(p)) for p in unique_paths]
        wandb.log({f"{tag_prefix}/grids": images, "step": step}, step=step)


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(FLAGS.num_workers > 0),
    )
    datalooper = infiniteloop(dataloader)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is slightly worse than single GPU due to stats in DataParallel."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = sum(p.data.nelement() for p in net_model.parameters())
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    # ----- Flow Matching family selection
    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "diffusion":
        FM = DiffusionMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si', 'diffusion']"
        )

    savedir = os.path.join(FLAGS.output_dir, FLAGS.model)
    os.makedirs(savedir, exist_ok=True)

    # ----- W&B init
    run = _maybe_init_wandb(
        {
            "model": FLAGS.model,
            "num_channel": FLAGS.num_channel,
            "lr": FLAGS.lr,
            "grad_clip": FLAGS.grad_clip,
            "total_steps": FLAGS.total_steps,
            "warmup": FLAGS.warmup,
            "batch_size": FLAGS.batch_size,
            "num_workers": FLAGS.num_workers,
            "ema_decay": FLAGS.ema_decay,
            "parallel": FLAGS.parallel,
            "save_step": FLAGS.save_step,
        }
    )
    if run is not None:
        # Record model size once
        wandb.summary["model/params_M"] = model_size / 1e6

    # ----- Training loop
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)

            # grad norm (pre-clipping) for logging
            loss.backward()
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip
            )
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # EMA loss on the same mini-batch (no grad) as a lightweight eval metric
            with torch.no_grad():
                vt_ema = ema_model(t, xt)
                ema_loss = torch.mean((vt_ema - ut) ** 2)

            # progress bar
            lr = sched.get_last_lr()[0]
            pbar.set_description(f"step {step} | loss {loss.item():.4f} | ema {ema_loss.item():.4f} | lr {lr:.2e}")

            # ---- W&B logging
            if _HAS_WANDB and FLAGS.use_wandb and (step % FLAGS.log_every == 0):
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/ema_loss": float(ema_loss.item()),
                        "train/lr": float(lr),
                        "train/grad_norm": float(getattr(total_grad_norm, "item", lambda: total_grad_norm)()),
                        "step": step,
                    },
                    step=step,
                )

            # sample + save + log images
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                # For diffusion, we need to wrap the model to output velocity for the ODE solver
                if FLAGS.model == "diffusion":
                    net_model_to_sample = DiffusionVelocityWrapper(net_model)
                    ema_model_to_sample = DiffusionVelocityWrapper(ema_model)
                else:
                    net_model_to_sample = net_model
                    ema_model_to_sample = ema_model

                generate_samples(net_model_to_sample, FLAGS.parallel, savedir, step, net_="normal")
                generate_samples(ema_model_to_sample, FLAGS.parallel, savedir, step, net_="ema")

                if _HAS_WANDB and FLAGS.use_wandb:
                    _wandb_log_images_if_exist(savedir, step, tag_prefix="samples")

                # checkpoints (handle DataParallel transparently)
                to_save_net = net_model.module if isinstance(net_model, torch.nn.DataParallel) else net_model
                to_save_ema = ema_model.module if isinstance(ema_model, torch.nn.DataParallel) else ema_model
                torch.save(
                    {
                        "net_model": to_save_net.state_dict(),
                        "ema_model": to_save_ema.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    os.path.join(savedir, f"{FLAGS.model}_cifar10_weights_step_{step}.pt"),
                )

    if _HAS_WANDB and FLAGS.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    app.run(train)
