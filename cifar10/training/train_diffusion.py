import copy
import math
import os
import sys

import torch
import torch.nn.functional as F
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import trange

# Dependencies for logging and metrics
import wandb
from cleanfid import fid

# Assumes utils_cifar.py exists in the same directory
from utils_cifar import ema, infiniteloop, setup
from torchcfm.models.unet.unet import UNetModelWrapper

print("IMPORTS DONE SUCCESSFULLY",flush=True)

FLAGS = flags.FLAGS

# ==============================================================================
# Flags
# ==============================================================================

flags.DEFINE_string("output_dir", "./results_ddpm/", help="output_directory")

# Model Architecture
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training Hyperparameters
flags.DEFINE_float("lr", 2e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 200001, help="total training steps")
flags.DEFINE_integer("batch_size", 256, help="batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")

# Warmup
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup steps")

# DDP
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_string("master_addr", "localhost", help="master address for DDP")
flags.DEFINE_string("master_port", "12355", help="master port for DDP")

# DDPM Specifics
flags.DEFINE_integer("timesteps", 1000, help="diffusion timesteps")
flags.DEFINE_float("beta_start", 1e-4, help="start of linear beta schedule")
flags.DEFINE_float("beta_end", 0.02, help="end of linear beta schedule")

# Eval / Sampling / FID / WandB
flags.DEFINE_integer("save_step", 50000, help="frequency of evaluation (sampling & saving)")
flags.DEFINE_integer("num_samples", 64, help="number of images to generate for grid visualization")

flags.DEFINE_bool("use_wandb", True, help="enable wandb logging")
flags.DEFINE_string("wandb_project", "cifar10-ddpm", help="wandb project name")
flags.DEFINE_string("wandb_entity", 'harvardml', help="wandb entity/username")

# FID Specifics
flags.DEFINE_bool("compute_fid", True, help="whether to compute FID during eval")
flags.DEFINE_integer("fid_num_gen", 10000, help="number of samples to generate for FID calculation (50k is standard but slow)")
flags.DEFINE_integer("fid_batch_size", 128, help="batch size for FID generation")


def get_lr_schedule(step):
    # 1. Linear Warmup
    if step < FLAGS.warmup:
        return float(step) / float(max(1, FLAGS.warmup))
    
    # 2. Cosine Decay (1.0 -> 0.1)
    progress = float(step - FLAGS.warmup) / float(max(1, FLAGS.total_steps - FLAGS.warmup))
    alpha_min = 0.1  # Decay to 10%
    return alpha_min + 0.5 * (1.0 - alpha_min) * (1.0 + math.cos(math.pi * progress))

# ==============================================================================
# DDPM Scheduler
# ==============================================================================
class DDPMScheduler:
    def __init__(self, timesteps, beta_start, beta_end, device):
        self.timesteps = timesteps
        self.device = device
        
        # Linear Beta Schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def add_noise(self, x_start, t):
        noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_noisy = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
        return x_noisy, noise

    @torch.no_grad()
    def sample(self, model, shape):
        device = self.device
        img = torch.randn(shape, device=device)
        
        # Iterate from T-1 down to 0
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # Predict noise
            pred_noise = model(t, img) 
            
            sqrt_recip_alpha = self.sqrt_recip_alphas[i]
            beta_t = self.betas[i]
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[i]
            
            # Mean
            model_mean = sqrt_recip_alpha * (
                img - (beta_t / sqrt_one_minus_alpha_bar) * pred_noise
            )
            
            if i > 0:
                noise = torch.randn_like(img)
                # Fixed sigma = sqrt(beta)
                sigma = torch.sqrt(beta_t) 
                img = model_mean + sigma * noise
            else:
                img = model_mean
                
        return img

# ==============================================================================
# Training Logic
# ==============================================================================
def train(rank, total_num_gpus, argv):
    # --- 1. Determine if this process is the "Master" (Rank 0) ---
    is_master = False
    if FLAGS.parallel:
        # In DDP, rank is an integer (0, 1, 2...)
        if rank == 0:
            is_master = True
    else:
        # In Single GPU, rank is a torch.device object, so we are always master
        is_master = True

    # --- 2. Initialize WandB (Only on Master) ---
    if is_master:
        print(f"Training Config: LR={FLAGS.lr}, Steps={FLAGS.total_steps}, Batch={FLAGS.batch_size}")
        if FLAGS.use_wandb:
            wandb.init(
                project=FLAGS.wandb_project, 
                entity=FLAGS.wandb_entity,
                config=FLAGS.flag_values_dict()
            )

    if FLAGS.parallel and total_num_gpus > 1:
        batch_size_per_gpu = FLAGS.batch_size // total_num_gpus
        setup(rank, total_num_gpus, FLAGS.master_addr, FLAGS.master_port)
    else:
        batch_size_per_gpu = FLAGS.batch_size

    # --- Data ---
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )
    sampler = DistributedSampler(dataset) if FLAGS.parallel else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=False if FLAGS.parallel else True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )
    datalooper = infiniteloop(dataloader)

    # --- Model ---
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(rank)


    ema_model = copy.deepcopy(net_model)
    
    # Optimizer & Warmup
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=get_lr_schedule)
    
    if FLAGS.parallel:
        net_model = DistributedDataParallel(net_model, device_ids=[rank])
        ema_model = DistributedDataParallel(ema_model, device_ids=[rank])

    # --- DDPM Scheduler ---
    ddpm = DDPMScheduler(FLAGS.timesteps, FLAGS.beta_start, FLAGS.beta_end, rank)
    savedir = FLAGS.output_dir
    
    if is_master:
        os.makedirs(savedir, exist_ok=True)

    # --- Training Loop ---
    steps_per_epoch = math.ceil(len(dataset) / FLAGS.batch_size)
    num_epochs = math.ceil(FLAGS.total_steps / steps_per_epoch)
    global_step = 0

    # Helper function for FID generation (passed to clean-fid)
    def fid_generator_fn(unused_z):
        bs = FLAGS.fid_batch_size
        model_to_use = ema_model.module if FLAGS.parallel else ema_model
        
        with torch.no_grad():
            imgs = ddpm.sample(model_to_use, shape=(bs, 3, 32, 32))
            
        imgs = (imgs.clamp(-1, 1) + 1) * 127.5
        imgs = imgs.to(torch.uint8)
        return imgs

    with trange(num_epochs, dynamic_ncols=True) as epoch_pbar:
        for epoch in epoch_pbar:
            if is_master:
                epoch_pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            if sampler: sampler.set_epoch(epoch)

            with trange(steps_per_epoch, dynamic_ncols=True) as step_pbar:
                for step in step_pbar:
                    global_step += 1

                    # 1. Update Weights
                    optim.zero_grad()
                    x0 = next(datalooper).to(rank)
                    t = torch.randint(0, FLAGS.timesteps, (x0.shape[0],), device=rank).long()
                    
                    xt, noise = ddpm.add_noise(x0, t)
                    predicted_noise = net_model(t, xt)
                    
                    loss = F.mse_loss(predicted_noise, noise)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
                    optim.step()
                    sched.step()
                    ema(net_model, ema_model, FLAGS.ema_decay)
                    
                    # Log Loss to WandB
                    if is_master and FLAGS.use_wandb and global_step % 100 == 0:
                        wandb.log({"train_loss": loss.item(), "lr": sched.get_last_lr()[0]}, step=global_step)

                    # 2. Eval Loop
                    if FLAGS.save_step > 0 and global_step % FLAGS.save_step == 0 and global_step > 0:
                        
                        if is_master:
                            print(f"\nRunning evaluation at step {global_step}...")
                            eval_model = ema_model.module if FLAGS.parallel else ema_model
                            eval_model.eval()

                            # --- A. Generate Grid Samples ---
                            with torch.no_grad():
                                nrow = int(math.sqrt(FLAGS.num_samples))
                                x_gen = ddpm.sample(eval_model, shape=(FLAGS.num_samples, 3, 32, 32))
                                x_gen = (x_gen.clamp(-1, 1) + 1) / 2 # [0, 1]
                                
                                grid = make_grid(x_gen, nrow=nrow)
                                save_path = os.path.join(savedir, f"samples_ema_{global_step}.png")
                                save_image(grid, save_path)
                                
                                if FLAGS.use_wandb:
                                    wandb.log({"generated_images": wandb.Image(grid)}, step=global_step)

                            # --- B. Compute FID ---
                            if FLAGS.compute_fid:
                                print(f"Computing FID with {FLAGS.fid_num_gen} samples...")
                                try:
                                    score = fid.compute_fid(
                                        gen=fid_generator_fn,
                                        dataset_name="cifar10",
                                        batch_size=FLAGS.fid_batch_size,
                                        dataset_res=32,
                                        num_gen=FLAGS.fid_num_gen,
                                        dataset_split="train",
                                        mode="legacy_tensorflow",
                                        device=torch.device(rank) if isinstance(rank, int) else rank
                                    )
                                    print(f"Step {global_step} FID: {score}")
                                    if FLAGS.use_wandb:
                                        wandb.log({"FID": score}, step=global_step)
                                except Exception as e:
                                    print(f"FID computation failed: {e}")

                            # --- C. Save Weights ---
                            torch.save(
                                {
                                    "net_model": net_model.state_dict(),
                                    "ema_model": ema_model.state_dict(),
                                    "sched": sched.state_dict(),
                                    "optim": optim.state_dict(),
                                    "step": global_step,
                                },
                                os.path.join(savedir, f"ddpm_cifar10_weights_step_{global_step}.pt"),
                            )
                            eval_model.train()

def main(argv):
    total_num_gpus = int(os.getenv("WORLD_SIZE", 1))
    if FLAGS.parallel and total_num_gpus > 1:
        train(rank=int(os.getenv("RANK", 0)), total_num_gpus=total_num_gpus, argv=argv)
    else:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        train(rank=device, total_num_gpus=total_num_gpus, argv=argv)

if __name__ == "__main__":
    app.run(main)
