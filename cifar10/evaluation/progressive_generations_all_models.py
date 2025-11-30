import math
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from torchdiffeq import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt

from torchcfm.models.unet.unet import UNetModelWrapper
# Only needed if you ever want to use prob. flow for a diffusion baseline:
# from utils_cifar import DiffusionVelocityWrapper


# ---------------------------------------------------------------------------
# Matplotlib style (your config)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "text.usetex": True,
})


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODELS = [
    {
        "label": "OT-CFM",
        "kind": "flow",
        "model_type": "otcfm",
        "checkpoint": "/n/netscratch/pehlevan_lab/Everyone/maryCIFARcheckpoints/outputs/model_cifar_runs_mary_otcfm/otcfm/otcfm_cifar10_weights_step_300000.pt",  # noqa: E501
    },
    {
        "label": "FM",
        "kind": "flow",
        "model_type": "fm",
        "checkpoint": "/n/netscratch/pehlevan_lab/Everyone/maryCIFARcheckpoints/outputs/model_cifar_runs_mary_fm/fm/fm_cifar10_weights_step_300000.pt",  # noqa: E501
    },
    {
        "label": "ICFM",
        "kind": "flow",
        "model_type": "icfm",
        "checkpoint": "/n/netscratch/pehlevan_lab/Everyone/maryCIFARcheckpoints/outputs/model_cifar_runs_mary_icfm/icfm/icfm_cifar10_weights_step_300000.pt",  # noqa: E501
    },
    {
        "label": "DDPM",
        "kind": "ddpm",
        "model_type": "diffusion",
        "checkpoint": "/n/netscratch/pehlevan_lab/Everyone/pranavOTdiffusion/ddpm_cifar10_weights_step_100000.pt",  # noqa: E501
    },
]

OUTPUT_FIG = "progressive_generations_all_models.png"

NUM_SAMPLES = 4      # samples per panel (arranged as a grid)
NUM_TIMES = 5         # number of time points (columns)
SEED = 5
IMG_SIZE = (3, 32, 32)  # C, H, W

FLOW_SOLVER = "dopri5"
FLOW_TOL = 1e-5

# DDPM defaults (must match training)
DDPM_TIMESTEPS = 1000
DDPM_BETA_START = 1e-4
DDPM_BETA_END = 0.02


# ---------------------------------------------------------------------------
# Utility: infer num_channels from checkpoint
# ---------------------------------------------------------------------------
def infer_model_config(state_dict):
    first_conv_weight = state_dict.get("input_blocks.0.0.weight")
    if first_conv_weight is None:
        first_conv_weight = state_dict.get("module.input_blocks.0.0.weight")

    if first_conv_weight is not None:
        num_channels = first_conv_weight.shape[0]
    else:
        num_channels = 32
        print(f"Warning: Could not infer num_channels, using default: {num_channels}")

    print(f"Inferred model config: num_channels={num_channels}")
    return {"num_channels": num_channels}


# ---------------------------------------------------------------------------
# Load flow-type models (OT-CFM, FM, ICFM)
# ---------------------------------------------------------------------------
def load_flow_model(checkpoint_path, device):
    print(f"Loading flow model from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("ema_model") or ckpt.get("model") or ckpt

    inferred = infer_model_config(state_dict)
    num_channels = inferred["num_channels"]

    net = UNetModelWrapper(
        dim=IMG_SIZE,
        num_res_blocks=2,
        num_channels=num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    try:
        net.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # strip "module." if present
            new_state_dict[k[7:]] = v
        net.load_state_dict(new_state_dict)

    net.eval()
    return net


# ---------------------------------------------------------------------------
# Load DDPM model (noise-prediction UNet)
# ---------------------------------------------------------------------------
def load_ddpm_model(checkpoint_path, device):
    print(f"Loading DDPM model from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Training script saved:
    # {"net_model": net_model.state_dict(), "ema_model": ema_model.state_dict(), ...}
    state_dict = ckpt.get("ema_model") or ckpt.get("net_model") or ckpt

    inferred = infer_model_config(state_dict)
    num_channels = inferred["num_channels"]

    net = UNetModelWrapper(
        dim=IMG_SIZE,
        num_res_blocks=2,
        num_channels=num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    try:
        net.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        net.load_state_dict(new_state_dict)

    net.eval()
    return net


# ---------------------------------------------------------------------------
# Flow trajectory: ODE integration from t=0..1
# ---------------------------------------------------------------------------
def generate_flow_trajectory(model, device, z0, num_times=5, solver="dopri5", tol=1e-5):
    t_span = torch.linspace(0, 1, num_times, device=device)
    with torch.no_grad():
        traj = odeint(
            model,
            z0,
            t_span,
            rtol=float(tol),
            atol=float(tol),
            method=solver,
            options={"dtype": torch.float32} if device.type == "mps" else {},
        )
    return traj  # [T, B, C, H, W]


# ---------------------------------------------------------------------------
# DDPM scheduler + trajectory
# ---------------------------------------------------------------------------
class DDPMScheduler:
    def __init__(self, timesteps, beta_start, beta_end, device):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    @torch.no_grad()
    def sample_trajectory(self, model, z0, num_times):
        """
        Run the reverse DDPM chain starting from z0 ~ N(0,I) and
        return `num_times` snapshots along the denoising trajectory.

        We sample at equally spaced steps in the *reverse* process.
        """
        device = self.device
        T = self.timesteps

        img = z0.clone().to(device)

        # We want snapshots including initial noise and final sample,
        # so we consider T steps and T+1 states. We'll index "progress"
        # from 0..T (0 = initial noise, T = fully denoised).
        step_indices = np.linspace(0, T, num_times, dtype=int)
        save_set = set(step_indices)

        saved = []
        progress = 0

        # Save initial pure noise state (progress=0)
        if progress in save_set:
            saved.append(img.clone())

        # Reverse diffusion: x_T -> x_0
        for i in reversed(range(0, T)):
            t = torch.full((img.shape[0],), i, device=device, dtype=torch.long)

            pred_noise = model(t, img)
            sqrt_recip_alpha = self.sqrt_recip_alphas[i]
            beta_t = self.betas[i]
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[i]

            model_mean = sqrt_recip_alpha * (
                img - (beta_t / sqrt_one_minus_alpha_bar) * pred_noise
            )

            if i > 0:
                noise = torch.randn_like(img)
                sigma = torch.sqrt(beta_t)  # fixed sigma
                img = model_mean + sigma * noise
            else:
                img = model_mean

            progress += 1
            if progress in save_set:
                saved.append(img.clone())

        traj = torch.stack(saved, dim=0)  # [num_times, B, C, H, W]
        return traj


# ---------------------------------------------------------------------------
# Main plotting logic
# ---------------------------------------------------------------------------
def main():
    # Device
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Shared initial noise
    torch.manual_seed(SEED)
    B, C, H, W = (NUM_SAMPLES,) + IMG_SIZE
    z0 = torch.randn(B, C, H, W, device=device)

    num_models = len(MODELS)
    num_times = NUM_TIMES

    # Global normalized time labels (0..1) for columns
    t_values = np.linspace(0.0, 1.0, num_times)

    # Prepare figure
    fig, axes = plt.subplots(
        nrows=num_models,
        ncols=num_times,
        figsize=(3.0 * num_times, 3.0 * num_models),
    )

    if num_models == 1:
        axes = np.expand_dims(axes, 0)
    if num_times == 1:
        axes = np.expand_dims(axes, 1)

    for i, model_info in enumerate(MODELS):
        label = model_info["label"]
        kind = model_info["kind"]
        ckpt = model_info["checkpoint"]

        if kind == "flow":
            model = load_flow_model(ckpt, device)
            traj = generate_flow_trajectory(
                model, device, z0.clone(), num_times=num_times,
                solver=FLOW_SOLVER, tol=FLOW_TOL
            )
        elif kind == "ddpm":
            model = load_ddpm_model(ckpt, device)
            ddpm = DDPMScheduler(
                timesteps=DDPM_TIMESTEPS,
                beta_start=DDPM_BETA_START,
                beta_end=DDPM_BETA_END,
                device=device,
            )
            traj = ddpm.sample_trajectory(model, z0.clone(), num_times=num_times)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        # map from [-1,1] to [0,1] for plotting
        traj_imgs = ((traj + 1) / 2).clamp(0, 1).cpu()  # [T, B, C, H, W]

        for j in range(num_times):
            ax = axes[i, j]
            ax.axis("off")

            imgs_t = traj_imgs[j]  # [B, C, H, W]

            nrow = int(math.sqrt(NUM_SAMPLES))
            grid = vutils.make_grid(imgs_t, nrow=nrow, padding=2)
            grid_np = grid.permute(1, 2, 0).numpy()  # [H, W, C]

            ax.imshow(grid_np)

            # Column titles: normalized "time"
            if i == 0:
                ax.set_title(r"$t = {:.2f}$".format(t_values[j]), fontsize=8)

            # Left-side method label ("y-axis")
            if j == 0:
                ax.set_ylabel(label, fontsize=8)

    plt.tight_layout()
    out_path = Path(OUTPUT_FIG)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
