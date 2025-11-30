"""
Comprehensive experiment script for evaluating flow matching models.

This script runs systematic experiments varying:
- ODE solvers (Euler, Dopri5, RK45, etc.)
- Number of function evaluations (NFE)
- Integration tolerances (for adaptive solvers)

Metrics computed:
- FID (Fréchet Inception Distance)
- KID (Kernel Inception Distance, if available)
- NFE (Number of Function Evaluations)
- Path Straightness

Outputs:
- Comprehensive plots (NFE vs FID, NFE vs Straightness, FID vs Straightness)
- Detailed JSON results
- Summary statistics
"""

import sys
import torch
import numpy as np
import pandas as pd
import argparse
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

from torchcfm.models.unet.unet import UNetModelWrapper
from utils_cifar import DiffusionVelocityWrapper

print("imported everything",flush=True)

# Parse arguments
parser = argparse.ArgumentParser(description='Run comprehensive experiments on flow matching model')
parser.add_argument('checkpoint', type=str, help='Path to checkpoint file (.pt)')
parser.add_argument('--num_gen', type=int, default=10000, help='Number of samples for FID (default: 10000)')
parser.add_argument('--num_straightness_samples', type=int, default=100, help='Samples for straightness computation (default: 100)')
parser.add_argument('--batch_size_fid', type=int, default=100, help='Batch size for FID (default: 100)')
parser.add_argument('--skip_kid', action='store_true', default=True, help='Skip KID computation (default: True)')
parser.add_argument('--output_dir', type=str, default='./experiments', help='Output directory (default: ./experiments)')
parser.add_argument('--model_type', type=str, default=None, choices=['diffusion', 'otcfm', 'fm', 'icfm', 'si'], 
                   help='Model type (auto-detected from checkpoint name if not specified)')

args = parser.parse_args()

# Solver configurations
SOLVER_CONFIGS = [
    # Fixed-step solvers with varying steps
    {"solver": "euler", "steps": 5},
    {"solver": "euler", "steps": 10},
    {"solver": "euler", "steps": 20},
    {"solver": "euler", "steps": 50},
    {"solver": "euler", "steps": 100},
    
    {"solver": "midpoint", "steps": 5},
    {"solver": "midpoint", "steps": 10},
    {"solver": "midpoint", "steps": 20},
    
    {"solver": "rk4", "steps": 5},
    {"solver": "rk4", "steps": 10},
    {"solver": "rk4", "steps": 20},
    
    # Adaptive solvers with varying tolerances
    {"solver": "dopri5", "tol": 1e-3},
    {"solver": "dopri5", "tol": 1e-4},
    {"solver": "dopri5", "tol": 1e-5},
]

# Device setup
use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class NFEWrapper(torch.nn.Module):
    """Wrapper to count number of function evaluations."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        return self.model(t, x, *args, **kwargs)


def compute_straightness(traj):
    """
    Compute path straightness metric.
    
    Straightness = direct_distance / path_length
    where:
    - direct_distance = ||x_1 - x_0||
    - path_length = sum ||x_{i+1} - x_i||
    
    Returns average straightness across batch.
    """
    # traj: [steps, batch, channels, height, width]
    # Flatten each sample to a vector: [steps, batch, -1]
    traj_flat = traj.reshape(traj.shape[0], traj.shape[1], -1)
    
    # Compute direct distance for each sample in the batch
    direct_dist = torch.norm(traj_flat[-1] - traj_flat[0], dim=-1)  # [batch]
    
    # Compute path length
    path_len = torch.zeros_like(direct_dist)
    for i in range(len(traj_flat) - 1):
        path_len += torch.norm(traj_flat[i+1] - traj_flat[i], dim=-1)  # [batch]
        
    return (direct_dist / path_len).mean().item()


def infer_model_config(state_dict):
    """
    Infer model configuration from state_dict tensor shapes.
    
    Returns dict with inferred config parameters.
    """
    # Get num_channels from first conv layer
    # Shape is [num_channels, in_channels, kernel, kernel]
    first_conv_weight = state_dict.get('input_blocks.0.0.weight')
    if first_conv_weight is None:
        # Try with 'module.' prefix (DataParallel)
        first_conv_weight = state_dict.get('module.input_blocks.0.0.weight')
    
    if first_conv_weight is not None:
        num_channels = first_conv_weight.shape[0]
    else:
        # Fallback to default
        num_channels = 32
        print(f"Warning: Could not infer num_channels, using default: {num_channels}")
    
    print(f"Inferred model config: num_channels={num_channels}")
    return {'num_channels': num_channels}


def detect_model_type(checkpoint_path):
    """Auto-detect model type from checkpoint filename."""
    filename = Path(checkpoint_path).name.lower()
    
    if 'diffusion' in filename:
        return 'diffusion'
    elif 'otcfm' in filename or 'cfm' in filename:
        return 'otcfm'
    elif 'icfm' in filename:
        return 'icfm'
    elif 'fm' in filename:
        return 'fm'
    elif 'si' in filename:
        return 'si'
    else:
        print(f"Warning: Could not detect model type from filename, assuming 'otcfm'")
        return 'otcfm'


def load_model(checkpoint_path, model_type=None):
    """Load the trained model, automatically inferring architecture from checkpoint."""
    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("ema_model") or checkpoint.get("model") or checkpoint
    
    # Infer model config from checkpoint
    inferred_config = infer_model_config(state_dict)
    num_channels = inferred_config['num_channels']
    
    # Auto-detect model type if not specified
    if model_type is None:
        model_type = detect_model_type(checkpoint_path)
    
    print(f"Model type: {model_type}")
    
    new_net = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    try:
        new_net.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        new_net.load_state_dict(new_state_dict)
    new_net.eval()
    
    # Wrap for diffusion models
    if model_type == "diffusion":
        new_net = DiffusionVelocityWrapper(new_net)
        
    return new_net, model_type


def run_experiment(model, config):
    """
    Run a single experiment with given solver configuration.
    
    Returns dict with all metrics.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {config}")
    print(f"{'='*60}")
    
    # Wrap model for NFE counting
    nfe_model = NFEWrapper(model)
    
    # Setup integration
    if config["solver"] in ["euler", "midpoint", "rk4"]:
        node = NeuralODE(nfe_model, solver=config["solver"])
        t_span = torch.linspace(0, 1, config["steps"] + 1, device=device)
        is_adaptive = False
    else:
        # Adaptive solver
        is_adaptive = True
        t_span_gen = torch.linspace(0, 1, 2, device=device)  # Only start/end for generation
    
    # Define generator function for cleanfid
    def gen_images(z):
        with torch.no_grad():
            x = torch.randn(z.shape[0], 3, 32, 32, device=device)
            
            if is_adaptive:
                # Use scalar floats (float32 compatible) for tolerances
                traj = odeint(
                    nfe_model,
                    x,
                    t_span_gen,
                    rtol=float(config["tol"]),
                    atol=float(config["tol"]),
                    method=config["solver"],
                    options={'dtype': torch.float32} if device.type == 'mps' else {}
                )
            else:
                traj = node.trajectory(x, t_span=t_span)
            
            final_img = traj[-1, :]
            return (final_img * 127.5 + 128).clip(0, 255).to(torch.uint8)
    
    # Compute FID
    print("Computing FID...")
    fid_device = "cpu" if device.type == "mps" else device
    score_fid = fid.compute_fid(
        gen=gen_images,
        dataset_name="cifar10",
        batch_size=args.batch_size_fid,
        dataset_res=32,
        num_gen=args.num_gen,
        dataset_split="train",
        mode="legacy_tensorflow",
        device=fid_device,
    )
    
    # Compute KID (optional)
    score_kid = None
    if not args.skip_kid:
        try:
            print("Computing KID...")
            score_kid = fid.compute_kid(
                gen=gen_images,
                dataset_name="cifar10",
                batch_size=args.batch_size_fid,
                dataset_res=32,
                num_gen=args.num_gen,
                dataset_split="train",
                mode="legacy_tensorflow",
                device=fid_device,
            )
        except Exception as e:
            print(f"Warning: KID computation failed: {e}")
    
    # Compute NFE and Straightness on a separate batch
    print("Computing NFE and Straightness...")
    nfe_model.nfe = 0
    with torch.no_grad():
        x = torch.randn(args.num_straightness_samples, 3, 32, 32, device=device)
        if is_adaptive:
            # Use more points for trajectory to compute straightness
            t_span_straightness = torch.linspace(0, 1, 50, device=device)
            # Use scalar floats (float32 compatible) for tolerances
            traj = odeint(
                nfe_model,
                x,
                t_span_straightness,
                rtol=float(config["tol"]),
                atol=float(config["tol"]),
                method=config["solver"],
                options={'dtype': torch.float32} if device.type == 'mps' else {}
            )
        else:
            traj = node.trajectory(x, t_span=t_span)
    
    # NFE is total function evaluations (batched ops share evals)
    avg_nfe = nfe_model.nfe
    straightness = compute_straightness(traj)
    
    result = {
        "config": config,
        "fid": float(score_fid),
        "kid": float(score_kid) if score_kid is not None else None,
        "nfe": float(avg_nfe),
        "straightness": float(straightness),
    }
    
    print(f"\nResults:")
    print(f"  FID: {result['fid']:.2f}")
    print(f"  KID: {result['kid']:.4f}" if result['kid'] else "  KID: N/A")
    print(f"  NFE: {result['nfe']:.1f}")
    print(f"  Straightness: {result['straightness']:.4f}")
    
    return result


def plot_results(results, output_dir):
    """Generate comprehensive plots from experiment results."""
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)
    df['solver'] = df['config'].apply(lambda x: x['solver'])
    df['config_str'] = df['config'].apply(lambda x: str(x))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    
    # 1. NFE vs FID
    ax1 = plt.subplot(2, 3, 1)
    for solver in df['solver'].unique():
        solver_df = df[df['solver'] == solver]
        ax1.plot(solver_df['nfe'], solver_df['fid'], 'o-', label=solver, markersize=8)
    ax1.set_xlabel('NFE', fontsize=12)
    ax1.set_ylabel('FID ↓', fontsize=12)
    ax1.set_title('NFE vs FID (lower is better)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. NFE vs Straightness
    ax2 = plt.subplot(2, 3, 2)
    for solver in df['solver'].unique():
        solver_df = df[df['solver'] == solver]
        ax2.plot(solver_df['nfe'], solver_df['straightness'], 'o-', label=solver, markersize=8)
    ax2.set_xlabel('NFE', fontsize=12)
    ax2.set_ylabel('Straightness ↑', fontsize=12)
    ax2.set_title('NFE vs Straightness (higher is better)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. FID vs Straightness
    ax3 = plt.subplot(2, 3, 3)
    for solver in df['solver'].unique():
        solver_df = df[df['solver'] == solver]
        ax3.plot(solver_df['straightness'], solver_df['fid'], 'o-', label=solver, markersize=8)
    ax3.set_xlabel('Straightness ↑', fontsize=12)
    ax3.set_ylabel('FID ↓', fontsize=12)
    ax3.set_title('Straightness vs FID', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficiency plot: FID vs NFE (scatter with color-coded straightness)
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(df['nfe'], df['fid'], c=df['straightness'], 
                         s=100, cmap='viridis', alpha=0.7)
    for idx, row in df.iterrows():
        ax4.annotate(row['solver'][:3], (row['nfe'], row['fid']), 
                    fontsize=8, ha='center')
    plt.colorbar(scatter, ax=ax4, label='Straightness')
    ax4.set_xlabel('NFE', fontsize=12)
    ax4.set_ylabel('FID ↓', fontsize=12)
    ax4.set_title('Quality-Efficiency Trade-off', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Pareto frontier (NFE vs FID)
    ax5 = plt.subplot(2, 3, 5)
    # Sort by NFE
    df_sorted = df.sort_values('nfe')
    # Find Pareto frontier (minimize both NFE and FID)
    pareto_points = []
    min_fid = float('inf')
    for idx, row in df_sorted.iterrows():
        if row['fid'] < min_fid:
            pareto_points.append(idx)
            min_fid = row['fid']
    
    ax5.scatter(df['nfe'], df['fid'], alpha=0.3, s=50, label='All configs')
    ax5.scatter(df.loc[pareto_points, 'nfe'], 
               df.loc[pareto_points, 'fid'],
               c='red', s=100, label='Pareto frontier', zorder=5)
    ax5.plot(df.loc[pareto_points, 'nfe'], 
            df.loc[pareto_points, 'fid'],
            'r--', alpha=0.5, zorder=4)
    ax5.set_xlabel('NFE', fontsize=12)
    ax5.set_ylabel('FID ↓', fontsize=12)
    ax5.set_title('Pareto Frontier (Quality vs Efficiency)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary table (best configs)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Find best configurations
    best_fid_idx = df['fid'].idxmin()
    best_straight_idx = df['straightness'].idxmax()
    best_efficient_idx = (df['fid'].rank() + df['nfe'].rank()).idxmin()
    
    summary_data = [
        ['Metric', 'Config', 'FID', 'NFE', 'Straight'],
        ['Best FID', df.loc[best_fid_idx, 'solver'], 
         f"{df.loc[best_fid_idx, 'fid']:.2f}",
         f"{df.loc[best_fid_idx, 'nfe']:.1f}",
         f"{df.loc[best_fid_idx, 'straightness']:.3f}"],
        ['Best Straight', df.loc[best_straight_idx, 'solver'],
         f"{df.loc[best_straight_idx, 'fid']:.2f}",
         f"{df.loc[best_straight_idx, 'nfe']:.1f}",
         f"{df.loc[best_straight_idx, 'straightness']:.3f}"],
        ['Most Efficient', df.loc[best_efficient_idx, 'solver'],
         f"{df.loc[best_efficient_idx, 'fid']:.2f}",
         f"{df.loc[best_efficient_idx, 'nfe']:.1f}",
         f"{df.loc[best_efficient_idx, 'straightness']:.3f}"],
    ]
    
    table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Best Configurations', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    checkpoint_name = Path(args.checkpoint).stem
    plot_path = Path(output_dir) / f"comprehensive_results_{checkpoint_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {plot_path}")
    
    plt.close()


def main():
    """Run all experiments and generate comprehensive analysis."""
    
    # Create output directory
    checkpoint_name = Path(args.checkpoint).stem
    output_dir = Path(args.output_dir) / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE EXPERIMENT SUITE")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load model once
    model, model_type = load_model(args.checkpoint, args.model_type)
    
    # Run all experiments
    results = []
    for i, config in enumerate(SOLVER_CONFIGS, 1):
        print(f"\n[Experiment {i}/{len(SOLVER_CONFIGS)}]")
        result = run_experiment(model, config)
        results.append(result)
    
    # Save raw results
    results_path = output_dir / "detailed_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Save as CSV for easy analysis
    df = pd.DataFrame(results)
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV results saved to: {csv_path}")
    
    # Generate plots
    plot_results(results, output_dir)
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total experiments: {len(results)}")
    print(f"FID range: {df['fid'].min():.2f} - {df['fid'].max():.2f}")
    print(f"NFE range: {df['nfe'].min():.1f} - {df['nfe'].max():.1f}")
    print(f"Straightness range: {df['straightness'].min():.4f} - {df['straightness'].max():.4f}")
    print(f"\nBest FID: {df['fid'].min():.2f} (config: {df.loc[df['fid'].idxmin(), 'config']})")
    print(f"Best Straightness: {df['straightness'].max():.4f} (config: {df.loc[df['straightness'].idxmax(), 'config']})")
    print(f"{'='*70}\n")
    
    print("✅ All experiments completed successfully!")


if __name__ == "__main__":
    main()
