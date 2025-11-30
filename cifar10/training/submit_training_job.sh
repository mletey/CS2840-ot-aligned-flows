#!/bin/bash
# model_cifar_runs_mary.sbatch
#
#SBATCH --job-name=model_cifar_runs_mary
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-8:00:00
#SBATCH -p kempner_h100
#SBATCH --account kempner_grads
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G
#SBATCH -o log_files/diffusion.out
#SBATCH -e log_files/diffusion.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module purge
module load python/3.10.12-fasrc01
source activate torchenv

parentdir="outputs"
OUTPUT_DIR="$parentdir/${SLURM_JOB_NAME}_diffusion"
mkdir -p "$OUTPUT_DIR"

python train_NONdiffusion.py \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "${SLURM_JOB_NAME}" \
    --model diffusion

