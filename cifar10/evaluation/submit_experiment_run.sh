#!/bin/bash
# evaluate_cifar_runs_mary.sbatch
#
#SBATCH --job-name=evaluate_cifar_runs_mary
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-8:00:00
#SBATCH -p kempner_h100
#SBATCH --account kempner_grads
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G
#SBATCH -o log_files/fm.out
#SBATCH -e log_files/fm.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module purge
module load python/3.10.12-fasrc01
source activate torchenv_proj

python run_experiments.py \
    marys_shit/outputs/model_cifar_runs_mary_fm/fm/fm_cifar10_weights_step_300000.pt \
    --output_dir marys_shit/evaluations/fm300 \
    --model fm
