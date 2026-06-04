#!/bin/bash
#SBATCH --job-name=cof-landscaper
#SBATCH --account=ctb-moosavi5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=slurm-%j.log
#SBATCH --error=slurm-%j.log
#SBATCH --mail-type=FAIL

set -euo pipefail

module purge
module load StdEnv/2023 python/3.12.4 rdkit/2025.09.4

source ~/venvs/cof-landscaper-dev/bin/activate

cd "$SLURM_SUBMIT_DIR"

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"

python cof-landscaper.py > cof-landscaper.log 2>&1

echo "Job finished on $(date)"