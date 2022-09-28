#!/bin/bash
#SBATCH --job-name="Simclr_32_lands_s2_P_500eph"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=170gb
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu 
#SBATCH --account=generalGrp
#SBATCH --output=my_job-%j.o
#SBATCH --error=my_error-%j.e
#SBATCH --mail-user=camalas@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1


cd /deac/csc/paucaGrp/camalas/Contrastive/pythorch/SimCLR_S2_P/
source /deac/csc/paucaGrp/camalas/software/venvs/pytorch/bin/activate

python -u main.py --batch_size 256 --epochs 500
  
slurm_mem_report -g

exit 0


