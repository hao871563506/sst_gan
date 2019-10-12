#!/bin/sh
#
# run file for capstone project
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=ocp      # The account name for the job.
#SBATCH --job-name=longer_s    # The job name.
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=24:00:00              # The time the job will take to run (here, 1 min)
#SBATCH --mem-per-cpu=32gb        # The memory the job will use per cpu core.
module load cuda10.0/toolkit
module load cuda10.0/blas
module load cudnn/cuda_10.0_v7.6.2

cd /rigel/ocp/users/lw2827/sst_superresolution/srgan/
python runner.py --epochs=100000 --batch_size=1
 
# End of script
