#!/bin/sh
#
# run file for capstone project
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=ocp      # The account name for the job.
#SBATCH --job-name=srgan_source_code    # The job name.
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH -c 4                     # The number of cpu cores to use.
#SBATCH --time=20:00              # The time the job will take to run (here, 1 min)
#SBATCH --mem-per-cpu=8gb        # The memory the job will use per cpu core.
module load cuda10.0/toolkit
module load cuda10.0/blas
module load cudnn/cuda_10.0_v7.6.2

cd /rigel/ocp/users/yl4089/sst_superresolution/srgan/
python srgan.py
 
# End of script
