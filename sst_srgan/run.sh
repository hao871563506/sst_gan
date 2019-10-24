#!/bin/sh
#
# run file for capstone project
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=ocp      # The account name for the job.
#SBATCH --job-name=srgan_source_code    # The job name.
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=12:00:00              # The time the job will take to run (here, 1 min)
#SBATCH --mem-per-cpu=32gb        # The memory the job will use per cpu core.
#SBATCH --constraint=p100

module load cuda10.0/toolkit
module load cuda10.0/blas
module load cudnn/cuda_10.0_v7.6.2

cd /rigel/ocp/users/zw2533/sst_superresolution/sst_srgan

source activate sst

python runner.py --job_name=ocean_4x --dataset_name=hr_lr_2timesteps --epochs=20 --batch_size=4

# End of script
