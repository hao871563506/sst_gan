#!/bin/sh
#
# run file for capstone project
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=ocp      # The account name for the job.
#SBATCH --job-name=generate_data_numpy    # The job name.
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=12:00:00              # The time the job will take to run (here, 1 min)
#SBATCH --mem-per-cpu=64gb        # The memory the job will use per cpu core.


cd /rigel/ocp/users/zw2533/sst_superresolution/sst_srgan

source activate sst

python runner.py --job_name=test --epochs=1 --batch_size=1
 
# End of script
