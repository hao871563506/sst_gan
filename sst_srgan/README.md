# SST_SRGAN



## Runing in local computer

1. First, install anaconda or miniconda installed your local computer, create a new env named "sst":

`conda create -n sst python=3.5 anaconda` or `conda create -n sst python=3.5` (light-weighted)

2. Then activate sst env by:

`conda activate sst`

3. Once entered the environment, run the command below:

   ```
   pip install tensorflow==1.14 --user
   pip install keras==2.3.0
   pip install git+https://www.github.com/keras-team/keras-contrib.git
   pip install scipy==1.1.0
   pip install pillow
   pip install click
   pip install git-repo
   # to preprocess sst data from zarr format to npz format we need
   pip install xarray==0.12.3 zarr gcsfs intake intake-xarray
   conda install -c conda-forge numcodecs
   ```

4. After that, you can test by running the following. Currently, we assume to use preprocessed ocean data (numpy array) by default

`python runner.py --job_name=test --dataset_name=hr_lr_2timesteps --epochs=10 --batch_size=1`

5. Check the result under `face_img_rslts` folder!

## Runing in Habanero

*cuda==10.0*

*cudnn==7.6.2*

*python==3.5*

*tensorflow-gpu==1.14*

*keras==2.3.0*

*pip install git+https://www.github.com/keras-team/keras-contrib.git*

<br>
Same as above, you need to have anaconda or miniconda under your path in habanero first, then create conda env sst in habanero and install all packages except 

`pip install tensorflow-gpu==1.14` instead of `tensorflow==1.14`.



 Once finished, enter:  

`srun --pty -t 0-02:00:00 --constraint=p100 --gres=gpu -A ocp /bin/bash`

Now you can submit interactive jobs!



To submit batch jobs please use:

`sbatch run.sh`

which would produce a output file like `slurm.oxxxx.out`  To see the dynamic output result, you can use:

`tail -f slurm.oxxx.out`



## Notice 
1. For the setting of gpu and cpu resource, you can modify run.sh file. 

2. Try to put all folders under` /rigel/ocp/users/` or `/rigel/ocp/projects/` to save the disk space of your home directory. 

3. To install anaconda or miniconda to your habanero, you can download the installation package of, say miniconda, in your local computer, and scp that file to the path in your habanero. After that, you can unzip the installation package and begin the installation process. 
