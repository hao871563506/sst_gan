## SRGAN

This folder is for testing srgan source code in a gpu and cpu environment. 
<br>

### Runing in local computer

*python==3.5*

*tensorflow==1.14*

*keras==2.3.0*

*pip install git+https://www.github.com/keras-team/keras-contrib.git*

<br>
After having anaconda or miniconda installed in your local computer, create a new env named "sst":

->**conda create -n sst python=3.5 anaconda**

Then activate sst env by:

->**conda activate sst**

Once entered, run the command below:

->**pip install tensorflow==1.14 --user**

->**pip install keras==2.3.0**

->**pip install git+https://www.github.com/keras-team/keras-contrib.git**

After that, you can successfully run srgan.py.

<br>

### Runing in Habanero

*cuda==10.0*

*cudnn==7.6.2*

*python==3.5*

*tensorflow-gpu==1.14*

*keras==2.3.0*

*pip install git+https://www.github.com/keras-team/keras-contrib.git*

<br>
Same as above, you need to create conda env sst in habanero and install all packages except tensorflow-gpu==1.14 instead of tensorflow==1.14. Once finished, enter:  

->**srun --pty -t 0-02:00:00 --constraint=p100 --gres=gpu -A ocp /bin/bash**

Now, we can import cuda and cudnn module on a p100 gpu node. I write all the command in the run.sh file so that we can just sbatch run.sh to load cuda, cudnn, and run the srgan.py file. 

->**sbatch run.sh**

To see the dynamic output result, you can use:

->**tail -f (the output file of slrum)**

### Notice 
1. For the setting of gpu and cpu resource, you can modify run.sh file. 

2. Try to put all folders under /rigel/ocp/users/ or /rigel/ocp/projects/ to save the disk space of your home directory. 
