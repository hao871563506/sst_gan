# SST_SRGAN

This folder is for testing srgan source code for our sst datasets. 

## Runing in local computer

In order to run srgan for our sst datasets, you need to first follow the link below to install all environment packages required for srgan. 

https://github.com/rabernat/sst_superresolution/blob/master/srgan/README.md

Then, because sst data is stored in zarr format, you should install xarray and numcodecs to read the data successfully:

->**pip install xarray==0.12.3 zarr gcsfs intake intake-xarray**

->**conda install -c conda-forge numcodecs**

Notice all the commands above should be run in your virtual conda environment. 

## Runing in Habanero
