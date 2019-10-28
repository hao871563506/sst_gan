import fsspec
import xarray as xr 
import scipy
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

#np_array_save_path = "./numpy_array"
#path = os.path.join(np_array_save_path, "hr_lr_1region.npz")


class DataLoader():
    def __init__(self, dataset_name, img_res=(512, 512), downsize_factor=(4, 4), local_path=None):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.downsize_factor = downsize_factor
        uris = ['/rigel/ocp/projects/shared_data/sst_superresolution/LLC4320/SST.{tstep:010d}.zarr' for tstep in range(0, 4088+1, 73)][:2]
        dsets = [xr.open_zarr(uri, consolidated=True) for uri in uris]
        ds = xr.combine_nested(dsets, 'timestep')
        print(ds)
        sst_coarse = ds.SST.coarsen(x=self.downsize_factor[0], y=self.downsize_factor[1]).mean()       
        print(sst_coarse)
        region = 306
        self.hr = ds.SST[0, region].load().values
        self.lr = sst_coarse[timestep, region].load().values

    def load_data(self, batch_size=1, is_testing=False):
        return np.array([self.hr]), np.array([self.lr])

    # the following is using generator to load data
    def load_data_generator(self, batch_size=1, is_testing=False):

        self.num_batch = int(len(self.hr.shape[0]) / batch_size)
        self.hr_stream = self.hr[:self.num_batch * batch_size,]
        self.hr_sequence_batch = np.split(self.hr_stream, self.num_batch, 0)
        self.lr_stream = self.lr[:self.num_batch * batch_size,]
        self.lr_sequence_batch = np.split(self.lr_stream, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        imgs_hr = self.hr_sequence_batch[self.pointer]
        imgs_lr = self.lr_sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch

        return imgs_hr, imgs_lr

    def reset_pointer(self):
        self.pointer = 0

