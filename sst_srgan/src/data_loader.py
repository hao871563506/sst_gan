import fsspec
import xarray as xr 
import scipy
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

SST_DATASETS_PATH = "./numpy_array"

class DataLoader():
    def __init__(self, dataset_name, img_res=(512, 512), downsize_factor=(4, 4), local=False, local_path=None):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.downsize_factor = downsize_factor
        self.use_local_data = True

        if self.use_local_data:
            # load from test dataset
            assert local_path is not None
            sst_dataset_path = local_path + "/{}.npz".format(self.dataset_name)
        else:
            sst_dataset_path = os.path.join(SST_DATASETS_PATH, "{}.npz".format(dataset_name))
            if not os.path.exists(sst_dataset_path):
                uris = ['gcs://pangeo-ocean-ml/LLC4320/SST.{id}:010d.zarr'.format(id=tstep) for tstep in range(0, 4088+1, 73)][:10]
                dsets = [xr.open_zarr(fsspec.get_mapper(uri), consolidated=True) for uri in uris]
                ds = xr.combine_nested(dsets, 'timestep')
                print(ds)
                # to use ds.SST[0] to calculate nan value for all timestep
                num_nans = ds.SST[0].isnull().sum(dim=['x', 'y']).load()
                sst_valid = ds.SST.where(num_nans == 0, drop=True)
                print(sst_valid)
                sst_coarse = sst_valid.coarsen(x=self.downsize_factor[0], y=self.downsize_factor[1]).mean()
                print(sst_coarse)

                # hr = sst_valid.load().values
                # lr = sst_coarse.load().values
                hr = []
                lr = []
                for timestep in range(sst_valid.shape[0]):
                    for region in range(sst_valid.shape[1]):
                        hr.append(sst_valid[timestep, region].load().values)
                        lr.append(sst_coarse[timestep, region].load().values)
                print("got values!")
                if not os.path.exists(SST_DATASETS_PATH):
                    try:
                        os.makedirs(SST_DATASETS_PATH)
                    except:
                        print(SST_DATASETS_PATH + " created error")

                np.savez(sst_dataset_path, name1=np.array(hr), name2=np.array(lr))
                print("numpy array successfully saved")

        #hr,lr = np.load(sst_dataset_path)
        data = np.load(sst_dataset_path)
        self.hr = data['name1']
        self.lr = data['name2']
        print("numpy array successfully loaded")

    def load_data(self, batch_size=1, is_testing=False):
        #data_type = "train" if not is_testing else "test"
        batch_index = np.random.choice(self.hr.shape[0], size=batch_size)
        batch_hr = self.hr[batch_index,]
        batch_lr = self.lr[batch_index,]
        # If training => do random flip
        imgs_hr = []
        imgs_lr = []
        for hr_image, lr_image in zip(batch_hr, batch_lr):
            if not is_testing and np.random.random() < 0.5:
                hr_image = np.fliplr(hr_image)
                lr_image = np.fliplr(lr_image)
            imgs_hr.append(hr_image)
            imgs_lr.append(lr_image)

        #self.num_batch = int(len(imgs_hr) / batch_size)
        # imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        # imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return np.array(imgs_hr), np.array(imgs_lr)

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



