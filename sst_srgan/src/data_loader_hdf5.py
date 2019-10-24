import fsspec
import xarray as xr 
import scipy
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tables

SST_DATASETS_PATH = "./numpy_array"

class DataLoader():
    def __init__(self, dataset_name, img_res=(512, 512), downsize_factor=(4, 4), local_path=None):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.downsize_factor = downsize_factor
        self.use_local_data = True

        if self.use_local_data:
            # load from test dataset
            assert local_path is not None
            #sst_dataset_path = local_path + "/{}.npz".format(self.dataset_name)
            self.sst_dataset_path = local_path
        else:
            #sst_dataset_path = os.path.join(SST_DATASETS_PATH, "{}.npz".format(dataset_name))
            self.sst_dataset_path = SST_DATASETS_PATH
            if not os.path.exists(SST_DATASETS_PATH):
                try:
                    os.makedirs(SST_DATASETS_PATH)
                except:
                    print(SST_DATASETS_PATH + " created error")
            uris = ['gcs://pangeo-ocean-ml/LLC4320/SST.{id:010d}.zarr'.format(id=tstep) for tstep in range(0, 4088+1, 73)][:]
            #uris = [f'gcs://pangeo-ocean-ml/LLC4320/SST.{tstep:010d}.zarr' for tstep in range(0, 4088+1, 73)][:10]
            dsets = [xr.open_zarr(fsspec.get_mapper(uri), consolidated=True) for uri in uris]
            ds = xr.combine_nested(dsets, 'timestep')
            print(ds)
            # to use ds.SST[0] to calculate nan value for all timestep
            num_nans = ds.SST[0].isnull().sum(dim=['x', 'y']).load()
            sst_valid = ds.SST.where(num_nans == 0, drop=True)
            print(sst_valid)
            sst_coarse = sst_valid.coarsen(x=self.downsize_factor[0], y=self.downsize_factor[1]).mean()
            print(sst_coarse)

            temp = sst_valid[0][0].load().values
            length = img_res[0]*img_res[1]+img_res[0]//downsize_factor[0]*img_res[1]//downsize_factor[1]
            print(length)
            hdf5_path = SST_DATASETS_PATH+ "/output.hdf5"
            # 和普通文件操作类似，'w'、'r'、'a' 分别表示写数据、读数据、追加数据
            hdf5_file = tables.open_file(hdf5_path, mode='w')
            # 设定压缩级别和压缩方法
            filters = tables.Filters(complevel=5, complib='blosc')
            earray = hdf5_file.create_earray(
                hdf5_file.root,
                'data', # 数据名称，之后需要通过它来访问数据
                tables.Atom.from_dtype(temp.dtype), # 设定数据格式（和data1格式相同）
                shape=(0, length), # 第一维的 0 表示数据可沿行扩展
                filters=filters,
                expectedrows=15000 # 完整数据大约规模，可以帮助程序提高时空利用效率
            )

            for timestep in range(sst_valid.shape[0]):
                for region in range(sst_valid.shape[1]):
                    # hr.append(sst_valid[timestep, region].load().values)
                    # lr.append(sst_coarse[timestep, region].load().values)
                    hr = sst_valid[timestep, region].load().values
                    lr = sst_coarse[timestep, region].load().values
                    hr = hr.flatten()
                    lr = lr.flatten()
                    temp = np.append(hr, lr).reshape((1, -1))
                    print(temp.shape)
                    earray.append(temp)
            print("got values!")
            hdf5_file.close()

            #np.savez(sst_dataset_path, name1=np.array(hr), name2=np.array(lr))
            print("hdf5 successfully saved")

        #hr,lr = np.load(sst_dataset_path)
        #data = np.load(sst_dataset_path)
        #self.hr = data['name1']
        #self.lr = data['name2']
        #print("numpy array successfully loaded")

    def load_data(self, batch_size=1, is_testing=False):
        
        hdf5_path = self.sst_dataset_path + "output.hdf5"
        hdf5_file = tables.open_file(hdf5_path, mode='r')
        # 数据名称为 'data'，我们可以通过 .root.data 访问到它
        hdf5_data = hdf5_file.root.data
        print(hdf5_data.shape) # (1000, 4096)
        # 像在内存中一样自由读取数据切片！
        batch_index = np.random.choice(hdf5_data.shape[0], size=batch_size)
        batch = np.array([hdf5_data[x] for x in batch_index])
        print(batch)
        hr = batch[:, 0:self.img_res[0]*self.img_res[1]].reshape(-1, self.img_res[0], self.img_res[1])
        lr = batch[:, self.img_res[0]*self.img_res[1]:].reshape(-1, int(self.img_res[0]/self.downsize_factor[0]), \
            int(self.img_res[1]/self.downsize_factor[1]))
        #print(hdf5_data[:10])
        hdf5_file.close()
        return hr, lr
        
        # imgs_hr = []
        # imgs_lr = []
        # for nparray in batch_nparrays:
        #     data = np.load(nparray)
        #     hr_image = data['name1']
        #     lr_image = data['name2']
        # #for hr_image, lr_image in zip(batch_hr, batch_lr):
        #     if not is_testing and np.random.random() < 0.5:
        #         hr_image = np.fliplr(hr_image)
        #         lr_image = np.fliplr(lr_image)
        #     imgs_hr.append(hr_image)
        #     imgs_lr.append(lr_image)

        #self.num_batch = int(len(imgs_hr) / batch_size)
        # imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        # imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        #return np.array(imgs_hr), np.array(imgs_lr)

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



