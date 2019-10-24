import fsspec
import xarray as xr 
import scipy
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

SST_DATASETS_PATH = "./numpy_array2"

class DataLoader():
    def __init__(self, dataset_name, img_res=(512, 512), downsize_factor=(4, 4), local=False, local_path=None, batch_size=1):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.downsize_factor = downsize_factor
        self.use_local_data = True
        self.iterator = None

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
            
            writer = tf.python_io.TFRecordWriter(SST_DATASETS_PATH+"/output.tfrecords")

            def _bytes_feature(value):
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

            # hr = []
            # lr = []
            for timestep in range(sst_valid.shape[0]):
                for region in range(sst_valid.shape[1]):
                    hr=sst_valid[timestep, region].load().values
                    lr=sst_coarse[timestep, region].load().values
                    #np.savez(SST_DATASETS_PATH + "/{}_{}.npz".format(timestep,region), name1=hr, name2=lr)
                    feature = {'hr':  _bytes_feature(tf.compat.as_bytes(hr.tostring())), \
                    'lr':  _bytes_feature(tf.compat.as_bytes(lr.tostring()))}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    print("got values!")
            # hr = np.array(hr)
            # lr = np.array(lr)

            print("tf record successfully saved")

        self.create_dataset(batch_size)

    def create_dataset(self, batch_size=1):
        #print("-----------------------------")
        def _parse_function(proto):
            # define your tfrecord again. Remember that you saved your image as a string.
            keys_to_features = {'hr': tf.FixedLenFeature([], tf.string),
                                'lr': tf.FixedLenFeature([], tf.string)}
            
            # Load one example
            parsed_features = tf.parse_single_example(proto, keys_to_features)
            
            # Turn your saved image string into an array
            parsed_features['hr'] = tf.decode_raw(
                parsed_features['hr'], tf.uint8)

            parsed_features['lr'] = tf.decode_raw(
                parsed_features['lr'], tf.uint8)
            
            return parsed_features['hr'], parsed_features['lr']


        dataset = tf.data.TFRecordDataset(self.sst_dataset_path+"/output.tfrecords")
        # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
        dataset = dataset.map(_parse_function, num_parallel_calls=8)
        # This dataset will go on forever
        dataset = dataset.repeat()
        # Set the number of datapoints you want to load and shuffle 
        dataset = dataset.shuffle(20000)
        # Set the batchsize
        dataset = dataset.batch(batch_size)

        # Create an iterator
        self.iterator = dataset.make_one_shot_iterator()
        
    def load_data(self, batch_size=1):
        # Create your tf representation of the iterator
        hr, lr = self.iterator.get_next()
        hr = tf.reshape(hr, [-1, 512, 512, 1])
        lr = tf.reshape(lr, [-1, int(512/self.downsize_factor[0]), int(512/self.downsize_factor[1]), 1])
        print("###############################")
        print(hr.shape)
        return hr, lr

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



