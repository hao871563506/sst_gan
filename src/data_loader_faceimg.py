import scipy.misc
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, dataset_dir, img_res=(128, 128), lr_res=(64, 64)):
        self.dataset_dir = dataset_dir
        self.img_res = img_res
        self.lr_res = lr_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob(self.dataset_dir + "/*")

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w = self.img_res
            low_h, low_w = self.lr_res

            img_hr = scipy.misc.imresize(img, self.img_res)  # original size, 218, 173
            img_lr = scipy.misc.imresize(img, (low_h, low_w))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)


if __name__ == '__main__':
    print(scipy.__version__)
    img = scipy.misc.imread("/Users/leah/Columbia/courses/19fall/capstone/sst_superresolution/datasets/img_align_celeba_small/000001.jpg")
    print(img.shape)
