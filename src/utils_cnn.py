import gc
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

import matplotlib
matplotlib.use('Agg') #for display on remote node
import matplotlib.pyplot as plt

NumDigitsImg=3

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


# numpy file containing a list (array) of patches
class train_data():
    def __init__(self, filepath='./data/image_clean_pat.npy'):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath='./data/image_clean_pat.npy'):
    return train_data(filepath=filepath)

# loads a list of images
def load_images_png(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data

# loads a list of images
def load_images_bin(filelist, Nz, Ny, Nx):
    data = []
    vl = np.zeros(Nz, dtype=np.float32)
    vh = np.zeros(Nz, dtype=np.float32)
    for i in range(Nz):
        f = open(filelist[i], 'rb')
        img = np.fromfile(f, '<f4') #np.float32
        img = np.reshape(img, ((1,Ny,Nx,1)) )
        data.append(img)
        vl[i] = np.percentile(img, 0.5)
        vh[i] = np.percentile(img, 99.5)
        f.close()
    return data, vl, vh    

# saves single image
def save_images_png(filepath, ground_truth, noisy_image=None, clean_image=None, psnr=None, clims_flag=0):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)

    if psnr is None:
        im = Image.fromarray(cat_image.astype('uint8')).convert('L')
        im.save(filepath+'.png', 'png')
    else:
        if(clims_flag):
            vl = np.percentile(ground_truth, 0.5)
            vh = np.percentile(ground_truth, 99.5)
            plt.imshow(cat_image, clim=(vl, vh), cmap='gray')
            plt.colorbar()
        else:
            plt.imshow(cat_image, cmap='gray')
        plt.title(('PSNR = %.2f' % psnr))
        plt.show(block=False)
        plt.savefig(filepath+'.pdf', format='pdf')
        plt.clf()

# saves single image (a pdf showing noisy+clean image and writing out clean image as binary file)
# Note that compared to save_images_png the order of arguments is altered
def save_images_bin(filepath, noisy_image, clean_image, ground_truth=None, psnr=None):
    # Write clean image as bin file
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if ground_truth is not None:
        ground_truth = np.squeeze(ground_truth)

    f = open(filepath+'.2Dimgdata', 'wb')
    data = clean_image.flatten()
    data.tofile(f)
    f.close()

    # Concatenate images for plot
    if ground_truth is not None:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
        vl = np.percentile(ground_truth, 0.5)
        vh = np.percentile(ground_truth, 99.5)
    else:
        cat_image = np.concatenate([noisy_image, clean_image], axis=1)
        vl = np.percentile(clean_image, 0.5)
        vh = np.percentile(clean_image, 99.5)
    
    # Plot image
    plt.imshow(cat_image, clim=(vl, vh), cmap='gray')
    plt.colorbar()
    if psnr is not None:
        plt.title(('PSNR = %.2f' % psnr))
    plt.show(block=False)
    plt.savefig(filepath+'.pdf', format='pdf')
    plt.clf()


# VS: Why is the signal power fixed here ? Makes no sense. 
def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse) 
    return psnr

# VS: This PSNR function allows any dynamic range and computes signal power correctly
def cal_psnr_new(im1, im2, vh=255):
    mse  = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    spow = vh ** 2

    psnr = 10 * np.log10(spow / mse) 
    return psnr


# VS: Did not change. Not too important, works as along as pixel range is 0-1.
def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))
