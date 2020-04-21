"""
Grayscale BM3D denoising demo file, based on Y. Mäkinen, L. Azzari, A. Foi, 2019.
Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise.
In IEEE International Conference on Image Processing (ICIP), pp. 185-189
"""

import sys
sys.path.insert(0,'../src') # wrt runDemo.sh

import numpy as np
from PIL import Image
from bm3d import bm3d, BM3DProfile
#from experiment_funcs import get_experiment_noise, get_psnr, get_cropped_psnr
import getopt, sys 

import matplotlib
matplotlib.use('Agg') #for display on remote node
import matplotlib.pyplot as plt

def read_image_3D(fname_base, FirstSliceNumber, SliceNumDigits, Nz, Ny, Nx):
    image = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    for i in range(Nz):
        fname = '%s_slice%0*d.2Dimgdata' % (fname_base, SliceNumDigits, FirstSliceNumber+i)
        f = open(fname, 'rb')
        data = np.fromfile(f, '<f4')
        data = np.reshape(data, ((Ny,Nx)) )
        image[i] = data
        f.close()
    return image    


def write_image_3D(image, fname_base, FirstSliceNumber, SliceNumDigits):
    Nz, Ny, Nx = image.shape
    for i in range(Nz):
        fname = '%s_slice%0*d.2Dimgdata' % (fname_base, SliceNumDigits, FirstSliceNumber+i)
        f = open(fname, 'wb')
        data = image[i].flatten()
        data.tofile(f)
        f.close()

def display_image_3D(image, fname_base, FirstSliceNumber, SliceNumDigits):
    Nz, Ny, Nx = image.shape
    for i in range(Nz):
        data = image[i]
        plt.title('%s_slice%0*d.2Dimgdata'%(fname_base, SliceNumDigits, FirstSliceNumber+i))
        plt.imshow(data, cmap='gray')
        plt.colorbar()
        plt.show(block=False)
        plt.savefig('%s_slice%0*d.pdf'%(fname_base, SliceNumDigits, FirstSliceNumber+i), format='pdf')
        plt.clf()

def bm3d_denoise_slicewise(image, FirstSliceNumber, SliceNumDigits, Sigma_n, vl, vh):
    Nz, Ny, Nx = image.shape
    data_out   = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    for i in range(Nz):
        data  = np.atleast_3d( ((image[i]-vl)/(vh-vl)) ) #scale to (0,1)
        data  = np.minimum(np.maximum(data, 0), 1)       #clip
        data_out[i] = bm3d(data, Sigma_n)                #denoise

    data_out  =  np.minimum(np.maximum(data_out, 0), 1) #clip
    data_out  =  data_out*(vh-vl)+vl                    #re-scale    
    return data_out

def read_cmdline(argumentList):
    # Options 
    options = "hf:l:u:s:z:y:x:i:d:"
      
    # Long options (in this example --Help, ...)
    long_options = ["Help", "FileName", "LowerBound", "UpperBound", "Sigma_n", "Nz", "Ny", "Nx", "FirstSliceNumber", "SliceNumDigits"] 
      
    try: 
        # Parsing argument 
        opts, args = getopt.getopt(argumentList, options, long_options) 
          
        # checking each argument 
        for opt, arg in opts: 
      
            if opt in ("-h", "--Help"): 
                print ("Diplaying command line format: -f <FileName> -l <LowerBound> -u <UpperBound> -s <Sigma_n> \
                 -z <Nz> -y <Ny> -x <Nx> -i <FirstSliceNumber> -d <SliceNumDigits>\n") 
                  
            elif opt in ("-f", "--FileName"): 
                fname_base = arg

            elif opt in ("-l", "--LowerBound"):
                LowerBound = float(arg)

            elif opt in ("-u", "--UpperBound"):
                UpperBound = float(arg)

            elif opt in ("-s", "--Sigma_n"):
                Sigma_n = float(arg)/255.0    

            elif opt in ("-z", "--Nz"):
                Nz = int(arg)  

            elif opt in ("-y", "--Ny"):
                Ny = int(arg)

            elif opt in ("-x", "--Nx"):
                Nx = int(arg)  

            elif opt in ("-i", "--FirstSliceNumber"):
                FirstSliceNumber = int(arg)

            elif opt in ("-d", "--NumSliceDigits"):
                SliceNumDigits = int(arg)  
              
    except getopt.error as err: 
        # output error, and return with an error code 
        print (str(err)) 

    return fname_base, LowerBound, UpperBound, Sigma_n, Nz, Ny, Nx, FirstSliceNumber, SliceNumDigits

def main():
    # Experiment specifications
    argumentList = sys.argv[1:] 
    filename_base, vl, vh, Sigma_n, Nz, Ny, Nx, FirstSliceNumber, SliceNumDigits = read_cmdline(argumentList)

    # read image
    image_noisy=read_image_3D(filename_base+'_in', FirstSliceNumber, SliceNumDigits, Nz, Ny, Nx)

    #denoise
    image_clean = bm3d_denoise_slicewise(image_noisy, FirstSliceNumber, SliceNumDigits, Sigma_n, vl, vh)    

    #display images
    #display_image_3D(image_noisy, filename_base+'_noisy', FirstSliceNumber, SliceNumDigits)
    #display_image_3D(image_clean, filename_base+'_clean', FirstSliceNumber, SliceNumDigits)

    #write image 
    write_image_3D(image_clean, filename_base+'_out', FirstSliceNumber, SliceNumDigits)

if __name__ == '__main__':
    main()

