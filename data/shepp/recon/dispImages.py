"""
Grayscale BM3D denoising demo file, based on Y. Mäkinen, L. Azzari, A. Foi, 2019.
Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise.
In IEEE International Conference on Image Processing (ICIP), pp. 185-189
"""

import numpy as np
import getopt, sys 
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


def read_cmdline(argumentList):
    # Options 
    options = "hf:z:y:x:i:d:"
      
    # Long options (in this example --Help, ...)
    long_options = ["Help", "FileName", "Nz", "Ny", "Nx", "FirstSliceNumber", "SliceNumDigits"] 
      
    try: 
        # Parsing argument 
        opts, args = getopt.getopt(argumentList, options, long_options) 
          
        # checking each argument 
        for opt, arg in opts: 
      
            if opt in ("-h", "--Help"): 
                print ("Diplaying command line format: -f <FileName> -z <Nz> -y <Ny> -x <Nx> -i <FirstSliceNumber> -d <SliceNumDigits>\n") 
                  
            elif opt in ("-f", "--FileName"): 
                fname_base = arg  

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

    return fname_base, Nz, Ny, Nx, FirstSliceNumber, SliceNumDigits

def main():
    # Experiment specifications
    argumentList = sys.argv[1:] 
    filename_base, Nz, Ny, Nx, FirstSliceNumber, SliceNumDigits = read_cmdline(argumentList)

    # read
    image=read_image_3D(filename_base, FirstSliceNumber, SliceNumDigits, Nz, Ny, Nx)

    #display images
    display_image_3D(image, filename_base, FirstSliceNumber, SliceNumDigits)
 

if __name__ == '__main__':
    main()

