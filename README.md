# sv-mbirct

### HIGH PERFORMANCE MODEL BASED IMAGE RECONSTRUCTION (MBIR) </br> FOR PARALLEL-BEAM COMPUTED TOMOGRAPHY WITH ADVANCED PRIOR MODELS
*Optimized for Intel multi-core processors*

Demo scripts and data files for running this program are available under this repository.

Further references on MBIR and the technology in the high-performance implementation of this
code can be found at the bottom of this page, in the documentation accompanying the OpenMBIR
project and on Charles Bouman's website:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
https://github.com/cabouman/OpenMBIR  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
http://engineering.purdue.edu/~bouman/publications/pub_tomography.html  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
http://engineering.purdue.edu/~bouman/publications/pub_security.html  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
http://engineering.purdue.edu/~bouman/publications/pdf/MBIP-book.pdf

## SYSTEM REQUIREMENTS

1. Intel-based CPU(s)
2. Intel "icc" compiler (included in "Parallel Studio XE", available from Intel for Linux, macOS)

For using advanced prior models (BM3D and Deep Resnet CNN) the following Python version and libararies are required:
1. Python 3.5 (recommended) or greater
2. numpy
3. scipy
4. matplotlib
5. pillow
6. pywavelets
7. tensorflow-gpu or tensorflow 

## COMPILING

From a terminal prompt, enter the *src* folder and type *make*. If
compiling is successful the binary *mbir_ct* will be created and moved into
the *bin* folder. 
```
cd src  
make
```
Note: In case of compilation issues when using Parallel Studio XE, refer to
https://github.com/HPImaging/sv-mbirct


## RUNNING

For local computer: After compilation, from the terminal prompt enter the *run* folder and type *./runDemo.sh*
```
cd ../run
./runDemo.sh
```

For submitting job on the Purdue cluster: After compilation, from the terminal prompt enter the *run* folder and submit the job via a submission script.
```
cd ../run
sbatch jobrun.sub
```
The next section explains the parameter and data files required for running the MBIR algorithm.
The demo-script thoroughly explains how the various files must be specified via command line.
To print the usage statement for the command line, from the *bin* directory type:
```
./mbir_ct -help
```

## DESCRIPTION OF PARAMETER AND DATA FILES

The following 4 parameter files are required, all in simple text:

     <basename>.sinoparams  
     <basename>.imgparams  
     <basename>.reconparams  
     <basename>.priorparams
     <view_angles_file.txt>

While it is recommended to use the same *basename* for all parameter files, 
note that these input filenames are independent as they're specified in different
arguments in the command line.

However, the [.reconparams] and [.priorparams] files must share the same basename. 
The [.reconparams] specifies high level parameters pertaining to the iterative MBIR algorithim, including the choice of prior model.
The [.priorparams] file specifies parameters for the prior-model selected in the [.reconparams] file.

For the files containing sinogram or image data,
the associated 3D data is split across files, one file per slice.
The naming convention for the different files is as follows:

     <basename>_sliceNNN.2Dimgdata
     <basename>_sliceNNN.2Dsinodata
     <basename>_sliceNNN.2Dweightdata
     <basename>_sliceNNN.2Dprojection

where "NNN" is a slice index. The slice indices must be a non-negative 
integer sequence, where the indices include leading zeros and no 
spaces (e.g. 0000 to 0513). 
The number of digits used for the slice indices is flexible (up to 4) 
but must be consistent across all the files used in a given reconstruction call.

Note that the accompanying demo scripts include a utility that detects whether
the necessary sytem matrix file has already been computed and is available, 
given the input image/sino parameters, and the script automatically reads
the file if available, or computes/stores it if not.

## References

##### Xiao Wang, Amit Sabne, Putt Sakdhnagool, Sherman J. Kisner, Charles A. Bouman, and Samuel P. Midkiff, "Massively Parallel 3D Image Reconstruction," *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC'17)*, November 13-16, 2017. (One of three finalists for 2017 ACM Gordon Bell Prize.)

##### Amit Sabne, Xiao Wang, Sherman J. Kisner, Charles A. Bouman, Anand Raghunathan, and Samuel P. Midkiff, "Model-based Iterative CT Imaging Reconstruction on GPUs," *22nd ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '17)*, February 4-8, 2017.

##### Xiao Wang, K. Aditya Mohan, Sherman J. Kisner, Charles Bouman, and Samuel Midkiff, "Fast voxel line update for time-space image reconstruction," *Proceedings of the IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP)*, pp. 1209-1213, March 20-25, 2016.

##### Xiao Wang, Amit Sabne, Sherman Kisner, Anand Raghunathan, Charles Bouman, and Samuel Midkiff, "High Performance Model Based Image Reconstruction," *21st ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '16)*, March 12-16, 2016. 

##### Suhas Sreehari, S. Venkat Venkatakrishnan, Brendt Wohlberg, Gregery T. Buzzard, Lawrence F. Drummy, Jeffrey P. Simmons, and Charles A. Bouman, "Plug-and-Play Priors for Bright Field Electron Tomography and Sparse Interpolation," *IEEE Transactions on Computational Imaging*, vol. 2, no. 4, Dec. 2016. 

##### Jean-Baptiste Thibault, Ken Sauer, Charles Bouman, and Jiang Hsieh, "A Three-Dimensional Statistical Approach to Improved Image Quality for Multi-Slice Helical CT," *Medical Physics*, pp. 4526-4544, vol. 34, no. 11, November 2007.
