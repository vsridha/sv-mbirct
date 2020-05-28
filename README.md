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

To print a usage statement:

     ./mbir_ct -help

The program is able to run completely with a single command call, but it's 
usually preferrable to run the reconstruction in two stages. In the 
first stage, the sytem matrix is precomputed and stored, and the second
stage performs the actual reconstruction. 
Both stages use the executable *mbir_ct*.
The system matrix can take a significant time to compute,
however the matrix is fixed for a given geometry and data/image 
dimensions, so the matrix file can be reused for any scan that uses the 
same sinogram and image parameters.
The initial stage can also pre-compute 
the forward projection of the default initial condition (constant image)
to save additional time in the reconstruction stage.

(Note the accompanying demo scripts include a utility that detects whether
the necessary sytem matrix file has already been computed and is available, 
given the input image/sino parameters, and the script automatically reads
the file if available, or computes/stores it if not.)

### Stage 1: Compute and store the System Matrix (and initial projection)

    ./mbir_ct
       -i <basename>[.imgparams]     : Input image parameters
       -j <basename>[.sinoparams]    : Input sinogram parameters
    (plus one or more of the following)
       -m <basename>[.2Dsvmatrix]   : Output matrix file
       -f <basename>[.2Dprojection]  : Output projection of default or input IC
       -f <basename>[_sliceNNN.2Dprojection] -t <basename>[_sliceNNN.2Dimgdata]

In the above arguments, the exensions given in the '[]' symbols must be part
of the file names but should be omitted from the command line.
In the last line that includes both -f and -t arguments, the initial 
projection of the initial condition provided by -t is computed and 
saved to a file(s). Further description of data/image filenames is provided
further down.

Examples: (written as if file names have been assigned 
           to variables in a shell script)

To compute/write the system matrix and the projection of the default initial condition:  

    ./mbir_ct -i $parName -j $parName -m $matName -f $matName

To compute/write only the system matrix:  
 
    ./mbir_ct -i $parName -j $parName -m $matName


Similar to the above but the initial projection is computed for the supplied input image (-t):  

    ./mbir_ct -i $parName -j $parName -m $matName -f proj/$imgName -t init/$imgName

The -m option can be omitted if you only want to compute/store the
projection, however the system matrix will need to be computed in any case.


### Stage 2: Compute MBIR Reconstruction

There are files that specify the parameters for MBIR reconstruction: [.imgparams], [.sinoparams], [.reconparams] and [.priorparams]
The [.priorparams] file must have the same filename as the [.reconparams] file. 
The type of prior-model is specified by the latter, based on which relevant fields are parsed from the former during execution. 

    ./mbir_ct
       -i <basename>[.imgparams]           : Input image parameters
       -j <basename>[.sinoparams]          : Input sinogram parameters
       -k <basename>[.reconparams]         : Input reconstruction parameters 
       -s <basename>[_sliceNNN.2Dsinodata] : Input sinogram projection file(s)
       -r <basename>[_sliceNNN.2Dimgdata]  : Output reconstructed image file(s)
    (following are optional)
       -m <basename>[.2Dsvmatrix]          : INPUT matrix (params must match!)
       -w <basename>[_sliceNNN.2Dweightdata] : Input sinogram weight file(s)
       -t <basename>[_sliceNNN.2Dimgdata]  : Input initial condition image(s)
       -e <basename>[_sliceNNN.2Dprojection] : Input projection of init. cond.
                                           : ** default IC if -t not specified
       -f <basename>[_sliceNNN.2Dprojection] : Output projection of final image
       -p <basename>[_sliceNNN.2Dimgdata]  : Proximal map for Plug & Play
                                           : * -p will apply proximal prior
                                           : * generally use with -t -e -f

Examples:

    ./mbir_ct -i $parName -j $parName -k $parName -s $sinoName \
       -w $wgtNname -r $recName -m $matName -e $projName

If either -m or -e are omitted, the corresponding entity (matrix or
projection) will be computed prior to starting the reconstruction.
The default prior model is a q-QGGMRF with a 10-pt 3D neighborhood, unless
the -p argument is included (Plug & Play).


## DESCRIPTION OF PARAMETER AND DATA FILES

For a detailed description of the contents and format for all the data and parameter
files used in this program, see the documentation in the OpenMBIR project
referenced at the top of this readme, directly linked to 
[here](https://github.com/cabouman/OpenMBIR/raw/master/Documentation/MBIR-Modular-specification.docx).
Also see the [demos](https://github.com/sjkisner/mbir-demos)
for specific examples.

The following parameter files are required, all in simple text:

     <basename>.sinoparams  
     <basename>.imgparams  
     <basename>.reconparams  
     <basename>.priorparams
     <view_angles_file.txt>

Note these show the same generic *basename* but the names of all
the input files are independent as they're specified in different
arguments in the command line.

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



## References

##### Xiao Wang, Amit Sabne, Putt Sakdhnagool, Sherman J. Kisner, Charles A. Bouman, and Samuel P. Midkiff, "Massively Parallel 3D Image Reconstruction," *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC'17)*, November 13-16, 2017. (One of three finalists for 2017 ACM Gordon Bell Prize.)

##### Amit Sabne, Xiao Wang, Sherman J. Kisner, Charles A. Bouman, Anand Raghunathan, and Samuel P. Midkiff, "Model-based Iterative CT Imaging Reconstruction on GPUs," *22nd ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '17)*, February 4-8, 2017.

##### Xiao Wang, K. Aditya Mohan, Sherman J. Kisner, Charles Bouman, and Samuel Midkiff, "Fast voxel line update for time-space image reconstruction," *Proceedings of the IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP)*, pp. 1209-1213, March 20-25, 2016.

##### Xiao Wang, Amit Sabne, Sherman Kisner, Anand Raghunathan, Charles Bouman, and Samuel Midkiff, "High Performance Model Based Image Reconstruction," *21st ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '16)*, March 12-16, 2016. 

##### Suhas Sreehari, S. Venkat Venkatakrishnan, Brendt Wohlberg, Gregery T. Buzzard, Lawrence F. Drummy, Jeffrey P. Simmons, and Charles A. Bouman, "Plug-and-Play Priors for Bright Field Electron Tomography and Sparse Interpolation," *IEEE Transactions on Computational Imaging*, vol. 2, no. 4, Dec. 2016. 

##### Jean-Baptiste Thibault, Ken Sauer, Charles Bouman, and Jiang Hsieh, "A Three-Dimensional Statistical Approach to Improved Image Quality for Multi-Slice Helical CT," *Medical Physics*, pp. 4526-4544, vol. 34, no. 11, November 2007.
