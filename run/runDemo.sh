#!/bin/bash

# This is a script for running the "sv-mbirct" program for 
# parallel beam computed tomography. Specifically, it will 
# reconstruct sample data available for download at 
# "http://github.com/sjkisner/mbir-demos.git".
#
# More information on the command line usage can be found
# in the Readme file, and also by running "./mbir_ct -help".

export OMP_NUM_THREADS=20
export OMP_DYNAMIC=true

# Changes directory to where run-script (represented here by $0) is
cd "$(dirname $0)"

### Set executable and data locations
execdir="../bin"

# Default data-directory (for all datasets) and dataset-name. 
# This can also be altered through additional arguments in command line.
dataDir="../data"
dataName="shepp"

# Here $# means no. of arguments (excluding run-script). 
# If no. of arguments > 0, then extra argument is of form "path-for-all-datasets/dataset-name" 
if [[ "$#" -gt 0 ]]; then
  dataDir="$(dirname $1)" 
  dataName="$(basename $1)"
fi

# Path for parameter files
parName="$dataDir/$dataName/par/$dataName"
# Path for sinogram data
sinoName="$dataDir/$dataName/sino/$dataName"
wgtName="$dataDir/$dataName/weight/$dataName"
# Path for reconstruction
recName="$dataDir/$dataName/recon/$dataName"
# Path for system-matrix
matDir="$dataDir/sysmatrix"
matName="$matDir/$dataName"


#---If using CNN as a prior model provide the following info--
# Path for TensorFlow (TF) checkpoint files
TF_ckpt_dir="../data/TF_checkpoint" 
# TensorFlow args
# Required: -c to indicate TF checkpoint directory  
# Optional: -g to disable GPU acceleration, -d to use specific TF checkpoint file instead of most recent
# If -d is included, then -l needed to indicate epoch number for specifying TF checkpoint file 
TF_args="-c ${TF_ckpt_dir}"


# create folders that hold pre-computed items and output if they don't exist
# -d usage: check if directory exists
if [[ ! -d "$matDir" ]]; then
  echo "Creating directory $matDir"
  mkdir "$matDir"
fi
if [[ ! -d "$(dirname $recName)" ]]; then
  echo "Creating directory $(dirname $recName)" 
  mkdir "$(dirname $recName)" 
fi

### Compute reconstruction

### Form 1: Reconstruct with a single call (uncomment the next two lines to use)
# $execdir/mbir_ct -i $parName -j $parName -k $parName -s $sinoName \
#    -w $wgtName -r $recName -m $matName -e $matName
# exit 0

### Form 2: Pre-compute system matrix and initial projection and write to file.
###   Then reconstruct. First call only has to be done once for a given set of
###   image/sinogram dimensions--resolution, physical size, offsets, etc.
# $execdir/mbir_ct -i $parName -j $parName -m $matName -f $matName
# $execdir/mbir_ct -i $parName -j $parName -k $parName -s $sinoName \
#    -w $wgtName -r $recName -m $matName -e $matName
# exit 0

### Form 3: The code below checks if the matrix for the input problem 
###   dimensions was previouly computed and saved in the $matDir folder. If no,
###   it computes and saves the matrix; If yes, it skips to the reconstruction.

### PRE-COMPUTE STAGE

# In below script "$?" means exit status. General use is as follows
#run_some_command
#EXIT_STATUS=$?
#if [ "$EXIT_STATUS" -eq "0" ]
#then
    ## Do work when command exists on success
#else
    ## Do work for when command has a failure exit
#fi

# generate the hash value and check the exit status
# The below line invokes a separate run-script on its own
HASH="$(./genMatrixHash.sh $parName)"
if [[ $? -eq 0 ]]; then
   matName="$matDir/$HASH"
else
   echo "Matrix hash generation failed. Can't read parameter files?"
   [[ -f "$matName.2Dsvmatrix" ]] && /bin/rm "$matName.2Dsvmatrix" 
fi

# check for matrix file (-f usage), and compute if not present
if [[ ! -f "$matName.2Dsvmatrix" ]]; then
   echo "Generating system matrix file: $matName.2Dsvmatrix"
   echo "Generating projection file: $matName.2Dprojection"
   $execdir/mbir_ct -i $parName -j $parName -m $matName -f $matName
else
   echo "System matrix file found: $matName.2Dsvmatrix"
fi

touch $matName.lastused

### RECONSTRUCTION STAGE

$execdir/mbir_ct -i $parName -j $parName -k $parName -s $sinoName -w $wgtName \
   -r $recName -m $matName -e $matName ${TF_args}
#   2>&1 | tee $(dirname $recName)/out

exit 0


