#!/bin/bash
#
# This generates a truncated hash label for creating a unique tag for 
# a Sytem Matrix file. It pulls out the relevant fields from the image
# and sinogram sino parameter files including the list of view angles.
#
# usage 1:  ./genMatrixHash.sh <basename>
#    The argument <basename> includes the relative or full path to the files
#       <basename>.imgparams  and  <basename>.sinoparams
#
# usage 2:  ./genMatrixHash.sh <imgparams_filename> <sinoparams_filename>
#    The filenames include the path (relative or full) and extension.
#    This allows different basenames between the two files.
#
# examples: 
#    If there are parameter files "../data/shepp.{imgparams,sinoparams}"
#    Ex.1: ./genMatrixHash.sh ../data/shepp
#    Ex.2: ./genMatrixHash.sh ../data/shepp.imgparams ../data/shepp.sinoparams

# parse arguments. "$#" represents  no. of arguments beside run-script name
if [[ "$#" -lt 1 ]]; then
  echo "Not enough arguments"
  exit 1
elif [[ "$#" -eq 1 ]]; then
  imgfile="$1.imgparams"
  sinofile="$1.sinoparams"
else
  imgfile="$1"
  sinofile="$2"
fi
#echo $imgfile
#echo $sinofile

# check img/sino params file existance
if [[ ! -f "$imgfile" ]] || [[ ! -f "$sinofile" ]]; then
  echo "Can't read params files"
  exit 1
fi

# It's forgiving if sino/img arguments are swapped, HOWEVER the views 
# file path is pulled specifically from the last argument

# The below line is a piped command - executes multiple shell commands seprated by | sequentially
# 1) cat lists the contents of the file line-by-line ($sinofile should suffice $imgile not needed)
# 2) grep shows only those lines of the above file that have a particular word or pattern
# 3) cut splits a given line into 2 parts at each occurence of a delimiter specified by -d option, and, then from all parts selects the one specified by -f
# 4) tr deletes from a string occurence of a delimiter specified by -d 
viewsfile="$(cat $imgfile $sinofile |grep ViewAngle |cut -d : -f 2 |tr -d ' ')"
# File for list of view-angles
viewsfile="$(dirname $sinofile)/$viewsfile"


# check views list file existance 
if [[ ! -f "$viewsfile" ]]; then
  echo "Can't read view list file $viewsfile"
  exit 1
fi

# Pull the relevant parameter fields; remove spaces; sort to remove field order 
# depencence; add views file; generate and truncate sha1 hash
# In below piped command 
# 1) grep -e if u have multiple patterns for which relevant lines must be extracted, specify each of them separated by -e option
# 2) tr -d '[:blank:]' deletes spaces (same as tr -d ' ')
# 3) sort command sorts above lines in alphabetical order
# 4) shasum generates a 'checksum code' based on above contents ...
#    The idea here is each time you make a change to any of the parameters --> you get a unique code
# 5) cut -c 1-8 selects only first 8 characters from the long checksum code that is generated.

cat "$imgfile" "$sinofile" \
  | grep -e Nx -e Ny -e Deltaxy -e ROIRadius -e NChannels -e NViews -e DeltaChannel -e CenterOffset \
  | tr -d '[:blank:]' \
  | sort \
  | cat "$viewsfile" - \
  | shasum -a 1 \
  | cut -c 1-8

exit 0

