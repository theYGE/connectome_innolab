#!/bin/bash

module purge
module load ants/2.3.4
module load fsl/6.0.3
module load freesurfer/7.1.1
module load R/4.0.0

inputPath="fMRI/"
outfolder="output_data"
reference="MNI152_T1_2mm_brain.nii.gz"
mask="MNI152_T1_2mm_brain_mask_bin.nii.gz"


echo "host name : " `hostname`

# create an array variable containing all the file names

# get specific file name, assign it to $file
#file=${FILES[$SLURM_ARRAY_TASK_ID]}

#step1: unzip the files "filtered_func_data_clean.nii.gz" and "example_func2standard_warp.nii.gz"
mkdir $outfolder"/"${file%.*}
unzip ./fMRI/rfMRI.ica/filtered_func_data_clean.nii.gz -d $outfolder"/"${file%.*}
unzip ./fMRI/rfMRI.ica/reg/example_func2standard_warp.nii.gz -d $outfolder"/"${file%.*}

#step2: normalize file "filtered_func_data_clean.nii.gz" using package ants and package fsl 
workPath="$outfolder"/"${file%.*}"
input="$workPath/filtered_func_data_clean.nii.gz"
output="$workPath/filtered_func_data_clean_MNI.nii.gz"
warp="$workPath/example_func2standard_warp.nii.gz"



# antsApplyTransforms -r $reference -i $input -o $output -t $warp -n bSpline -e 3
# fslmaths $output -mas $mask $output
#step3: apply the R file
echo ${file%.*}
Rscript test.R "${file%.*}"

# write the input filenames to the backlogs.
echo $file 
echo Files $file were aligned by task number $SLURM_ARRAY_TASK_ID on $(date)