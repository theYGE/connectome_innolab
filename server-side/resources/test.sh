#!/bin/bash
#SBATCH --job-name=task_unzip_2
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH --array=[0-4]%20
#SBATCH -o %x_%A_%a.out

# module load ants/2.3.4
# module load fsl/6.0.3

inputPath="/NASRauch/mini_data/"
outfolder="mini_data"
reference="template/MNI152_T1_2mm_brain.nii.gz"
mask="template/MNI152_T1_2mm_brain_mask_bin.nii.gz"


echo "host name : " `hostname`
echo This is array task number $SLURM_ARRAY_TASK_ID

# create an array variable containing all the file names
FILES=($(ls $inputPath))

echo $FILES
# get specific file name, assign it to $file
file=${FILES[$SLURM_ARRAY_TASK_ID]}

#step1: unzip the files "filtered_func_data_clean.nii.gz" and "example_func2standard_warp.nii.gz"
mkdir $outfolder"/"${file%.*}
# unzip -j $inputPath"/"$file fMRI/rfMRI.ica/filtered_func_data_clean.nii.gz -d $outfolder"/"${file%.*}
# unzip -j $inputPath"/"$file fMRI/rfMRI.ica/reg/example_func2standard_warp.nii.gz -d $outfolder"/"${file%.*}

#step2: normalize file "filtered_func_data_clean.nii.gz" using package ants and package fsl 
# workPath="$outfolder"/"${file%.*}"
# input="$workPath/filtered_func_data_clean.nii.gz"
# output="$workPath/filtered_func_data_clean_MNI.nii.gz"
# warp="$workPath/example_func2standard_warp.nii.gz"



# antsApplyTransforms -r $reference -i $input -o $output -t $warp -n bSpline -d 4


# fslmaths $output -mas $mask $output
#step3: apply the R file


# write the input filenames to the backlogs.
echo $file 
echo Files $file were aligned by task number $SLURM_ARRAY_TASK_ID on $(date)
