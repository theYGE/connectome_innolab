# connectome_innolab_preprocess_ukb_complete
This script preprocesses large resting-state functional magnetic resonance imaging (fMRI) data and creates a connectivity matrix using the Nilearn and Nipype libraries. The following instructions will help you use this script.
## Requirements
* Python 3.7 or higher
* Nilearn>=0.9.2
* Numpy>=1.22.4
* Nipype>=1.8.5
* FSL in your local machine(If you want to run the whole preprocess step than FSL is necessary to be installed on your local machine,please follow the intall step [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation#Installing_FSL).
## Input
The preprocess.py script requires the following inputs:
* path_atlas_file: Path to the atlas file.
```bash
path_atlas_file = "../../data/atlas/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
```
* path_normalize_refrence_file: Path to the reference file for normalization.
```bash
path_normalize_refrence_file = "../../data/mask/MNI152_T1_2mm_brain.nii.g"
```
* path_mask_file: Path to the mask file.
```bash
path_normalize_refrence_file = "../../data/mask/MNI152_T1_2mm_brain_bin.nii.g"
```
* path_out: Path to the output directory.
```bash
path_out = "../../data/example_preprocess_ukb_output"
```

* participants_path: Path to the directory containing participants' data.
***To be noticed***: there is no file inside the ***participants_path*** folder currently due to the security reason. If you would like to test the preprocess data please download the data from large fMRI datset and unzip each particpants' "***filtered_func_data_clean.nii.gz***" file and "***example_func2standard_warp.nii.gz***" file to ***participants_path/test** folder.
```bash
participants_path = "../../data/Participants/"
```
## Usage
To use the script, follow these steps:<br />
    - Install all required dependencies.<br />
    - Run the script using the following command:<br />
```bash
python preprocess_ukb.py path_atlas_file path_normalize_refrence_file path_mask_file path_out participants_path

```
Replace path_atlas_file, path_normalize_refrence_file, path_mask_file, path_out, and participants_path with the corresponding file paths in above.
## Output

The script outputs a connectivity matrix for each participant in CSV format, connectivity matrix figure and a report.html file. The output directory (path_out) will contain the 3 products for each participant.
