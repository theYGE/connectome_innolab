# connectome_innolab_preProcess

## Files
As the "filtered_func_data_clean.nii.gz" file is too large, so please add the nii file manually into the following image_file path. The Atlas file is already exist in the repository.
### Input

```bash
image_file = "./Data/UKBiobank/filtered_func_data_clean.nii.gz"
Atlas_file = "./Data/Atlas/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
```

### Output

```bash
connectivity_output = "./Data/Connectivity_Output/output_sample.csv"
```

## How to run the code?
- first: pleas compile the code in vscode and go into the r terminal.
```bash
Example: source("~/preProcess/atlas_based_connectivity_computation.R", encoding = "UTF-8")
```
- second: run the following command 

```bash
FC_computation_atlas_based(image_file, Atlas_file, Fisher_z_transform, Mask_atlas_with_GM, GM_mask, detrend, bandpass, TR, motion_regression, motion_file, WM_nuisance, WM_mask, CSF_nuisance,CSF_mask)
```
