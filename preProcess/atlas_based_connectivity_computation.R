# This function computes atlas based functional connectivity 
# output path=
connectivity_output = "./Data/Connectivity_Output/output_sample.csv"
# required inputs are
### Main input ###
# image_file = path to a nifti image (string, indicate the filtered_func_data_clean.nii.gz file)
image_file = "./Data/UKBiobank/filtered_func_data_clean.nii.gz"
# Atlas_file = path to a nifti image (string), where ROIs are specified via numeric indices (i.e. in an atlas including 100 ROIs, ROI indices are coded from 1:100)
# Current Atlas is 1mm, 7-Networks, 400 parcels 
Atlas_file ="./Data/Atlas/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
# Fisher_z_transform = boolean (T or F), indicates whether the Pearson-Moment functional connectivity matrix should be Fisher-z transformed
Fisher_z_transform = F
### Atlas preprocessing ###
# Mask_atlas_with_GM = boolean (T or F)
# => if T => GM_mask = path to GM mask nifti (string)
Mask_atlas_with_GM = F
GM_mask=""
### fMRI preprocessing ###
## filtering
# detrend = boolean (T or F), indicates whether linear trend should be removed from the images
detrend = F
# bandpass = boolean (T or F), indicates whether data should be band-pass filtered. The pre-set filter is set to 0.01-0.8 Hz (Usually use 0.8 Hz)
# => if T => TR = numeric value indicating the TR in seconds
bandpass = T
TR = 0.8
# motion_regression = boolean (T or F), indicates whether motion parameters should be regressed out from ROI specific timecourses
# => if T => motion_file = path to .txt file with motion parameters (string) organized as rows=volumes, columns=translations/rotations etc. //motion correction-> values of motion backend
motion_regression = F
motion_file = ""
# WM_nuisance = boolean (T of F), indicates whether White matter signal should be regressed out from ROI specific timecourses 
# => if T => WM_mask = path to WM mask nifti (string) (White matter brain regions)
WM_nuisance = F
WM_mask =""
# CSF_nuisance = boolean (T of F), indicates whether CSF signal should be regressed out from ROI specific timecourses
# => if T => CSF_mask = path to CSF mask nifti (string) (water regions)
CSF_nuisance = F
CSF_mask = ""




FC_computation_atlas_based <- function(image_file, Atlas_file, Fisher_z_transform, Mask_atlas_with_GM, GM_mask, detrend, bandpass, TR, motion_regression, motion_file, WM_nuisance, WM_mask, CSF_nuisance,CSF_mask){
  
  # Probably need to install the following files in your local machine with command line install.packages("packages name")
  require(neurobase)
  require(RSEIS)
  require(plyr)
  require(dplyr)
  require(psych)
  require(RNifti) 
  require(tictoc)
  
  # load image
  print("load image")  
  tic()
  image=RNifti::readNifti(image_file)
  toc()
  
  # load atlas
  print("load atlas")
  tic()
  Atlas=RNifti::readNifti(Atlas_file)
  ROIs <- unique(as.vector(Atlas))
  ROIs <- ROIs[which(ROIs!=0)]
  ROIs <- sort(ROIs)
  print(paste0("number of ROIs ", length(ROIs)))
  
  if (Mask_atlas_with_GM==T){
    GM_mask_readin=RNifti::readNifti(GM_mask)
    Atlas=Atlas*GM_mask_readin
  }
  Atlas_vectorized=Atlas[Atlas!=0]
  toc()
  
  
  #### Step 2: transform image to matrix ####
  # determine rows and columns of matrix
  dim_rows=length(Atlas_vectorized)
  dim_cols=dim(image)[4]
  image_matrix=matrix(NA, dim_rows, dim_cols)
  
  # transform image to matrix
  tic()
  for (i in 1:(dim(image)[4])){
    # volume_vectorized=image[,,,i]*(Atlas!=0)
    volume_vectorized=as.vector(image[,,,i])*(Atlas!=0)
    image_matrix[,i]=volume_vectorized[Atlas!=0]
  } 
  toc()
  
  #### Step 3: compute median TS per ROI ####
  print("extracting ROI timeseries")
  tic()
  # determine rows and columns ROI of matrix
  dim_rows=length(ROIs)
  dim_cols=dim(image)[4]
  current_TS_ROI=matrix(NA, dim_rows, dim_cols)
  n=0
  for (i in 1:length(ROIs)){
    n=n+1
    current_ROI=ROIs[i]
    current_voxels=which(Atlas_vectorized==current_ROI)
    
    if (length(current_voxels>1)){current_TS_ROI[i,]=colMeans(image_matrix[current_voxels,], na.rm=T)}
    # if (length(current_voxels==1)){current_TS_ROI[i,]=image_matrix[current_voxels,]}
    if (length(current_voxels==1)){ifelse(is.na(current_TS_ROI[i,]), image_matrix[current_voxels,], current_TS_ROI[i,])}
    
    
  }
  toc()
  
  
  input_TS=current_TS_ROI
  
  #### Step 4: detrend ####
  if(detrend==T){
    print("detrending")
    output_TS=matrix(0, dim(input_TS)[1], dim(input_TS)[2])
    n=0
    for (i in 1:length(ROIs)){
      n=n+1
      output_TS[i,]=detrend(input_TS[i,])
    }
    input_TS=output_TS
  }
  
  #### Step 5: band-pass filter ####
  if(bandpass==T){
    print("band-pass filtering")
    output_TS=matrix(0, dim(input_TS)[1], dim(input_TS)[2])
    n=0
    for (i in 1:length(ROIs)){
      output_TS[i,]=butfilt(input_TS[i,], fl=0.01, fh=0.08, deltat=TR, type="BP")
    }
    input_TS=output_TS 
  }
  
  #### Step 6: extract nuisance timeseries #### 
  # white matter
  tic()
  if (WM_nuisance==T){
    print("Prepare WM nuisance regression")
    WM_mask_readin=RNifti::readNifti(WM_mask)
    WM_mask_vectorized=WM_mask_readin[WM_mask_readin!=0]
    WM_TS=numeric()
    # transform image to matrix
    
    for (i in 1:(dim(image)[4])){
      volume_vectorized=image[,,,i]*(WM_mask_readin!=0)
      WM_TS[i]=mean(volume_vectorized[WM_mask_readin!=0], na.rm=T)
    } 
  }
  toc()
  
  # CSF
  tic()
  if (CSF_nuisance==T){
    print("Prepare CSF nuisance regression")
    CSF_mask_readin=RNifti::readNifti(CSF_mask)
    CSF_mask_vectorized=CSF_mask_readin[CSF_mask_readin!=0]
    CSF_TS=numeric()
    # transform image to matrix
    
    for (i in 1:(dim(image)[4])){
      volume_vectorized=image[,,,i]*(CSF_mask_readin!=0)
      CSF_TS[i]=mean(volume_vectorized[CSF_mask_readin!=0], na.rm=T)
    } 
  }
  toc()
  
  # Motion
  if (motion_regression==T){
    print("Prepare Motion regression")
    motion_covariates=read.table(motion_file)
  }
  
  #### Step 7: concatenate nuisance regressors #### 
  # all three nuisance variables
  if(motion_regression==T & CSF_nuisance==T & WM_nuisance==T){
    print("concatenate motion, CSF and WM timeseries")
    nuisance_covariates=cbind(motion_covariates, CSF_TS, WM_TS)
  }
  
  # two nuisance variables
  if(motion_regression==T & CSF_nuisance==T & WM_nuisance==F){
    print("concatenate motion and CSF")
    nuisance_covariates=cbind(motion_covariates, CSF_TS)
  }
  
  if(motion_regression==T & CSF_nuisance==F & WM_nuisance==T){
    print("concatenate motion and WM")
    nuisance_covariates=cbind(motion_covariates, WM_TS)
  }
  
  if(motion_regression==F & CSF_nuisance==T & WM_nuisance==T){
    print("concatenate CSF and WM")
    nuisance_covariates=cbind(CSF_TS, WM_TS)
  }
  
  # one nuisance variable
  if(motion_regression==T & CSF_nuisance==F & WM_nuisance==F){
    print("use motion only")
    nuisance_covariates=motion_covariates
  }
  
  if(motion_regression==F & CSF_nuisance==T & WM_nuisance==F){
    print("use CSF only")
    nuisance_covariates=CSF_TS
  }
  
  if(motion_regression==F & CSF_nuisance==F & WM_nuisance==T){
    print("use WM only")
    nuisance_covariates=WM_TS
  }
  
  
  #### Step 8: regress out nuisance variables ####
  if((motion_regression+CSF_nuisance+WM_nuisance)>0){
    print("Perform Nuisance Regression")
    output_TS=matrix(0, dim(input_TS)[1], dim(input_TS)[2])
    n=0
    for (i in 1:length(ROIs)){
      current_TS=data.frame(ROI_TS=input_TS[i,])
      current_TS_incl_nuisance=cbind(current_TS, nuisance_covariates)
      regr_model=lm(current_TS_incl_nuisance)
      output_TS[i,]=regr_model$residuals
    }
    input_TS=output_TS 
  }
  
  #### Step 9: compute connectivity matrix ####
  print("Compute Functional Connectivity")
  
  input_TS_transpose=t(input_TS)
  FC_matrix=cor(input_TS_transpose)
  rownames(FC_matrix) <- ROIs
  colnames(FC_matrix) <- ROIs
  
  if(Fisher_z_transform==T){
    print("Apply Fisher z transformation")
    zFC_matrix=fisherz(FC_matrix)
    # return(zFC_matrix)
    write.csv(zFC_matrix, file = "output_sample.csv", row.names = FALSE)
  }
  
  if(Fisher_z_transform==F){
    # return(FC_matrix)
    write.csv(FC_matrix, file = connectivity_output, row.names = FALSE)
  }
}


