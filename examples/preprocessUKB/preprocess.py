import nibabel as nib
import numpy as np
import os
import sys

from nilearn.maskers import NiftiMasker
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nipype.interfaces import fsl
from nipype.interfaces.fsl.maths import ApplyMask
from nipype.testing import example_data
from sklearn.covariance import GraphicalLassoCV
from sklearn.covariance import GraphLassoCV as GraphicalLassoCV
def preprocess_ukb(
    path_atlas_file: str,
    path_normalize_refrence_file:str,
    path_mask_file:str,
    path_out: str,
    participants_path: str,
):
    """
    Preprocess UKB data and create connectivity matrix.

    Parameters:
    path_atlas_file (str): Path to atlas file.
    path_normalize_refrence_file (str): Path to the reference file for normalization.
    path_mask_file (str): Path to the mask file.
    path_out (str): Path to the output directory.
    participants_path (str): Path to the directory containing participant data.

    Returns:
    None
    """

    participants = [folder for folder in os.listdir(participants_path) if os.path.isdir(os.path.join(participants_path, folder))]

    atlas = nib.load(path_atlas_file)
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize='zscore',
        detrend=True,
        memory="nilearn_cache",
        verbose=5,
    )
    connectome_measure = ConnectivityMeasure(kind='correlation')
    for p in participants:
        # initialize some file path
        path_input_file = os.path.join(participants_path, p,"filtered_func_data_clean_MNI.nii.gz")
        path_warp_file = os.path.join(participants_path, p,"example_func2standard_warp.nii.gz")
        path_output_file = os.path.join(participants_path, p,"filtered_func_data_clean_MNI.nii.gz")
        path_conn_result = os.path.join(path_out, p+".csv")
        # Nomalization step1: applywarp
        aw = fsl.ApplyWarp()
        aw.inputs.in_file = example_data(path_input_file)
        aw.inputs.ref_file = example_data(path_normalize_refrence_file)
        aw.inputs.field_file = example_data(path_warp_file)
        aw.inputs.interp = 'spline'
        aw.inputs.out_file = path_output_file
        applywarp_result = aw.run() 
        # Nomalization step2: fslmaths
        fm = fsl.maths().ApplyMask()
        fm.inputs.in_file = example_data(path_output_file)
        fm.inputs.mask_file =example_data(path_mask_file)
        fm.inputs.out_file = path_output_file
        #compute connectivity matric
        img = img = nib.load(path_output_file)
        time_series = masker.fit_transform(img)
        estimator = GraphicalLassoCV()
        estimator.fit(time_series)
        correlation_matrices = estimator.covariance_
        np.savetxt(path_conn_result, estimator.covariance_,delimiter=",")
