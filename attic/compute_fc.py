import logging
from utils import log_function_call
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nibabel as nb
from bids import BIDSLayout
# from nilearn import plotting, datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.maskers import NiftiLabelsMasker


@log_function_call
def compute_fc(path_atlas: str, path_out: str, participants: [str], low_pass_freq: float = 0.08, tr: float = 0.08,
               fwhm_smoothing: float = 8) -> None:
    """
    do fmriprep pipeline

    Based on https://nilearn.github.io/stable/connectivity/index.html and https://carpentries-incubator.github.io/SDC-BIDS-fMRI/aio/index.html
    No high-pass filter or detrend included in nilearn masker, as low-frequency predictors (cosine_XX) are included in load_confounds_strategy 'simple'.
    See https://github.com/SIMEXP/load_confounds or
    https://fmriprep.org/en/stable/outputs.html#confound-regressors-description or
    https://nilearn.github.io/stable/modules/generated/nilearn.interfaces.fmriprep.load_confounds_strategy.html
    fmriprep does not output cosines for low pass filtering, hence low_pass and t_r are passed to nilearn masker.

    TODO:
        participants
    """
    np.seterr(divide='ignore')
    layout = BIDSLayout(path_out, config=['bids', 'derivatives'])
    participants = [""]
    logging.info("\nSubject IDs: ", layout.get_subjects(), "\nn = ", len(layout.get_subjects()), "\nConditions: ",
                 layout.get_tasks())
    atlas = nb.load(path_atlas)
    masker = NiftiLabelsMasker(labels_img=atlas, memory="nilearn_cache", low_pass=low_pass_freq,
                               t_r=tr, standardize=True, smoothing_fwhm=fwhm_smoothing, verbose=0).fit()
    correlation_measure = ConnectivityMeasure(kind="correlation")
    for i, subj in enumerate(participants):
        logging.info("-----------------------------\nParticipant: ", subj, i, "/", len(participants))
        func_files = layout.get(subject=subj, datatype="func", desc="preproc", space="MNI152NLin2009cAsym",
                                extension="nii.gz", return_type="file")
        logging.info("\nLoading pre-processed fMRI file: ", func_files[0])
        img = nb.load(func_files[0])
        confounds, sample_mask = load_confounds_strategy(func_files[0], denoise_strategy="simple", motion="basic")
        logging.info("\nLoading confounds: ", list(confounds.head()))
        logging.info("\nFitting to atlas")
        time_series = masker.transform(img, confounds=confounds, sample_mask=sample_mask)
        logging.info("\nCalculating FC and Fisher-z transform")
        corr = np.row_stack(correlation_measure.fit_transform([time_series]))
        corr_fisher = pd.DataFrame(np.arctanh(corr))
        logging.info(corr_fisher)
