import logging
import os
from urllib.request import urlretrieve

import openneuro as on
from utils import log_function_call


@log_function_call
def load_data(dataset_name: str, target_dir: str) -> None:
    """
    load dataset from openFMRI

    TODO:
        check if data already exists so no additional (unnecessary) download is triggered
    """
    if not os.path.isdir(target_dir):
        logging.info(f"Download data to {target_dir}...")
        on.download(dataset=dataset_name, target_dir=target_dir)
        logging.info("data download finished")
    else:
        logging.info(f"{target_dir} directory already exists. No download triggered")


@log_function_call
def check_atlas(
    atlas_dir: str,
    atlas_url: str = "https://github.com/ThomasYeoLab/CBIG/blob/a8c7a0bb845210424ef1c46d1435fec591b2cf3d/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz?raw=true",
    atlas_name: str = "Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz",
) -> None:
    """
    uses:
    https://github.com/ThomasYeoLab/CBIG/blob/a8c7a0bb845210424ef1c46d1435fec591b2cf3d/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz

    TODO:
        exit code whether atlas file was found
        check for appropriate atlas
    """
    if not os.path.isfile(os.path.join(atlas_dir, atlas_name)):
        logging.info(f"Could not find atlas file in {atlas_dir}. Download atlas...")
        urlretrieve(atlas_url, os.path.join(atlas_dir, atlas_name))
        logging.info("")
    else:
        logging.info(f"atlas file found. Using {atlas_name}")
