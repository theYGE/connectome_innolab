"""
This script assumes all fc matrices are stored as one csv file in CORR_MAT_FOLDER.
To imitate BIDS structure it creates a new folder where for each subject/fc matrix a separate folder is created.
In each file one corresponding fc-mat will be saved.
"""
import os
import re
import shutil

import numpy as np

SCRIPT_LOCATION = os.getcwd()  # HOME-DIR/attic
PROJECT_FOLDER = os.path.dirname(SCRIPT_LOCATION)  # HOME-DIR
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
CORR_MAT_FOLDER = os.path.join(DATA_FOLDER, "corr_mat")
BIDS_FOLDER = os.path.join(DATA_FOLDER, "bids_struct")

if not os.path.isdir(BIDS_FOLDER):
    os.mkdir(BIDS_FOLDER)

if os.path.isdir(CORR_MAT_FOLDER):
    files = os.listdir(CORR_MAT_FOLDER)
    for file in files:
        filename = os.path.splitext(file)[0]
        filename = re.findall(r"\d+", filename)
        filename = filename.pop(0)  # re.findall returns list
        filename = "subject_" + filename
        foldername = os.path.join(BIDS_FOLDER, filename)
        if not os.path.isdir(foldername):
            os.mkdir(foldername)
        destination = os.path.join(foldername, file)
        source = os.path.join(CORR_MAT_FOLDER, file)
        # print(f"old location: {source}")
        # print(f"new location: {destination}")
        shutil.copy(source, destination)
