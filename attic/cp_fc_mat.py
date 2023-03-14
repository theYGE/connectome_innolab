import re

import torch

"""
This script writes fc matrices npy format from a BIDS structure folder into a single folder.
"""
import os

import numpy as np

SCRIPT_LOCATION = os.getcwd()  # HOME-DIR/attic
PROJECT_FOLDER = os.path.dirname(SCRIPT_LOCATION)  # HOME-DIR
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
BIDS_FOLDER = os.path.join(DATA_FOLDER, "bids_struct")
DESTINATION_FOLDER_FC_MAT = os.path.join(DATA_FOLDER, "fc_pt")

if not os.path.isdir(DESTINATION_FOLDER_FC_MAT):
    os.mkdir(DESTINATION_FOLDER_FC_MAT)

# if os.path.isdir(BIDS_FOLDER):
#     folders = os.listdir(BIDS_FOLDER)
#     for folder in folders:
#         folder_path = os.path.join(BIDS_FOLDER, folder)
#         folder_files = os.listdir(folder_path)
#         file_csv = [filename for filename in folder_files if filename.endswith(".csv")]
#         if len(file_csv) != 1:
#             print(f"expected exactly one csv file in {folder} but found {len(file_csv)}")
#             break
#         file_csv = file_csv.pop(0)
#         file_name = os.path.splitext(file_csv)[0]
#         source = os.path.join(folder_path, file_csv)
#         destination = os.path.join(DESTINATION_FOLDER_FC_MAT, file_name)
#         fc_mat = np.genfromtxt(source, delimiter=",", skip_header=1, dtype=np.float16)
#         fc_mat = torch.as_tensor(fc_mat, dtype=torch.float32)
#         torch.save(fc_mat, destination + ".pt")
#         # np.save(destination, fc_mat)


def rename(root: str) -> None:
    files = os.listdir(root)
    # files = [os.path.join(root, file) for file in files]
    for file in files:
        filename, ending = os.path.splitext(file)
        id = re.findall("\d+", filename)[-1]
        newname = os.path.join(root, "subject_" + id + ending)
        oldname = os.path.join(root, file)
        # print(f"oldname: {oldname}")
        # print(f"newname: {newname}")
        os.rename(oldname, newname)
