# -*- coding: utf-8 -*-
"""Do checks between certain steps in complete pipeline"""
import os
from typing import Tuple
import torch
import numpy as np


def check_pre_training(
    connectivity_csv_folder: str, min_train_files: int = 1
) -> Tuple[bool, str]:
    """
    Do checks before a (new) GNN model is trained.

    Args:
        connectivity_csv_folder (str): Absolute path to folder that contains connectivity csv files
        min_train_files (int): Minimal number of connectivity files to start training.

    Returns:
        (bool, message): Boolean value whether all prerequisites for training are fulfilled and message
    """
    assert os.path.isdir(connectivity_csv_folder)
    csv_folder_name = os.path.basename(connectivity_csv_folder)
    csv_folder_env = os.listdir(os.path.dirname(connectivity_csv_folder))
    n_files = len(os.listdir(connectivity_csv_folder))
    if n_files < min_train_files:
        msg = csv_folder_name + "contains only " + str(n_files) + "files."
        return False, msg
    # all files are .csv?
    if check_all_ending(connectivity_csv_folder, ".csv"):
        # if yes then check if there is corresponding folder with '.pt' files.
        pt_folder_name = os.path.splitext(csv_folder_name)[0] + ".pt"
        if pt_folder_name not in csv_folder_env:
            msg = (
                "Could not find "
                + pt_folder_name
                + " in "
                + os.path.dirname(connectivity_csv_folder)
                + "."
            )
            return False, msg
    else:
        # else return False and a message
        msg = "Not all files in " + csv_folder_name + " appear to be csv files"
        return False, msg

    return True, "Everything seems to be okay to start training."


def check_same_dimensions(data_folder: str, ending: str = ".csv") -> Tuple[bool, str]:
    """
    Do all files have same dimensionality?

    Depending on whetercsv or pt files are considered this function checks if all files have same dimensionality.
    If there are issues a tuple (False, msg) will be returned where msg will indicate the file causing issues.
    Some random file from folder is chosen to be baseline for comparison.

    Args:
        data_folder (str): path to folder to check.
        ending (str): either '.csv' or '.pt'.

    Returns:
        (bool, str): check_status and corresponding message.
    """
    assert isinstance(data_folder, str)
    assert isinstance(ending, str)
    file_names = os.listdir(data_folder)
    file_names = [os.path.join(data_folder, file) for file in file_names]
    assert len(file_names) > 0
    assert ending == ".csv" or ending == ".pt"
    if ending == ".csv":
        data = np.genfromtxt(file_names[0], delimiter=",")
        dim_check = data.shape
        for file in file_names:
            data = np.genfromtxt(file, delimiter=",")
            condition = data.shape == dim_check
            if not condition:
                msg = "No coherent dimension: " + os.path.splitext(file)[0]
                return False, msg
    if ending == ".pt":
        data = torch.load(file_names[0])
        dim_check = data.shape
        for file in file_names:
            data = torch.load(file)
            condition = data.shape == dim_check
            if not condition:
                msg = "No coherent dimension: " + os.path.splitext(file)[0]
                return False, msg
    return True, "OK"


def check_all_ending(train_data_folder: str, ending: str) -> bool:
    """
    Are all files in <data_folder> .csv files?

    Args:
        train_data_folder (str): Absolute path to folder that contains data to be trained on.
        ending (str): File endings to be checked (e.g. '.csv' or '.pt')

    Returns:
        bool: whether all files are csv.
    """
    assert isinstance(train_data_folder, str)
    assert isinstance(ending, str)
    files = os.listdir(train_data_folder)
    assert len(files) >= 1
    to_check = [os.path.splitext(file)[1] for file in files]
    condition = all(check == ending for check in to_check)
    return condition


if __name__ == "__main__":
    root = os.getcwd()
    root = os.path.dirname(root)
    root = os.path.dirname(root)
    root = os.path.dirname(root)
    check_same_dimensions(data_folder=os.path.join(root, "data", "fc_pt"), ending=".pt")
