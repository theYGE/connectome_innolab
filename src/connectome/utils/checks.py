# -*- coding: utf-8 -*-
"""Do checks between certain steps in complete pipeline"""
import os
from typing import Tuple


def check_pre_training(
    connectivity_csv_folder: str, min_train_files: int = 1
) -> Tuple[bool, str]:
    """
    Do checks before a (new) GNN model is trained.

    Args:
        connectivity_csv_folder (str): Absolute path to folder that contains
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


def check_all_ending(train_data_folder: str, ending: str) -> bool:
    """
    Are all files in <data_folder> .csv files?

    Args:
        train_data_folder (str): Absolute path to folder that contains data to be trained on.
        ending (str): File endings to be checked (e.g. '.csv' or '.pt')

    Returns:
        bool: whether all files are csv.
    """
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
    data_path = os.path.join(root, "data")
    log_folder = os.path.join(root, ".logs")
    train_data = os.path.join(data_path, "corr_mat")
    check_pre_training(train_data)
