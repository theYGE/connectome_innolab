# -*- coding: utf-8 -*-
""" preprocess connectivity matrices"""
import os

import numpy as np
import torch


def csv_to_pt(
    source_path: str, destination_path: str, skip_col: int = 1, skip_row: int = 1
) -> None:
    """
    This function reads csv files from <source_path> and copies those as a pytorch tensor (.pt) in <destination_path>.

    Args:
        source_path (str): Absolute path of data source directory
        destination_path (str): Absolute path  for destination directory
        skip_col (int): number of first n column to skip while reading csv file.
        skip_row (int): number of first n rows to skip while reading csv file.
    """
    assert os.path.isdir(source_path)
    # TODO: check if all are csv
    if not os.path.isdir(destination_path):
        os.mkdir(destination_path)
    files = os.listdir(source_path)
    files = [os.path.join(source_path, file) for file in files]
    for file in files:
        name = os.path.basename(file)
        name = os.path.splitext(name)[0]
        data = torch.from_numpy(
            np.genfromtxt(
                file,
                delimiter=",",
                skip_header=skip_row,
            )
        )
        data = data[:, skip_col:]
        data.fill_diagonal_(0)
        new_name = os.path.join(destination_path, name) + ".pt"
        torch.save(data, new_name)


if __name__ == "__main__":
    FOLDER_PATH = os.getcwd()
    ROOT_PATH = os.path.dirname(FOLDER_PATH)  # src/connectome/
    ROOT_PATH = os.path.dirname(ROOT_PATH)  # src/
    DATA_PATH = os.path.join(os.path.dirname(ROOT_PATH), "data")
    raw_path = os.path.join(DATA_PATH, "matrix_last_year")
    pt_path = os.path.join(DATA_PATH, "fc_pt")
    csv_to_pt(
        raw_path,
        pt_path,
    )
