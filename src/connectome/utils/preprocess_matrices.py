# -*- coding: utf-8 -*-
""" preprocess connectivity matrices"""
import math
import os
import torch
import numpy as np
import pandas as pd
from typing import List
from functools import partial


def csv_to_pt(source_path: str, destination_path: str) -> None:
    """
    This function reads csv files from <source_path> and copies those as a pytorch tensor (.pt) in <destination_path>.
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
        data = torch.from_numpy(np.genfromtxt(file, delimiter=",", skip_header=1))
        new_name = os.path.join(destination_path, name) + ".pt"
        torch.save(data, new_name)


def standardize_matrix_list(mat_list: List[torch.tensor]) -> List[torch.tensor]:
    """
    This function takes a list of matrices and normalizes each matrix entry by the corresponding entry in all matrices.
    Take for example 3 matrices A, B, C with entry A_{i,j} etc. For entry A_{i,j} this function computes
    \tilde{A}_{i,j} = (A_{i,j} - MEAN(A_{i,j}, B_{i,j}, C_{i,j})) / VAR(A_{i,j}, B_{i,j}, C_{i,j}).
    """
    matrices = torch.stack(mat_list)
    mean_mat = torch.mean(matrices, dim=0)
    var_mat = torch.var(matrices, dim=0)
    matrices = torch.unbind(matrices)
    fun = partial(standardize_matrix, var=var_mat, mean=mean_mat)
    processed = [fun(x) for x in matrices]
    return processed


def standardize_matrix(mat: torch.tensor, var: torch.tensor, mean: torch.tensor) -> torch.tensor:
    """
    standardize a given matrix <mat> entry-wise by corresponding mean and variance given in matrices <mean> and <var>.
    """
    z = torch.subtract(mat, mean)
    z = torch.div(z, var)  # diagonal elements have zero var
    z.fill_diagonal_(1)
    return z


def preprocess_folder(matrices_raw_path: str, matrices_processed_path: str, metadata_path: str,
                      age_bins: np.ndarray = np.linspace(0, 100, 11)) -> None:
    """
    This function normalizes matrices group-wise. Each group is defined by age range and sex.
    """
    if not os.path.exists(matrices_processed_path):
        os.mkdir(matrices_processed_path)
    assert os.path.isdir(matrices_raw_path)
    assert os.path.isdir(matrices_processed_path)
    assert os.path.isfile(metadata_path)
    assert isinstance(age_bins, np.ndarray)

    metadata = pd.read_csv(metadata_path)
    metadata["age_group"] = pd.cut(metadata.age, age_bins, include_lowest=True)
    split = metadata.groupby(["sex", "age_group"])

    for split_cols, table in split:
        subjects = [file + ".pt" for file in table.subject_id.values]
        # to make sure subject in csv file is actually also in fc folder
        subjects = list(set(subjects).intersection(set(os.listdir(matrices_raw_path))))
        processed_names = [os.path.join(matrices_processed_path, filename) for filename in subjects]
        subjects = [os.path.join(matrices_raw_path, filename) for filename in subjects]
        subjects = [torch.load(file) for file in subjects]
        processed_values = standardize_matrix_list(subjects)
        processed = zip(processed_names, processed_values)
        for path, file in processed:
            torch.save(file, path)


def show_some_matrices(matrices_path: List[str], n=7) -> None:
    """
    show some matrices
    """
    # TODO: always same plot in subplots due to lazy evaluation? not sure on that...
    from matplotlib import pyplot as plt
    import seaborn as sns
    matrices = os.listdir(matrices_path)
    matrices = [os.path.join(matrices_path, file) for file in matrices]
    matrices = [torch.load(file) for file in matrices]
    nrow = math.floor(n / 2)
    ncol = n - nrow
    if nrow > 1:
        fig, axes = plt.subplots(nrow, ncol)
        for row_idx in range(nrow):
            for col_idx in range(ncol):
                idx = row_idx + col_idx
                matrix = matrices[idx]
                # TODO: check if distinctive matrices are plotted and not last one
                sns.distplot(matrix, ax=axes[row_idx, col_idx], kde=True, rug=False)
    plt.show()


if __name__ == "__main__":
    FOLDER_PATH = os.getcwd()
    ROOT_PATH = os.path.dirname(FOLDER_PATH)  # src/connectome/
    ROOT_PATH = os.path.dirname(ROOT_PATH)  # src/
    DATA_PATH = os.path.join(os.path.dirname(ROOT_PATH), "data")
    # metadata_path = os.path.join(DATA_PATH, "covariates", "covariates.csv")
    raw_path = os.path.join(DATA_PATH, "fc_pt")
    # processed_path = os.path.join(DATA_PATH, "fc_pt_processed")
    # preprocess_folder(matrices_raw_path=raw_path, matrices_processed_path=processed_path, metadata_path=metadata_path)
    show_some_matrices(raw_path)
    #csv_to_pt(os.path.join(os.path.dirname(ROOT_PATH), "data", "corr_mat"),os.path.join(os.path.dirname(ROOT_PATH), "data", "test_out"))
