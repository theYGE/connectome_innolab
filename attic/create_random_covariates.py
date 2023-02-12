# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

root = os.getcwd()
root = os.path.dirname(root)
adjacencies_folder = os.path.join(root, "data", "fc_pt")
covariates_folder = os.path.join(root, "data", "covariates")
covariates_name = "covariates.csv"
covariates_path = os.path.join(covariates_folder, covariates_name)


def fun(x):
    return os.path.splitext(x)[0]


files = os.listdir(adjacencies_folder)
files = list(map(fun, files))
n = len(files)
age = np.random.choice(list(range(91)), size=n, replace=True)
sex = np.random.choice(["male", "female"], size=n, replace=True)
data = {"subject_id": files, "age": age, "sex": sex}
dataframe = pd.DataFrame.from_dict(data)
dataframe.to_csv(covariates_path, sep=",", index=False)

