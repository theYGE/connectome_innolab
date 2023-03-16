# This script exports all files names from given folders

import os

import pandas as pd

# list of all folders that save fMRI scan
path_list = [
    "/run/user/1000/gvfs/smb-share:server=nas645cbc.local,share=public/ukb_rs_part1",
    "/run/user/1000/gvfs/smb-share:server=nas645cbc.local,share=public/rs_part2",
    "/run/user/1000/gvfs/smb-share:server=nas645cbc.local,share=public/rs_part3",
    "/run/user/1000/gvfs/smb-share:server=nas645cbc.local,share=public/rs_part4",
    "/run/user/1000/gvfs/smb-share:server=nas645cbc.local,share=public/rs_part5",
    "/run/user/1000/gvfs/smb-share:server=nas645cbc.local,share=public/rs_part6",
    "/run/user/1000/gvfs/smb-share:server=nas645cbc.local,share=public/rs_part7",
]


all_files = pd.DataFrame(columns=["filename", "resources"])

for idx, path in enumerate(path_list):
    # find all zip files
    curr_files = [f for f in os.listdir(path) if f.endswith("zip") or f.endswith("7z")]
    folder = path.split("/")[-1]  # extract resources name
    tmp_df = pd.DataFrame(curr_files, columns=["filename"])
    tmp_df["resources"] = folder
    all_files = pd.concat([all_files, tmp_df], axis=0)

all_files.to_csv("all_zipfiles.csv")
