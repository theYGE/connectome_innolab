# This script extracts target file from zip file, in given folders
# copy this script to the folder for saving the extracted files, then run it.

import multiprocessing as mp
import os
import time
from itertools import cycle
from zipfile import ZipFile


def extract_zipfile(zip_info):
    # extract target file from given zip
    zip_path, zipname = zip_info[0], zip_info[1]
    zip_file_path = os.path.join(zip_path, zipname)
    # print(zip_file_path)
    with ZipFile(zip_file_path, "r") as zipObj:
        # Extract all the contents of zip file in different directory
        listOfFileNames = zipObj.namelist()

        for fileName in listOfFileNames:
            if fileName.endswith("filtered_func_data_clean.nii.gz"):
                # Extract a single file from zip
                zipObj.extract(fileName, zipname[:-4])


def extract_all(zip_path):
    # find out all zips in given path, then extract all of them
    print("working on folder {}".format(zip_path))
    zip_filename_list = []
    for file in os.listdir(zip_path):
        if file.endswith(".zip"):
            zip_filename_list.append(file)

    # find the number of cpus of current laptop
    print(mp.cpu_count())
    with mp.Pool(mp.cpu_count() - 5) as pool:  # multi-processing
        # for i in zip(cycle([zip_path]), zip_filename_list):
        # print(i)
        pool.map(extract_zipfile, zip(cycle([zip_path]), zip_filename_list))

    # num_zip = len(zip_filename_list)
    # i = 1
    # for zip_filename in zip_filename_list:

    # print("Extracting file: {} out of {}".format(i, num_zip))
    # start_time = time.time()
    # extract_zipfile(zip_path, zip_filename)
    # end_time = time.time()
    # duration = end_time - start_time
    # total_duration = num_zip * duration
    # remain_duration = total_duration - i * duration
    # print("Hey Bruh, it take this long {} for a zip.".format(duration))
    # i += 1
    # except:
    # print("folder {} is not extracted!".format(zip_filename))


# zip_path_list = [r"/home/demenzbild/Downloads/Innolabs_tmp2",]

zip_path_list = [
    "/run/user/1000/gvfs/smb-share:server=nas645cbc.local,share=public/ukb_rs_part1",
    # "/run/user/1000/gvfs/smb-share:server=nas645cbc.local,share=public/rs_part2",
]

for zip_path in zip_path_list:
    extract_all(zip_path)
