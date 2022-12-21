"""
This script writes fc matrices from a BIDS structure folder into a single folder.
"""
import os
import shutil

SCRIPT_LOCATION = os.getcwd()  # HOME-DIR/attic
PROJECT_FOLDER = os.path.dirname(SCRIPT_LOCATION)  # HOME-DIR
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
BIDS_FOLDER = os.path.join(DATA_FOLDER, "bids_struct")
DESTINATION_FOLDER = os.path.join(DATA_FOLDER, "fc_mat_folder")

if not os.path.isdir(DESTINATION_FOLDER):
    os.mkdir(DESTINATION_FOLDER)

if os.path.isdir(BIDS_FOLDER):
    folders = os.listdir(BIDS_FOLDER)
    for folder in folders:
        folder_path = os.path.join(BIDS_FOLDER, folder)
        #print(f"folder_path: {folder_path}")
        folder_files = os.listdir(folder_path)
        #print(f"folder_files: {folder_files}")
        file_csv = [filename for filename in folder_files if filename.endswith(".csv")]
        #print(f"csv: {file_csv}")
        if len(file_csv) != 1:
            print(f"expected exactly one csv file in {folder} but found {len(file_csv)}")
            break
        file_csv = file_csv.pop(0)
        source = os.path.join(folder_path, file_csv)
        destination = os.path.join(DESTINATION_FOLDER, file_csv)
        shutil.copy(source, destination)
