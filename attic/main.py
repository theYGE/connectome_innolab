"""
set parameters and run main
"""
import logging
import os
from pathlib import Path  # included in SL since 3.4

from load_data import check_atlas, load_data

SCRIPT_LOCATION = os.getcwd()  # HOME-DIR/attic
PROJECT_FOLDER = os.path.dirname(SCRIPT_LOCATION)  # HOME-DIR
LOG_FOLDER = os.path.join(PROJECT_FOLDER, "logs")
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
ATLAS_FOLDER = os.path.join(DATA_FOLDER, "atlas")
DATASET_ID = "ds000114"
FMRIPREP_OUT_FOLDER = os.path.join(DATA_FOLDER, "fmriprep_out")


logging.basicConfig(level=logging.INFO)


def main() -> None:
    """
    main routine to handle our different steps in connectome-GNN project

    TODO:
        logging of function by entrance on global level
    """
    print("\nCALL MAIN\n")
    # create data folder if necessary
    Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    # check for atlas
    Path(ATLAS_FOLDER).mkdir(parents=True, exist_ok=True)
    check_atlas(ATLAS_FOLDER)
    # load data
    load_data(DATASET_ID, os.path.join(DATA_FOLDER, "sample_data_1"))
    # fmriprep pipeline
    # SKIP
    # create connectivity matrix


if __name__ == "__main__":
    main()
