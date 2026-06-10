#!/usr/bin/env python3

import os
import argparse

#import numpy as np
#from matplotlib import pyplot as plt

from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.database import database_manager
#from LoLIM.pipeline.initial_statistics import readInitialStatisticsFile

#from LoLIM import prettytable

if __name__ == "__main__":
    print('program clear_stage_completeFiles')


    flash_state_categories = ["ready_download_data", "download_data", "initial_statistics", "find_RFI", "impulsive_imager", "LTA_upload", "local_cleanup"]

    parser = argparse.ArgumentParser(description='delete the completion files for a processing stage of a flash. This is so the processor will not skip this stage if it is re-run. NOTE: many stages also skip jobs if final results still exist, which are deleted via different scripts.')
    parser.add_argument("settings", help="location of the settings JSON file")
    parser.add_argument("flash", help="name of the flash. E.G 20B-1")

    parser.add_argument("stage", help="stage for which to delete the complete files", choices=flash_state_categories)

    args = parser.parse_args()



    print('read settings')
    databaseSettings = readSettings( args.settings )

    print('read database')
    with database_manager(databaseSettings['flashDatabase_location']) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()

    dataRow = flashDatabase.loc[args.flash]
    flashYear = dataRow['year']


    print('deleting completion files')

    flashOutputFolder = os.path.join(  databaseSettings["processed_data_loc"], flashYear, args.flash)
    data_outputFolder = os.path.join(  flashOutputFolder, databaseSettings[ args.stage ]['output_folderName'])
    logFolderLocation = os.path.join( data_outputFolder, databaseSettings[ args.stage ]['log_folderName'])

    completion_file_symbol = databaseSettings[ args.stage ]['succesfulCompletion_FileName']

    files_to_delete = available_datafiles = [ f for f in os.listdir(logFolderLocation) if f.startswith(completion_file_symbol) ]

    print()
    print('deleted:')
    print(files_to_delete)

    for f in files_to_delete:
        
        os.remove( os.path.join(logFolderLocation, f) )







