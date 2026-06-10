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
    print('program clear_intialStatistics_results')


    parser = argparse.ArgumentParser(description='Delete initial statistics data for all or one station. Need to run this if want to redo results.')
    parser.add_argument("settings", help="location of the settings JSON file")
    parser.add_argument("flash", help="name of the flash. E.G 20B-1")

    parser.add_argument("--station", help="station for which to delete resuls. Leave empty to delete for all stations", default=None)

    args = parser.parse_args()


    print('read settings')
    databaseSettings = readSettings( args.settings )

    print('read database')
    with database_manager(databaseSettings['flashDatabase_location']) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()

    dataRow = flashDatabase.loc[args.flash]
    flashYear = dataRow['year']


    ## folder operations
    flashOutputFolder = os.path.join(  databaseSettings["processed_data_loc"], flashYear, args.flash)
    outputFolder = os.path.join(  flashOutputFolder, databaseSettings["initial_statistics"]['output_folderName'])

    output_files = [ f for f in os.listdir(outputFolder) if os.path.isfile(os.path.join(outputFolder, f)) and f.endswith('_stats.json') ]

    if args.station is None:
       files_to_remove =  output_files
    else:
        files_to_remove = [ f for f in output_files if f.startswith(args.station) ] 


    print('from', outputFolder, 'deleting:', files_to_remove)
    for f in files_to_remove:
         os.remove( os.path.join(outputFolder, f) )


