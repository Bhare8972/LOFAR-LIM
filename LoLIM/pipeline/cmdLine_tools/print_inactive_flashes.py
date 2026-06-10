#!/usr/bin/env python3


import argparse

from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.database import database_manager

if __name__ == "__main__":
    print('program print inactive flashes')
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", help="location of the settings JSON file")
    parser.add_argument("year", help="year of interest",type=str)
    args = parser.parse_args()

    print('read settings')
    databaseSettings = readSettings( args.settings )

    #stormDatabase_location = databaseSettings['stormDatabase_location']
    flashDatabase_location = databaseSettings['flashDatabase_location']


    with database_manager(flashDatabase_location) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()



    yearFlashDatabase = flashDatabase[ flashDatabase["year"]==args.year ]
    #yearFlashDatabase = flashDatabase
    inactiveFlashDatbase = yearFlashDatabase[ yearFlashDatabase["pipeline_mode"]=="off" ] ## ignore flashes that ought to be ignored

    print(len(inactiveFlashDatbase), 'to print')
    for flashName, dataRow in inactiveFlashDatbase.iterrows():
        print(flashName, ':', dataRow['TimeID'], dataRow['state'], dataRow['sub_state'], dataRow['storm'], dataRow['pipeline_mode'], dataRow['year'])

