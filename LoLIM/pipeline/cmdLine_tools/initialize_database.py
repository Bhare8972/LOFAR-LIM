#!/usr/bin/env python3

import argparse

from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.database import makeEmptyDatabase

if __name__ == "__main__":
    print('program initalize_database')
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", help="location of the settings JSON file")
    #parser.add_argument("settings", help="location of the settings JSON file")
    args = parser.parse_args()

    print('read settings')
    databaseSettings = readSettings( args.settings )

    print('running')
    makeEmptyDatabase(flashDatabase_fname=databaseSettings['flashDatabase_location'], stormDatabase_fname=databaseSettings['stormDatabase_location'])



