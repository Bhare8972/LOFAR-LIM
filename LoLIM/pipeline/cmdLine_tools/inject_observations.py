#!/usr/bin/env python3


import argparse

from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.pipeline_flash_injection import inject_observations

if __name__ == "__main__":
    print('program inject observations')
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", help="location of the settings JSON file")
    #parser.add_argument("settings", help="location of the settings JSON file")
    args = parser.parse_args()

    print('read settings')
    databaseSettings = readSettings( args.settings )

    print('running')
    inject_observations(databaseSettings, newFlash_pipelineMode="off")