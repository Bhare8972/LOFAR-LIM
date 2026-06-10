#!/usr/bin/env python3

""" Pipeline stage that readys a flash for downloading data. Namely, makes a pandas database / JSON table to hold information about the files """

import sys
import os
from time import sleep
import json

import pandas as pd

from LoLIM import utilities as utils
from LoLIM.pipeline.database import database_manager
from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.download_raw_data import LTAlocations
from LoLIM.pipeline.jobInterface import standard_jobInterface

class readyDownloadData_jobInterface(standard_jobInterface):
    def stage_name(self):
        """return the name of this stage"""
        raise "ready_download_data"

    def command(self):
        """return string of command to run"""
        raise "python -m LoLIM.pipeline.ready_download_data"


def RunStage_readyDownloadData(flashName, outputFolder, logFolderLocation, flashDatabaseData, databaseSettings, stageSettings):

    stormName = flashDatabaseData['storm']
    year = flashDatabaseData['year']


    LTA_locs_folder = databaseSettings['download_data']['LTA_locations_folder']
    LTA_locs_subfolder = os.path.join(LTA_locs_folder, year) if databaseSettings['download_data']['LTA_locations_folder_organized_byYear'] else LTA_locs_folder


    with database_manager(databaseSettings['stormDatabase_location']) as stormdatabase_manager:
        stormDatabase = stormdatabase_manager.read_database()
    stormRow = stormDatabase.loc[stormName]



    LTAloc_file = stormRow['LTAloc_file']
    if not (isinstance(LTAloc_file, str) and len(LTAloc_file)>0):
        print("LTAloc_file in database is no good:", LTAloc_file)
        return False

    LTAloc_file = os.path.join( LTA_locs_subfolder, LTAloc_file )


    MD5Sum_file = stormRow['MD5Sum_file']
    if not (isinstance(MD5Sum_file, str) and len(MD5Sum_file)>0):
        MD5Sum_file = None
    else:
        MD5Sum_file = os.path.join( LTA_locs_subfolder, MD5Sum_file)


    LTAlocationsObject = LTAlocations( LTAloc_file,  MD5Sum_file, year=year, flash_timeID=flashDatabaseData['TimeID']  ) 
    fullTimeID = LTAlocationsObject.getTimeIDs()[0] ##handful of reasons to do this

    file_locs = LTAlocationsObject.locs_by_TimeID( fullTimeID )
    file_fnames = [ LTAlocationsObject.loc_to_fname(floc) for floc in file_locs ]

    file_MD5Sum =  [ LTAlocationsObject.getMD5_per_fLoc(floc) for floc in file_locs ]
    file_MD5Sum = [ '' if M is None else M for M in file_MD5Sum ]  ## make sure is string

    current_MD5Sum = ['' for i in range(len(file_MD5Sum))]
    fileDownloaded = [False for i in range(len(file_MD5Sum))]
    prefixes = ['%default' for i in range(len(file_MD5Sum))]

    DF =  pd.DataFrame( { 
    'LTA_location':pd.Series(file_locs,dtype=str),
    'LTA_MD5Sum':pd.Series(file_MD5Sum,dtype=str),
    'fileName':pd.Series(file_fnames,dtype=str),
    'local_MD5Sum':pd.Series(current_MD5Sum,dtype=str),
    'file_downloaded':pd.Series(fileDownloaded,dtype=bool),
    #'LTA_loc_prefix':pd.Series(prefixes,dtype=str),
    }, 
    )

    tableOutputFname = os.path.join(outputFolder, databaseSettings['ready_download_data']['dataFileTableName'])

    with open(tableOutputFname, 'w') as fout:
        fout.write( DF.to_json(orient='table') )

    return True


if __name__ == "__main__":
    settingsFname  = sys.argv[1]
    flashName      = sys.argv[2]
    jobNumber      = sys.argv[3]
    maxJobs        = sys.argv[4]
    numCPUs        = sys.argv[5]

## open database and pipeline settings
    stageName = "ready_download_data"

    databaseSettings = readSettings( settingsFname )

    with database_manager(databaseSettings['flashDatabase_location']) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()

    dataRow = flashDatabase.loc[flashName]
    flashYear = dataRow['year']

    ## setup overall folder and logging
    flashOutputFolder = os.path.join(  databaseSettings["processed_data_loc"], flashYear, flashName)

    outputFolder = os.path.join(  flashOutputFolder, databaseSettings[stageName]['output_folderName'])
    logFolderLocation = os.path.join( outputFolder, databaseSettings[stageName]['log_folderName'])

    logFileName = os.path.join( logFolderLocation, databaseSettings[stageName]['log_fileName'])
    logFileName = logFileName + '_job'+str(jobNumber)+'.txt'

    log = utils.logger( logFileName, False )
    log.take_stdout()
    log.take_stderr()

    print("stage: ready_download_data")
    print("  flash:", flashName, "PipelineSettings:", settingsFname, "jobNumber:",jobNumber, "maxJobs:",maxJobs, "numCPUs:",numCPUs)


## now check for stage settings
    stage_settings_fname = os.path.join( outputFolder, databaseSettings[stageName]['settingsFileName'] ) 
    if os.path.isfile(stage_settings_fname):
        stageSettings = json.load( open(stage_settings_fname, 'r') )
    else:
        stageSettings = databaseSettings[stageName]['settings']


## now run!
    if int(jobNumber)==0:
        succsses = RunStage_readyDownloadData(flashName, outputFolder, logFolderLocation, dataRow, databaseSettings, stageSettings)
    else:
        print('not running due to jobNumber != 0')
        succsses = True


## check if good!
    if succsses:
        print('is succsesful!')

        completionFile = os.path.join( outputFolder, databaseSettings[stageName]['log_folderName'], databaseSettings[stageName]['succesfulCompletion_FileName']+"_"+jobNumber)
        with open(completionFile, 'w') as fout:
            fout.write("1")
    else:
        print('not succsesful')


    del log