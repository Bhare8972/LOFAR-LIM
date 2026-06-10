#!/usr/bin/env python3

"""FIX THIS"""

import sys
import os
import json

import time

import numpy as np

from LoLIM import utilities as utils
from LoLIM.pipeline.readSettings import readSettings
from LoLIM.iterativeMapper import iterative_mapper as IM # make_header
from LoLIM.pipeline.jobInterface import standard_jobInterface
from LoLIM.pipeline.run_find_RFI import readFindRFI_results

class impulsiveImager_jobInterface(standard_jobInterface):
    def stage_name(self):
        """return the name of this stage"""
        raise "impulsive_imager"

    def command(self):
        """return string of command to run"""
        raise "python -m LoLIM.pipeline.run_impulsiveImager"

    def initialize(self, use_slurm, slurm_settings, num_processes, maxCPU_per_process):
        """this will create the impulsive imager settings file, if it does not exist"""

        output_folder = os.join( self.flashOutputFolder, self.settings_data["impulsive_imager"]["output_folderName"] )
        output_fname = os.join(output_folder, "header.json")

## if already exists, do not overwrite
        if os.path.isfile(output_fname):
            return True


## make with default settings
        calibration_fname = self.flashDatabase_row['calibration_file']
        timeID = self.flashDatabase_row['TimeID']

        headerOBJ = IM.make_header(timeID,  total_cal=calibration_fname)
        headerOBJ.output_directory = output_folder

        ## WHAT TO DO HERE? RFI is spread over many files
        headerOBJ.RFI_info = readFindRFI_results( self.settings_data, self.flashOutputFolder )


## now check for stage settings
        stage_settings_fname = os.path.join( output_folder, self.settings_data["impulsive_imager"]['settingsFileName'] ) 
        if os.path.isfile(stage_settings_fname):
            stageSettings = json.load( open(stage_settings_fname, 'r') )
        else:
            stageSettings = self.settings_data[stageName]['settings']

        ## and apply!

        for setting_name, setting_data in stageSettings:
            if setting_name[0] == '%':  ## every setting that starts with a '%' we pass to the header. This works becouse JSON can store more than strings!
                headerOBJ.__dict__[setting_name[1:]] = setting_data


        ## then run

        headerOBJ.run()


        return True






def RunStage_ImIm(flashName, flashOutputFolder, outputFolder, logFolderLocation, flashDatabaseData, jobNumber, maxJobs, databaseSettings, stageSettings):

    jobNumber = int(jobNumber)
    maxJobs = int(maxJobs)

    num_consecutive_blocks = 100
    if 'num_consecutive_blocks' in stageSettings:
        num_consecutive_blocks = stageSettings['num_consecutive_blocks']


    skipBlocksDone = True
    if 'skip_blocks_done' in stageSettings:
        skipBlocksDone = bool(stageSettings['skip_blocks_done'])



    working_folder = os.join( flashOutputFolder, databaseSettings["impulsive_imager"]["output_folderName"] )

    inHeader = read_header(working_folder)
    log_fname = inHeader.next_log_file()
    
    logger_function = logger()
    logger_function.set( log_fname, False )
    logger_function.take_stderr()
    
    logger_function("process", jobNumber, '/', maxJobs)
    logger_function("date and time run:", time.strftime("%c") )

    print_lock.acquire()
    print('loading process', process_i, 'logfile:', log_fname)
    print_lock.release()
    
    mapper = iterative_mapper(inHeader, logger_function)

    blockRanges = mapper.getBlocksToProcess(jobNumber, maxJobs, num_Consecutive_Blocks=num_consecutive_blocks)
    print_lock.acquire()
    print(' numBlockRanges:', len(blockRanges))
    print_lock.release()


    for startBlok, endBlock in blockRanges:

            print_lock.acquire()
            print('process', jobNumber, 'blocks', startBlok, '-', endBlock)
            print_lock.release()

            logger_function('process', jobNumber, 'blocks', startBlok, '-', endBlock)
            mapper.process_blocks(startBlok, endBlock, print_func=logger_function, skip_blocks_done=skipBlocksDone)
        
    logger_function("done!" )
    print_lock.acquire()
    print('process', jobNumber, 'done!!')
    print_lock.release()



if __name__ == "__main__":
    settingsFname  = sys.argv[1]
    flashName      = sys.argv[2]
    jobNumber      = sys.argv[3]
    maxJobs        = sys.argv[4]
    numCPUs        = sys.argv[5]


## open database and pipeline settings
    stageName = "find_RFI"

    databaseSettings = readSettings( settingsFname )

    with database_manager(databaseSettings['flashDatabase_location']) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()

    dataRow = flashDatabase.loc[flashName]
    flashYear = dataRow['year']

    flashOutputFolder = os.path.join(  databaseSettings["processed_data_loc"], flashYear, flashName)

    outputFolder = os.path.join(  flashOutputFolder, databaseSettings[stageName]['output_folderName'])

## setup logging
    logFolderLocation = os.path.join( outputFolder, databaseSettings[stageName]['log_folderName'])

    logFileName = os.path.join( logFolderLocation, databaseSettings[stageName]['log_fileName'])
    logFileName = logFileName + '_job'+str(jobNumber)+'.txt'

    log = utils.logger( logFileName, False )
    log.take_stdout()
    log.take_stderr()

    print("stage:", stageName)
    print("  flash:", flashName, "settings:", settingsFname, "jobNumber:",jobNumber, "maxJobs:",maxJobs, "numCPUs:",numCPUs)


## now check for stage settings
    stage_settings_fname = os.path.join( outputFolder, databaseSettings[stageName]['settingsFileName'] ) 
    if os.path.isfile(stage_settings_fname):
        stageSettings = json.load( open(stage_settings_fname, 'r') )
    else:
        stageSettings = databaseSettings[stageName]['settings']


## now run!
    succsses = RunStage_ImIm(flashName, flashOutputFolder, outputFolder, logFolderLocation, dataRow, jobNumber, maxJobs, databaseSettings, stageSettings)



## check if good!
    if succsses:
        print('is succsesful!')

        completionFile = os.path.join( outputFolder, databaseSettings[stageName]['log_folderName'], databaseSettings[stageName]['succesfulCompletion_FileName']+"_"+jobNumber)
        with open(completionFile, 'w') as fout:
            fout.write("1")
    else:
        print('not succsesful')


    del log

