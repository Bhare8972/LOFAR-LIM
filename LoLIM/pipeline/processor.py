#!/usr/bin/env python3

import sys
import os
import subprocess
import traceback

import numpy as np

from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.database import database_manager, backupFile, pipelineMode_categories, cleanupBackups
import LoLIM.utilities as utils


#from LoLIM.pipeline.doNothing import doNothing
from LoLIM.pipeline.ready_download_data import readyDownloadData_jobInterface #postProcessing_readyDownloadData   ## this should be combined with downloading data
from LoLIM.pipeline.download_raw_data import LTAlocations, downloadData_jobInterface #postProcessing_downloadRawData
from LoLIM.pipeline.initial_statistics import initialStatistics_jobInterface #postProcessing_initialStatistics
from LoLIM.pipeline.run_find_RFI import findRFI_jobInterface #postProcessing_findRFI
from LoLIM.pipeline.run_impulsiveImager import impulsiveImager_jobInterface


## CONSTRUCTRUCT interfaces dictionary
_interfaces_ = [readyDownloadData_jobInterface, downloadData_jobInterface, initialStatistics_jobInterface, findRFI_jobInterface, impulsiveImager_jobInterface]
interfaces = {}
for i in interfaces:
    name = i.stage_name()
    if name in interfaces:
        print("ERROR in processor. Two stages with same name:", name)
        quit()

    interfaces[name] = i



### for getting process IDs
def getSLURM_processIDs( settings ):

    ret = []
    subp = subprocess.run(['squeue', '-u', settings['slurm_user']], stdout=subprocess.PIPE)
    text = subp.stdout.decode()
    for line in text.split("\n"):
        words = line.split()
        if len(words)>0 and words[0].isdigit():
            ret.append( int(words[0]) )

    return ret

def getSystem_processIDs( ):
    """calls 'ps -A' and returns list of integers representing all process IDs"""

    ret = []

    subp = subprocess.run(['ps', '-A'], stdout=subprocess.PIPE)
    text = subp.stdout.decode()
    for line in text.split("\n"):
        words = line.split()
        if len(words)>0 and words[0].isdigit():
            ret.append( int(words[0]) )

    return ret


def run_processor(settings_fname):

#### THIS DATA DEFINES THE STEPS IN THE PIPELINE
    ## probably a more ellagent way to do this. 


### I feel like each pipeline stage should have a number of module-level variables that contain this information. ALso includding the run-class to be designed

    ## must be same as in database.flashState_categories
    #stage_names =          ["ready_download_data",                          "download_data",                              "initial_statistics",                          "find_RFI"] 

    ## associated command line commands to run
    #stage_commands =       ["python -m LoLIM.pipeline.ready_download_data", "python -m LoLIM.pipeline.download_raw_data", "python -m LoLIM.pipeline.initial_statistics", "python -m LoLIM.pipeline.run_find_RFI"]

    ## finish-up functions 
    #stage_postProcessing = [postProcessing_readyDownloadData,                postProcessing_downloadRawData,               postProcessing_initialStatistics,              postProcessing_findRFI]

####

### initialization things
    databaseSettings = readSettings( settings_fname )

    ## get rellavent settings
    useSLURM = databaseSettings["processor_uses_slurm"]
    maxAllowedProcesses = databaseSettings["maxAllowedProcesses"]

    #stormDatabase_location = databaseSettings['stormDatabase_location']
    flashDatabase_location = databaseSettings['flashDatabase_location']

    processedDataFolder = databaseSettings["processed_data_loc"]


    ## TODO: check if any backups are too old and remove them

    #backupFile(stormDatabase_location)
    backupFile(flashDatabase_location)
    cleanupBackups(flashDatabase_location, min_num_backups=databaseSettings["minNumDatabaseBackups"], maxBackupDays=databaseSettings["maxDatabaseBackup_Days"])

    all_process_IDs = getSLURM_processIDs( databaseSettings ) if useSLURM else getSystem_processIDs( )
    spinupJob_function = SLURM_spinupJob if useSLURM else SYSTEM_spinupJob

### open database
    #with database_manager(stormDatabase_location) as stormdatbase_manager:

    with database_manager(flashDatabase_location) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()

        activeFlashDatbase = flashDatabase[ flashDatabase["pipeline_mode"]!="off" ] ## ignore flashes that ought to be ignored

        ## loop over every flash and decide if then need more prcessing

        flashNamesToProcess = []
        flashPrioritiesToProcess = []
        flashStageToProcess = []

        numberRunningProcesses = 0
        for flashName, dataRow in activeFlashDatbase.iterrows():

            current_state = dataRow["state"]
            current_subState = dataRow["sub_state"]
            flash_pipelineMode = dataRow["pipeline_mode"]

            flashOutputFolder = os.path.join(  processedDataFolder, dataRow['year'], flashName)

            print('processing', flashName)
            print('  state', current_state, current_subState)


            currentJobInterface = interfaces[current_state]( settings_fname, databaseSettings, flashOutputFolder, flashName, dataRow )


            if current_subState == "error":
                print('  flash is currently in error sub_state. Skipping.')
                continue

            if current_subState == "running":
                ## check if still running

                flashProcess_IDs = dataRow['slurm_processes']

                num_stage_running_processes, flashStage_stillRunning = currentJobInterface.job_is_running(flashProcess_IDs, all_process_IDs)
                numberRunningProcesses += num_stage_running_processes

                if not flashStage_stillRunning:
                    current_subState = 'complete'



            if current_subState == "complete":
                ### check if indeed completed
                completionFname = os.path.join(  flashOutputFolder, databaseSettings[current_state]['output_folderName'], 
                    databaseSettings[current_state]['log_folderName'], databaseSettings[current_state]['succesfulCompletion_FileName'] )


                #complete_succsesfully = False
                if os.path.isfile(completionFname+"_ALL"):
                    complete_succsesfully = True

                else:
                    flashProcess_IDs = dataRow['slurm_processes']
                    complete_succsesfully = currentJobInterface.all_processes_succsesful( flashProcess_IDs )

                    #have_all_fnames = True
                    #for i in range(len(flashProcess_IDs)):
                    #    fname = completionFname+"_"+str(i)
                    #    if not os.path.isfile(fname):
                    #        have_all_fnames = False
                    #        break

                    #if have_all_fnames:
                    #    complete_succsesfully = True
#
                    #else:
                    #    complete_succsesfully = False


                ## do any requird post-processing
                if not complete_succsesfully:
                    print('  not complete succsesfully')

                    current_subState = 'error'

                else:
                    print('  completed. Doing finish-up')
                    ## do finish-up function
                    stageIndex = stage_names.index( current_state )

                    try:
                        #funcIsGood = function(flashName, flashDatabase, databaseSettings )
                        funcIsGood = currentJobInterface.postProcessing( )
                    except Exception as e:  
                        funcIsGood = False
                        print('callback function for flash', flashName, 'stage', current_state, 'has errored')
                        print(traceback.format_exc())



                    if not funcIsGood:
                        current_subState = 'error'
                    else:
                        with open(completionFname+"_ALL", 'w') as fout:
                            fout.write("1")

                        ## find next stage
                        processingIsComplete = (flash_pipelineMode=='all' and current_state==stage_names[-1]) or ( flash_pipelineMode==current_state )

                        if processingIsComplete:
                            flash_pipelineMode = 'off'

                        else:
                            current_state = stage_names[stageIndex+1]
                            current_subState = 'ready'
                            flashDatabase.at[ flashName, "slurm_processes" ] = ''

                            currentJobInterface = interfaces[current_state]( settings_fname, databaseSettings, flashOutputFolder, flashName, dataRow )



            if current_subState == "ready" and flash_pipelineMode!='off':

                ### check if this stage was already done
                completionFname = os.path.join(  flashOutputFolder, databaseSettings[current_state]['output_folderName'], 
                    databaseSettings[current_state]['log_folderName'], databaseSettings[current_state]['succesfulCompletion_FileName']+"_ALL" )

                if os.path.isfile(completionFname):
                    current_subState = 'complete'

                else:
                    print('  adding to job list')

                    stageIndex = stage_names.index( current_state )

                    flashNamesToProcess.append( flashName )
                    flashPrioritiesToProcess.append(  dataRow['pipeline_priority']*len(stage_names) + stageIndex  )
                    flashStageToProcess.append( current_state )


            flashDatabase.at[ flashName, "state" ] = current_state
            flashDatabase.at[ flashName, "sub_state" ] = current_subState
            flashDatabase.at[ flashName, "pipeline_mode" ] = flash_pipelineMode

            #with database_manager(flashDatabase_location) as flashdatabase_manager:
            flashdatabase_manager.save_database( flashDatabase )

            print('Flash:', flashName)
            print("    pipeline mode:", flash_pipelineMode)
            print("    current state:", current_state)
            print("        sub-state:", current_subState)

        print()
        print('-------------------------')
        print()

        ### sort jobs to do by priority
        sorter = np.argsort(flashPrioritiesToProcess)[::-1] ## higher numbers are higher priority

        for i in sorter:
        ## calc num processes needed
            flashName = flashNamesToProcess[i]
            flashStage = flashStageToProcess[i]
            flashStageIndex = stage_names.index( flashStage )

            dataRow = activeFlashDatbase.loc[flashName]

            minProcessesRequired = databaseSettings[ flashStage ]["minNumProcesses"]
            maxProcessesRequired = databaseSettings[ flashStage ]["maxNumProcesses"]

            numAllowedProcesses = maxAllowedProcesses - numberRunningProcesses

            if numAllowedProcesses < minProcessesRequired:
                continue


        ## make folders if needed
            yearFolder =  os.path.join(  processedDataFolder, dataRow['year'])
            if not os.path.isdir( yearFolder ):
                os.mkdir( yearFolder )

            flashFolder = os.path.join( yearFolder, flashName)
            if not os.path.isdir( flashFolder ):
                os.mkdir( flashFolder )

            outputFolder = os.path.join( flashFolder, databaseSettings[current_state]['output_folderName'])
            if not os.path.isdir( outputFolder ):
                os.mkdir( outputFolder )

            outputLogFolder = os.path.join( outputFolder, databaseSettings[current_state]['log_folderName'])
            if not os.path.isdir( outputLogFolder ):
                os.mkdir( outputLogFolder )



        ## remove all previous completion file
            ## if there are completion files, they need to be removed first
            completionFname = databaseSettings[current_state]['succesfulCompletion_FileName']
            onlyLogFiles = [f for f in os.listdir(outputLogFolder) if os.path.isfile( os.path.join(outputLogFolder, f))]
            for f in onlyLogFiles:
                if f.startswith( completionFname ):
                    os.remove( os.path.join(outputLogFolder, f) )



        ## spin up processes
            actualNumProcesses = min(numAllowedProcesses, maxProcessesRequired)
            cmd = stage_commands[flashStageIndex]
            numCPUs = databaseSettings[ flashStage ]["maxCPU_perProcess"]

            slurm_settings = databaseSettings[ flashStage ]["slurm_settings"]
            slurm_settings["_output_filePrefix"] = os.path.join(outputLogFolder, 'slurm_out_')

            #jobIDs = [ spinupJob_function(cmd, settings_fname, flashName, jobNumber, actualNumProcesses, numCPUs, slurm_settings) for jobNumber in range(actualNumProcesses)]
            currentJobInterface = interfaces[current_state]( settings_fname, databaseSettings, flashFolder, flashName, dataRow )
            jobIDs = currentJobInterface.launch_job(use_slurm, slurm_settings, actualNumProcesses, numCPUs )

            if jobIDs is None:
                flashDatabase.at[ flashName, "sub_state" ] = "error"
                print('Flash:', flashName)
                print("    has error when attempting to start:", current_state)
                print()

            else:

                numberRunningProcesses += actualNumProcesses

                flashDatabase.at[ flashName, "slurm_processes" ] = jobIDs
                flashDatabase.at[ flashName, "sub_state" ] = "running"

                print('Flash:', flashName)
                print("    now running with:", actualNumProcesses, "jobs")
                print()

            #with database_manager(flashDatabase_location) as flashdatabase_manager:
            flashdatabase_manager.save_database( flashDatabase )



def deleteCompletionFiles(flashName, stage, settings_fname):

    databaseSettings = readSettings( settings_fname )

    with database_manager(databaseSettings['flashDatabase_location']) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()
    dataRow = flashDatabase.loc[ flashName ]

    outputFolder = os.path.join( databaseSettings["processed_data_loc"], dataRow['year'], flashName, databaseSettings[stage]['output_folderName'])
    outputLogFolder = os.path.join( outputFolder, databaseSettings[stage]['log_folderName'])

    completionFname = databaseSettings[stage]['succesfulCompletion_FileName']
    onlyLogFiles = [f for f in os.listdir(outputLogFolder) if os.path.isfile( os.path.join(outputLogFolder, f))]
    for f in onlyLogFiles:
        if f.startswith( completionFname ):
            os.remove( os.path.join(outputLogFolder, f) )




if __name__ == "__main__":
    ### TODO, save output to a file?
    run_processor( sys.argv[1] )