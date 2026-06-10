#!/usr/bin/env python3

"""this file contains utilties for the initial_statisics stage of the pipeline. This calculates the maximum per block of data per station, and the fraction of dataloss and saturation"""

import sys
import os
import json

import pandas as pd
import numpy as np

from LoLIM import utilities as utils
from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.database import database_manager
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, open_L1_or_L2_data


import LoLIM.pipeline.cythonTools as cytool 


from LoLIM.pipeline.jobInterface import standard_jobInterface

class initialStatistics_jobInterface(standard_jobInterface):
    def stage_name(self):
        """return the name of this stage"""
        raise "initial_statistics"

    def command(self):
        """return string of command to run"""
        raise "python -m LoLIM.pipeline.initial_statistics"



def stationStatsCore(fileList, blockSize):

    #TBB_data = MultiFile_Dal1( filename_list=fileList )

    TBB_data = open_L1_or_L2_data(file_list=fileList)

    sname = TBB_data.get_station_name()
    ant_names = TBB_data.get_antenna_names()
    num_antennas = len(ant_names)
    saturation_max, saturation_min = TBB_data.getSaturationValues()

    num_blocks = int( np.min(TBB_data.get_nominal_data_lengths()) /blockSize ) ### note that this throws away the last partial data block

    maxOverBlocks = np.empty(num_blocks, dtype=int)          
    fracSaturation_perAntenna = {n:0 for n in ant_names}
    fracDblZero_perAntenna = {n:0 for n in ant_names}

    frac_saturation_by_block = np.empty(num_blocks, dtype=float) 
    frac_dblZero_by_block = np.empty(num_blocks, dtype=float) 


    data = None  ## so this will be allocated as an array of correct size and type, and then reused

    for block_i in range(num_blocks):
        if ( block_i%1000 )==0:
            print( 'initial stats:', block_i, '/', num_blocks)

        blockMax = 0
        sum_num_saturation = 0 
        sum_dblZero = 0 
        for antenna_i in range(num_antennas):
                    
            data = TBB_data.get_data( block_i*blockSize, blockSize, antenna_index=antenna_i, out=data )
            num_saturation, num_dbl_zeros, maximum = cytool.rawDataStatistics(data, satMax=saturation_max, satMin=saturation_min)

            if maximum > blockMax:
                blockMax = maximum

            sum_num_saturation += num_saturation
            sum_dblZero += num_dbl_zeros


            fracSaturation_perAntenna[ ant_names[antenna_i] ] += num_saturation#/float(len(data))
            fracDblZero_perAntenna[ ant_names[antenna_i] ] += num_dbl_zeros/float(len(data))

        maxOverBlocks[block_i] = blockMax
        frac_saturation_by_block[block_i] = sum_num_saturation/num_antennas
        frac_dblZero_by_block[block_i] = sum_dblZero/num_antennas



    fracSaturation_perAntenna = {n:d/num_blocks for n,d in fracSaturation_perAntenna.items()}
    fracDblZero_perAntenna = {n:d/num_blocks for n,d in fracDblZero_perAntenna.items()}


    d = {'sname':sname, 'ant_names':ant_names, 'saturation_max':saturation_max, 'saturation_min':saturation_min, 
        'blockSize':blockSize, 'num_blocks':num_blocks, 'maxOverBlocks':maxOverBlocks, 
        'fracSaturation_perAntenna':fracSaturation_perAntenna, 'fracDblZero_perAntenna':fracDblZero_perAntenna,
        'saturation_perBlock':frac_saturation_by_block,  'dblZero_perBlock':frac_dblZero_by_block }

    return d



def doStationStats(output_fname, fileList, stageSettings):

    blockSize = stageSettings['blockSize']

    d = stationStatsCore(fileList, blocksize)

    json.dump( d, fp=open(output_fname, 'w'),  cls=utils.JSON_CustomEncoder) 

def readInitialStatisticsFile( databaseSettings, flashOutputFolder, sname ):

    outputFolder = os.path.join(  flashOutputFolder, databaseSettings["initial_statistics"]['output_folderName'])
    fname = os.path.join( outputFolder, sname+'_stats.json' )

    return json.load( fp=open(fname, 'r'), cls=utils.JSON_CustomDecoder_maker() )



def RunStage_InitialStatistics(flashName, outputFolder, logFolderLocation, flashDatabaseData, jobNumber, maxJobs, databaseSettings, stageSettings):

    jobNumber = int(jobNumber)
    maxJobs = int(maxJobs)

    timeID = flashDatabaseData["TimeID"]
    if '.' not in timeID:
        print("timeID for flash", flashName, "is not complete (", timeID,")")
        return False

    raw_fpaths_dict = filePaths_by_stationName(timeID, raw_data_loc=databaseSettings['raw_data_loc'])

    sorted_stations = utils.natural_sort( list(raw_fpaths_dict.keys()) ) 


    Num = len(sorted_stations)
    Num_todo = Num/maxJobs
    workIndex = [ i*Num_todo for i in range(maxJobs) ]
    workIndex.append( Num )
    initial_index = int( workIndex[jobNumber] )
    final_index = int( workIndex[jobNumber+1] )

    for sname in sorted_stations[initial_index:final_index]:
        file_list = raw_fpaths_dict[sname]

        output_fname = os.path.join( outputFolder, sname+'_stats.json' )

        if os.path.isfile(output_fname):
            print('station:', sname, 'has stats file aleady. SKIPPING')
        else:
            print('doing', sname)
            doStationStats(output_fname, file_list, stageSettings)
            print('complete', sname)

    return True



if __name__ == "__main__":
    settingsFname  = sys.argv[1]
    flashName      = sys.argv[2]
    jobNumber      = sys.argv[3]
    maxJobs        = sys.argv[4]
    numCPUs        = sys.argv[5]


## open database and pipeline settings
    stageName = "initial_statistics"

    databaseSettings = readSettings( settingsFname )

    with database_manager(databaseSettings['flashDatabase_location']) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()

    dataRow = flashDatabase.loc[flashName]
    flashYear = dataRow['year']

    flashOutputFolder = os.path.join(  databaseSettings["processed_data_loc"], flashYear, flashName)

    outputFolder = os.path.join(  flashOutputFolder, databaseSettings[stageName]['output_folderName'])
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
    succsses = RunStage_InitialStatistics(flashName, outputFolder, logFolderLocation, dataRow, jobNumber, maxJobs, databaseSettings, stageSettings)



## check if good!
    if succsses:
        print('is succsesful!')

        completionFile = os.path.join( outputFolder, databaseSettings[stageName]['log_folderName'], databaseSettings[stageName]['succesfulCompletion_FileName']+"_"+jobNumber)
        with open(completionFile, 'w') as fout:
            fout.write("1")
    else:
        print('not succsesful')


    del log

