#!/usr/bin/env python3

"""FIX THIS"""

import sys
import os
import json

import pandas as pd
import numpy as np

from LoLIM import utilities as utils
from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.database import database_manager
from LoLIM.IO.raw_tbb_IO import open_L1_or_L2_data, filePaths_by_stationName
from LoLIM import findRFI_adv as RFI_adv

from LoLIM.pipeline.initial_statistics import readInitialStatisticsFile


from LoLIM.pipeline.jobInterface import standard_jobInterface

class findRFI_jobInterface(standard_jobInterface):
    def stage_name(self):
        """return the name of this stage"""
        raise "find_RFI"

    def command(self):
        """return string of command to run"""
        raise "python -m LoLIM.pipeline.run_find_RFI"


def readFindRFI_results( databaseSettings, flashOutputFolder, sname_list=None ):


    if isinstance( stageSettings['useRFIData_fromAnoutherFlash'], str)  and len(stageSettings['useRFIData_fromAnoutherFlash'])>0:

        OF = flashOutputFolder[:-1] if flashOutputFolder[-1]=='/' else flashOutputFolder
        year_folder = os.path.dirname(OF)

        flashOutputFolder = os.path.join( year_folder, stageSettings['useRFIData_fromAnoutherFlash'] )


    outputFolder = os.path.join(  flashOutputFolder, databaseSettings["find_RFI"]['output_folderName'])

    if sname_list is None:
        sname_list = [ f[0:5] for f in os.listdir(outputFolder) if f.endswith('_advFindRFI.json')]


    out_dict = {}

    for sname in sname_list:
        output_fname = os.path.join( outputFolder, sname+'_advFindRFI.json' )

        statRFI = RFI_adv.open_advFindRFI( timeID=None, folder=None, fname=output_fname )

        out_dict = RFI_adv.fold_advFindRFI_resultsTogether( out_dict, statRFI )

    return out_dict




def statsData_to_RFIBlocks(statsData, numBins, noise_max, threshold_F, min_consective_blocks):
    """
    This algorithm uses max_over_blocks data to find regions with minimal lightning. Typically for RFI-finding.
    The idea is that max over blocks follows a distribution, that is a sum of two distributions. There is a spike at low values due to noise plus a low-level uniform distribution extending over the full voltage range due to lightning
    
    Therefore we have a value (noise_max), that is larger than all noise, and we assume the distribution of max-over-blocks is flat above this level. So we find the average of max-over-blocks above this voltage level, and the max of 
    distributino of max-over-blocks below this level. Finally, the threshold is a fraction (XYX) between the two. This function than outputs a list of continuous blocks that are all below this threshold

    statsData should be a dictionary, that is output by the initial_statistics stage.
    other settings:
        numBins is number of histogram bins to bin the max-over-blocks data (should be roughly 150)
        noise_max should be the value of the voltages that is larger than all noise. E.G. all voltages above this cannot be due to normal noise.
        threshold_F is the fraction of distance from the ave over noise_max to the maximum below noise_max that the threshold is set to. Typically 0.25. If threshold_F is 0, the threshold is average above noise_max (i.e. higher, more permisive). If 1, than threshold is max below noise_max (i.e lower, less permiive)
        min_consective_blocks is minimun number of blocks that are consecutively below the chosen threshold. Typically 100.

    return blocks_start, block_end   Both of which are block-indeces. Such that the blocks between these two indeces (includeing blocks_start but not blocks_start) are all consistant with background and block_end-blocks_start >= min_consective_blocks
        If no such blocks are found, than blocks_start, block_end are both None
    """

    ## stats data:
    ## {'sname':sname, 'ant_names':ant_names, 'saturation_max':saturation_max, 'saturation_min':saturation_min, 
    ##  'blockSize':blockSize, 'num_blocks':num_blocks, 'maxOverBlocks':maxOverBlocks, 
    ##  'fracSaturation_perAntenna':fracSaturation_perAntenna, 'fracDblZero_perAntenna':fracDblZero_perAntenna }


    max_o_blocks = statsData['maxOverBlocks']
   
    ### bin the data
    hist, histedges = np.histogram(max_o_blocks, bins=numBins, range=[0, statsData['saturation_max']])

    ## cut off the upper bin of historgram, as that will be high due to saturating blocks
    hist_noSat = hist[:-1]
    histedges_noSat = histedges[:-1]
    histCenters_noSat = (histedges_noSat[1:] + histedges_noSat[:-1])*0.5


    ## find index that cuts at noise_max
    noise_max_index = np.searchsorted(histCenters_noSat,  noise_max)


    ## we calculate the average value of the "flat" part of the distrubtion above noise_max
    if noise_max_index >= len(hist_noSat):
        ave_level = 0
    else:
        ave_level = np.average( hist_noSat[noise_max_index:] )

    ## find peak of distribution below noise_max
    distNoisePeak_index = np.argmax( hist_noSat[:noise_max_index] )
    distNoisePeak = hist_noSat[ distNoisePeak_index ]

    ## and caluclate threshold
    number_threshold = ave_level + threshold_F*(distNoisePeak - ave_level)
    for i in range(distNoisePeak_index, noise_max_index):
        if hist_noSat[i] < number_threshold:
            voltage_threshold = histCenters_noSat[i]
            break


    ## finally find stretch of blocks below background
    current_block_start = None 
    current_block_end = None

    for block_i in range(len(max_o_blocks)): 

        if current_block_start is None:  ## previous block was below threshold
            if max_o_blocks[ block_i ] > voltage_threshold:
                current_block_start = block_i

        else:   ## previous block was above threshold
            if max_o_blocks[ block_i ] < voltage_threshold:
                current_block_end = block_i

                if (current_block_end-current_block_start) >= min_consective_blocks:
                    return current_block_start, current_block_end

    ## if this is reached, we never found the end of a suffienct train of blocks
    if current_block_start is None: ## we also did not find the start
        return None, None
    else:  ## we did have a start. Is the run long enough?
        current_block_end = block_i
        if (current_block_end-current_block_start) >= min_consective_blocks:
            return current_block_start, current_block_end

        else: ## run is NOT long enough
            return None, None





def doStationFindRFI(sname, flashOutputFolder, databaseSettings, output_fname, fileList, stageSettings):

    if isinstance( stageSettings['useRFIData_fromAnoutherFlash'], str)  and len(stageSettings['useRFIData_fromAnoutherFlash'])>0:
        ## then we do nothing 
        return True



    if stageSettings['use_userDefinedBlocks_forRFI']:
        start_block = stageSettings['userSet_startBlock']
        end_block = stageSettings['userSet_endBlock']

        block_size = databaseSettings['initial_statistics']['settings']['blockSize']

    else:
        ## use max-over-blocks to determine RFI block range

        initialStatsFile = readInitialStatisticsFile( databaseSettings, flashOutputFolder, sname )
        ## {'sname':sname, 'ant_names':ant_names, 'saturation_max':saturation_max, 'saturation_min':saturation_min, 
        ##  'blockSize':blockSize, 'num_blocks':num_blocks, 'maxOverBlocks':maxOverBlocks, 
        ##  'fracSaturation_perAntenna':fracSaturation_perAntenna, 'fracDblZero_perAntenna':fracDblZero_perAntenna }

        block_size = initialStatsFile['blockSize']

        start_block, end_block = statsData_to_RFIBlocks(initialStatsFile, numBins=stageSettings["auto_numBins"], noise_max=stageSettings["auto_noiseMax"], \
            threshold_F=stageSettings["auto_thresholdF"], min_consective_blocks=stageSettings["min_consective_blocks"])

        if start_block is None:
            print('cannot automatically find quiet region')
            return False


        
    num_blocks = stageSettings['num_blocks']


    #TBB_data = MultiFile_Dal1( fileList, force_metadata_ant_pos=True ) #, total_cal=cal_file?
    TBB_data = open_L1_or_L2_data(fileList)
    RFIdata_out = RFI_adv.FindRFI_adv(TBB_data, block_size, start_block, num_blocks, end_block-start_block, verbose=True, figure_location=None, num_dbl_z=1000)

    RFI_adv.save_advFindRFI( RFIdata_out, timeID=None, folder=None, fname=output_fname )

    return True


def RunStage_FindRFI(flashName, flashOutputFolder, outputFolder, logFolderLocation, flashDatabaseData, jobNumber, maxJobs, databaseSettings, stageSettings):

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

    all_succsesful = True
    for sname in sorted_stations[initial_index:final_index]:
        file_list = raw_fpaths_dict[sname]

        output_fname = os.path.join( outputFolder, sname+'_advFindRFI.json' )

        if os.path.isfile(output_fname):
            print('station:', sname, 'has RFI output file aleady. SKIPPING')
        else:
            print('doing', sname)
            succsesful = doStationFindRFI(sname,  flashOutputFolder, databaseSettings, output_fname, file_list, stageSettings)
            print('complete', sname, 'succsesful:', succsesful)

            all_succsesful = all_succsesful and succsesful

    return all_succsesful


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
    succsses = RunStage_FindRFI(flashName, flashOutputFolder, outputFolder, logFolderLocation, dataRow, jobNumber, maxJobs, databaseSettings, stageSettings)



## check if good!
    if succsses:
        print('is succsesful!')

        completionFile = os.path.join( outputFolder, databaseSettings[stageName]['log_folderName'], databaseSettings[stageName]['succesfulCompletion_FileName']+"_"+jobNumber)
        with open(completionFile, 'w') as fout:
            fout.write("1")
    else:
        print('not succsesful')


    del log

