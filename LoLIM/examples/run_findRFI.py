#!/usr/bin/env python3

"""this is a short script that runs findRFI and saves the results to a python pickle file"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir, logger
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
#from LoLIM.findRFI_OLD import FindRFI
from LoLIM.findRFI import FindRFI
#from LoLIM.findRFI_TST import FindRFI

from os import mkdir
from os.path import isdir
from pickle import dump

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = "D20210618T174657.311Z"
    output_folder = "/findRFI"
    out_fname = "/findRFI_results"
    block_size = 2**16
    initial_block = 1000
    num_blocks = 10
    max_blocks = 500
    
    skip_stations = []
    
    
    processed_data_dir = processed_data_dir(timeID)
    
    output_fpath = processed_data_dir + output_folder
    if not isdir(output_fpath):
        mkdir(output_fpath)
    
    log = logger()
    log.set(output_fpath+'/log.txt')
    log("timeID:", timeID)
    log("initial_block:", initial_block)
    log("num_blocks:", num_blocks)
    log("max_blocks:", max_blocks)
    log("skip_stations:", skip_stations)
#    log.take_stdout()
        
    
    #### get paths to raw data by station ####
    raw_fpaths = filePaths_by_stationName(timeID)
    
    output = {}
    station_log = logger()
    station_log.take_stdout()
    for station in raw_fpaths.keys():
        
        if station in skip_stations:
            continue
        
        path = output_fpath + '/' + station
        if not isdir(path):
            mkdir(path)
        
        station_log.set(path+'/log.txt')
        print("station", station)
            
        TBB_data = MultiFile_Dal1( raw_fpaths[station], force_metadata_ant_pos=True )
        fig_loc = path
        # fig_loc = "show"
        out = FindRFI(TBB_data, block_size, initial_block, num_blocks, max_blocks, verbose=True, figure_location=fig_loc, num_dbl_z=1000)
        
        if out is None:
            log("cannot find RFI for station:", station)
        else:
            output[station] = out

                
        print()
        
    with open(output_fpath+out_fname, 'wb') as fout:
        dump(output, fout)
        
        
        
        
        
        
        
        
        
        
        
        