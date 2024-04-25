#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir, logger
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.signal_processing import num_double_zeros

from os import mkdir
from os.path import isdir


## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/data/lightning_data"
utilities.default_processed_data_loc = "/home/hare/processed_files"

if __name__ == "__main__":
    timeID = "D20230713T110220.737Z"
    block_size = 2**16
    output_folder = "/find_percent_data_dropped"
    initial_block = 1000
    final_block = 2000
    
    level_to_catch = 50  # if data dropeed is larger than this, then print antennas again once done

    raw_fpaths = filePaths_by_stationName(timeID)
 
    processed_data_dir = processed_data_dir(timeID)
    
    output_fpath = processed_data_dir + output_folder
    if not isdir(output_fpath):
        mkdir(output_fpath)
        
    log = logger()
    log.set(output_fpath+'/log.txt', True)
    log.take_stdout()
        
   
    bad_ants = []
    for station, fpaths in raw_fpaths.items():
        print( station )
        
        TBB_data = MultiFile_Dal1( fpaths )
        
        for i,ant_name in enumerate(TBB_data.get_antenna_names()):
            num_data_points = 0
            num_zeros = 0
            for block in range(initial_block,final_block):
            
                data = TBB_data.get_data(block*block_size, block_size, antenna_index=i)
                
                num_zeros += num_double_zeros( data )
                num_data_points += block_size
                
            F = 100.0*float(num_zeros)/num_data_points
            print(i, ant_name, F)

            if F>level_to_catch:
                bad_ants.append( [ ant_name, F ] )
            
        print()

    print('bad antennas')
    for a,f in bad_ants:
        print(a, f)
