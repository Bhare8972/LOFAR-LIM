#!/usr/bin/env python3

""" This script loops over every antenna and every station, and plots the maximum in each block of data. Used to find location of lightning in a file"""

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName

from os import mkdir
from os.path import isdir

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/data/lightning_data"
utilities.default_processed_data_loc = "/home/hare/processed_files"

if __name__ == "__main__":
    timeID = "D20230713T110220.737Z"
    output_folder = "/max_over_blocks"
    block_size = 2**16
    
    X_axis_is_time_from_second = False ## false gives block number
    
    processed_data_dir = processed_data_dir(timeID)
    
    output_fpath = processed_data_dir + output_folder
    if not isdir(output_fpath):
        mkdir(output_fpath)
    
    #### get paths to raw data by station ####
    raw_fpaths = filePaths_by_stationName(timeID)
    for station in raw_fpaths.keys():
        
        print( "station:", station )
        
        #### open the data for this station ####
        TBB_data = MultiFile_Dal1( raw_fpaths[station] )
        TBB_data_start_times = TBB_data.get_time_from_second(force_file_delays=True) ## only needed if X_axis_is_time_from_second
        
        num_antennas = len( TBB_data.get_antenna_names() )
        num_blocks = int( np.min(TBB_data.get_nominal_data_lengths()) /block_size ) ### note that this throws away the last partial data block
        
        plt.figure()
        for antenna_i in range(num_antennas):
#            if antenna_i == 1:
#                break
            print("  antenna:", antenna_i, "/", num_antennas)
            
            data = np.zeros(num_blocks, dtype=float)
        
            for block_i in range(num_blocks):
                if not block_i % 1000:
                    print("    block percent:",block_i/float(num_blocks))
                    
                try:
                    antenna_block_max = np.max( TBB_data.get_data( block_i*block_size, block_size, antenna_index=antenna_i ) )
                except:
                    print("    error in block:", block_i)
                    antenna_block_max = 0
                    
                data[block_i] = antenna_block_max
                

            if X_axis_is_time_from_second:
                t0 = TBB_data_start_times[antenna_i]
                T = np.arange( num_blocks,dtype=np.double )
                T *= block_size*(5.0e-9)
                T += t0
                plt.plot(T, data)

            else:
                plt.plot(data)
#            plt.show()
        
        print("saving figure:", output_fpath+'/'+station+'.png')
        plt.savefig(output_fpath+'/'+station+'.png')
#        plt.show()
        plt.close()