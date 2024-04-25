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
    timeID = "D20190424T194432.504Z"

    XYZ = np.array([-23000.0, -10000.0, 4000.0])
    
    print('center:', XYZ)
    print()


    
    #### get paths to raw data by station ####
    raw_fpaths = filePaths_by_stationName(timeID)
    for station in raw_fpaths.keys():
        
       
        #### open the data for this station ####
        TBB_data = MultiFile_Dal1( raw_fpaths[station] )
        
        XYZs = TBB_data.get_LOFAR_centered_positions()
        stat_XYZ = np.average(XYZs,axis=0)
        print(station)
        print('  XYZ:', stat_XYZ)
        print(' D[km]:', np.linalg.norm(stat_XYZ-XYZ)/1000 )
        print()

        