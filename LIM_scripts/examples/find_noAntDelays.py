#!/usr/bin/env python3

""" Some antennas don't have a callibrated timing delay. This file checks for those antennas """
import numpy as np

from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    
    timeID = "D20180809T141413.250Z"
    threshold = 1.0E-16
    
    raw_fpaths = filePaths_by_stationName(timeID)
    
    for station, fpaths in raw_fpaths.items():
        print(station)
        TBB_data = MultiFile_Dal1( fpaths)
        antenna_names = TBB_data.get_antenna_names()
        dipole_delays = TBB_data.get_timing_callibration_delays()
        for pair_i in range( int(len(antenna_names)/2) ):
            even_delay = dipole_delays[ pair_i*2 ]
            odd_delay  = dipole_delays[ pair_i*2 + 1]
            
            even_check= np.abs(even_delay) < threshold
            odd_check = np.abs(odd_delay) < threshold
            if even_check and odd_check:
                print(antenna_names[pair_i*2], even_delay, odd_delay)
        print()
        print()
        
        