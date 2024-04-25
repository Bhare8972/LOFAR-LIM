#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.signal_processing import remove_saturation
from LoLIM.findRFI import window_and_filter
from LoLIM.read_pulse_data import  read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays



## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/data/lightning_data"
utilities.default_processed_data_loc = "/home/hare/processed_files"

if __name__ == "__main__":
    timeID = "D20210605T055555.042Z"
    
    
    raw_fpaths = filePaths_by_stationName(timeID)
    
    for station, fpaths in raw_fpaths.items():
        TBB_data = MultiFile_Dal1( fpaths)
        locs = TBB_data.get_LOFAR_centered_positions()
        names = TBB_data.get_antenna_names()
        
        print(station)
        for even_ant_i in  range(0,len(locs),2):
            name = names[even_ant_i]
            loc = locs[even_ant_i]
            print(name, loc)
        print()
        
        