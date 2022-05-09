#!/usr/bin/env python3


from datetime import datetime
from os import mkdir
from os.path import isdir

from LoLIM.get_phase_callibration import download_phase_callibrations
import LoLIM.utilities as utils
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    #### a full working example of opening a file, checking if it needs metadata, and downloading if necisary
    
    timeID = "D20180308T170417.500Z"
    history_folder = "./svn_phase_cal_history"
    get_all_timings = True ## if false, only gets ones needed
    mode = 'LBA_OUTER' ## set to None to get ALL files.
    
    skip = [] ##stations to skip
    
    
    if not isdir(history_folder):
        mkdir(history_folder)
        
    fpaths = filePaths_by_stationName(timeID)
    stations = fpaths.keys()
    
    for station in stations:
        if station in skip:
            continue
    
        TBB_data = MultiFile_Dal1( fpaths[station] )
        timestamp = datetime.fromtimestamp( TBB_data.get_timestamp() )
        
        if get_all_timings or TBB_data.needs_metadata():
            print("downloading for station:", station)
            download_phase_callibrations(station, history_folder, timestamp, utils.raw_data_dir(timeID), mode )
            
        print( station)
        print( TBB_data.get_timing_callibration_delays() )