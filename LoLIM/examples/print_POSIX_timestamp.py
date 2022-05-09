#!/usr/bin/env python3

from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    
    timeID = "D20170929T202255.000Z"
    
    raw_fpaths = filePaths_by_stationName(timeID)
    TBB_data = MultiFile_Dal1( raw_fpaths["CS002"] )
    print(TBB_data.get_timestamp())
        
        