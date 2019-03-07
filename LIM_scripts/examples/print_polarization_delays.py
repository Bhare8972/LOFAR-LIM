#!/usr/bin/env python3

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName


## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = 'D20180813T153001.413Z'
    
    
    raw_fpaths = filePaths_by_stationName(timeID)
    
    for station, fpaths in raw_fpaths.items():
        TBB_data = MultiFile_Dal1( fpaths)
        print(station)
        above_tol = TBB_data.find_and_set_polarization_delay( verbose=True )
        print("even antennas above tolerance:", above_tol[::2])
        print()
        print()