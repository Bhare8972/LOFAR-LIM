#!/usr/bin/env python3
from matplotlib import pyplot as plt

from LoLIM.make_planewave_fits import planewave_fits
from LoLIM.IO.raw_tbb_IO import  filePaths_by_stationName
from LoLIM.utilities import v_air

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    
    timeID = "D20180813T153001.413Z"
    stations = filePaths_by_stationName(timeID)
    
    for sname in ['RS503']:#['RS306','RS106', "RS310", "RS305"]:#stations.keys():
        
        print("processing", sname)
        
        RMSs, zeniths, azimuths = planewave_fits(timeID = timeID, 
                       station = sname, 
                       polarization  = 0, 
                       initial_block = 2500, 
                       number_of_blocks = 1000, 
                       pulses_per_block = 100, 
                       pulse_length = 50 + int(100/(v_air*5.0E-9)), 
                       min_amplitude = 50, 
                       upsample_factor = 4, 
                       min_num_antennas = 4,
                       verbose = False, ## doesn't do anything anyway
                        polarization_flips="polarization_flips.txt", bad_antennas="bad_antennas.txt", additional_antenna_delays = "ant_delays.txt",  
                        positive_saturation = 2046, negative_saturation = -2047, saturation_post_removal_length = 50, saturation_half_hann_length = 50)
        
        print(len(RMSs), "found planewaves")
        plt.hist(RMSs, bins=50, range=[0,10e-9])
        plt.show()
    
    