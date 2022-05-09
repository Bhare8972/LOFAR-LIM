#!/usr/bin/env python3


from os import mkdir
from os.path import isdir

from pickle import load

from matplotlib import pyplot as plt
import numpy as np

from LoLIM.make_planewave_fits import planewave_fits
from LoLIM.IO.raw_tbb_IO import  filePaths_by_stationName
from LoLIM.utilities import v_air, processed_data_dir

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"


### TODO: plot a horizontal line at 1.5 ns, then save plots into a folder automatically

if __name__ == "__main__":
    
#    timeID = "D20180809T145549.143Z"
    timeID = "D20170929T202255.000Z"
    output_folder = "/planewave_RMS_histograms"
    

    stations = filePaths_by_stationName(timeID)
    
    processed_data_dir = processed_data_dir(timeID)
    output_fpath = processed_data_dir + output_folder
    if not isdir(output_fpath):
        mkdir(output_fpath)
        
    with open( processed_data_dir + "/findRFI/findRFI_results", 'rb' ) as fin:
        find_RFI = load(fin)
    
    for sname in stations.keys():
#        if (sname in ['CS002', 'CS003', 'CS004', 'CS005', 'CS006', 'CS007',
#                      'CS011', 'CS013', 'CS017', 'CS021', 'CS026']) or (sname not in find_RFI):
#            continue
    
        sname= 'CS002'
        
        print("processing", sname)
        
        RMSs, zeniths, azimuths = planewave_fits(timeID = timeID, 
                       station = sname, 
                       polarization  = 0, 
                       initial_block = 3500,
                       number_of_blocks = 500, 
                       pulses_per_block = 2, 
                       pulse_length = 50 + int(100/(v_air*5.0E-9)), 
                       min_amplitude = 50, 
                       upsample_factor = 4, 
                       min_num_antennas = 4,
                       max_num_planewaves = 1000,
                       verbose = False, ## doesn't do anything anyway
                       polarization_flips="polarization_flips.txt", bad_antennas="bad_antennas.txt", additional_antenna_delays = "ant_delays.txt",  
                       positive_saturation = 2046, negative_saturation = -2047, saturation_post_removal_length = 50, saturation_half_hann_length = 50)
        
        print(np.average(zeniths),np.average(azimuths) )
        
        print(len(RMSs), "found planewaves")
        plt.hist(RMSs, bins=50, range=[0,10e-9])
        plt.savefig(output_fpath+'/'+sname+'.png')
        ### TODO: fit a distribution and print the max and width
#        plt.show()
        plt.close()
        
        break
    
    