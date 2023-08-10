#!/usr/bin/env python3

"""use this script to make a new header for running the iterative mapper. First, run this script, then actually run the mapper"""


from LoLIM.iterativeMapper.iterative_mapper import make_header

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
# utilities.default_raw_data_loc = "/home/brian/local_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

out_folder = 'iterMapper_9Aug2023Cal_evenAnts'
timeID = 'D20190424T194432.504Z'

initial_datapoint = 500*(2**16)
            
outHeader = make_header(timeID, initial_datapoint, 
   total_cal_fname = 'Cal_9Aug2023.txt'
   )
        
outHeader.stations_to_exclude = [ ]
                                 
outHeader.max_events_perBlock = 500

## other settings:

#outHeader.max_antennas_per_station = 6
outHeader.referance_station = 'CS002'

outHeader.use_even_antennas = True
#
#outHeader.blocksize = 2**16
#
#outHeader.remove_saturation = True
#outHeader.remove_RFI = True
#outHeader.positive_saturation = 2046
#outHeader.negative_saturation = -2047
#outHeader.saturation_removal_length = 50
#outHeader.saturation_half_hann_length = 50
#
#outHeader.hann_window_fraction = 0.1
#
#outHeader.num_zeros_dataLoss_Threshold = 10
#
#outHeader.min_amplitude = 15
#
#outHeader.upsample_factor = 8
outHeader.min_pulse_length_samples = 50
#outHeader.erasure_length = 20 ## num points centered on peak to not image again
#
#outHeader.guess_distance = 10000
#
#
#outHeader.kalman_devations_toSearch = 3 # +/- on either side of prediction
#
#outHeader.pol_flips_are_bad = True
#
#outHeader.antenna_RMS_info = "find_calibration_out"
#outHeader.default_expected_RMS = 1.0E-9
#outHeader.max_planewave_RMS = 1.0E-9
#outHeader.stop_chi_squared = 100.0
#
#outHeader.max_minimize_itters = 1000
#outHeader.minimize_ftol = 1.0E-11 #3.0E-16
#outHeader.minimize_xtol = 1.0E-7 # 3.0E-16
#outHeader.minimize_gtol = 3.0E-16


outHeader.run( out_folder )





