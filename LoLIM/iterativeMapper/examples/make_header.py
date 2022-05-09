#!/usr/bin/env python3

"""use this script to make a new header for running the iterative mapper. First, run this script, then actually run the mapper"""


from LoLIM.iterativeMapper.iterative_mapper import make_header

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

out_folder = 'iterMapper_50_CS002'
#timeID = 'D20180813T153001.413Z'
#timeID = 'D20170929T202255.000Z'
timeID = 'D20180809T141413.250Z'
initial_datapoint = 1000*(2**16)
            
outHeader = make_header(timeID, initial_datapoint, 
   station_delays_fname = 'station_delays_CHRIS.txt', 
#   station_delays_fname = 'station_delays4.txt', 
   additional_antenna_delays_fname = 'ant_delays_CHRIS.txt', bad_antennas_fname = 'bad_antennas_CHRIS.txt', 
   pol_flips_fname = 'polarization_flips_CHRIS.txt'
   )
        
outHeader.stations_to_exclude = []
                     #[ 'RS407', 'RS409']#
                                 #'CS002', 'CS003', 'CS004', 'CS005', 'CS006', 'CS007', 'CS011', 'CS013', 'CS017']
                                 #'RS208', 'RS306', 'RS307', 'RS310', 'RS406', 'RS409', 'RS503', 'RS508', 'RS509'] 
                                 
#outHeader.stations_to_exclude = [ ]
                                 
outHeader.max_events_perBlock = 500

## other settings:

#outHeader.max_antennas_per_station = 6
outHeader.referance_station = 'CS002'
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





