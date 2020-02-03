#!/usr/bin/env python3

from LoLIM.stationTimings.autoCorrelator3_fromLOC import save_EventByLoc

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__=="__main__":
    
    station_delays = {
'CS002':  0.0,
'CS003':   1.40436380151e-06 ,
'CS004':   4.31343360778e-07 ,
'CS005':   -2.18883924536e-07 ,
'CS006':   4.33532992523e-07 ,
'CS007':   3.99644095007e-07 ,
'CS011':   -5.85451477265e-07 ,
'CS013':   -1.81434735154e-06 ,
'CS017':   -8.4398374875e-06 ,
'CS021':   9.23663075135e-07 ,
'CS030':   -2.74255354078e-06,
'CS032':   -1.57305580305e-06,
'CS101':   -8.17154277682e-06,
'CS103':   -2.85194082718e-05,
'RS208':   6.97951240511e-06 ,
'CS301':   -7.15482701536e-07 ,
'CS302':   -5.35024064624e-06 ,
'RS306':   7.04283154727e-06,
'RS307':   6.96315727897e-06 ,
'RS310':   7.04140267551e-06,
'CS401':   -9.5064990747e-07 ,
'RS406':   6.96866309712e-06,
'RS409':   7.02251772331e-06,
'CS501':   -9.61256584076e-06 ,
'RS503':   6.93934919654e-06 ,
'RS508':   6.98208245779e-06 ,
'RS509':   7.01900854365e-06,
}
    
    station_delays['CS002'] = 0.0 ## need to add referance station
    
    save_EventByLoc(timeID = "D20170929T202255.000Z", 
                    XYZT = [-16794.30127223  , 9498.38995127  , 3297.47036309, 1.2642410207364205 ], 
                    station_timing_delays = station_delays, 
                    pulse_index = 24, 
                    output_folder = "callibrator_fromLOC_intOut4", 
                    pulse_width=50,
                    min_ant_amp=5,
                    upsample_factor = 4,
                    polarization_flips="polarization_flips.txt", 
                    bad_antennas="bad_antennas.txt", 
                    additional_antenna_delays = "ant_delays.txt")
    
    
    