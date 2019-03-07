#!/usr/bin/env python3

from LoLIM.stationTimings.autoCorrelator3_fromLOC import save_EventByLoc

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__=="__main__":
    
    station_delays = {
'CS001' :  2.22387557646e-06 , ## diff to guess: 3.18517622545e-10
'CS003' :  1.40455863627e-06 , ## diff to guess: 1.53521313723e-11
'CS004' :  4.30900574133e-07 , ## diff to guess: 1.38365536384e-10
'CS005' :  -2.18815284186e-07 , ## diff to guess: 2.6508625035e-11
'CS006' :  4.34980864494e-07 , ## diff to guess: -9.330207262e-11
'CS007' :  4.01729497298e-07 , ## diff to guess: -8.09325165755e-11
'CS011' :  -5.823430971e-07 , ## diff to guess: -1.15198613492e-10
'CS013' :  -1.81465773882e-06 , ## diff to guess: -2.00763538069e-11
'CS017' :  -8.43258169953e-06 , ## diff to guess: -3.85056108665e-10
'CS021' :  9.19941039479e-07 , ## diff to guess: 2.05397653314e-10
'CS024' :  2.32318085224e-06 , ## diff to guess: 2.17960259389e-10
'CS030' :  -2.74560701447e-06 , ## diff to guess: -1.77137589853e-11
'CS032' :  -1.58157668878e-06 , ## diff to guess: 6.87814027615e-10
'CS101' :  -8.15953877184e-06 , ## diff to guess: -1.04463662949e-09
'CS103' :  -2.84990371249e-05 , ## diff to guess: -1.37752973241e-09
'CS201' :  -1.04757682248e-05 , ## diff to guess: -4.92204636901e-10
'CS301' :  -7.21792199501e-07 , ## diff to guess: 6.93388604766e-10
'CS302' :  -5.3731028034e-06 , ## diff to guess: 1.98583644899e-09
'CS501' :  -9.60968609077e-06 , ## diff to guess: -5.26058814995e-10
'RS106' :  7.08056273311e-06 , ## diff to guess: -2.21303338139e-09
'RS205' :  6.98980285689e-06 , ## diff to guess: 2.80223827456e-09
'RS306' :  6.9378252342e-06 , ## diff to guess: 9.87807334176e-09
'RS307' :  6.70507155702e-06 , ## diff to guess: 2.59243527917e-08
'RS310' :  2.45363311428e-05 , ## diff to guess: 0.0
'RS406' :  6.9239217611e-06 , ## diff to guess: 4.20859492066e-10
'RS407' :  7.02096706021e-06 , ## diff to guess: -4.72098717648e-09
'RS409' :  6.57692512977e-06 , ## diff to guess: 2.98049180677e-08
'RS503' :  6.94972035701e-06 , ## diff to guess: -8.5438566934e-10
'RS508' :  7.08148673507e-06 , ## diff to guess: -1.48805065579e-08
'RS208' :  6.84965668876e-06 , ## diff to guess: 2.93529464353e-08
}
    
    station_delays['CS002'] = 0.0 ## need to add referance station
    
    save_EventByLoc(timeID = "D20180813T153001.413Z", 
                    XYZT = [747.991021321, -51273.5997815, 6596.09365446, 1.56124616374], 
                    station_timing_delays = station_delays, 
                    pulse_index =0, 
                    output_folder = "correlate_foundPulses_byLOC_PolDelayTest", 
                    pulse_width=50,
                    min_ant_amp=5,
                    upsample_factor = 4,
                    polarization_flips="polarization_flips.txt", 
                    bad_antennas="bad_antennas.txt", 
                    additional_antenna_delays = "ant_delays.txt")
    
    
    