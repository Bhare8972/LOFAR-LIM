#!/usr/bin/env python3

from LoLIM.stationTimings.autoCorrelator3_stochastic_fitter import run_fitter   

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

guess_timings = {
#'RS305' :  7.20258598963e-06 , ## diff to guess: 1.37699440122e-09 ## station has issues

'CS001' :  2.244054293e-06 , ## diff to guess: 4.05925699175e-11
'CS003' :  1.40680800439e-06 , ## diff to guess: -2.64288001635e-10
'CS004' :  4.38983148243e-07 , ## diff to guess: -1.11651850796e-09
'CS005' :  -2.17227785889e-07 , ## diff to guess: 4.28812235945e-10
'CS006' :  4.292087742e-07 , ## diff to guess: -1.4552343702e-09
'CS007' :  3.97447965631e-07 , ## diff to guess: -1.94971279817e-09
'CS011' :  -5.91817696814e-07 , ## diff to guess: -2.26507290072e-09
'CS013' :  -1.81184640369e-06 , ## diff to guess: 1.06583355832e-09
'CS017' :  -8.45987209634e-06 , ## diff to guess: -4.67066994288e-09
'CS021' :  9.42348878836e-07 , ## diff to guess: 1.86858037013e-08
'CS024' :  2.32902043866e-06 , ## diff to guess: -1.42377481358e-09
'CS026' :  -9.26849457257e-06 , ## diff to guess: -9.26849457257e-06
'CS030' :  -2.74076325588e-06 , ## diff to guess: -1.08734205093e-09
'CS031' :  6.27642515058e-07 , ## diff to guess: 6.27642515058e-07
'CS032' :  -1.54339968072e-06 , ## diff to guess: -2.35697068994e-06
'CS101' :  -8.21658280993e-06 , ## diff to guess: -1.10456258121e-09
'CS103' :  -2.85898201335e-05 , ## diff to guess: -1.17407311792e-09
'CS201' :  -1.05134347239e-05 , ## diff to guess: -2.94392486115e-09
'CS301' :  -6.88248885802e-07 , ## diff to guess: -1.03586837929e-09
'CS302' :  -5.2700695843e-06 , ## diff to guess: -1.37398765225e-09
'CS501' :  -9.6299972151e-06 , ## diff to guess: 3.04189787168e-10
'RS106' :  6.75857868514e-06 , ## diff to guess: -1.19989551401e-08
'RS205' :  7.08871134252e-06 , ## diff to guess: -1.67397554153e-09
'RS306' :  7.38267965332e-06 , ## diff to guess: -1.35640391422e-08
'RS307' :  7.71032534451e-06 , ## diff to guess: -4.24458592156e-08
'RS310' :  7.13329881171e-06 , ## diff to guess: -3.47171528565e-07
'RS406' :  6.94875734975e-06 , ## diff to guess: -7.67966538289e-09
'RS407' :  6.82301851379e-06 , ## diff to guess: -3.75606984269e-09
'RS409' :  7.165912578e-06 , ## diff to guess: -1.13161412681e-07
'RS503' :  6.92229806939e-06 , ## diff to guess: -1.27556263665e-09
'RS508' :  6.35829867091e-06 , ## diff to guess: -2.61655916099e-09

        }
    
    
known_sources = [0,10] ## note that the index here is file_index + source_index*10

### locations of fitted sources
guess_source_locations = {
 0 :[ 2.77925728772 , -50538.649715 , 0.0 , 1.56125004432 ],
10 :[ 2.8286091023 , -73029.5420716 , 0.0 , 1.56440919296 ],
}

### polarization of fitted sources
known_polarizations = {
 0 : 0 ,
10 : 0 ,
}
    
    
## these are stations to exclude
stations_to_exclude = {
        0:[],
       10:["CS302", "CS301", "CS031", "CS026","RS106","RS205","RS306","RS307","RS310","RS406","RS407","RS409","RS503","RS508", "CS101", "CS032"],
        }

antennas_to_exclude = {
        0:[],
       10:[],
        }


bad_antennas = [
        ##CS002
        ##CS003
        ##CS004
        ##CS005
        ##CS006
        ##CS007
        ##CS011
        ##CS013
        ##CS017
        ##CS021
        ##CS030
        ##CS032
        ##CS101
        ##CS103
        ##CS301
        ##CS302
        ##CS401
        ##CS501
        ##RS208
        ##RS306
        ##RS307
        ##RS310
        ##RS406
        ##RS409
        ##RS503  
        ##RS508
        ##RS509
                ]



run_fitter(timeID="D20180813T153001.413Z", 
           output_folder = "autoCorrelator_fitter",
           pulse_input_folders = ["correlate_foundPulses"],
           guess_timings = guess_timings,
           souces_to_fit=known_sources, ## note that the index here is file_index + source_index*10
           guess_source_locations=guess_source_locations,
           source_polarizations=known_polarizations,
           source_stations_to_exclude=stations_to_exclude,
           source_antennas_to_exclude=antennas_to_exclude,
           bad_ants=bad_antennas,
           ref_station="CS002",
           min_ant_amplitude=10,
           max_stoch_loop_itters = 2000,
           min_itters_till_convergence = 100,
           initial_jitter_width = 100000E-9,
           final_jitter_width = 1E-9,
           cooldown_fraction = 10.0,
           strong_cooldown_fraction = 100.0,
           fitter = "dt" ##CHOOSE: dt, dt2, and locs,default is dt. Oscilate between dt and dt2 for best results. locs only fits locations and keeps delays constant
           )