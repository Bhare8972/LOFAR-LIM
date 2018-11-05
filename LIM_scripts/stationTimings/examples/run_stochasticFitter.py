#!/usr/bin/env python3

from LoLIM.stationTimings.autoCorrelator3_stochastic_fitter import run_fitter   

guess_timings = {
#'RS305' :  7.20258598963e-06 , ## diff to guess: 1.37699440122e-09 ## station has issues

'CS001' :  2.24401370043e-06 , ## diff to guess: 1.29235299643e-11
'CS003' :  1.40707229239e-06 , ## diff to guess: 7.82040168843e-12
'CS004' :  4.40099666751e-07 , ## diff to guess: 9.61678512336e-12
'CS005' :  -2.17656598125e-07 , ## diff to guess: 4.84705827503e-12
'CS006' :  4.3066400857e-07 , ## diff to guess: 4.14197713031e-12
'CS007' :  3.99397678429e-07 , ## diff to guess: 5.94789283395e-12
'CS011' :  -5.89552623913e-07 , ## diff to guess: 7.1660845422e-12
'CS013' :  -1.81291223725e-06 , ## diff to guess: 1.51169225377e-11
'CS017' :  -8.4552014264e-06 , ## diff to guess: 2.2686025574e-11
'CS021' :  9.23663075135e-07 , ## diff to guess: 0.0
'CS024' :  2.33044421347e-06 , ## diff to guess: 1.05806343653e-11
'CS026': 0.0,
'CS030' :  -2.73967591383e-06 , ## diff to guess: 5.67213694396e-11
'CS031':    0.0,
'CS032' :  8.13571009217e-07 , ## diff to guess: 7.6275878917e-11
'CS101' :  -8.21547824735e-06 , ## diff to guess: 5.62878278313e-11
'CS103' :  -2.85886460604e-05 , ## diff to guess: 3.66485536647e-10
'CS201' :  -1.0510490799e-05 , ## diff to guess: 7.84117944321e-11
'CS301' :  -6.87213017423e-07 , ## diff to guess: 1.12597989423e-11
'CS302' :  -5.26869559665e-06 , ## diff to guess: 2.91420217558e-10
'CS501' :  -9.63030140489e-06 , ## diff to guess: 1.96524840723e-11
'RS106' :  6.77057764028e-06 , ## diff to guess: 8.64777505687e-09
'RS205' :  7.09038531806e-06 , ## diff to guess: 4.90898285216e-10
'RS306' :  7.39624369246e-06 , ## diff to guess: 1.00577633859e-08
'RS307' :  7.75277120373e-06 , ## diff to guess: 3.30790236347e-08
'RS310' :  7.48047034028e-06 , ## diff to guess: 2.75667095084e-07
'RS406' :  6.95643701513e-06 , ## diff to guess: 5.33683986172e-09
'RS407' :  6.82677458363e-06 , ## diff to guess: 2.2068065121e-09
'RS409' :  7.27907399068e-06 , ## diff to guess: 8.93964436744e-08
'RS503' :  6.92357363203e-06 , ## diff to guess: 2.13873992035e-10
'RS508' :  6.36091523007e-06 , ## diff to guess: 0.000429378832772
        }
    
    
known_sources = [0,10] ## note that the index here is file_index + source_index*10

### locations of fitted sources
guess_source_locations = {
 0 :[ 2.82860910223 , -50739.3349291 , 0.0 , 1.56124937368 ],
10 :[ 2.82860910223 , -50739.3349291 , 0.0 , 1.56124937368 ],
}

### polarization of fitted sources
known_polarizations = {
 0 : 0 ,
10 : 0 ,
}
    
    
## these are stations to exclude
stations_to_exclude = {
        0:[],
       10:["CS302", "CS301", "CS031", "CS026","RS106","RS205","RS306","RS307","RS310","RS406","RS407","RS409","RS503","RS508", "CS101", "CS032", "CS030"],
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
           strong_cooldown_fraction = 100.0)