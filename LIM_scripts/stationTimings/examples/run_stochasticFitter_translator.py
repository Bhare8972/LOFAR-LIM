#!/usr/bin/env python3

import numpy as np

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

#refractivity = 0.000293*2
#utilities.v_air = utilities.C/(1.0+refractivity)

from LoLIM.stationTimings.autoCorrelator3_stochastic_fitter_translater import run_fitter   
guess_timings = {

'CS001' :  2.22594112593e-06 , ## diff to guess: -2.03808220893e-11
'CS003' :  1.40486809199e-06 , ## diff to guess: 1.38253423356e-12
'CS004' :  4.30660422831e-07 , ## diff to guess: -6.23906907784e-12
'CS006' :  4.34184493916e-07 , ## diff to guess: 2.14758738162e-12
'CS007' :  4.01114093971e-07 , ## diff to guess: 4.13871757537e-12
'CS011' :  -5.8507310781e-07 , ## diff to guess: 2.48144545939e-12
'CS013' :  -1.81346936224e-06 , ## diff to guess: 6.68830181871e-12
'CS017' :  -8.4374268431e-06 , ## diff to guess: 1.77463333928e-11
'CS024' :  2.31979457875e-06 , ## diff to guess: -1.9906186238e-11
'CS026' :  -9.23248915119e-06 , ## diff to guess: 3.03730975849e-11
'CS030' :  -2.74190858905e-06 , ## diff to guess: 1.39799986061e-11
'CS032' :  -1.5759048054e-06 , ## diff to guess: -3.19342188075e-11
'CS101' :  -8.16744875658e-06 , ## diff to guess: 5.33916553344e-11
'CS103' :  -2.85149677531e-05 , ## diff to guess: 5.70630260641e-11
'CS201' :  -1.04838101677e-05 , ## diff to guess: 1.54806624706e-11
'RS205' :  7.00165432704e-06 , ## diff to guess: -1.95632200609e-10
'RS208' :  6.87005906191e-06 , ## diff to guess: -9.82176687537e-10
'CS301' :  -7.21061332925e-07 , ## diff to guess: -4.32337696006e-11
'CS302' :  -5.36006158172e-06 , ## diff to guess: -1.02567927779e-10
'CS501' :  -9.60769807486e-06 , ## diff to guess: 3.95090180021e-11
'RS503' :  6.92169621854e-06 , ## diff to guess: 8.61637302128e-11
'RS210' :  6.77016324355e-06 , ## diff to guess: -1.95983879239e-09
'RS307' :  6.84081748485e-06 , ## diff to guess: -1.40219978817e-09
'RS406' :  6.96855897059e-06 , ## diff to guess: 1.61461113386e-10
'RS407' :  7.03053077398e-06 , ## diff to guess: 4.46846181079e-10
'RS508' :  6.99307140404e-06 , ## diff to guess: 8.81735114658e-10
        }
    
known_sources = [60, 70, 90, 110, 180, 200]


### locations of fitted sources
guess_source_locations = {

60 :[ -37271.6815412 , -11373.6155251 , 7249.50552983 , 0.987339222361 ],
70 :[ -39825.7854173 , -9939.98606304 , 7346.71602204 , 0.987597345271 ],
90 :[ -39149.3745702 , -9738.89193919 , 7595.47885902 , 0.989187161303 ],
110 :[ -39111.2435955 , -9738.28744472 , 7521.93504761 , 0.989723242319 ],
180 :[ -37660.8913437 , -11662.5335688 , 7414.11898808 , 0.981013659193 ],
200 :[ -38443.5588294 , -10039.9482497 , 7160.94046621 , 0.979045605781 ],

}

### polarization of fitted sources
known_polarizations = {
60 : 1 ,
70 : 1 ,
90 : 0 ,
110 : 0 ,
180 : 0 ,
200 : 1 ,
}
    
#0. 2,1    

## these are stations to exclude
stations_to_exclude = {
        60: [],
        70: [],
        90: [],
        110: ['RS208'], ## I can't figure this one out
        180: [],
        200: [],
        }

antennas_to_exclude = {
        60: ['001011094'],
        70: ['147011094'],
        90: ['017009078', '130009078', '130001014'],
        110: [],
        180: [],
        200: ['147003030', '147009078'],
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
        ##CS026
        ##CS030
        ##CS032
        ##CS101
        ##CS103
        ##CS301
        ##CS302
        ##CS401
        ##CS501
        ##RS106
        ##RS205
        ##RS208
        ##RS305
        ##RS306
        ##RS307
        '147001014',
        ##RS310
        ##RS406
        ##RS409
        ##RS503  
        ##RS508
        ##RS509
                ]


run_fitter(timeID="D20180921T194259.023Z", 
           output_folder = "autoCorrelator_fitter_translation_test",
           pulse_input_folders = ["pulse_finding"],
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
           X_deviations = np.linspace(-1000,1000,50),
           Y_deviations = np.linspace(-1000,1000,50),
           Z_deviations = np.linspace(-2000,5000,100),
           
           )