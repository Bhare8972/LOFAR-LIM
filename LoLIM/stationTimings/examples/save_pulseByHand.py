#!/usr/bin/env python3

from LoLIM.stationTimings.autoCorrelator3_part1_selfPeakFind2 import save_Pulse

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__=="__main__":
    
    timeID = "D20180813T153001.413Z"
    output_folder = "correlate_foundPulses"
    
    event_index = 7
    
    pulse_time = 1.1601864
    window_width = 5E-6 ## in s, centered on pulse_time
    
    
    skip_stations = []
    
    ## time delay, in seconds, used to align the peaks
    guess_station_delays = {
#"CS002": 0.0,
#            
#            "RS208":  6.97951240511e-06 , ##??
#            "RS210": 0.0, ##??
            
#'RS305' :  7.1934989871e-06 , ## diff to guess: 1.37699440122e-09 ## station has issues
            
#'CS001' :  2.22608533961e-06 , ## diff to guess: 4.78562753903e-11
#'CS003' :  1.40501165264e-06 , ## diff to guess: 2.56900601842e-11
#'CS004' :  4.32211945847e-07 , ## diff to guess: 3.20360474459e-11
#'CS005' :  -2.18380807142e-07 , ## diff to guess: -3.01646125334e-12
#'CS006' :  4.34522296073e-07 , ## diff to guess: -2.94819663886e-11
#'CS007' :  4.01400128492e-07 , ## diff to guess: -1.83203313815e-11
#'CS011' :  -5.8302961409e-07 , ## diff to guess: -7.32960828329e-11
#'CS013' :  -1.81462826555e-06 , ## diff to guess: 5.9737204973e-11
#'CS017' :  -8.43481910796e-06 , ## diff to guess: -1.54596823704e-10
#'CS021' :  9.21050369068e-07 , ## diff to guess: 1.24261191845e-10
#'CS024' :  2.32427672269e-06 , ## diff to guess: -8.77497333155e-11
#'CS026' :  -9.22825332003e-06 , ## diff to guess: -2.66372216731e-10
#'CS030' :  -2.7456803478e-06 , ## diff to guess: 1.39204321291e-10
#'CS031' :  6.01413419127e-07 , ## diff to guess: 2.53209023786e-10
#'CS032' :  -1.57741838792e-06 , ## diff to guess: 1.846326379e-10
#'CS101' :  -8.16615435706e-06 , ## diff to guess: -2.53208768089e-10
#'CS103' :  -2.85091400004e-05 , ## diff to guess: -7.27782381276e-10
#'CS201' :  -1.04793635499e-05 , ## diff to guess: -3.0681327272e-10
#'CS301' :  -7.17561119226e-07 , ## diff to guess: 5.88567275111e-11
#'CS302' :  -5.36129780715e-06 , ## diff to guess: 3.35483144602e-10
#'CS501' :  -9.61278903257e-06 , ## diff to guess: 6.00305103156e-11
#'RS106' :  7.02924807891e-06 , ## diff to guess: -7.25879447838e-09
#'RS205' :  7.0038838679e-06 , ## diff to guess: -8.49291760387e-10
#'RS306' :  6.96711967265e-06 , ## diff to guess: -2.62120449015e-09
#'RS307' :  6.75696202824e-06 , ## diff to guess: -1.3075950508e-08
#'RS310' :  2.41269896248e-05 , ## diff to guess: -1.10397627992e-07
#'RS406' :  6.9110087838e-06 , ## diff to guess: -9.59008754502e-10
#'RS407' :  6.98525430598e-06 , ## diff to guess: -3.00606097676e-10
#'RS409' :  6.46668496272e-06 , ## diff to guess: -4.04928527945e-08
#'RS503' :  6.94443898031e-06 , ## diff to guess: 3.84182560508e-10
#'RS508' :  6.98408223074e-06 , ## diff to guess: -2.67436047276e-09
    
'CS002': 0.0, 'RS208': 6.97951240511e-06, 'RS210': 0.0, 'RS305': 7.1934989871e-06, 'CS001': 2.24398052932e-06, 'CS003': 1.40671433963e-06, 'CS004': 4.39254549298e-07, 'CS005': -2.17392408269e-07, 'CS006': 4.29289039189e-07, 'CS007': 3.97558509721e-07, 'CS011': -5.91906970684e-07, 'CS013': -1.81204525376e-06, 'CS017': -8.45993703135e-06, 'CS021': 9.41558820961e-07, 'CS024': 2.32937078961e-06, 'CS026': -9.27244674102e-06, 'CS030': -2.73605732527e-06, 'CS031': 6.35936979893e-07, 'CS032': -1.54249405457e-06, 'CS101': -8.21575780092e-06, 'CS103': -2.85893646507e-05, 'CS201': -1.05133457098e-05, 'CS301': -6.86328550514e-07, 'CS302': -5.26891242658e-06, 'CS501': -9.62998566046e-06, 'RS106': 6.76765397484e-06, 'RS205': 7.08992058615e-06, 'RS306': 7.39393907628e-06, 'RS307': 7.7449641133e-06, 'RS310': 7.41170595821e-06, 'RS406': 6.95522873621e-06, 'RS407': 6.8262423552e-06, 'RS409': 7.25738861722e-06, 'RS503': 6.92336265463e-06, 'RS508': 6.36019852868e-06
   
            }
    
    referance_station = "CS002" ## only needed if using real delays, via the location on next line
    guess_flash_location = [2.43390916381, -50825.1364969, 0.0, 1.56124908739] #[ -3931.11275309 , -50046.0631816 , 3578.27013805 , 1.57679214101 ] ##ONLY USE THIS IF USED IN plot_multiple_data_all_stations.py!!!
    
    
    save_Pulse(timeID, 
               output_folder, 
               event_index, 
               pulse_time, 
               guess_station_delays, 
               skip_stations=skip_stations, 
               window_width=window_width,
               pulse_width=50, 
               block_size=2**16, 
               min_ant_amp=5.0, 
               guess_flash_location=guess_flash_location, 
               referance_station=referance_station, 
               polarization_flips="polarization_flips.txt", 
               bad_antennas="bad_antennas.txt", 
               additional_antenna_delays = "ant_delays.txt")