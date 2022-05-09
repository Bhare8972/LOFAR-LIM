#!/usr/bin/env python3

from LoLIM.stationTimings.plot_multiple_data_all_stations import plot_blocks


## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = "D20180813T153001.413Z"
    initial_block =int(   1.1600369/((2**16)*(5.0E-9)) - 7  )#3600
    num_blocks = 5#15
    block_size = 2**16
    points = [ int(i*block_size + initial_block*block_size) for i in range(num_blocks) ]
    
    guess_flash_location = [2.43390916381, -50825.1364969, 0.0, 1.56124908739] #[ -3931.11275309 , -50046.0631816 , 3578.27013805 , 1.57679214101 ] ## use this if time delays are real and not apparent. Set to None when delays above are apparent delays, not real delays
    
    guess_station_delays = {
#"CS002": 0.0,
            
#            "RS208":  6.97951240511e-06 , ##??
#            "RS210": 0.0, ##??
#            
#'RS305' :  7.1934989871e-06 , ## diff to guess: 1.37699440122e-09 ## station has issues
#            
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

#'CS002': 0.0, 'RS208': 6.97951240511e-06, 'RS210': 0.0, 'RS305': 7.1934989871e-06, 'CS001': 2.24398052932e-06, 'CS003': 1.40671433963e-06, 'CS004': 4.39254549298e-07, 'CS005': -2.17392408269e-07, 'CS006': 4.29289039189e-07, 'CS007': 3.97558509721e-07, 'CS011': -5.91906970684e-07, 'CS013': -1.81204525376e-06, 'CS017': -8.45993703135e-06, 'CS021': 9.41558820961e-07, 'CS024': 2.32937078961e-06, 'CS026': -9.27244674102e-06, 'CS030': -2.73605732527e-06, 'CS031': 6.35936979893e-07, 'CS032': -1.54249405457e-06, 'CS101': -8.21575780092e-06, 'CS103': -2.85893646507e-05, 'CS201': -1.05133457098e-05, 'CS301': -6.86328550514e-07, 'CS302': -5.26891242658e-06, 'CS501': -9.62998566046e-06, 'RS106': 6.76765397484e-06, 'RS205': 7.08992058615e-06, 'RS306': 7.39393907628e-06, 'RS307': 7.7449641133e-06, 'RS310': 7.41170595821e-06, 'RS406': 6.95522873621e-06, 'RS407': 6.8262423552e-06, 'RS409': 7.25738861722e-06, 'RS503': 6.92336265463e-06, 'RS508': 6.36019852868e-06}
'CS002': 0.0, 
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
            
    referance_station = "CS002" ## only needed if using real delays, via the location on next line
    
    plot_blocks(timeID, 
                block_size, 
                points, 
                guess_station_delays, 
                guess_location = guess_flash_location, 
                bad_stations=["CS401", "CS031", "CS026", "RS210", "RS106"], 
                polarization_flips="polarization_flips.txt", 
                bad_antennas = "bad_antennas.txt", 
                additional_antenna_delays = "ant_delays.txt", 
                do_remove_saturation = True, 
                do_remove_RFI = True, 
                positive_saturation = 2046, 
                negative_saturation = -2047, 
                saturation_post_removal_length = 50, 
                saturation_half_hann_length = 5, 
                referance_station = "CS002")
    
    
    