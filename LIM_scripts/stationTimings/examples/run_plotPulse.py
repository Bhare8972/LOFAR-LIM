#!/usr/bin/env python3

from LoLIM.stationTimings.autoCorrelator3_plotPulse import processed_data_dir, plot_stations, plot_one_station


timeID = "D20180813T153001.413Z"
polarization = 0 ## 0 for even, 1 for odd
processed_data_folder = processed_data_dir(timeID)
known_station_delays = {
'CS001' :  2.23545892775e-06 , ## diff to guess: 4.85371845818e-11
'CS003' :  1.40706114e-06 , ## diff to guess: 3.76512003389e-11
'CS004' :  4.40087678149e-07 , ## diff to guess: 4.06188421883e-11
'CS005' :  -2.17661571097e-07 , ## diff to guess: 2.4838391542e-11
'CS006' :  4.30668300056e-07 , ## diff to guess: 1.33514751606e-11
'CS007' :  3.99396015938e-07 , ## diff to guess: 1.99298165766e-11
'CS011' :  -5.89526252769e-07 , ## diff to guess: -3.55778543384e-12
'CS013' :  -1.81292414096e-06 , ## diff to guess: 5.3208524009e-11
'CS017' :  -8.45511666843e-06 , ## diff to guess: -3.2275726996e-11
            "CS026": 0.0,
            "CS030":  -2.74255354078e-06,
            "CS031":    0.0,
'CS024' :  2.33048426443e-06 , ## diff to guess: -1.20320577008e-11
#'CS030' :  -2.73964850878e-06 , ## diff to guess: 1.01919814397e-10
'CS032' :  8.13623908736e-07 , ## diff to guess: 1.17584958794e-10
'CS101' :  -8.21529470206e-06 , ## diff to guess: -6.33049157782e-11
'CS103' :  -2.85877168735e-05 , ## diff to guess: -2.0718875951e-10
'CS201' :  -1.05102568468e-05 , ## diff to guess: -8.62680343692e-11
'CS301' :  -6.87225513955e-07 , ## diff to guess: 4.63029402975e-11
'CS302' :  -5.26830684558e-06 , ## diff to guess: 2.31595712663e-10
'CS501' :  -9.63031122715e-06 , ## diff to guess: 6.09644328428e-11
'RS106' :  6.77266076081e-06 , ## diff to guess: -4.42419893354e-10
'RS205' :  7.09159338981e-06 , ## diff to guess: -1.9994579321e-10
'RS210' :  0.0 , ## diff to guess: 0.0
'RS305' :  7.20258598963e-06 , ## diff to guess: 1.37699440122e-09 ## station may be bad
'RS306' :  7.41517356774e-06 , ## diff to guess: 1.99906885173e-09
'RS307' :  7.81654906363e-06 , ## diff to guess: 4.67483493181e-09
'RS310' :  8.02640787741e-06 , ## diff to guess: 2.70571125987e-08
'RS406' :  6.96640988338e-06 , ## diff to guess: 0.000249997746786

            "RS407":  -1.14843+1.14849,
            "RS409":  7.02251772331e-06-1.14843+1.14848, 
            "RS503":  6.93934919654e-06-1.14843+1.14802,
            "RS508":  6.98208245779e-06-1.14843+1.148,
    }


## uncomment line below to load station delays from a file
#known_station_delays = read_station_delays(processed_data_folder + '/station_delays.txt') 



plot_stations(timeID, 
              polarization=0, ##0 is even. 1 is odd 
              input_file_name=processed_data_folder + "/correlate_foundPulses/potSource_0.h5", 
              source_XYZT = [ 0.0565507911682 , -4741.43610828 , 0.0 , 1.56140285104 ], 
              known_station_delays = known_station_delays, 
              stations = "RS", ## all, RS, or CS
              referance_station="CS002", 
              min_antenna_amplitude=10)



plot_one_station(timeID, 
              polarization=0, ##0 is even. 1 is odd 
              input_file_name=processed_data_folder + "/correlate_foundPulses/potSource_0.h5", 
              source_XYZT = [ 0.0565507911682 , -4741.43610828 , 0.0 , 1.56140285104 ], 
              known_station_delays = known_station_delays, 
              station = "CS002",
              referance_station="CS002", 
              min_antenna_amplitude=10)



