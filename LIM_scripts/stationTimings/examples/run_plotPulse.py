#!/usr/bin/env python3

from LoLIM.stationTimings.autoCorrelator3_plotPulse import processed_data_dir, plot_stations, plot_one_station


timeID = "D20180813T153001.413Z"
polarization = 0 ## 0 for even, 1 for odd
processed_data_folder = processed_data_dir(timeID)
known_station_delays = {
'CS001' :  2.24402739732e-06 , ## diff to guess: -2.68956819291e-11
'CS003' :  1.40676569926e-06 , ## diff to guess: -4.23051273218e-11
'CS004' :  4.39115646134e-07 , ## diff to guess: 1.32497890812e-10
'CS005' :  -2.17309775134e-07 , ## diff to guess: -8.19892447879e-11
'CS006' :  4.29252905591e-07 , ## diff to guess: 4.41313905417e-11
'CS007' :  3.97501953962e-07 , ## diff to guess: 5.39883307237e-11
'CS011' :  -5.91845933518e-07 , ## diff to guess: -2.82367035135e-11
'CS013' :  -1.81193091401e-06 , ## diff to guess: -8.45103235983e-11
'CS017' :  -8.45984892957e-06 , ## diff to guess: 2.31667660655e-11
'CS021' :  9.42037965136e-07 , ## diff to guess: -3.10913700271e-10
'CS024' :  2.32921021773e-06 , ## diff to guess: 1.89779072741e-10
'CS026' :  -9.26794152956e-06 , ## diff to guess: 5.53043008843e-10
'CS030' :  -2.73596745154e-06 , ## diff to guess: 4.79580434118e-09
'CS031' :  6.28184866031e-07 , ## diff to guess: 5.42350972736e-10
'CS032' :  -1.5428791754e-06 , ## diff to guess: 5.20505319831e-10
'CS101' :  -8.21607551862e-06 , ## diff to guess: 5.07291310758e-10
'CS103' :  -2.85888087157e-05 , ## diff to guess: 1.01141780869e-09
'CS201' :  -1.05132077289e-05 , ## diff to guess: 2.2699504661e-10
'CS301' :  -6.8785267902e-07 , ## diff to guess: 3.96206782286e-10
'CS302' :  -5.26911060716e-06 , ## diff to guess: 9.58977136518e-10
'CS501' :  -9.62997534216e-06 , ## diff to guess: 2.18729360558e-11
'RS106' :  6.77721513352e-06 , ## diff to guess: 1.86364483837e-08
'RS205' :  7.09015980791e-06 , ## diff to guess: 1.44846539114e-09
'RS306' :  7.40395520476e-06 , ## diff to guess: 2.12755514376e-08
'RS307' :  7.77981562261e-06 , ## diff to guess: 6.94902781034e-08
'RS310' :  7.71125768236e-06 , ## diff to guess: 5.77958870648e-07
'RS406' :  6.96030539125e-06 , ## diff to guess: 1.1548041502e-08
'RS407' :  6.82795434715e-06 , ## diff to guess: 4.93583335514e-09
'RS409' :  7.35369200865e-06 , ## diff to guess: 1.87779430653e-07
'RS503' :  6.92309493137e-06 , ## diff to guess: 7.96861976983e-10
'RS508' :  6.36132886902e-06 , ## diff to guess: 3.03019810557e-09
    }


## uncomment line below to load station delays from a file
#known_station_delays = read_station_delays(processed_data_folder + '/station_delays.txt') 



plot_stations(timeID, 
              polarization=0, ##0 is even. 1 is odd 
              input_file_name=processed_data_folder + "/correlate_foundPulses/potSource_1.h5", 
              source_XYZT = [ 2.82860910062 , -74260.9789353 , 0.0 , 1.56440508452 ], 
              known_station_delays = known_station_delays, 
              stations = "RS", ## all, RS, or CS
              referance_station="CS002", 
              min_antenna_amplitude=10)



plot_one_station(timeID, 
              polarization=0, ##0 is even. 1 is odd 
              input_file_name=processed_data_folder + "/correlate_foundPulses/potSource_1.h5", 
              source_XYZT = [ 2.82860910062 , -74260.9789353 , 0.0 , 1.56440508452 ], 
              known_station_delays = known_station_delays, 
              station = "CS0030",
              referance_station="CS002", 
              min_antenna_amplitude=10)



