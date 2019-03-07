#!/usr/bin/env python3

from LoLIM.stationTimings.autoCorrelator3_plotPulse import processed_data_dir, plot_stations, plot_one_station, plot_one_station_allData, plot_stations_AllData


## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

timeID = "D20180921T194259.023Z"
processed_data_folder = processed_data_dir(timeID)
known_station_delays = {
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


#fname = processed_data_folder + "/pulse_finding/potSource_0.h5"
#loc = [ -25383.6550768 , -8163.11570266 , 0.175016843601 , 0.984046325655 ]
#polarization = 1
#stations_to_skip = []
#bad_antennas = []

#fname = processed_data_folder + "/pulse_finding/potSource_1.h5"
#loc =[ -24428.0016061 , -7410.53707223 , 0.175016843552 , 0.984337125506 ]
#polarization = 0
#stations_to_skip = []
#bad_antennas = []#['146009078']

#fname = processed_data_folder + "/pulse_finding/potSource_2.h5"
#loc = [ -36964.5478698 , -8415.53036896 , -0.421147762007 , 0.984916477788 ]
#polarization = 1
#stations_to_skip = ['CS006']
#bad_antennas = []#['146009078']

#fname = processed_data_folder + "/pulse_finding/potSource_3.h5"
#loc = [ 1531.90572559 , 24244.2591802 , 308.6369871 , 0.985483408847 ]
#polarization = 1
#stations_to_skip = []
#bad_antennas = []

#fname = processed_data_folder + "/pulse_finding/potSource_4.h5"
#loc = [ -33510.2161972 , -10657.5606301 , 3938.93020211 , 0.986665863365 ]
#polarization = 1
#stations_to_skip = []
#bad_antennas = ['146009078']

#fname = processed_data_folder + "/pulse_finding/potSource_5.h5"
#loc =[ -36910.0475907 , -13662.6502376 , 5228.5489493 , 0.986750427195 ]
#polarization = 0
#stations_to_skip = []
#bad_antennas = []
#
#fname = processed_data_folder + "/pulse_finding/potSource_6.h5"
#loc = [ -33420.7732969 , -10238.6593789 , 4203.17649729 , 0.987354008232 ]
#polarization = 1
#stations_to_skip = []
#bad_antennas = ['001011094']

#fname = processed_data_folder + "/pulse_finding/potSource_7.h5"
#loc = [ -40324.8612313 , -10036.5721567 , 5780.00522225 , 0.987596492976 ]
#polarization = 1
#stations_to_skip = []
#bad_antennas = ['147011094', '147001014']

#fname = processed_data_folder + "/pulse_finding/potSource_8.h5"
#loc = [ -39940.661778 , -9889.297566 , 7277.26681281 , 0.988907427765 ]
#polarization = 0
#stations_to_skip = ['RS210']
#bad_antennas = []

fname = processed_data_folder + "/pulse_finding/potSource_9.h5"
loc = [ -39149.3745702 , -9738.89193919 , 7595.47885902 , 0.989187161303 ]
polarization = 0
stations_to_skip = ['RS307']
bad_antennas = ['017009078', '130009078', '130001014']

#fname = processed_data_folder + "/pulse_finding/potSource_11.h5"
#loc = [ -39327.9164621 , -9756.00581782 , 7253.49060651 , 0.989722699084 ]
#polarization = 0
#stations_to_skip = []
#bad_antennas = []

#fname = processed_data_folder + "/pulse_finding/potSource_15.h5"
#loc = [ -38264.4688068 , -9377.01594644 , 7248.94353376 , 0.991015075495 ]
#polarization = 0
#stations_to_skip = []
#bad_antennas = []

#fname = processed_data_folder + "/pulse_finding/potSource_17.h5"
#loc = [ -37145.1550538 , -11173.7953293 , 8830.13213504 , 0.982066435114 ]
#polarization = 0
#stations_to_skip = []
#bad_antennas = []

#fname = processed_data_folder + "/pulse_finding/potSource_18.h5"
#loc = [ -37726.5052238 , -11708.9547491 , 7838.73790662 , 0.981013140205 ]
#polarization = 0
#stations_to_skip = []
#bad_antennas = []

#fname = processed_data_folder + "/pulse_finding/potSource_20.h5"
#loc = [ -38575.0533521 , -10117.1002307 , 7045.42564838 , 0.979045191678 ]
#polarization = 1
#stations_to_skip = []
#bad_antennas = ['147003030', '147009078']

#fname = processed_data_folder + "/pulse_finding/potSource_21.h5"
#loc = [ -34461.9254663 , -13164.2950072 , -266.300418148 , 0.979027680169 ]
#polarization = 1
#stations_to_skip = []
#bad_antennas = []



stations_to_skip += ['RS305', 'RS306']
bad_antennas += ['147001014']

plot_stations(timeID, 
              polarization=polarization, ##0 is even. 1 is odd 
              input_file_name=fname, 
              source_XYZT = loc, 
              known_station_delays = known_station_delays, 
              stations = "all", ## all, RS, or CS
              referance_station="CS002", 
              min_antenna_amplitude=10,
              ### next three lines only needed to make plot very pretty
              skip_stations=stations_to_skip, 
              antennas_to_exclude = bad_antennas,
              plot_peak_time=True,
              plot_real=False,
              seperation_factor=1)#0.25)


#plot_stations_AllData(timeID, 
#              input_file_name=fname, 
#              source_XYZT = loc, 
#              known_station_delays = known_station_delays, 
#              stations = "RS", ## all, RS, or CS
#              referance_station="CS002", 
#              min_antenna_amplitude=10,
#              ### next three lines only needed to make plot very pretty
#              skip_stations=stations_to_skip, 
#              antennas_to_exclude = bad_antennas,
#              seperation_factor=1)#0.25)


plot_one_station(timeID, 
              polarization=polarization, ##0 is even. 1 is odd 
              input_file_name=fname, 
              source_XYZT = loc, 
              known_station_delays = known_station_delays, 
              station ='RS307',
              referance_station="CS002", 
              min_antenna_amplitude=10,
              plot_real=True)


#plot_one_station_allData(timeID,
#              input_file_name=fname, 
#              source_XYZT = loc, 
#              known_station_delays = known_station_delays, 
#              station = "CS002",
#              referance_station="CS002", 
#              min_antenna_amplitude=10)



