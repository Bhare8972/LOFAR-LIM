#!/usr/bin/env python3

#python
import time
from os import mkdir
from os.path import isdir, isfile
from itertools import chain
from pickle import dump

#external
import numpy as np
from scipy.optimize import least_squares, minimize
from matplotlib import pyplot as plt

#mine
from LoLIM.prettytable import PrettyTable
from LoLIM.utilities import log, processed_data_dir, v_air, SId_to_Sname, Sname_to_SId_dict
from LoLIM.IO.binary_IO import read_long, write_long, write_double_array, write_string, write_double
from porta_code import code_logger, pyplot_emulator
#from RunningStat import RunningStat

from LoLIM.read_pulse_data import writeTXT_station_delays,read_station_info, curtain_plot_CodeLog

from LoLIM.planewave_functions import read_SSPW_timeID



#### source object ####
## represents a potential source
## keeps track of a SSPW on the prefered station, and SSPW on other stations that could correlate and are considered correlated
## contains utilities for fitting, and for finding RMS in total and for each station
## also contains utilities for plotting and saving info

## need to handle inseartion of random error, and that choosen SSPE can change

class source_object():
## assume: guess_location , ant_locs, station_to_antenna_index_list, station_to_antenna_index_dict, referance_station, station_order, SSPW_dict,
#    sorted_antenna_names, station_locations
    # are global
    def __init__(self, SSPW):
        self.SSPW = SSPW
        guess_time = SSPW.ZAT[2] - np.linalg.norm( station_locations[referance_station]-guess_location )/v_air
        self.guess_XYZT = np.append( guess_location, [guess_time] )
        
    def find_viable_SSPW(self, timing_error):
        self.viable_SSPW = {referance_station:[self.SSPW]}
        self.num_stations_with_unique_viable_SSPW = 0
        
        for sname, offset in guess_timings.items():
            station_location = station_locations[sname]
            arrival_time = self.guess_XYZT[3] + np.linalg.norm( station_location-self.guess_XYZT[:3] )/v_air
            arrival_time += offset
            
            SSPW_list = SSPW_dict[sname]
            SSPW_times = np.array([SSPW.ZAT[2] for SSPW in SSPW_list])
            
            SSPW_times -= arrival_time
            np.abs( SSPW_times, out=SSPW_times)
            
            
            error = timing_error+4.0E-7 ## additional error due to station size
            correlatable = np.where( SSPW_times<error )[0]
            
            correlatable_SSPW = [SSPW_list[i] for i in correlatable]
            
            if len(correlatable_SSPW) == 1:
                self.num_stations_with_unique_viable_SSPW += 1
                
            self.viable_SSPW[sname] = correlatable_SSPW
    
            
        
if __name__ == "__main__":
    timeID = "D20170929T202255.000Z"
    output_folder = "autoCorrelator_Part1_2"
    
    SSPW_folder = 'SSPW'
    first_block = 3800
    num_blocks = 100
    
    #### SSPW cuts ####
    min_antennas = 4
    max_RMS = 2E-9
    
    #### source quality requirments ####
    min_stations = 4
    max_station_RMS = 5.0E-9
    
    #### initial guesses ####
    referance_station = "CS002"
    guess_location = np.array( [1.72389621e+04,   9.50496918e+03, 2.37800915e+03] )
    
#    guess_timings = {
##        "CS002":0.0,
#        "CS003":1.0E-6,
#        "CS004":0.0,
#        "CS005":0.0,
#        "CS006":0.0,
#        "CS007":0.0,
#        "CS011":0.0,
#        "CS013":-0.000003,
#        "CS017":-7.0E-6,
#        "CS021":-8E-7,
##        "CS026":-7E-6,
#        "CS030":-5.5E-6,
#        "CS032":-5.5E-6 + 2E-6 + 1E-7,
#        "CS101":-7E-6,
#        "CS103":-22.5E-6-20E-8,
##        "RS106":35E-6 +30E-8 +12E-7,
##        "CS201":-7.5E-6,
##        "RS205":25E-6,
#        "RS208":8E-5+3E-6,
#        "CS301":6.5E-7,
#        "CS302":-3.0E-6-35E-7,
##        "RS305":-6E-6,
#        "RS306":-7E-6,
#        "RS307":175E-7+8E-7,
#        "RS310":8E-5+6E-6,
#        "CS401":-3E-6,
#        "RS406":-25E-6,
##        "RS407":5E-6 -8E-6 -15E-7,
#        "RS409":8E-6,
#        "CS501":-12E-6,
#        "RS503":-30.0E-8-10E-7,
#        "RS508":6E-5+5E-7,
#        "RS509":10E-5+15E-7,
#        }
    
    guess_timings = {
        'CS003' :  1.40526456194e-06 , ## diff to guess: 4.55627447874e-07
        'CS004' :  4.31356266729e-07 , ## diff to guess: 7.2887697412e-07
        'CS005' :  -2.19897336711e-07 , ## diff to guess: -1.49882129358e-07
        'CS006' :  4.332573673e-07 , ## diff to guess: 4.52513564994e-08
        'CS007' :  3.99972691508e-07 , ## diff to guess: 7.43711881803e-08
        'CS011' :  -5.86722283776e-07 , ## diff to guess: -1.17466092483e-06
        'CS013' :  -1.81265918331e-06 , ## diff to guess: 1.25841170957e-06
        'CS017' :  -8.43957694903e-06 , ## diff to guess: -3.00884667607e-06
        'CS021' :  9.26045460667e-07 , ## diff to guess: 2.5847711269e-06
        'CS030' :  -2.73892861781e-06 , ## diff to guess: 3.18405329258e-06
        'CS032' :  -1.57297425699e-06 , ## diff to guess: 4.20784483686e-06
        'CS101' :  -8.17066368731e-06 , ## diff to guess: -4.70254289258e-06
        'CS103' :  -2.85210517625e-05 , ## diff to guess: -1.12033199939e-05
        'RS208' :  6.93121576765e-06 , ## diff to guess: -1.05271584873e-05
        'CS301' :  -7.19152577351e-07 , ## diff to guess: 6.15814624918e-07
        'CS302' :  -5.35311982205e-06 , ## diff to guess: 7.53810275436e-06
        'RS306' :  7.03855700545e-06 , ## diff to guess: 4.3416568682e-05
        'RS307' :  6.92931506523e-06 , ## diff to guess: 4.66092155666e-05
        'RS310' :  1.74531641247e-05 , ## diff to guess: 0.00010394828689
        'CS401' :  -9.4885765996e-07 , ## diff to guess: 4.83000647468e-06
        'RS406' :  4.28410077858e-06 , ## diff to guess: 4.80074511009e-05
        'RS409' :  7.03814795037e-06 , ## diff to guess: 0.000105366498689
        'CS501' :  -9.60793966089e-06 , ## diff to guess: 1.29396922611e-06
        'RS503' :  6.95244122356e-06 , ## diff to guess: 7.70290088305e-06
        'RS508' :  2.92036819094e-05 , ## diff to guess: 0.0
        'RS509' :  -2.31004961137e-06 , ## diff to guess: 0.0
    }
    

    
    guess_timing_error = 5E-6
    guess_is_from_curtain_plot = False ## if true, must take station locations into account in order to get true offsets
    
    if referance_station in guess_timings:
        del guess_timings[referance_station]
    
    
    
    #### setup directory variables ####
    processed_data_dir = processed_data_dir(timeID)
    
    data_dir = processed_data_dir + "/" + output_folder
    if not isdir(data_dir):
        mkdir(data_dir)
        
    logging_folder = data_dir + '/logs_and_plots'
    if not isdir(logging_folder):
        mkdir(logging_folder)

    #Setup logger and open initial data set
    log.set(logging_folder + "/log_out.txt") ## TODo: save all output to a specific output folder
    log.take_stderr()
    log.take_stdout()
    
    
    #### read SSPW ####
    print("reading SSPW")
    SSPW_data = read_SSPW_timeID(timeID, SSPW_folder, data_loc="/home/brian/processed_files", min_block=first_block, max_block=first_block+num_blocks, load_timeseries=False)
    SSPW_dict = SSPW_data["SSPW_dict"]
    ant_loc_dict = SSPW_data["ant_locations"]
    
    station_locations = {}
    for ant_name,ant_loc in ant_loc_dict.items():
        ID = int( ant_name[0:3] )
        sname = SId_to_Sname[ ID ]
        
        if sname not in station_locations:
            station_locations[sname] = ant_loc
            
            
    if guess_is_from_curtain_plot:
        reference_propagation_delay = np.linalg.norm( guess_location - station_locations[referance_station] )/v_air
        for sname in guess_timings.keys():
            propagation_delay =  np.linalg.norm( guess_location - station_locations[sname] )/v_air
            guess_timings[sname] -= propagation_delay - reference_propagation_delay
    
    #### filter all SSPW ####
    print("filter and sort SSPW") ## should we sort?
    for sname in chain( guess_timings.keys(), [referance_station]):
        SSPW_list = SSPW_dict[sname]
        filtered_list = [SSPW  for SSPW in SSPW_list if SSPW.fit<max_RMS and len(SSPW.ant_data)>=min_antennas]
        filtered_list.sort( key=lambda x: x.ZAT[2] )
        SSPW_dict[sname] = filtered_list
        
    #### make the initial sources ####
    initial_sources = []
    for SSPW in SSPW_dict[referance_station]:
        new_source_object = source_object( SSPW )
        initial_sources.append( new_source_object )
        new_source_object.find_viable_SSPW( guess_timing_error )
        
    initial_sources.sort(key=lambda x: x.num_stations_with_unique_viable_SSPW, reverse=True)
    
    
    print("saving")
    
    data = [ ]
    for source in initial_sources:
        ID = source.SSPW.unique_index
        
        viable_SSPW_indeces = {}
        for sname, SSPW_list in source.viable_SSPW.items():
            viable_SSPW_indeces[sname] = [SSPW.unique_index for SSPW in SSPW_list]
            
        data.append( [ID, viable_SSPW_indeces] )
    
    with open(data_dir+'/out', 'wb') as fout:
        dump(data, fout)

        
        
    
    
    
    
    