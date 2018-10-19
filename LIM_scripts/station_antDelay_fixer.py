#!/usr/bin/env python3

import time
from os import mkdir
from os.path import isdir

#external
import numpy as np
#from scipy.optimize import least_squares
import matplotlib.pyplot as plt

#mine
from utilities import log, processed_data_dir, v_air
#from binary_IO import read_long

from read_PSE import read_PSE_timeID
#from read_pulse_data import read_station_info
from planewave_functions import read_SSPW_timeID_multiDir

if __name__=="__main__":
    ##opening data
    timeID = "D20160712T173455.100Z"
    output_folder = "antDelays3_RS205"
    SSPW_folders = ["excluded_SSPW_delay_fixed", "excluded_SSPW_delay_fixed_later"] #"excluded_planewave_data" #"RS509_planewave_data"
    PSE_folder = "allPSE"
    
    station_to_correlate = "RS205"
    
    max_RMS = 7.0E-9
    # SSPW : PSE
    correlation_matrix = {
            3482:  63,
            3341:  52,
            3529:  69,
            3306:  21,
            3322:  22,
            5914: 456,
            5191: 362,
            6306: 497,
            5507: 429,
            6162: 501,
            6297: 509, 
            }
    
    
    
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
    
    
    log("Time ID:", timeID)
    log("output folder name:", output_folder)
    log("date and time run:", time.strftime("%c") )
    log("SSPW folder:", SSPW_folders)
    log("PSE folder:", PSE_folder)
    log("station to correlate:", station_to_correlate)
    log("correlations:", correlation_matrix)

    PSE_data_dict = read_PSE_timeID(timeID, PSE_folder, data_loc="/home/brian/processed_files")
    SSPW_data_dict = read_SSPW_timeID_multiDir(timeID, SSPW_folders, data_loc="/home/brian/processed_files", stations=[station_to_correlate]) ## still need to implement this...
    
    PSE_ant_locations = PSE_data_dict["ant_locations"]
    PSE_list = PSE_data_dict["PSE_list"]
    
    SSPW_list = SSPW_data_dict["SSPW_dict"][station_to_correlate]
    SSPW_locs_dict = SSPW_data_dict["ant_locations"]
    
    def get_PSE(index):
        for PSE in PSE_list:
            if PSE.unique_index == index:
                return PSE
    
    PolE_ant_diffs = {}
    PolO_ant_diffs = {}
    
    SSqE = 0.0
    num_ant = 0
    
    for SSPW in SSPW_list:
        if SSPW.unique_index in correlation_matrix:
            PSE_index = correlation_matrix[SSPW.unique_index]
            PSE = get_PSE( PSE_index )
            
            print(PSE.unique_index)
            print("  even RMS:", PSE.PolE_RMS)
            print("  odd RMS:", PSE.PolO_RMS)
            
            if PSE.PolE_RMS<PSE.PolO_RMS:
                polarity = 0
            else:
                polarity = 1
            
            
            diffs = []
            for ant_name, ant_data in SSPW.ant_data.items():
                ant_loc = SSPW_locs_dict[ant_name]
                
                if (ant_data.antenna_status==0 or ant_data.antenna_status==2) and PSE.PolE_RMS<max_RMS:
                    if ant_name not in PolE_ant_diffs:
                        PolE_ant_diffs[ant_name] = []
                        
                    model_time = np.linalg.norm( PSE.PolE_loc[:3]-ant_loc)/v_air + PSE.PolE_loc[3]
                    delta = ant_data.PolE_peak_time-model_time
                    PolE_ant_diffs[ant_name].append(delta)
                    if polarity == 0:
                        diffs.append( delta )
                
                if (ant_data.antenna_status==0 or ant_data.antenna_status==1) and PSE.PolO_RMS<max_RMS:
                    if ant_name not in PolO_ant_diffs:
                        PolO_ant_diffs[ant_name] = []
                        
                    model_time = np.linalg.norm( PSE.PolO_loc[:3]-ant_loc)/v_air + PSE.PolO_loc[3]
                    delta = ant_data.PolO_peak_time-model_time
                    PolO_ant_diffs[ant_name].append(delta)
                    if polarity == 1:
                        diffs.append( delta )
                
            diffs = np.array( diffs )
            diffs -= np.average(diffs)
            diffs *= diffs
            SSqE += np.sum( diffs )
            num_ant += len( diffs )
                
            PSE.load_antenna_data( False )
            for ant_info in PSE.antenna_data.values():
                ant_loc = PSE_ant_locations[ant_info.ant_name]
                if (ant_data.antenna_status==0 or ant_data.antenna_status==2):
                    model_time = np.linalg.norm( PSE.PolE_loc[:3]-ant_loc)/v_air + PSE.PolE_loc[3]
                    diff = model_time - ant_info.PolE_peak_time
                    if polarity == 0 and np.isfinite( diff ):
                        num_ant += 1
                        SSqE += diff*diff
                    
                if (ant_data.antenna_status==0 or ant_data.antenna_status==1):
                    model_time = np.linalg.norm( PSE.PolO_loc[:3]-ant_loc)/v_air + PSE.PolO_loc[3]
                    diff = model_time - ant_info.PolO_peak_time
                    if polarity == 1 and np.isfinite( diff ):
                        num_ant += 1
                        SSqE += diff*diff
                    
                    
                    
    ##first, lets average EVERYTHING
    total = 0.0
    num_even = 0
    num_odd = 0
    
    for ant_diffs in PolE_ant_diffs.values():
        total += np.sum( ant_diffs )
        num_even += len(ant_diffs)
    
    for ant_diffs in PolO_ant_diffs.values():
        total += np.sum( ant_diffs )
        num_odd += len(ant_diffs)
        
    average_offsets = total/(num_even+num_odd)
    
    print()
    print("tot RMS:", np.sqrt(SSqE/num_ant))
    print("average offset:", average_offsets)
    print("num_even:", num_even)
    print("num_odd:", num_odd)
    
    ## now for each antenna
    print(" just even ")
    for ant_name, ant_diffs in PolE_ant_diffs.items():
        ave = np.average(ant_diffs)-average_offsets
        num = len(ant_diffs)
        STD = np.std( ant_diffs )
        print(ant_name, "ave:", ave, "+/-", STD/np.sqrt(num))
        print("   num:", num, "std:", STD)
    print()
    print()
    
    
    print(" just odd ")
    for ant_name, ant_diffs in PolO_ant_diffs.items():
        ave = np.average(ant_diffs)-average_offsets
        num = len(ant_diffs)
        STD = np.std( ant_diffs )
        print(ant_name, "ave:", ave, "+/-", STD/np.sqrt(num))
        print("   num:", num, "std:", STD)
    print()
    print()
        
    
    
    print("both pols")
    for ant_name in PolO_ant_diffs.keys():
        polE_diffs = PolE_ant_diffs[ant_name]
        polO_diffs = PolO_ant_diffs[ant_name]
        ant_diffs = np.append(polE_diffs, [polO_diffs])
        
        ave = np.average(ant_diffs)-average_offsets
        num = len(ant_diffs)
        STD = np.std( ant_diffs )
        print(ant_name, "ave:", ave, "+/-", STD/np.sqrt(num))
        print("   num:", num, "std:", STD)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    