#!/usr/bin/env python3

#python
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
from planewave_functions import read_SSPW_timeID

def get_best_delay(antenna_locs, SSPW, PSE):
                    
    PSE.load_antenna_data()
        
    if PSE.PolE_RMS<PSE.PolO_RMS:
        polarity = 0
        PSE_fit = PSE.PolE_RMS
    else:
        polarity = 1
        PSE_fit = PSE.PolE_RMS
        
    point_source_XYZT = PSE.PolE_loc if polarity==0 else PSE.PolO_loc
    
    
    delta_X = []
    delta_Y = []
    delta_Z = []
    delta_T = []
    
    for ant_name, ant_data in SSPW.ant_data.items():
        pt = ant_data.PolE_peak_time if polarity==0 else ant_data.PolO_peak_time 
        if np.isfinite(pt):
            X,Y,Z = antenna_locs[ant_name]
            delta_X.append( X - point_source_XYZT[0] )
            delta_Y.append( Y - point_source_XYZT[1] )
            delta_Z.append( Z - point_source_XYZT[2] )
            delta_T.append( pt - point_source_XYZT[3] )
        
    delta_X = np.array( delta_X )
    delta_Y = np.array( delta_Y )
    delta_Z = np.array( delta_Z )
    delta_T = np.array( delta_T )
    
    delta_X *= delta_X
    delta_Y *= delta_Y
    delta_Z *= delta_Z
    
    model_T = delta_X
    model_T += delta_Y
    model_T += delta_Z
    np.sqrt(model_T, out=model_T)
    model_T *= 1.0/v_air
    
    delta_T -= model_T
    delay = np.average(delta_T)
    
    return delay, PSE_fit

if __name__=="__main__":
    ##opening data
    timeID = "D20160712T173455.100Z"
    output_folder = "matrixCorrelate_RS509_late"
    
    SSPW_folder = "old_stats_SSPW_late"
    PSE_folder = "allPSE"
    
    station_to_correlate = "RS509"
    
    max_delay = 1.0E-5
    max_RMS = 6E-9
    
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
    log("SSPW folder:", SSPW_folder)
    log("PSE folder:", PSE_folder)
    log("station to correlate:", station_to_correlate)

    PSE_data_dict = read_PSE_timeID(timeID, PSE_folder, data_loc="/home/brian/processed_files")
    SSPW_data_dict = read_SSPW_timeID(timeID, SSPW_folder, data_loc="/home/brian/processed_files", stations=[station_to_correlate]) 
    
    PSE_ant_locations = PSE_data_dict["ant_locations"]
    PSE_list = PSE_data_dict["PSE_list"]
    
    SSPW_list = SSPW_data_dict["SSPW_dict"][station_to_correlate]
    SSPW_locs_dict = SSPW_data_dict["ant_locations"]
    
    
#    correlated_PSE = [PSE for PSE in PSE_list if PSE.unique_index==23][0]
#    correlated_SSPW = [SSPW for SSPW in SSPW_list if SSPW.unique_index==296][0]
#    
#    print( get_best_delay( SSPW_locs_dict, correlated_SSPW, correlated_PSE ) )
#    quit()
    
    
    delay_array = np.zeros( len(SSPW_list)*len(PSE_list), dtype=np.double )
    
    i = 0
#    num=0
    for SSPW in SSPW_list:
#        print("testing SSPW", SSPW.unique_index)
        for PSE in PSE_list:
            delay, PSE_fit = get_best_delay( SSPW_locs_dict, SSPW, PSE )
            if PSE_fit > max_RMS:
                continue
            
#            if abs(delay-1.38E-6) < 0.2E-6:
#                print("sspw pse pair:", SSPW.unique_index, PSE.unique_index)
#                num+=1
            
            if abs(delay) < max_delay:
                delay_array[i] = delay
                i += 1
                
    delay_array = delay_array[:i]
            
    print( np.average(delay_array), np.std(delay_array) )
    print( np.min(delay_array), np.max(delay_array) )
    
    n_bins = int( (np.max(delay_array)-np.min(delay_array))/1.0E-7 ) + 1
    print(n_bins, "bins")
#    print(num, "in bin!")
    
            
    histogram, edges, throw = plt.hist(delay_array, 180)
    plt.show()
    
    
    max_bin = np.argmax(histogram)
    min_delay = edges[max_bin]
    max_delay = edges[max_bin+1]
    print(histogram[max_bin], "delays between", min_delay, max_delay)
    
    #### TODO:  need to check for duplicates!!
    
    
    data_list = []
    for SSPW in SSPW_list:
        for PSE in PSE_list:
            delay, PSE_fit = get_best_delay( SSPW_locs_dict, SSPW, PSE )
            
            if delay>min_delay and delay<max_delay:
#                print(SSPW.unique_index, ":", PSE.unique_index)
                data_list.append( [SSPW.unique_index, PSE.unique_index, SSPW.best_ave_amp()] )
                
    print()
    data_list.sort(key=lambda X: X[2], reverse=True)
    print("SSPW vs PSE")
    for SSPW, PSE, amp in data_list:
        print( SSPW, ":", PSE)
        
    print()
    print("PSE vs SSPW")
    for SSPW, PSE, amp in data_list:
        print( PSE, ":", SSPW)
        
    print()
    print("SSPW vs best ave amp")
    for SSPW, PSE, amp in data_list:
        print( SSPW, ":", amp)
    
                
    
    
    
    
    
    
    
    





