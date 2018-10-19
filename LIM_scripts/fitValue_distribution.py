#!/usr/bin/env python3

#python
import time
from os import mkdir
from os.path import isdir, isfile

#external
import numpy as np
from matplotlib import pyplot as plt

#mine
from utilities import log, processed_data_dir, v_air
from read_PSE import read_PSE_timeID

if __name__=="__main__":
    ##opening data
    timeID = "D20160712T173455.100Z"
    output_folder = "fitValue_dist_allPSE"
    
    PSE_folder = "allPSE"
    
    
    
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
    
    
    #### print details
    print("Time ID:", timeID)
    print("output folder name:", output_folder)
    print("date and time run:", time.strftime("%c") )
    print("PSE folder:", PSE_folder)
    
    
    
    PSE_info = read_PSE_timeID(timeID, PSE_folder)
    PSE_list = PSE_info["PSE_list"]
    ant_locs = PSE_info["ant_locations"]
    old_ant_delays = PSE_info["ant_delays"]
    
    PolE_RMS_sq = np.array( [PSE.PolE_RMS**2 for PSE in PSE_list] )
    PolE_n_ant = np.array( [PSE.num_even_antennas for PSE in PSE_list] )
    PolO_RMS_sq = np.array( [PSE.PolO_RMS**2 for PSE in PSE_list] )
    PolO_n_ant = np.array( [PSE.num_odd_antennas for PSE in PSE_list] )
    
    PolE_error_sq = np.average(PolE_RMS_sq)
    PolO_error_sq = np.average(PolO_RMS_sq)
    
    print("sqrt of ave of PolE_RMS sq:", np.sqrt(PolE_error_sq))
    print("sqrt of ave of PolO_RMS sq:", np.sqrt(PolO_error_sq))
    
    PolE_scaled_chi_squared = PolE_RMS_sq/PolE_error_sq
    PolO_scaled_chi_squared = PolO_RMS_sq/PolO_error_sq
    
    plt.hist(PolE_scaled_chi_squared, bins=500)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    