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
from read_pulse_data import curtain_plot_CodeLog, read_station_info

if __name__=="__main__":
    ##opening data
    timeID = "D20160712T173455.100Z"
    output_folder = "plot_SSPWvsPSE"
    
    SSPW_folder = "RS509_planewave_data"#"excluded_planewave_data" 
    PSE_folder = "allPSE"
    
    station_to_correlate = "RS509"
    station_delay = 1.38709655077e-06
    
    #### additional data files
    ant_timing_calibrations = "cal_tables/TxtAntDelay"
    polarization_flips = "/polarization_flips.txt"
    bad_antennas = "/bad_antennas.txt"
    antenna_OneSample_shifts = None #"/antenna_OneSample_shifts.txt"
    
    
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
    
    polarization_flips = processed_data_dir + '/' + polarization_flips
    bad_antennas = processed_data_dir + '/' + bad_antennas
    ant_timing_calibrations = processed_data_dir + '/' + ant_timing_calibrations
    
    
    

    PSE_data_dict = read_PSE_timeID(timeID, PSE_folder, data_loc="/home/brian/processed_files")
    SSPW_data_dict = read_SSPW_timeID(timeID, SSPW_folder, data_loc="/home/brian/processed_files", stations=[station_to_correlate]) ## still need to implement this...
    
    PSE_ant_locations = PSE_data_dict["ant_locations"]
    PSE_list = PSE_data_dict["PSE_list"]
    
    SSPW_list = SSPW_data_dict["SSPW_dict"][station_to_correlate]
    SSPW_locs_dict = SSPW_data_dict["ant_locations"]
    
    StationInfo_dict = read_station_info(timeID, station_names=[station_to_correlate], bad_antennas_fname=bad_antennas, pol_flips_fname=polarization_flips, txt_cal_table_folder=ant_timing_calibrations)
    StationInfo = StationInfo_dict[station_to_correlate]
    
    
    CP = curtain_plot_CodeLog(StationInfo_dict, logging_folder + "/SSPWvsPSE_"+station_to_correlate)
    min_y = np.min( CP.station_offsets[station_to_correlate] )
    max_y = np.max( CP.station_offsets[station_to_correlate] )

    #### plot the planewave data
    data_SSPW  = [e.get_DataTime(StationInfo.sorted_antenna_names) for e in SSPW_list]
    annotations  = [e.unique_index for e in SSPW_list]
    
    CP.addEventList(station_to_correlate, data_SSPW, 'b', marker='o', size=50, annotation_list=annotations, annotation_size=20)
    
    for PSE in PSE_list:
        if PSE.PolE_RMS<PSE.PolO_RMS:
            polarity = 0
            PSE_fit = PSE.PolE_RMS
        else:
            polarity = 1
            PSE_fit = PSE.PolE_RMS
            
        point_source_XYZT = PSE.PolE_loc if polarity==0 else PSE.PolO_loc
        
        station_location = StationInfo.get_station_location()
        
        recieved_time = np.linalg.norm( station_location-point_source_XYZT[:3] )/v_air + point_source_XYZT[3] + station_delay
        
        CP.CL.add_function('plt.plot', [recieved_time,recieved_time],  [min_y,max_y], 'r')
        CP.CL.add_function("plt.annotate", str(PSE.unique_index), xy=(recieved_time,max_y), size=20)
    
        
#    CP.annotate_station_names(size=30, t_offset=-3.0E-6)
    CP.save()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    