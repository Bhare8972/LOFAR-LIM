#!/usr/bin/env python3

##ON APP MACHINE

from os import mkdir
from os.path import isdir
import bisect
import pickle
import time

##import external packages
import numpy as np

#import matplotlib as mpl
#mpl.use('Agg')

#from matplotlib import pyplot as plt
#from scipy.optimize import brute, minimize
#from scipy.sparse.linalg import lsmr

##my packages
from LoLIM.utilities import log, processed_data_dir
from LoLIM.porta_code import code_logger
from LoLIM.IO.binary_IO import write_long, write_string, write_double_array

from LoLIM.read_pulse_data import curtain_plot_CodeLog, AntennaPulse_dict__TO__AntennaTime_dict, read_station_info, refilter_pulses, getNwriteBin_modData
from LoLIM.planewave_functions import find_planewave_events, find_planewave_events_OLD


if __name__=="__main__":
    
    ##global vars
    min_signal_SNR = 100 ##10? 15? 20?
    
    ##opening data
    timeID = "D20180728T135703.246Z"
    input_folder_name = "/pulse_data"
    output_folder = "SSPWB"
    stations_to_find_SSPW = None ## leave at None for all stations, otherwise a list of station names
    stations_to_exclude = ["CS006", "CS101", "CS401", "RS305", "RS503", "CS007"]
    initial_block = 5500
    num_blocks_per_step = 100
    num_steps = 5
    
    locations_are_local = True ## is True if the antennas locations in the pulse data are in local coordinates. should be True unless error
    plot_station_map = True
    
    #### additional data files
    ant_timing_calibrations = "cal_tables/TxtAntDelay"
    polarization_flips = "/polarization_flips.txt"
    bad_antennas = "/bad_antennas.txt"
    additional_antenna_delays = "/ant_delays.txt"
    
    #### one sample shift controls
    max_RMS = 3.0E-9
    shift_threshold = 4.0E-9
    OneSample_shifts_output = "antenna_OneSample_shifts.txt"


    
    
    #### setup directory variables ####
    processed_data_dir = processed_data_dir(timeID)
    
    data_dir = processed_data_dir + "/" + output_folder
    if not isdir(data_dir):
        mkdir(data_dir)
        
    logging_folder = data_dir + '/logs_and_plots'
    if not isdir(logging_folder):
        mkdir(logging_folder)

    #Setup logger and open initial data set
#    log = logger()
    log.set(logging_folder + "/log_out.txt") ## TODo: save all output to a specific output folder
    log.take_stderr()
#    log.take_stdout()
    
    
    log("Time ID:", timeID)
    log("output folder name:", data_dir)
    log("input folder name:", input_folder_name)
    log("signal snr:", min_signal_SNR)
    log("date and time run:", time.strftime("%c") )
    log("max RMS for 1 sample shifts:", max_RMS)
    log("1 sample shift threshold:", shift_threshold)
    log("stations to find SSPW", stations_to_find_SSPW)
    log("stations to exclude", stations_to_exclude)
    log("initial block:", initial_block)
    log("num blocks per step:", num_blocks_per_step)
    log("num steps:", num_steps)
    log("antenna timing calibrations:", ant_timing_calibrations)
    log("antenna polarization flips file:", polarization_flips)
    log("bad antennas file:", bad_antennas)
    log("ant delays file:", additional_antenna_delays)
    
    polarization_flips = processed_data_dir + '/' + polarization_flips
    bad_antennas = processed_data_dir + '/' + bad_antennas
    ant_timing_calibrations = processed_data_dir + '/' + ant_timing_calibrations
    additional_antenna_delays = processed_data_dir + '/' + additional_antenna_delays
    
    
    
    
    #### open station info ####
    StationInfo_dict = read_station_info(timeID, input_folder_name, station_names=stations_to_find_SSPW, stations_to_exclude=stations_to_exclude, ant_delays_fname=additional_antenna_delays, 
                                         bad_antennas_fname=bad_antennas, pol_flips_fname=polarization_flips, txt_cal_table_folder=ant_timing_calibrations, locations_are_local=locations_are_local)
    
    if plot_station_map:
        CL = code_logger(logging_folder+ "/station_map")
        CL.add_statement("import numpy as np")
        CL.add_statement("import matplotlib.pyplot as plt")
        
        for sname,sdata in StationInfo_dict.items():
            
            
            ant_X = np.array([ ant.location[0] for ant in sdata.AntennaInfo_dict.values() ])
            ant_Y = np.array([ ant.location[1] for ant in sdata.AntennaInfo_dict.values() ])
            station_X = np.average(ant_X)
            station_Y = np.average(ant_Y)
            
            CL.add_function("plt.scatter", ant_X, ant_Y)
            CL.add_function("plt.annotate", sname, xy=(station_X, station_Y), size=30)
            
        CL.add_statement( "plt.tick_params(axis='both', which='major', labelsize=30)" )
        CL.add_statement( "plt.show()" )
        CL.save()
    
    
    ####process data ####
    antenna_shifts = []
    for sname, sdata in StationInfo_dict.items():
        
        log("processing station:", sname )
        
        
        antenna_diffs = {}
        for step_i in range(num_steps):
            first_block = initial_block + step_i*num_blocks_per_step
            final_block = first_block + num_blocks_per_step
            log("   step", step_i, "block", first_block, "to", final_block)
        
            #### open data, filter pulses, find planewaves ####
            antennaPulse_dict = sdata.read_pulse_data(approx_min_block=first_block, approx_max_block=final_block )
            antennaPulse_dict = refilter_pulses(antennaPulse_dict, min_signal_SNR)
            
            for ant_name, pulses in antennaPulse_dict.items():
                print(ant_name, len(pulses))
            
            planewave_events = find_planewave_events_OLD(sdata, antennaPulse_dict)
            
            print("found:", len(planewave_events), "SSPW")
            
            #### look for one-sample offsets ####
            log("Search antenna data for 5ns shifts")
            for SSPW in planewave_events:
                if SSPW.fit < max_RMS:
                    ant_names = SSPW.station_info.sorted_antenna_names
                    model_times = SSPW.get_ModelTime()
                    data_times = SSPW.get_DataTime()
                    for ant_name, MT, DT in zip(ant_names, model_times, data_times):
                        key = (ant_name,SSPW.polarization)
                        if key not in antenna_diffs:
                            antenna_diffs[key] = []
                        if np.isfinite(DT-MT):
                            antenna_diffs[key].append( DT-MT )
                   
                        
                        
            log("plotting pulses vs SSPW")    
            #### plot all PSE and SSPW ####
            CP = curtain_plot_CodeLog(StationInfo_dict, logging_folder + "/all_singleStation_events_"+sname + "_" + str(first_block))
            ####plot the pulses
            ATD = AntennaPulse_dict__TO__AntennaTime_dict(antennaPulse_dict, sdata)
            CP.add_AntennaTime_dict(sname, ATD, color='g', marker='o', size=50)
            #### plot the planewave fits
            data_SSPW  = [e.get_DataTime() for e in planewave_events]
            model_SSPW = [e.get_ModelTime() for e in planewave_events]
            annotations  = [e.unique_index for e in planewave_events]
            
            CP.addEventList(sname, data_SSPW, 'b', marker='o', size=50)
            CP.addEventList(sname, model_SSPW, 'r', marker='+', size=100, annotation_list=annotations, annotation_size=20)
                
            CP.annotate_station_names(size=30, t_offset=-3.0E-6)
            CP.save()
            
           
            #### plot and save each SSPW ####
            with open(data_dir+'/SSPW_'+sname+'_'+str(first_block), 'wb') as fout:
                
                
                write_long(fout, 1) ## means mod data is next
                getNwriteBin_modData(fout, StationInfo_dict, stations_to_write=[sname] )
                
                write_long(fout, 2) ## means antenna location data is next
                write_long(fout, len(sdata.AntennaInfo_dict))
                for ant_name, ant_data in sdata.AntennaInfo_dict.items():
                    write_string(fout, ant_name)
                    write_double_array(fout, ant_data.location)
                
                
                write_long(fout,6) ## means planewave data is next
                write_long(fout, len(planewave_events))
                
                for SSPW in planewave_events:
#                    ## plot 
#                    CL = code_logger(logging_folder+ "/SSPW_"+str(SSPW.unique_index))
#                    CL.add_statement("import numpy as np")
#                    CL.add_statement("import matplotlib.pyplot as plt")
#                            
#                    SSPW.data_CodeLogPlot(CL)
#                            
#                    CL.add_statement( "plt.tick_params(axis='both', which='major', labelsize=30)" )
#                    CL.add_statement( "plt.show()" )
#                    CL.save()
                        
                    ## save data
                    SSPW.save_binary(fout)
                    
                    
        ## finish looking for ant diffs 
        for (ant_name, pol), diffs in antenna_diffs.items():
            N = len(diffs)
            if N>0:
                ave = np.average(diffs)
                std = np.std(diffs)
                log( ant_name, pol, ave/(1.0E-9), std/(1.0E-9), N, std/(np.sqrt(N)*1.0E-9) )
                if np.abs(ave) > shift_threshold:
                    log( "   5 ns shift!" )
                    S = '+'
                    if ave < 0:
                        S = '-'
                    antenna_shifts.append( (ant_name, pol, S) )
              
            
                
    #### save one-sample shifts ####
    with open(data_dir+'/'+OneSample_shifts_output, 'w') as antenna_shifts_fout:
        for ant_name, pol, sign in antenna_shifts:
            antenna_shifts_fout.write(ant_name+" "+str(pol)+" "+sign+"\n")
        
#        if antenna_OneSample_shifts is not None:
#            for ant_name, (even_shift, odd_shift) in known_antenna_shifts.items():
#                if np.abs(even_shift) > 0.1E-9: ### we do not want to record shifts of zero
#                    even_sign = '+' if even_shift>0 else '-'
#                    antenna_shifts_fout.write(ant_name+" 0 "+even_sign+"\n")
#                if np.abs(odd_shift) > 0.1E-9:
#                    odd_sign = '+' if odd_shift>0 else '-'
#                    antenna_shifts_fout.write(ant_name+" 1 "+odd_sign+"\n")
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
