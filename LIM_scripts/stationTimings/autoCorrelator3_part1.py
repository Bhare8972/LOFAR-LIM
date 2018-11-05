#!/usr/bin/env python3

##internal
import time
from os import mkdir
from os.path import isdir

##import external packages
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.linalg import lstsq
from matplotlib import pyplot as plt

import h5py

##my packages
from LoLIM.utilities import log, processed_data_dir, v_air
from LoLIM.read_pulse_data import read_station_info, refilter_pulses, curtain_plot_CodeLog, AntennaPulse_dict__TO__AntennaTime_dict, getNwriteBin_modData
from LoLIM.porta_code import code_logger
from LoLIM.IO.binary_IO import write_long, write_double_array, write_string, write_double

def find_subsection_pulse_list(pulse_list, start_T, end_T, search_start_i=0):
    start_i = search_start_i
    end_i = search_start_i
    last_T = None
    max_amp = -1
    max_i = None
#    print("SEARCH START")
    for i,pulse in enumerate(pulse_list[search_start_i:], start=search_start_i):
        
        T = pulse.peak_time()
        amp = pulse.best_SNR()
        
#        print("   ", start_T, T, end_T )
        
        if (last_T is None or last_T<start_T) and T>start_T:
            start_i = i
            max_amp = 0.0
            
        if T>end_T:
            end_i = i
            break
            
        if (max_amp > -1) and amp>max_amp: ## we have started time
            max_amp = amp
            max_i = i
        
        last_T = T
#    print("D", start_i, end_i, search_start_i)
#    print()
#    print()
        
    return start_i, end_i, max_amp, max_i

if __name__=="__main__":
    
    timeID = "D20180728T135703.246Z"
    output_folder = "autoCorrelator3_part1A"
    
    plot_station_map = True
    
    stations_to_exclude =  ["CS028","RS106", "RS305", "RS205", "CS201", "RS407"]
    
    num_blocks_per_step = 100
    initial_block = 1000
    num_steps = 30
    
    ant_timing_calibrations = "cal_tables/TxtAntDelay"
    polarization_flips = "polarization_flips.txt"
    bad_antennas = "bad_antennas.txt"
    additional_antenna_delays = "ant_delays.txt"
    
    
    amp_factor = 3
    time_window = 0.0001
    planewave_width = 300E-9
    
    guess_station_offsets = { ##these should be from lineing up planewaves, not accounting for source location
        "CS001":0.5036613,
        "CS002":0.5036603,
        "CS003":0.5036617 ,
        "CS004":0.5036603,
        "CS005":0.5036659,
#        "CS006":0.0,
#        "CS007":0.0,
        "CS011":0.5036600,
        "CS013":0.5036597,
        "CS017":0.5036533,
        "CS021":0.5036607,
        "CS024":0.5036617,
        "CS026":0.5036534,
        "CS028":0.5036538,
        "CS030":0.5036580,
        "CS032":0.5036565,
#        "CS101":-7E-6,
        "CS103":0.5036365,
        "RS106":0.5033261,
        "CS201":0.5036514,
        "RS205":0.5036237,
#        "RS208":8E-5+3E-6,
        "RS210":0.5038797,
        "CS301":0.5036571,
        "CS302":0.5036482,
#        "RS305":-6E-6,
        "RS306":0.5034770,
#        "RS307":175E-7+8E-7,
        "RS310":0.5037734,
#        "CS401":-3E-6,
#        "RS406":-25E-6,
        "RS407":0.5015300,
#        "RS409":8E-6,
        "CS501":0.5036528,
#        "RS503":-30.0E-8-10E-7,
        "RS508":0.5012425,
        "RS509":0.4959012,
    }
    
    
    prefered_station = "CS002"
    
        
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
    print("stations to exclude:", stations_to_exclude)
    print("initial block", initial_block, "num blocks per step:", initial_block, "num steps", num_steps)
    print("antenna timing calibrations:", ant_timing_calibrations)
    print("antenna polarization flips file:", polarization_flips)
    print("bad antennas file:", bad_antennas)
    print("ant delays file:", additional_antenna_delays)
    print()
    
    
        ##station data
    print()
    print("opening station data")
    
    polarization_flips = processed_data_dir + '/' + polarization_flips
    bad_antennas = processed_data_dir + '/' + bad_antennas
    ant_timing_calibrations = processed_data_dir + '/' + ant_timing_calibrations
    additional_antenna_delays = processed_data_dir + '/' + additional_antenna_delays
    
    StationInfo_dict = read_station_info(timeID, stations_to_exclude=stations_to_exclude, bad_antennas_fname=bad_antennas, ant_delays_fname=additional_antenna_delays, 
                                         pol_flips_fname=polarization_flips, txt_cal_table_folder=ant_timing_calibrations)
        
    
    pot_source_index = 0
    
    for iter_i in range(num_steps):
        print( "Opening Data. Itter:", iter_i )
        
        current_block = initial_block + iter_i*num_blocks_per_step
        
        AntennaPulse_dicts = {sname:sdata.read_pulse_data(approx_min_block=current_block, approx_max_block=current_block+num_blocks_per_step) for sname,sdata in StationInfo_dict.items()}

        prefered_antPulseDict = AntennaPulse_dicts[ prefered_station ]
        
#        for pulse_list in prefered_antPulseDict.values():
#            for pulse in pulse_list:
#                L = len(pulse.even_antenna_hilbert_envelope)
#                T_array = np.arange(pulse.starting_index, pulse.starting_index+L)
#                plt.plot( T_array, pulse.even_antenna_hilbert_envelope, 'r' )
#                plt.plot( T_array, pulse.odd_antenna_hilbert_envelope, 'g' )
#                plt.plot( T_array, pulse.even_antenna_data, 'r' )
#                plt.plot( T_array, pulse.odd_antenna_data, 'g' )
#                
#            plt.show()
        
        ### initialize ###
        past_lists = { ant_name:(0,0,0.0,None) for ant_name in prefered_antPulseDict.keys() }
        
        current_pulse = min( [pulse_list[0] for pulse_list in prefered_antPulseDict.values() if len(pulse_list)>0], key=lambda x: x.peak_time(), default=None )
        
        if current_pulse is None:
            continue
        
        current_T = current_pulse.peak_time()
        
        future_start_T =  current_T + planewave_width
        future_end_T =  future_start_T + time_window
        
        future_lists = { ant_name:find_subsection_pulse_list(pulse_list, future_start_T,future_end_T) for ant_name,pulse_list in prefered_antPulseDict.items() }
        
        while True:
            
            print("test T:", current_pulse.peak_time())
            
            ### first determine max past and max future
            max_past_ant, max_past = max( past_lists.items(), key=lambda x: x[1][2] )
            max_future_ant, max_future = max( future_lists.items(), key=lambda x: x[1][2] )
            
            ## is the present pulse large enough?
            max_amp = max(max_past[2], max_future[2])
            if current_pulse.best_SNR() > max_amp*amp_factor and ((max_past[3] is not None) or (max_future[3] is not None)):
                ### YAY WE FOUND A PULSE!
                ## find and save largest pulse on all stations for this time
                print("   found event", pot_source_index, current_pulse.best_SNR()/max_amp)
                print()
                print()
                
                start_T = current_pulse.peak_time() - time_window - guess_station_offsets[prefered_station]
                end_T = current_pulse.peak_time() + time_window  - guess_station_offsets[prefered_station]
                
                out_fname = data_dir + "/potSource_"+str(pot_source_index)+".h5"
                out_file = h5py.File(out_fname, "w")
                for stat_name, ant_pulse_dict in AntennaPulse_dicts.items():
                    h5_statGroup = out_file.create_group( stat_name )
                    
                    ## find pulses
                    pulses_to_save = []
                    ave_start_index = 0.0
                    for ant_name, pulse_list in ant_pulse_dict.items():
                        
                        start_i, end_t, max_amp, max_i = find_subsection_pulse_list( pulse_list, start_T+guess_station_offsets[stat_name], end_T+guess_station_offsets[stat_name]  )
                        if max_i is not None:
                            pulse_to_save = pulse_list[max_i]
                            
                            pulses_to_save.append( (ant_name,pulse_to_save) )
                            ave_start_index += pulse_to_save.starting_index
                            
                    if len(pulses_to_save) == 0:
                        continue
                    
                    ave_start_index /= len(pulses_to_save)
                    
                    ## save pulses
                    for ant_name, pulse_to_save in pulses_to_save:
                        if abs(pulse_to_save.starting_index - ave_start_index) > 5.0E-6/5.0E-9 :
                            continue ## deviates too much from planewave
                            
                        h5_Ant_dataset = h5_statGroup.create_dataset(ant_name, (4, len(pulse_to_save.even_antenna_data) ), dtype=np.double)
                        h5_Ant_dataset[0] = pulse_to_save.even_antenna_data
                        h5_Ant_dataset[1] = pulse_to_save.even_antenna_hilbert_envelope
                        h5_Ant_dataset[2] = pulse_to_save.odd_antenna_data
                        h5_Ant_dataset[3] = pulse_to_save.odd_antenna_hilbert_envelope
                        
                        h5_Ant_dataset.attrs['starting_index'] = pulse_to_save.starting_index
                        h5_Ant_dataset.attrs['PolE_peakTime'] =  pulse_to_save.PolE_peak_time
                        h5_Ant_dataset.attrs['PolO_peakTime'] =  pulse_to_save.PolO_peak_time
                        h5_Ant_dataset.attrs['PolE_timeOffset'] =  pulse_to_save.PolE_time_offset
                        h5_Ant_dataset.attrs['PolO_timeOffset'] =  pulse_to_save.PolO_time_offset
                        
                out_file.close()
                        
                
                
                pot_source_index +=1
                
                ### now we advance forward
#                current_pulse_options = [ prefered_antPulseDict[ ant_name ][ data[1] ] for ant_name,data in future_lists.items() if data[1] < len(prefered_antPulseDict[ ant_name ]) ]
#                
#                if len(current_pulse_options) == 0: ### no more pulses!!
#                    break
#                current_pulse = min( current_pulse_options, key=lambda x: x.peak_time()  )
                
                next_pulse = None
                for ant_name, data in future_lists.items():
                    pulse_list = prefered_antPulseDict[ ant_name ]
                    
                    if data[1] >= len(pulse_list):
                        continue
                    
                    for pulse in pulse_list[data[1]:]:
                        if pulse.peak_time() > (current_pulse.peak_time()+time_window) and (next_pulse is None or pulse.peak_time() < next_pulse.peak_time()):
                            next_pulse = pulse
                
                if next_pulse is None: ### no more pulses!!
                    break
                
                current_pulse = next_pulse
                
                
                past_lists = future_lists
                
                current_T = current_pulse.peak_time()
                
                future_start_T =  current_T + planewave_width
                future_end_T =  future_start_T + time_window
                
                future_lists = { ant_name:find_subsection_pulse_list(prefered_antPulseDict[ant_name], future_start_T, future_end_T, data[1]) for ant_name,data in past_lists.items() }
                
            else:
                if max_past[3] is not None and max_past[3]<len(prefered_antPulseDict[max_past_ant]):
                    max_past_pulse_T = prefered_antPulseDict[max_past_ant][max_past[3]].peak_time()
                else:
                    max_past_pulse_T = np.nan
                    
                if max_future[3] is not None and max_future[3]<len(prefered_antPulseDict[max_future_ant]):
                    max_future_pulse_T = prefered_antPulseDict[max_future_ant][max_future[3]].peak_time()
                else:
                    max_future_pulse_T = np.nan
                
                print("   No:", current_pulse.best_SNR(),  max_past[2], max_future[2], current_pulse.peak_time()-max_past_pulse_T, max_future_pulse_T-current_pulse.peak_time() )
                
                potential_pulses = [ (ant_name, data[3], data[2] ) for ant_name,data in future_lists.items() ]
                best_ant, best_pulse_i, amp = max( potential_pulses, key=lambda x:x[2]  )
                
                if best_pulse_i is None:
                    
                    next_pulse = None
                    for ant_name, data in future_lists.items():
                        pulse_list = prefered_antPulseDict[ ant_name ]
                        
                        if data[1] >= len(pulse_list):
                            continue
                        
                        for pulse in pulse_list[data[1]:]:
                            if pulse.peak_time() > current_pulse.peak_time() and (next_pulse is None or pulse.peak_time() < next_pulse.peak_time()):
                                next_pulse = pulse
                    
                    if next_pulse is None: ### no more pulses!!
                        break
                    current_pulse = next_pulse
                    
                else:
                    current_pulse = prefered_antPulseDict[ best_ant ][ best_pulse_i ]
                
                past_end_T = current_pulse.peak_time() - planewave_width
                past_start_T = past_end_T - time_window
                
                future_start_T = current_pulse.peak_time() + planewave_width
                future_end_T = future_start_T + time_window

                past_lists = { ant_name:find_subsection_pulse_list(prefered_antPulseDict[ant_name], past_start_T, past_end_T, data[0]) for ant_name,data in past_lists.items() }
                future_lists = { ant_name:find_subsection_pulse_list(prefered_antPulseDict[ant_name], future_start_T, future_end_T, data[0]) for ant_name,data in future_lists.items() }
                
                
                
                
                
                