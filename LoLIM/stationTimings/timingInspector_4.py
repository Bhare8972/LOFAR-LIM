#!/usr/bin/env python3

from os import listdir

import h5py
import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import  processed_data_dir, v_air, even_antName_to_odd
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1


print('WARNING!. timing inpector 4 not yet account for pol-flips!')

class input_manager:
    def __init__(self, processed_data_folder, input_files):
        self.max_num_input_files = 10
        if len(input_files) > self.max_num_input_files:
            print("TOO MANY INPUT FOLDERS!!!")
            quit()
        
        self.input_files = input_files
        
        self.input_data = []
        for folder_i, folder in enumerate(input_files):
            input_folder = processed_data_folder + "/" + folder +'/'
            
            file_list = [(int(f.split('_')[1][:-3])*self.max_num_input_files+folder_i ,input_folder+f) for f in listdir(input_folder) if f.endswith('.h5')] ## get all file names, and get the 'ID' for the file name
            file_list.sort( key=lambda x: x[0] ) ## sort according to ID
            self.input_data.append( file_list )
        
    def known_source(self, ID):
        
        file_i = int(ID/self.max_num_input_files)
        folder_i = ID - file_i*self.max_num_input_files

        file_list = self.input_data[ folder_i ]
        
        return [info for info in file_list if info[0]==ID][0]
    
    
def plot_all_stations(event_ID, 
               timeID, output_folder, pulse_input_folders, guess_timings, sources_to_fit, guess_source_locations,
               source_polarizations, source_stations_to_exclude, source_antennas_to_exclude, bad_ants,
               antennas_to_recalibrate={}, min_ant_amplitude=10, ref_station="CS002", input_antenna_delays=None):
    
    processed_data_folder = processed_data_dir( timeID )
    file_manager = input_manager( processed_data_folder, pulse_input_folders )
    throw, input_Fname = file_manager.known_source( event_ID )
    
    data_file = h5py.File(input_Fname, "r")
    
    fpaths = filePaths_by_stationName( timeID )
    
    source_XYZT = guess_source_locations[event_ID]
    source_polarization = source_polarizations[event_ID]
    antennas_to_exclude = bad_ants + source_antennas_to_exclude[event_ID]
    
    ref_station_delay = 0.0
    if ref_station in guess_timings:
        ref_station_delay = guess_timings[ref_station]
    
    current_offset = 0
    for sname in data_file.keys():
        if (sname not in guess_timings and sname !=ref_station) or (sname in source_stations_to_exclude[event_ID]):
            continue
            
        stat_group = data_file[ sname ]
        station_file = MultiFile_Dal1(fpaths[sname], force_metadata_ant_pos=True)
            
        stat_delay = 0.0
        if sname != ref_station:
            stat_delay = guess_timings[sname] - ref_station_delay
        
        even_HEs = []
        odd_HEs = []
        
        even_offet = []
        odd_offset = []
        
        even_peakT_offset = []
        odd_peakT_offset = []
        
        peak_amp = 0.0
        ## first we open the data
        print(sname)
        for ant_name, ant_loc in zip(station_file.get_antenna_names(),station_file.get_LOFAR_centered_positions()):
            if ant_name not in stat_group:
                continue
            
            even_ant_name = ant_name
            odd_ant_name = even_antName_to_odd( ant_name )
            
            ant_dataset = stat_group[ant_name]
#            even_trace = np.array( ant_dataset[0] )
            even_HE = np.array( ant_dataset[1] )
#            odd_trace = np.array( ant_dataset[2] )
            odd_HE = np.array( ant_dataset[3] )
            
            even_amp = np.max(even_HE)
            odd_amp = np.max(odd_HE)

            print(even_ant_name, 'amp:', even_amp)
            print(odd_ant_name, 'amp:', odd_amp)
            
            
            travel_delay = np.sqrt( (ant_loc[0]-source_XYZT[0])**2 + (ant_loc[1]-source_XYZT[1])**2 + (ant_loc[2]-source_XYZT[2])**2 )/v_air
            total_correction = travel_delay + stat_delay
            
            even_time =  - total_correction - source_XYZT[3]
            odd_time =  - total_correction
            
            if even_ant_name in antennas_to_recalibrate:
                even_time -= antennas_to_recalibrate[even_ant_name]
                
            if odd_ant_name in antennas_to_recalibrate:
                odd_time -= antennas_to_recalibrate[odd_ant_name]  
                
            if source_polarization==2 and len(source_XYZT)==5:
                odd_time -= source_XYZT[4]
            else:
                odd_time -= source_XYZT[3]



            StationSampleTime = station_file.get_nominal_sample_number()*5.0e-9
            filePolE_timeOffset = ant_dataset.attrs['PolE_timeOffset_CS'] + StationSampleTime
            filePolO_timeOffset = ant_dataset.attrs['PolO_timeOffset_CS'] + StationSampleTime
            polE_timeOffset = filePolE_timeOffset
            polO_timeOffset = filePolO_timeOffset
            if input_antenna_delays is not None:
                if even_ant_name in input_antenna_delays:
                    polE_timeOffset = input_antenna_delays[even_ant_name]
                else:
                    polE_timeOffset = 0.0

                if odd_ant_name in input_antenna_delays:
                    polO_timeOffset = input_antenna_delays[odd_ant_name]
                else:
                    polO_timeOffset = 0.0



                
            even_peak_time = ant_dataset.attrs["PolE_peakTime"] + even_time  + (filePolE_timeOffset - polE_timeOffset)
            odd_peak_time = ant_dataset.attrs["PolO_peakTime"] + odd_time    + (filePolO_timeOffset - polO_timeOffset)
                
            even_time += ant_dataset.attrs['starting_index']*5.0E-9  - ( polE_timeOffset - StationSampleTime )
            odd_time += ant_dataset.attrs['starting_index']*5.0E-9   - ( polO_timeOffset - StationSampleTime )
            
            even_is_good = (even_amp>min_ant_amplitude) and not (even_ant_name in antennas_to_exclude) 
            odd_is_good  = (odd_amp>min_ant_amplitude) and not (odd_ant_name in antennas_to_exclude)
                
            if (not even_is_good) and (not odd_is_good):
                continue
            
            M = max(even_amp, odd_amp)
            if M>peak_amp:
                peak_amp = M
                
            if even_is_good and source_polarization!=1:
                even_HEs.append( even_HE )
            else:
                even_HEs.append( None )
                
            if odd_is_good and source_polarization!=0:
                odd_HEs.append( odd_HE )
            else:
                odd_HEs.append( None )
        
            even_offet.append( even_time )
            odd_offset.append( odd_time )

            even_peakT_offset.append( even_peak_time )
            odd_peakT_offset.append( odd_peak_time )
        print()
            
        earliest_T = np.inf
        for pair_i in range(len(even_HEs)):
            
            if even_HEs[pair_i] is not None:
                even_T = np.arange( len(even_HEs[pair_i]) )*5.0E-9 + even_offet[pair_i]
                
                if even_T[0] < earliest_T:
                    earliest_T = even_T[0]
                
                p = plt.plot(even_T, even_HEs[pair_i]/peak_amp + current_offset)
                color = p[0].get_color()
                
                plt.plot( (even_peakT_offset[pair_i],even_peakT_offset[pair_i]), (current_offset,current_offset+1), color=color )
                
            if odd_HEs[pair_i] is not None:
                odd_T = np.arange( len(odd_HEs[pair_i]) )*5.0E-9 + odd_offset[pair_i]
                
                if odd_T[0] < earliest_T:
                    earliest_T = odd_T[0]
                
                p = plt.plot(odd_T, odd_HEs[pair_i]/peak_amp + current_offset)
                color = p[0].get_color()
                
                plt.plot( (odd_peakT_offset[pair_i],odd_peakT_offset[pair_i]), (current_offset,current_offset+1), color=color )

        plt.annotate(sname, xy=(earliest_T,current_offset+0.5))
        current_offset += 1
        
    plt.axvline(x=0)
    plt.show()
    
    
    
def plot_station(event_ID, sname_to_plot, plot_bad_antennas,
               timeID, output_folder, pulse_input_folders, guess_timings, sources_to_fit, guess_source_locations,
               source_polarizations, source_stations_to_exclude, source_antennas_to_exclude, bad_ants,
               antennas_to_recalibrate={}, min_ant_amplitude=10, ref_station="CS002", input_antenna_delays=None):
    
    processed_data_folder = processed_data_dir( timeID )
    file_manager = input_manager( processed_data_folder, pulse_input_folders )
    throw, input_Fname = file_manager.known_source( event_ID )
    
    
    data_file = h5py.File(input_Fname, "r")
    stat_group = data_file[ sname_to_plot ]
    
    fpaths = filePaths_by_stationName( timeID )[ sname_to_plot ]
    station_file = MultiFile_Dal1(fpaths, force_metadata_ant_pos=True)
    
    source_XYZT = guess_source_locations[event_ID]
    source_polarization = source_polarizations[event_ID]
    antennas_to_exclude = bad_ants + source_antennas_to_exclude[event_ID]
    
    ref_station_delay = 0.0
    if ref_station in guess_timings:
        ref_station_delay = guess_timings[ref_station]
        
    stat_delay = 0.0
    if sname_to_plot != ref_station:
        stat_delay = guess_timings[sname_to_plot] - ref_station_delay
    
    even_HEs = []
    odd_HEs = []
    even_sig = []
    odd_sig = []
    even_good = []
    odd_good = []
    even_offet = []
    odd_offset = []
    even_peakT_offset = []
    odd_peakT_offset = []
    even_ant_names = []
    odd_ant_names = []
    
    peak_amp = 0.0
    ## first we open the data
    even_SSqE = 0.0
    odd_SSqE = 0.0
    even_N = 0
    odd_N = 0
    for ant_name, ant_loc in zip(station_file.get_antenna_names(),station_file.get_LOFAR_centered_positions()):
        if ant_name not in stat_group:
            continue
        
        even_ant_name = ant_name
        odd_ant_name = even_antName_to_odd( ant_name )
        
        ant_dataset = stat_group[ant_name]
        even_trace = np.array( ant_dataset[0] )
        even_HE = np.array( ant_dataset[1] )
        odd_trace = np.array( ant_dataset[2] )
        odd_HE = np.array( ant_dataset[3] )
        
        even_amp = np.max(even_HE)
        odd_amp = np.max(odd_HE)
        
        M = max(even_amp, odd_amp)
        if M>peak_amp:
            peak_amp = M
        
        travel_delay = np.sqrt( (ant_loc[0]-source_XYZT[0])**2 + (ant_loc[1]-source_XYZT[1])**2 + (ant_loc[2]-source_XYZT[2])**2 )/v_air
        travel_delay_odd = travel_delay
        
        if source_polarization==3  and len(source_XYZT)==8:
            travel_delay_odd = np.sqrt( (ant_loc[0]-source_XYZT[4])**2 + (ant_loc[1]-source_XYZT[5])**2 + (ant_loc[2]-source_XYZT[6])**2 )/v_air
        
        even_time =  - travel_delay - stat_delay -source_XYZT[3]
        odd_time =  - travel_delay_odd - stat_delay
        
        if even_ant_name in antennas_to_recalibrate:
            even_time -= antennas_to_recalibrate[even_ant_name]
            
        if odd_ant_name in antennas_to_recalibrate:
            odd_time -= antennas_to_recalibrate[odd_ant_name]  
            
        if source_polarization==0 or source_polarization==1 or len(source_XYZT)==4:
            odd_time -= source_XYZT[3]
        elif (source_polarization==2 or source_polarization==3) and len(source_XYZT)==5:
            odd_time -= source_XYZT[4]
        elif (source_polarization==2 or source_polarization==3)  and len(source_XYZT)==8:
            odd_time -= source_XYZT[7]


        StationSampleTime = station_file.get_nominal_sample_number() * 5.0e-9
        filePolE_timeOffset = ant_dataset.attrs['PolE_timeOffset_CS'] + StationSampleTime
        filePolO_timeOffset = ant_dataset.attrs['PolO_timeOffset_CS'] + StationSampleTime
        polE_timeOffset = filePolE_timeOffset
        polO_timeOffset = filePolO_timeOffset
        if input_antenna_delays is not None:
            if even_ant_name in input_antenna_delays:
                polE_timeOffset = input_antenna_delays[even_ant_name]
            else:
                polE_timeOffset = 0.0

            if odd_ant_name in input_antenna_delays:
                polO_timeOffset = input_antenna_delays[odd_ant_name]
            else:
                polO_timeOffset = 0.0

        even_peak_time = ant_dataset.attrs["PolE_peakTime"] + even_time + (filePolE_timeOffset - polE_timeOffset)
        odd_peak_time = ant_dataset.attrs["PolO_peakTime"] + odd_time + (filePolO_timeOffset - polO_timeOffset)

        even_time += ant_dataset.attrs['starting_index'] * 5.0E-9 - (polE_timeOffset - StationSampleTime)
        odd_time += ant_dataset.attrs['starting_index'] * 5.0E-9 - (polO_timeOffset - StationSampleTime)
        
        
        even_is_good = even_amp>min_ant_amplitude
        if not even_is_good:
            print( even_ant_name, ": amp too low:", even_amp )
        else:
            even_is_good = not (even_ant_name in antennas_to_exclude)
            if not even_is_good:
                print( even_ant_name, ": excluded" )
            else:
                print( even_ant_name, ": good. amp:", even_amp )
            
        odd_is_good  = odd_amp>min_ant_amplitude
        if not odd_is_good:
            print( odd_ant_name, ": amp too low:", odd_amp  )
        else:
            odd_is_good = not (odd_ant_name in antennas_to_exclude)
            if not odd_is_good:
                print( odd_ant_name, ": excluded" )
            else:
                print( odd_ant_name, ": good. amp:", odd_amp  )
                
            
        even_HEs.append( even_HE )
        odd_HEs.append( odd_HE )
        even_sig.append( even_trace )
        odd_sig.append( odd_trace )
        even_good.append( even_is_good )
        odd_good.append( odd_is_good )
        even_offet.append( even_time )
        odd_offset.append( odd_time )
        even_ant_names.append( even_ant_name )
        odd_ant_names.append( odd_ant_name )
        even_peakT_offset.append( even_peak_time )
        odd_peakT_offset.append( odd_peak_time )
        
        if even_is_good:
            even_SSqE += even_peak_time*even_peak_time
            even_N += 1
            
        if odd_is_good:
            odd_SSqE += odd_peak_time*odd_peak_time
            odd_N += 1
    if even_N > 0:
        print('even RMS:', np.sqrt(even_SSqE/even_N),end=' ')
    if odd_N > 0:
        print('odd RMS:', np.sqrt(odd_SSqE/odd_N), end='')
    print()
        
    current_offset = 0
    for pair_i in range(len(even_HEs)):
        
        even_T = np.arange( len(even_HEs[pair_i]) )*5.0E-9 + even_offet[pair_i]
        odd_T = np.arange( len(odd_HEs[pair_i]) )*5.0E-9 + odd_offset[pair_i]
        
        if even_good[pair_i] or plot_bad_antennas:
            plt.plot(even_T, even_HEs[pair_i]/peak_amp + current_offset, 'g')
            plt.plot(even_T, even_sig[pair_i]/peak_amp + current_offset, 'g')
            
        if even_good[pair_i]:
            plt.plot( (even_peakT_offset[pair_i],even_peakT_offset[pair_i]), (current_offset,current_offset+1), 'g' )
            
        if odd_good[pair_i] or plot_bad_antennas:
            plt.plot(odd_T, odd_HEs[pair_i]/peak_amp + current_offset, 'm')
            plt.plot(odd_T, odd_sig[pair_i]/peak_amp + current_offset, 'm')
            
        if odd_good[pair_i]:
            plt.plot( (odd_peakT_offset[pair_i],odd_peakT_offset[pair_i]), (current_offset,current_offset+1), 'm' )
        
        even_str = even_ant_names[pair_i] + " "
        odd_str = odd_ant_names[pair_i] + " "
        
        if even_good[pair_i]:
            even_str += ': good'
        else:
            even_str += ': bad'
        
        if odd_good[pair_i]:
            odd_str += ': good'
        else:
            odd_str += ': bad'
            
        plt.annotate(odd_str,  xy=(odd_T[0], current_offset+0.3), c='m')
        plt.annotate(even_str, xy=(even_T[0], current_offset+0.6), c='g')
        
        current_offset += 2
        
    plt.axvline(x=0)
    plt.show()
            
        
        
        
    