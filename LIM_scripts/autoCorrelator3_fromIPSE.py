#!/usr/bin/env python3

##internal
import time
from os import mkdir
from os.path import isdir

##import external packages
import numpy as np
import h5py

from LoLIM.utilities import processed_data_dir, antName_is_even, odd_antName_to_even, v_air
from LoLIM.interferometry import read_interferometric_PSE as R_IPSE
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1
from LoLIM.read_pulse_data import  read_station_delays
from LoLIM.signal_processing import parabolic_fit


THIS FILE IS OUT-DATED AND SHOULD BE REMOVED. IT IS ONLY HERE FOR REFERANCE

if __name__ == "__main__":
    timeID = "D20170929T202255.000Z"
    input_folder = "interferometry_out3"
    station_delay_file = "station_delays.txt"
    polarity = 0
    
    output_folder = "autoCorrelator3_fromIPSE"
    
    min_amplitude = 200.0
    max_S1S2_distance = 10.0
    min_intensity = 90.0
    
    location_filter = [[-19597.037517, -14044.1312013],
                       [6862.37642846, 12129.4040068],
                       [957.983583362, 5670.02729203],
                       [1.15337247668, 1.29621696106]]
    
    ### TODO: fix even/odd antennas
    ### add log , inlcuding recording bad antnenas and pol flips
    ### fix starting index
    
    #### open ####
    
    processed_data_folder = processed_data_dir(timeID)
    input_dir = processed_data_folder + "/" + input_folder
    output_dir = processed_data_folder + "/" + output_folder
    interferometry_header, IPSE_list = R_IPSE.load_interferometric_PSE( input_dir )
    station_timing_offsets = read_station_delays( processed_data_folder+'/'+station_delay_file )
    
    
    if not isdir(output_dir):
        mkdir(output_dir)
    
    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(fpaths, force_metadata_ant_pos=True) \
                      for sname,fpaths in raw_fpaths.items() if sname in station_timing_offsets}
    
    #### filter ####
    
    filtered_IPSE = []
    for IPSE in IPSE_list:
        if (location_filter[0][0]<IPSE.loc[0]<location_filter[0][1]) and (location_filter[1][0]<IPSE.loc[1]<location_filter[1][1]) and (location_filter[2][0]<IPSE.loc[2]<location_filter[2][1]):
            if IPSE.amplitude>min_amplitude and IPSE.intensity>min_intensity and IPSE.S1_S2_distance<max_S1S2_distance:
                filtered_IPSE.append( IPSE )
                
    #### save ####
    ant_station_info = interferometry_header.antenna_info_by_station()
    for i,IPSE in enumerate(filtered_IPSE):
        print("saving", i, IPSE.unique_index)
        
        out_fname = output_dir + "/potSource_"+str(i)+".h5"
        out_file = h5py.File(out_fname, "w")
        
        for stat_name, antenna_list in ant_station_info.items():
            h5_statGroup = out_file.create_group( stat_name )
            data_file = raw_data_files[stat_name]
                                       
            antenna_delays = data_file.get_timing_callibration_delays()
            antenna_names = data_file.get_antenna_names()
            
            for ant_info in antenna_list:
                ant_name = ant_info.name
                
                station_ant_index = antenna_names.index( ant_name )
                
                modeled_time = IPSE.T + np.linalg.norm(ant_info.location-IPSE.loc)/v_air 
                station_time = modeled_time + station_timing_offsets[stat_name]
                antenna_offset =  - data_file.get_nominal_sample_number()*5.0E-9 + antenna_delays[station_ant_index]
                pulse_start = int( (station_time+antenna_offset)/5.0E-9 - interferometry_header.pulse_length*0.5)
                
                #### NOTE: do we want to reload the data to insure we are looking at the correct part?
                
                HE = np.abs(IPSE.file_dataset[ant_info.antenna_index])
                peak_finder = parabolic_fit(HE)
                
                
                if polarity==0 and antName_is_even(ant_name): ## even
                    polE_name = ant_name
                
                elif polarity==1 and not antName_is_even(ant_name): ## odd
                    polE_name = odd_antName_to_even(polE_name)
                    
                    
                if ant_name not in h5_statGroup:
                    h5_Ant_dataset = h5_statGroup.create_dataset(polE_name, (4, interferometry_header.pulse_length ), dtype=np.double)
                else: 
                    h5_Ant_dataset = h5_statGroup[polE_name]
                    
                    
                h5_Ant_dataset[0] = np.real(IPSE.file_dataset[ant_info.antenna_index])
                h5_Ant_dataset[1] = HE
                h5_Ant_dataset[2] = np.real(IPSE.file_dataset[ant_info.antenna_index])
                h5_Ant_dataset[3] = HE
                
                h5_Ant_dataset.attrs['starting_index'] = pulse_start
                h5_Ant_dataset.attrs['PolE_peakTime'] =  station_time -interferometry_header.pulse_length*0.5*5.0E-9 + peak_finder.peak_index*5.0E-9
                h5_Ant_dataset.attrs['PolE_timeOffset'] =  antenna_offset
                h5_Ant_dataset.attrs['PolO_peakTime'] =  h5_Ant_dataset.attrs['PolE_peakTime']
                h5_Ant_dataset.attrs['PolO_timeOffset'] =  h5_Ant_dataset.attrs['PolE_timeOffset']
            
            
#                h5_Ant_dataset[0] = pulse_to_save.even_antenna_data
#                h5_Ant_dataset[1] = pulse_to_save.even_antenna_hilbert_envelope
#                h5_Ant_dataset[2] = pulse_to_save.odd_antenna_data
#                h5_Ant_dataset[3] = pulse_to_save.odd_antenna_hilbert_envelope
#                
#                h5_Ant_dataset.attrs['starting_index'] = pulse_to_save.starting_index
#                h5_Ant_dataset.attrs['PolE_peakTime'] =  pulse_to_save.PolE_peak_time
#                h5_Ant_dataset.attrs['PolO_peakTime'] =  pulse_to_save.PolO_peak_time
#                h5_Ant_dataset.attrs['PolE_timeOffset'] =  pulse_to_save.PolE_time_offset
#                h5_Ant_dataset.attrs['PolO_timeOffset'] =  pulse_to_save.PolO_time_offset
                        
        out_file.close()
                        