#!/usr/bin/env python3


import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy import fftpack

from scipy.optimize import least_squares, minimize

from LoLIM.read_PSE import read_PSE_timeID
from LoLIM.utilities import v_air, RTD, processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.findRFI import FindRFI, window_and_filter
from LoLIM.IO.metadata import getClockCorrections
from LoLIM.read_pulse_data import read_antenna_delays, read_station_delays
from LoLIM.signal_processing import half_hann_window, correlateFFT, parabolic_fit

#### some algorithms for choosing pairs of antennas ####
def pairData_NumAntPerStat(input_files, num_ant_per_stat=1):
    """return the antenna-pair data for choosing one antenna per station"""
    num_stations = len(input_files)
    
    antennas = []
    for file_index in range(num_stations):
        for x in range(num_ant_per_stat):
            antennas.append( [file_index,x*2] )
    
    num_antennas = len( antennas )
    num_pairs = int( num_antennas*(num_antennas-1)*0.5 )
    pairs = np.zeros((num_pairs,2), dtype=np.int64)
    
    pair_i = 0
    for i in range(num_antennas):
        for j in range(num_antennas):
            if j<=i:
                continue
            
            pairs[pair_i,0] = i
            pairs[pair_i,1] = j
            
            pair_i += 1
            
    return np.array(antennas, dtype=int), pairs

if __name__=="__main__":
    
    timeID = "D20160712T173455.100Z"
    
    stations_to_exclude = ["RS509"] 
    ## what is wrong with RS509?
    
    
    station_delays = "station_delays_4.txt"
    additional_antenna_delays = "ant_delays.txt"
    
    
    block_size = 2**16
    pulse_length = 50 ## data points
    
    
    do_RFI_filtering = False
    
    findRFI_initial_block = 5
    findRFI_num_blocks = 20
    findRFI_max_blocks = 100
    
    
    
    PSE_id = 54
    
    
    bounding_box = np.array([[28000.0,36000.0], [23000.0,30000.0], [2000.0,7000.0]], dtype=np.double)
    
    hann_window_fraction = 0.1 ## the window length will be increased by this fraction
    
    upsample_factor = 8
    
    
    
    ### find PSE ###
    data = read_PSE_timeID(timeID, "allPSE_new3") ## we read PSE from analysis "allPSE_new3" of flash that has timeID of "D20160712T173455.100Z"
    PSE_list = data["PSE_list"] ## this is a list of point source event (PSE) objects
    ant_locs = data["ant_locations"] ## this is a dictionary of XYZ antenna locations
    
    
    for PSE in PSE_list:
        if PSE.unique_index == PSE_id:
            break
    PSE_loc = PSE.PolE_loc
    PSE.load_antenna_data(True)
    
    
    
    #### open data files and find RFI ####
    raw_fpaths = filePaths_by_stationName(timeID)
    station_names = []
    input_files = []
    RFI_filters = []
    CS002_index = None
    for station, fpaths  in raw_fpaths.items():
        if station not in stations_to_exclude:
            print("opening", station)
            station_names.append( station )
            
            if station=='CS002':
                CS002_index = len(station_names)-1
            
            input_files.append( MultiFile_Dal1( fpaths, force_metadata_ant_pos=True ) )
            
            RFI_result = None
            if do_RFI_filtering:
                RFI_result = FindRFI(input_files[-1], block_size, findRFI_initial_block, findRFI_num_blocks, findRFI_max_blocks, verbose=False, figure_location=None)
            RFI_filters.append( window_and_filter(block_size, find_RFI=RFI_result) )
            
    

    
    #### choose antenna pairs ####
    boundingBox_center = np.average(bounding_box, axis=-1)
    antennas_to_use, antenna_pairs = pairData_NumAntPerStat(input_files, 3 )
    num_antennas = len(antennas_to_use)
            
            
    
    #### get antenna locations and delays ####
    antenna_locations = np.zeros((num_antennas, 3), dtype=np.double)
    antenna_delays = np.zeros(num_antennas, dtype=np.double)
    antenna_data_offsets = np.zeros(num_antennas, dtype=np.long)
    
    clock_corrections = getClockCorrections()
    CS002_correction = 0#-clock_corrections["CS002"] - input_files[CS002_index].get_nominal_sample_number()*5.0E-9 ## wierd sign. But is just to correct for previous definiitions
    
    processed_data_dir = processed_data_dir(timeID)
    station_timing_offsets = read_station_delays( processed_data_dir+'/'+station_delays )
    
    
    extra_ant_delays = read_antenna_delays( processed_data_dir+'/'+additional_antenna_delays )
    
    for ant_i, (station_i, station_ant_i) in enumerate(antennas_to_use):
        data_file = input_files[station_i]
        station = station_names[station_i]
        
        ant_name = data_file.get_antenna_names()[ station_ant_i ]
        antenna_locations[ant_i] = data_file.get_LOFAR_centered_positions()[ station_ant_i ]
        antenna_delays[ant_i] = data_file.get_timing_callibration_delays()[ station_ant_i ]
        
        ## account for station timing offsets
        antenna_delays[ant_i]  += station_timing_offsets[station] +  (-clock_corrections[station] - data_file.get_nominal_sample_number()*5.0E-9) - CS002_correction 
        
        ## add additional timing delays
        ##note this only works for even antennas!
        if ant_name in extra_ant_delays:
            antenna_delays[ant_i] += extra_ant_delays[ ant_name ][0]
            
            
        ## now we accound for distance to the source
        travel_time = np.linalg.norm( antenna_locations[ant_i] - PSE_loc[0:3] )/v_air
        antenna_data_offsets[ant_i] = int( travel_time/5.0E-9 + PSE_loc[3]/5.0E-9 - block_size/2 )
            
        
        
        #### now adjust the data offsets and antenna delays so they are consistent
            
        ## first we amount to adjust the offset by time delays mod the sampling time
        offset_adjust = int( antenna_delays[ant_i]/5.0E-9 ) ##this needs to be added to offsets and subtracted from delays
        
        ## then we can adjust the delays accounting for the data offset
        antenna_delays[ant_i] -= antenna_data_offsets[ant_i]*5.0E-9 
        
        ##now we finally account for large time delays
        antenna_data_offsets[ant_i] += offset_adjust
        antenna_delays[ant_i] -= offset_adjust*5.0E-9
            
            
        
#    data_length = 2**( int(np.log2( pulse_length )) + 1 )
    half_hann_window = half_hann_window(pulse_length, hann_window_fraction)
        
        
        
    #### open and filter data###
    additional_shift =  int( -2.56e-6/5.0E-9 )
    factor = 2
#    data = np.empty((num_antennas,trace_length), dtype=np.complex)
    Tarray = np.arange(pulse_length)*5.0E-9
    tmp = np.empty( block_size, dtype=np.double )
    current_height = 0
    for ant_i, (station_i, station_ant_i) in enumerate(antennas_to_use):
        data_file = input_files[station_i]
        RFI_filter = RFI_filters[station_i]
        ant_name = data_file.get_antenna_names()[ station_ant_i ]
        
        offset = antenna_data_offsets[ant_i]
        tmp[:] = data_file.get_data(offset+additional_shift, block_size, antenna_index=station_ant_i) ## get the data. accounting for the offsets calculated earlier
            
        filtered = RFI_filter.filter( tmp, whiten=False )
        
        O = int(block_size/2 - pulse_length/2 )
        filtered = filtered[O:O+pulse_length  ]*half_hann_window
        
        HE = np.abs( filtered )
        HE_max = np.max( HE )
        
       
        plt.plot(Tarray-antenna_delays[ant_i]+O*5.0E-9, np.real(filtered)+current_height, 'r')
        plt.plot(Tarray-antenna_delays[ant_i]+O*5.0E-9, HE+current_height, 'b')
        
        if ant_name in PSE.antenna_data:
            ant_info = PSE.antenna_data[ ant_name ]
            
            pulse_T_array = (np.arange(len(ant_info.even_antenna_hilbert_envelope)) + ant_info.starting_index )*5.0E-9  + ant_info.PolE_time_offset
            plt.plot(pulse_T_array, ant_info.even_antenna_hilbert_envelope/factor+current_height, 'k' )
        
        current_height += HE_max
        
    
    
        
    plt.show()
    print('Data loaded and filtered')
            
            
            
            
            
    