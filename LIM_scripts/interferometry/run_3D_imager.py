#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

from LoLIM.utilities import v_air, RTD, processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.findRFI import FindRFI, window_and_filter
from LoLIM.signal_processing import half_hann_window, remove_saturation
from LoLIM.read_pulse_data import read_antenna_delays, read_station_delays
from LoLIM.IO.metadata import getClockCorrections

import helperfunctions as helpers
from LoLIM.interferometry.imager_3D import imager_3D_bypairs

#### TODO: need to account for antenna flips, and bad antennas

if __name__ == "__main__":
    timeID = "D20160712T173455.100Z"
    stations_to_exclude = []
    
    
    station_delays = "station_delays_4.txt"
    additional_antenna_delays = "ant_delays.txt"
    
    do_RFI_filtering = True
    
    block_size = 2**16
    
    findRFI_initial_block = 5
    findRFI_num_blocks = 20
    findRFI_max_blocks = 100
    
    block_index = int( (3.819/5.0E-9) )
    data_index = int( (2**16)*0.2 ) + 700
    
#    bounding_box = np.array([[30000,40000], [24500,30000], [0.0,10000.0]], dtype=np.double)
    bounding_box = np.array([[33125,33250], [26640,26680], [3100.0,3280.0]], dtype=np.double)
    num_points = np.array([100, 120, 150], dtype=np.int32)
    
    #### get known timing offsets and antenna timing adjustments
    processed_data_dir = processed_data_dir(timeID)
    station_timing_offsets = read_station_delays( processed_data_dir+'/'+station_delays )
    extra_ant_delays = read_antenna_delays( processed_data_dir+'/'+additional_antenna_delays )
    
    
    #### open data files and find RFI ####
    raw_fpaths = filePaths_by_stationName(timeID)
    station_names = []
    input_files = []
    data_filters = []
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
            data_filters.append( window_and_filter(block_size, find_RFI=RFI_result) )
            
    print("Data opened and FindRFI completed.")
    
    
    
    ## choose antenna pairs
    image_X = (bounding_box[0,0] + bounding_box[0,1])*0.5
    image_Y = (bounding_box[1,0] + bounding_box[1,1])*0.5
    image_Z = (bounding_box[2,0] + bounding_box[2,1])*0.5
    
    
#    antennas_to_use, antenna_pairs = helpers.pairData_OptimizedEvenAntennas(input_files, np.array([image_X,image_Y,image_Z]), 10)
#    antennas_to_use, antenna_pairs = helpers.pairData_OptimizedEvenAntennas_two(input_files, np.array([image_X,image_Y,image_Z]), 300)
#    antennas_to_use, antenna_pairs = helpers.pairData_everyEvenAntennaOnce(input_files, np.array([image_X,image_Y,image_Z]) )
    antennas_to_use, antenna_pairs = helpers.pairData_1antPerStat(input_files )
#    antennas_to_use, antenna_pairs = helpers.pairData_closest_antennas(input_files, 1000.0)
    num_antennas = len(antennas_to_use)
    print("using", num_antennas, "antennas over", len(antenna_pairs), "pairs")
    
    
    
    
    #### get antenna locations and delays ####
    antenna_locations = np.zeros((num_antennas, 3), dtype=np.double)
    antenna_delays = np.zeros(num_antennas, dtype=np.double)
    antenna_data_offsets = np.zeros(num_antennas, dtype=np.long)
    antenna_data_lengths = np.zeros(num_antennas, dtype=np.long)
    
    clock_corrections = getClockCorrections()
    CS002_correction = -clock_corrections["CS002"] - input_files[CS002_index].get_nominal_sample_number()*5.0E-9 ## wierd sign. But is just to correct for previous definiitions
    
    for ant_i, (station_i, station_ant_i) in enumerate(antennas_to_use):
        data_file = input_files[station_i]
        station = station_names[station_i]
        
        ant_name = data_file.get_antenna_names()[ station_ant_i ]
        antenna_locations[ant_i] = data_file.get_LOFAR_centered_positions()[ station_ant_i ]
        antenna_delays[ant_i] = data_file.get_timing_callibration_delays()[ station_ant_i ]
        
        ## account for station timing offsets
        antenna_delays[ant_i]  += station_timing_offsets[station] + (-clock_corrections[station]-data_file.get_nominal_sample_number()*5.0E-9) - CS002_correction
        ## and additional timing delays
        ##note this only works for even antnenas!
        if ant_name in extra_ant_delays:
            antenna_delays[ant_i] += extra_ant_delays[ ant_name ][0]
            
#        ## account for offsets of mod 5.0E-9
#        antenna_data_offsets[ant_i] = int( antenna_delays[ant_i]/5.0E-9 ) ##becouse if delay is 5.0E-9, then the 'zeroeth' point is at +1
#        antenna_delays[ant_i] -= antenna_data_offsets[ant_i]*5.0E-9
        
        ## now we accound for distance to the source
        smallest_time = helpers.closest_distance(antenna_locations[ant_i], bounding_box)/v_air
        largest_time = helpers.farthest_distance(antenna_locations[ant_i], bounding_box)/v_air
        
        dn = int(smallest_time/5.0E-9)
        
        antenna_data_offsets[ant_i] = dn
        antenna_data_lengths[ant_i] = int( (largest_time-smallest_time)/5.0E-9)+1
        
    
    
    ### adjust all antennas to handle same length
    data_length = 2**( int(np.log2(np.max(antenna_data_lengths))) +2 ) ##at least one additional factor of 2
    for ant_i, (timing_delay, offset, pref_length) in enumerate(zip(antenna_delays, antenna_data_offsets,antenna_data_lengths)):
        
        #### first we amount to adjust the offset by time delays mod the sampling time
        offset_adjust = int( timing_delay/5.0E-9 ) ##this needs to be added to offsets and subtracted from delays
        
        ## then we find find the beginning of the data, and adjust the offset accordingly
        excess_length = int((data_length-pref_length)*0.5)
        antenna_data_offsets[ant_i] -= excess_length
        ## then we can adjust the delays accounting for the beginning of the data
        antenna_delays[ant_i] -= antenna_data_offsets[ant_i]*5.0E-9 
        
        ##now we finally account for large time delays
        antenna_data_offsets[ant_i] += offset_adjust
        antenna_delays[ant_i] -= offset_adjust*5.0E-9
    antenna_data_lengths=None
    
    print("data length of:", data_length, "points and", data_length*5.0, "ns")
   
    
    
    
    
    #### open and filter data###
    data = np.empty((num_antennas,data_length), dtype=np.complex)
    tmp = np.empty( block_size, dtype=np.double )
    current_height = 0
    for ant_i, (station_i, station_ant_i) in enumerate(antennas_to_use):
        data_file = input_files[station_i]
        RFI_filter = data_filters[station_i]
        
        offset = antenna_data_offsets[ant_i]
        tmp[:] = data_file.get_data(block_index+offset, block_size, antenna_index=station_ant_i) ## get the data. accounting for the offsets calculated earlier
            
        filtered = RFI_filter.filter( tmp )
        data[ant_i] = filtered[data_index:data_index+data_length] ## filter out RFI, then select the bit we want
        
        plt.plot(np.real(data[ant_i])+current_height)
        plt.plot(np.abs(data[ant_i])+current_height)
        current_height += np.max(np.real(data[ant_i]))
        
    plt.show()
    print('Data loaded and filtered')
    
    
    
    ### now we window and FFT
    data[...,:] *= half_hann_window(data_length,0.1)
    data = fftpack.fft(data, n=data_length*2, axis=-1)
    print("FFT complete!")
    
        
    image = imager_3D_bypairs(data, antenna_locations, antenna_delays, antenna_pairs, bounding_box, num_points, upsample=2, report_spacing=50)
    
    
    X_bounds = np.arange(num_points[0]+1)* ( (bounding_box[0,1]-bounding_box[0,0])/num_points[0] ) + (bounding_box[0,0] - (bounding_box[0,1]-bounding_box[0,0])/(2*num_points[0]) )
    Y_bounds = np.arange(num_points[1]+1)* ( (bounding_box[1,1]-bounding_box[1,0])/num_points[1] ) + (bounding_box[1,0] - (bounding_box[1,1]-bounding_box[1,0])/(2*num_points[1]) )
    Z_bounds = np.arange(num_points[2]+1)* ( (bounding_box[2,1]-bounding_box[2,0])/num_points[2] ) + (bounding_box[2,0] - (bounding_box[2,1]-bounding_box[2,0])/(2*num_points[2]) )
    
#    YZ_image = np.sum(image, axis=0)
#    XZ_image = np.sum(image, axis=1)
#    XY_image = np.sum(image, axis=2)
#    
#    plt.pcolormesh(Y_bounds, Z_bounds, YZ_image.T)
#    plt.show()
#    
#    plt.pcolormesh(X_bounds, Z_bounds, XZ_image.T)
#    plt.show()
#    
#    plt.pcolormesh(X_bounds, Y_bounds, XY_image.T)
#    plt.show()
    
    veiwer = helpers.multi_slice_viewer( image, 2, X_bounds, Y_bounds, Z_bounds)
    plt.show()
#    veiwer.save_animation("anim.gif")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    