#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

from LoLIM.utilities import v_air, RTD, processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.findRFI import FindRFI, window_and_filter
from LoLIM.signal_processing import half_hann_window
from LoLIM.read_pulse_data import read_antenna_delays, read_station_delays
from LoLIM.IO.metadata import getClockCorrections

from LoLIM.interferometry.delayNsum_3D import beamform_3D
import helperfunctions as helpers
#### TODO: need to account for antenna flips, and bad antennas

import pstats, cProfile

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
    
    segment_index = int( (2**16)*0.2 ) + 700
    segment_time= 1.0E-6
    segment_size = 2**( int(np.log2(segment_time/5.0E-9))+1 )
    
    bounding_box = np.array([[31000,33000], [25500,27500], [2000,6000]], dtype=np.double)
    num_points = np.array([100, 100, 100], dtype=np.int32)
    
    #### get known timing offsets and antenna timing adjustments
    processed_data_dir = processed_data_dir(timeID)
    station_timing_offsets = read_station_delays( processed_data_dir+'/'+station_delays )
    extra_ant_delays = read_antenna_delays( processed_data_dir+'/'+additional_antenna_delays )
    
    
    #### open data files and find RFI ####
    raw_fpaths = filePaths_by_stationName(timeID)
    data_files = {}
    data_filters = {}
    num_antennas_total = 0
    for station, fpaths  in raw_fpaths.items():
        if station not in stations_to_exclude:
            print("opening", station)
            
            data_files[station] = MultiFile_Dal1( fpaths, force_metadata_ant_pos=True )
            num_antennas_total += len(data_files[station].get_antenna_names())
            
            RFI_result = None
            if do_RFI_filtering:
                RFI_result = FindRFI(data_files[station], block_size, findRFI_initial_block, findRFI_num_blocks, findRFI_max_blocks, verbose=False, figure_location=None)
            data_filters[station] = window_and_filter(block_size, find_RFI=RFI_result)
            
    print("Data opened and FindRFI completed.", num_antennas_total, 'antennas')
    
    
    
    
    
    
    #### get antenna locations and delays ####
    antenna_locations = np.empty((num_antennas_total, 3), dtype=np.double)
    antenna_delays = np.empty(num_antennas_total, dtype=np.double)
    antenna_data_offsets = np.zeros(num_antennas_total, dtype=np.long)
    
    clock_corrections = getClockCorrections()
    CS002_correction = -clock_corrections["CS002"] - data_files["CS002"].get_nominal_sample_number()*5.0E-9 ## wierd sign. But is just to correct for previous definiitions
    
    current_antenna = 0
    max_num_antennas = 0
    station_order = list( data_files.keys() )
    for station in station_order:
        
        data_file = data_files[station]
        
        antenna_names = data_file.get_antenna_names()
        num_station_ants = len( antenna_names )
        last_antenna = current_antenna + num_station_ants
        first_antenna = current_antenna
        current_antenna += num_station_ants
        
        if num_station_ants>max_num_antennas:
            max_num_antennas = num_station_ants
        
        data_file.get_LOFAR_centered_positions( out=antenna_locations[first_antenna:last_antenna] )
        data_file.get_timing_callibration_delays( out=antenna_delays[first_antenna:last_antenna] )
        
        
        ## account for station timing offsets
        antenna_delays[first_antenna:last_antenna]  += station_timing_offsets[station] + (-clock_corrections[station]-data_files[station].get_nominal_sample_number()*5.0E-9) - CS002_correction
        ## timing callibrations that are mod 5E-9 s can be adjusted by reading data from different point
        for ant_i in range(num_station_ants):
            ## account for additional antenna delays
            if antenna_names[ant_i] in extra_ant_delays:
                antenna_delays[first_antenna+ant_i] += extra_ant_delays[ antenna_names[ant_i] ][0]
                antenna_delays[first_antenna+ant_i+1] += extra_ant_delays[ antenna_names[ant_i] ][1]
            
            ## account for offsets of mod 5.0E-9
            antenna_data_offsets[first_antenna+ant_i] = int( antenna_delays[first_antenna+ant_i]/5.0E-9 ) ##becouse if delay is 5.0E-9, then the 'zeroeth' point is at +1
            antenna_delays[first_antenna+ant_i] -= antenna_data_offsets[first_antenna+ant_i]*5.0E-9

        
        ## account for distance to lightning
        station_X = np.average( antenna_locations[first_antenna:last_antenna, 0] )
        station_Y = np.average( antenna_locations[first_antenna:last_antenna, 1] )
        station_Z = np.average( antenna_locations[first_antenna:last_antenna, 2] )
        
        image_X = (bounding_box[0,0] + bounding_box[0,1])*0.5
        image_Y = (bounding_box[1,0] + bounding_box[1,1])*0.5
        image_Z = (bounding_box[2,0] + bounding_box[2,1])*0.5
        
        ave_delay = np.sqrt( (station_X-image_X)**2 + (station_Y-image_Y)**2 + (station_Z-image_Z)**2 )/v_air
        dn = int( ave_delay/5.0E-9 )
        antenna_data_offsets[first_antenna:last_antenna]  -= dn ## becouse if we want lightning at time T, then we want data from T-dt
        antenna_delays[first_antenna:last_antenna] += dn*5.0E-9 ## time delays subtract from T to get correct data time
        print(station, dn, dn*5.0E-9)

    print(max_num_antennas, "maximum num antennas")
    
    
    
    
    #### cut down to just even-polarized antennas
    even_antenna_locs = antenna_locations[::2]
    even_antenna_delays = antenna_delays[::2]
    even_antenna_offsets = antenna_data_offsets[::2]
    num_even_antennas = len(even_antenna_delays)
    
    

    #### open and filter data###
    block_data = np.empty((num_even_antennas,block_size), dtype=np.complex)
    current_even_ant = 0
    for station in station_order:
        data_file = data_files[station]
        RFI_filter = data_filters[station]
        num_even_ant = int( len(data_file.get_antenna_names())/2 )
        
        for even_ant_i in range(0,num_even_ant): ##loop over each even antenna
            offset = even_antenna_offsets[even_ant_i + current_even_ant]
            block_data[even_ant_i + current_even_ant] = data_file.get_data(block_index+offset, block_size, antenna_index=even_ant_i*2) ## get the data. accounting for the offsets calculated earlier
            
        block_data[current_even_ant:num_even_ant+current_even_ant] = RFI_filter.filter(block_data[current_even_ant:num_even_ant+current_even_ant] )
        current_even_ant += num_even_ant
        
    segment_data = block_data[:,segment_index:segment_index+segment_size]
    
    plt.plot(np.real(block_data[0,:]))
    plt.plot(np.abs(block_data[0,:]))
    plt.show()
    
    plt.plot(np.real(segment_data[0,:]))
    plt.plot(np.abs(segment_data[0,:]))
    plt.show()
    
    ### now we window and FFT
    segment_data[...,:] *= half_hann_window(segment_size,0.1)
    segment_data = fftpack.fft(segment_data, n=segment_size*2, axis=-1)
    print("FFT complete!")
    
    image = beamform_3D(segment_data, even_antenna_locs, even_antenna_delays, bounding_box, num_points)
    
#    cProfile.runctx("beamform_3D(segment_data, even_antenna_locs, even_antenna_delays, bounding_box, num_points)", globals(), locals(), "Profile.prof")
#    s = pstats.Stats("Profile.prof")
#    s.strip_dirs().sort_stats("time").print_stats()
     
    X_bounds = np.arange(num_points[0]+1)* ( (bounding_box[0,1]-bounding_box[0,0])/num_points[0] ) + (bounding_box[0,0] - (bounding_box[0,1]-bounding_box[0,0])/(2*num_points[0]) )
    Y_bounds = np.arange(num_points[1]+1)* ( (bounding_box[1,1]-bounding_box[1,0])/num_points[1] ) + (bounding_box[1,0] - (bounding_box[1,1]-bounding_box[1,0])/(2*num_points[1]) )
    Z_bounds = np.arange(num_points[2]+1)* ( (bounding_box[2,1]-bounding_box[2,0])/num_points[2] ) + (bounding_box[2,0] - (bounding_box[2,1]-bounding_box[2,0])/(2*num_points[2]) )
    
    veiwer = helpers.multi_slice_viewer( image, 2, X_bounds, Y_bounds, Z_bounds)
    plt.show()
    
    
    
    