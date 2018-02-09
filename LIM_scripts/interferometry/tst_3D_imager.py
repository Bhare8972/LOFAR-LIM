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

import LoLIM.interferometry.helperfunctions as helpers
from LoLIM.interferometry.imager_3D import imager_3D_bypairs

#from line_profiler import LineProfiler
import pstats, cProfile

class synth_data_gen:
    def __init__(self, source_location, source_time, width, frequency):
        self.loc = source_location
        self.time = source_time/5.0E-9
        self.width = width/5.0E-9
        self.freq = frequency*5.0E-9
        
    def get_data(self,  ant_loc,  offset, num_points):
        dR = self.loc-ant_loc
        dt = np.sqrt(np.sum(dR*dR))/v_air
        
        T = np.arange(num_points) + offset - self.time - dt/5.0E-9
        return np.exp( -0.5*(T/self.width)**2 )*np.sin( (2*np.pi*self.freq)*T)

if __name__ == "__main__":
    timeID = "D20160712T173455.100Z"
    stations_to_exclude = []
    

    
#    bounding_box = np.array([[31000,33000], [27500,29000], [2000.0,6000.0]], dtype=np.double)
    bounding_box = np.array([[33175,33225], [26640,26680], [3180.0,3280.0]], dtype=np.double)
#    bounding_box = np.array([[31900.0,32100.0], [27900.0,28100.0], [4500,5500.0]], dtype=np.double)
    num_points = np.array([100, 120, 150], dtype=np.int32)

    
#    synth_data = synth_data_gen(np.array( [ 0.0,  0.0,  5000.0] ), 0.5E-6, width=10.0E-9, frequency=10.0E6)
    synth_data = synth_data_gen(np.array( [ 33200.0,  26660.0,  3230.0] ), 0.0, width=10.0E-9, frequency=10.0E6)
#    synth_data2 = synth_data_gen(np.array( [ 31250.0,  28250.0,  2500.0] ), 0.5E-6, width=10.0E-9, frequency=10.0E6)
    
    
    #### open data files and find RFI ####
    raw_fpaths = filePaths_by_stationName(timeID)
    station_names = []
    input_files = []
    for station, fpaths  in raw_fpaths.items():
        if station not in stations_to_exclude:
            print("opening", station)
            
            station_names.append( station )
            input_files.append( MultiFile_Dal1( fpaths, force_metadata_ant_pos=True ) )

            
    print("Data opened.")
    
    
    
    ## choose antenna pairs
    image_X = (bounding_box[0,0] + bounding_box[0,1])*0.5
    image_Y = (bounding_box[1,0] + bounding_box[1,1])*0.5
    image_Z = (bounding_box[2,0] + bounding_box[2,1])*0.5
    
    antennas_to_use, antenna_pairs = helpers.pairData_everyEvenAntennaOnce(input_files, np.array([image_X,image_Y,image_Z]) )
#    antennas_to_use, antenna_pairs = helpers.pairData_1antPerStat(input_files )
#    antennas_to_use, antenna_pairs = helpers.pairData_OptimizedEvenAntennas(input_files, np.array([image_X,image_Y,image_Z]), 50)
#    antennas_to_use, antenna_pairs = helpers.pairData_OptimizedEvenAntennas_two(input_files, np.array([image_X,image_Y,image_Z]), 200)
#    antennas_to_use, antenna_pairs = helpers.pairData_closest_antennas(input_files, 1000.0)
    num_antennas = len(antennas_to_use)
    print("using", num_antennas, "antennas over", len(antenna_pairs), "pairs")
    
    
    
    
    #### get antenna locations and delays ####
    antenna_locations = np.empty((num_antennas, 3), dtype=np.double)
    antenna_delays = np.zeros(num_antennas, dtype=np.double)
    antenna_data_offsets = np.zeros(num_antennas, dtype=np.long)
    antenna_data_lengths = np.zeros(num_antennas, dtype=np.long)
    
    for ant_i, (station_i, station_ant_i) in enumerate(antennas_to_use):
        data_file = input_files[station_i]
        
        antenna_locations[ant_i] = data_file.get_LOFAR_centered_positions()[ station_ant_i ]
        
        smallest_time = helpers.closest_distance(antenna_locations[ant_i], bounding_box)/v_air
        largest_time = helpers.farthest_distance(antenna_locations[ant_i], bounding_box)/v_air
        
        dn = int(smallest_time/5.0E-9)
        
        antenna_data_offsets[ant_i] = dn
        antenna_data_lengths[ant_i] = int( (largest_time-smallest_time)/5.0E-9)+1
        
        
    ### adjust all antennas to handle same length
    data_length = 2**( int(np.log2(np.max(antenna_data_lengths))) +2 ) ##at least one additional factor of 2
    for ant_i, (offset, pref_length) in enumerate(zip(antenna_data_offsets,antenna_data_lengths)) :
        excess_length = int((data_length-pref_length)*0.5)
        antenna_data_offsets[ant_i] -= excess_length
        
        antenna_delays[ant_i] -= antenna_data_offsets[ant_i]*5.0E-9
        
        
    antenna_data_lengths=None
    
        
    print("data length of:", data_length, "points and", data_length*5.0, "ns")

    
    #### open and filter data###
    data = np.zeros((num_antennas,data_length), dtype=np.complex)
    height = 0
    for ant_i, (station_i, station_ant_i) in enumerate(antennas_to_use):
        data[ant_i] += synth_data.get_data( antenna_locations[ant_i],  antenna_data_offsets[ant_i], data_length)
#        data[even_ant_i] += synth_data2.get_data( even_antenna_locs[even_ant_i],  even_antenna_offsets[even_ant_i], num_points_per_window)
        
#        if not even_ant_i%6:
        plt.plot(np.real(data[ant_i])+height)
        height += np.max(np.real(data[ant_i]))
    plt.show()
    
    print('Data loaded and filtered')
    
    
    
    
    
    ### now we window and FFT
    data[...,:] *= half_hann_window(data_length,0.1)
    data = fftpack.fft(data, n=data_length*2, axis=-1)
    print("FFT complete!")
    
        
    image = imager_3D_bypairs(data, antenna_locations, antenna_delays, antenna_pairs, bounding_box, num_points, upsample=2, report_spacing=50)
    
#    lp = LineProfiler()
#    lp.runctx("imager_3D_bypairs(data, antenna_locations, antenna_delays, antenna_pairs, bounding_box, num_points, upsample=2, report_spacing=50)", globals(), locals())
#    lp.print_stats()
#    lp.dump_stats("Profile2.prof")
#    
#    cProfile.runctx("imager_3D_bypairs(data, antenna_locations, antenna_delays, antenna_pairs, bounding_box, num_points, upsample=2, report_spacing=50)", globals(), locals(), "Profile.prof")
#    s = pstats.Stats("Profile.prof")
#    s.strip_dirs().sort_stats("time").print_stats()
    
    
    X_bounds = np.arange(num_points[0]+1)* ( (bounding_box[0,1]-bounding_box[0,0])/num_points[0] ) + (bounding_box[0,0] - (bounding_box[0,1]-bounding_box[0,0])/(2*num_points[0]) )
    Y_bounds = np.arange(num_points[1]+1)* ( (bounding_box[1,1]-bounding_box[1,0])/num_points[1] ) + (bounding_box[1,0] - (bounding_box[1,1]-bounding_box[1,0])/(2*num_points[1]) )
    Z_bounds = np.arange(num_points[2]+1)* ( (bounding_box[2,1]-bounding_box[2,0])/num_points[2] ) + (bounding_box[2,0] - (bounding_box[2,1]-bounding_box[2,0])/(2*num_points[2]) )
    
#    YZ_image = np.sum(image, axis=0)
#    XZ_image = np.sum(image, axis=1)
#    XY_image = np.sum(image, axis=2)
#    
#    print("Z vs Y")
#    plt.pcolormesh(Y_bounds, Z_bounds, YZ_image.T)
#    plt.show()
#    
#    print("Z vs X")
#    plt.pcolormesh(X_bounds, Z_bounds, XZ_image.T)
#    plt.show()
#    
#    print("Y vs X")
#    plt.pcolormesh(X_bounds, Y_bounds, XY_image.T)
#    plt.show()

        
    veiwer = helpers.multi_slice_viewer( image, 2, X_bounds, Y_bounds, Z_bounds)
    plt.show()
    