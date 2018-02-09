#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

from LoLIM.utilities import v_air, RTD, processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.findRFI import FindRFI, window_and_filter
from LoLIM.interferometry.imager_3D import imager_3D
from LoLIM.signal_processing import half_hann_window, remove_saturation
from LoLIM.read_pulse_data import read_antenna_delays, read_station_delays
from LoLIM.IO.metadata import getClockCorrections

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
#    bounding_box = np.array([[29000,35000], [25500,31000], [0.0,10000.0]], dtype=np.double)
    bounding_box = np.array([[-200,200], [-200,200], [4500,5500.0]], dtype=np.double)
    num_points = np.array([100, 120, 150], dtype=np.int32)

    
    synth_data = synth_data_gen(np.array( [ 0.0,  0.0,  5000.0] ), 0.5E-6, width=10.0E-9, frequency=10.0E6)
    synth_data = synth_data_gen(np.array( [ 0.0,  0.0,  5000.0] ), 0.5E-6, width=10.0E-9, frequency=10.0E6)
#    synth_data2 = synth_data_gen(np.array( [ 31250.0,  28250.0,  2500.0] ), 0.5E-6, width=10.0E-9, frequency=10.0E6)
    
    
    #### open data files and find RFI ####
    raw_fpaths = filePaths_by_stationName(timeID)
    data_files = {}
    num_antennas_total = 0
    for station, fpaths  in raw_fpaths.items():
        if station not in stations_to_exclude:
            print("opening", station)
            
            data_files[station] = MultiFile_Dal1( fpaths, force_metadata_ant_pos=True )
            num_antennas_total += len(data_files[station].get_antenna_names())

            
    print("Data opened.", num_antennas_total, 'antennas')
    
    
    
    
    
    
    #### get antenna locations and delays ####
    antenna_locations = np.empty((num_antennas_total, 3), dtype=np.double)
    antenna_delays = np.empty(num_antennas_total, dtype=np.double)
    antenna_data_offsets = np.zeros(num_antennas_total, dtype=np.long)
    
    min_delay = np.inf ##minimum possible delay, compared to average, of all stations
    max_delay =-np.inf ##maximum possible delay, compared to average, of all stations
    ## together these tell us how big of a trace we need
    
    current_antenna = 0
    max_num_antennas = 0
    station_order = list( data_files.keys() )
    
        
    image_X = (bounding_box[0,0] + bounding_box[0,1])*0.5
    image_Y = (bounding_box[1,0] + bounding_box[1,1])*0.5
    image_Z = (bounding_box[2,0] + bounding_box[2,1])*0.5
    print("image center at", image_X, image_Y, image_Z)
    
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
        
       
        ## account for distance to lightning
        station_X = np.average( antenna_locations[first_antenna:last_antenna, 0] )
        station_Y = np.average( antenna_locations[first_antenna:last_antenna, 1] )
        station_Z = np.average( antenna_locations[first_antenna:last_antenna, 2] )
        
        ave_delay = np.sqrt( (station_X-image_X)**2 + (station_Y-image_Y)**2 + (station_Z-image_Z)**2 )/v_air
        dn = int( ave_delay/5.0E-9 )
        antenna_data_offsets[first_antenna:last_antenna]  += dn ## becouse if we want lightning at time T, then we want data from T+dt
        antenna_delays[first_antenna:last_antenna] -= dn*5.0E-9 ## time delays subtract from T to get correct data time
        print(station, dn, dn*5.0E-9)
        
        ## find smallest and largest possible delays
        for image_X_bound in bounding_box[0]:
            for image_Y_bound in bounding_box[1]:
                    for image_Z_bound in bounding_box[2]:
                        
                        dt = np.sqrt( (station_X-image_X_bound)**2 + (station_Y-image_Y_bound)**2 + (station_Z-image_Z_bound)**2 )/v_air
                        dt -= ave_delay
                        if dt<min_delay:
                            min_delay = dt
                        if dt>max_delay:
                            max_delay = dt
                            
    min_delay_points = int(min_delay/5.0E-9)
    max_delay_points = int(max_delay/5.0E-9)
        
    num_points_per_window = 2**( int(  np.log2(max_delay_points-min_delay_points)  +1) )
    print(num_points_per_window, "points per window. ", max_num_antennas, "maximum num antennas")
    
    antenna_data_offsets += min_delay_points
    
    
    
    
    #### cut down to just even-polarized antennas
    even_antenna_locs = antenna_locations[::2]
    even_antenna_delays = antenna_delays[::2]
    even_antenna_offsets = antenna_data_offsets[::2]
    num_even_antennas = len(even_antenna_delays)
    
    
    
    
    #### open and filter data###
    data = np.zeros((num_even_antennas,num_points_per_window), dtype=np.complex)
    height = 0
    for even_ant_i in range(num_even_antennas):
        data[even_ant_i] += synth_data.get_data( even_antenna_locs[even_ant_i],  even_antenna_offsets[even_ant_i], num_points_per_window)
#        data[even_ant_i] += synth_data2.get_data( even_antenna_locs[even_ant_i],  even_antenna_offsets[even_ant_i], num_points_per_window)
        
#        if not even_ant_i%6:
        plt.plot(np.real(data[even_ant_i])+height)
        height += np.max(np.real(data[even_ant_i]))
    plt.show()
    
    print('Data loaded and filtered')
    
    
    
    ### now we window and FFT
    data[...,:] *= half_hann_window(num_points_per_window,0.1)
    data = fftpack.fft(data, n=num_points_per_window*2, axis=-1)
    print("FFT complete!")
    
        
    image = imager_3D(data, even_antenna_locs, even_antenna_delays, bounding_box, num_points, upsample=2, max_baseline=1000)
    
    
    X_bounds = np.arange(num_points[0]+1)* ( (bounding_box[0,1]-bounding_box[0,0])/num_points[0] ) + (bounding_box[0,0] - (bounding_box[0,1]-bounding_box[0,0])/(2*num_points[0]) )
    Y_bounds = np.arange(num_points[1]+1)* ( (bounding_box[1,1]-bounding_box[1,0])/num_points[1] ) + (bounding_box[1,0] - (bounding_box[1,1]-bounding_box[1,0])/(2*num_points[1]) )
    Z_bounds = np.arange(num_points[2]+1)* ( (bounding_box[2,1]-bounding_box[2,0])/num_points[2] ) + (bounding_box[2,0] - (bounding_box[2,1]-bounding_box[2,0])/(2*num_points[2]) )
    
    YZ_image = np.sum(image, axis=0)
    XZ_image = np.sum(image, axis=1)
    XY_image = np.sum(image, axis=2)
    
    print("Z vs Y")
    plt.pcolormesh(Y_bounds, Z_bounds, YZ_image.T)
    plt.show()
    
    print("Z vs X")
    plt.pcolormesh(X_bounds, Z_bounds, XZ_image.T)
    plt.show()
    
    print("Y vs X")
    plt.pcolormesh(X_bounds, Y_bounds, XY_image.T)
    plt.show()
    
    def multi_slice_viewer(volume):
        remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = volume.shape[2] // 2
        ax.imshow(volume[:,:,ax.index])
        fig.canvas.mpl_connect('key_press_event', process_key)
        
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)
    
    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            previous_slice(ax)
        elif event.key == 'k':
            next_slice(ax)
        print(ax.index)
        fig.canvas.draw()
    
    def previous_slice(ax):
        """Go to the previous slice."""
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
        ax.images[0].set_array(volume[:,:,ax.index])
    
    def next_slice(ax):
        """Go to the next slice."""
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[2]
        ax.images[0].set_array(volume[:,:,ax.index])
        
    multi_slice_viewer( image  )
    plt.show()
    