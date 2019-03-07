#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.optimize import least_squares, brute

from LoLIM.utilities import processed_data_dir, v_air
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
from LoLIM.signal_processing import remove_saturation, num_double_zeros, parabolic_fit
from LoLIM.findRFI import window_and_filter

class locatefier:
    def __init__(self):
        self.pulse_times = []
        self.antenna_locations = []
        
    def add_antenna(self, pulse_time, location):
        self.pulse_times.append( pulse_time )
        self.antenna_locations.append( location )
        
    def num_measurments(self):
        return len(self.pulse_times)
    
    def prep_for_fitting(self):
        self.pulse_times = np.array( self.pulse_times )
        self.antenna_locations = np.array( self.antenna_locations )
        
    def model_arrival(self, ZA ):
        sin_Z= np.sin(ZA[0])
        cos_Z= np.cos(ZA[0])
        sin_A= np.sin(ZA[1])
        cos_A= np.cos(ZA[1])
        ref_loc = self.antenna_locations[0]
        ret = np.empty( self.num_measurments(), dtype=np.double )
        for ant_i, location in enumerate( self.antenna_locations ):
            dx = location[0] - ref_loc[0]
            dy = location[1] - ref_loc[1]
            dz = location[2] - ref_loc[2]
            dr = dz*cos_Z + dx*sin_Z*cos_A + dy*sin_Z*sin_A
            
            ret[ant_i] = -dr/v_air
        return ret
            
    def time_fits(self, ZA):
        model = self.model_arrival(ZA)
        model -= self.pulse_times
        model += self.pulse_times[0]
        return model
        
    def RMS(self, ZA, DOF=1):
        fits = self.time_fits( ZA )
        fits *= fits
        S = np.sum(fits)
        
        return np.sqrt( S/(self.num_measurments()-DOF) )
        
    

def planewave_fits(timeID, station, polarization, initial_block, number_of_blocks, pulses_per_block=10, pulse_length=50, min_amplitude=50, upsample_factor=0, min_num_antennas=4,
                    polarization_flips="polarization_flips.txt", bad_antennas="bad_antennas.txt", additional_antenna_delays = "ant_delays.txt", max_num_planewaves=np.inf,
                    positive_saturation = 2046, negative_saturation = -2047, saturation_post_removal_length = 50, saturation_half_hann_length = 50, verbose=True):
    
    left_pulse_length = int(pulse_length/2)
    right_pulse_length = pulse_length-left_pulse_length
    
    processed_data_folder = processed_data_dir(timeID)
    polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
    additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
    
    raw_fpaths = filePaths_by_stationName(timeID)
    TBB_data = MultiFile_Dal1( raw_fpaths[station], polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays )
    ant_names = TBB_data.get_antenna_names()
    antenna_locations = TBB_data.get_LOFAR_centered_positions()
    antenna_delays = TBB_data.get_timing_callibration_delays()
    num_antenna_pairs = int( len( ant_names )/2 ) 
    
    if num_antenna_pairs < min_num_antennas:
        return np.array([]), np.array([]), np.array([])

    RFI_filter = window_and_filter(timeID=timeID, sname=station)
    block_size = RFI_filter.blocksize
    
    out_RMS = np.empty( number_of_blocks*pulses_per_block, dtype=np.double )
    out_zenith = np.empty( number_of_blocks*pulses_per_block, dtype=np.double )
    out_azimuth = np.empty( number_of_blocks*pulses_per_block, dtype=np.double )
    N = 0
    data = np.empty( (num_antenna_pairs,block_size), dtype=np.double )
    for block in range(initial_block, initial_block+number_of_blocks):
        if N >= max_num_planewaves:
            break
        
        #### open and filter data
        for pair_i in range(num_antenna_pairs):
            ant_i = pair_i*2 + polarization
            
            data[pair_i,:] = TBB_data.get_data(block*block_size, block_size, antenna_index=ant_i)
            remove_saturation(data[pair_i,:], positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length)
            np.abs( RFI_filter.filter( data[pair_i,:] ), out=data[pair_i,:])
    
    
        #### loop over finding planewaves
        i = 0
        while i < pulses_per_block:
            if N >= max_num_planewaves:
                break
            
            pulse_location = 0
            pulse_amplitude = 0
            
            ## find highest peak
            for pair_i, HE in enumerate(data):
                loc = np.argmax( HE )
                amp = HE[ loc ]
                
                if amp > pulse_amplitude:
                    pulse_amplitude = amp
                    pulse_location = loc
                    
            ## check if is strong enough
            if pulse_amplitude < min_amplitude:
                break
            
            ## get 
            fitter = locatefier()
            for pair_i, HE in enumerate(data):
                ant_i = pair_i*2 + polarization
                
                signal = np.array( HE[ pulse_location-left_pulse_length : pulse_location+right_pulse_length] )
                if num_double_zeros(signal, threshold=0.1) == 0:
                    
                    sample_time = 5.0E-9
                    if upsample_factor > 1:
                        sample_time /= upsample_factor
                        signal = resample(signal, len(signal)*upsample_factor )
                        
                    peak_finder = parabolic_fit( signal )
                    peak_time = peak_finder.peak_index*sample_time - antenna_delays[ant_i]
                    fitter.add_antenna(peak_time, antenna_locations[ant_i])
                    
                HE[ pulse_location-left_pulse_length : pulse_location+right_pulse_length] = 0.0
            
            if fitter.num_measurments() < min_num_antennas:
                continue
            
            fitter.prep_for_fitting()
            
            X = brute(fitter.RMS, ranges=[[0,np.pi/2],[0,np.pi*2]], Ns=20)
            
            if X[0] >= np.pi/2:
                X[0] = (np.pi/2)*0.99
            if X[1] >= np.pi*2:
                X[1] = (np.pi*2)*0.99
                
            if X[0] < 0:
                X[0] = 0.00000001
            if X[1] < 0:
                X[1] = 0.00000001
            
            ret = least_squares( fitter.time_fits, X, bounds=[[0,0],[np.pi/2,2*np.pi]], ftol=3e-16, xtol=3e-16, gtol=3e-16, x_scale='jac' )
            
            out_RMS[N] = fitter.RMS(ret.x, 3)
            out_zenith[N] = ret.x[0] 
            out_azimuth[N] = ret.x[1] 
            N += 1
            
            i += 1
            
    return out_RMS[:N], out_zenith[:N], out_azimuth[:N]
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            