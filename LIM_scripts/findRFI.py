#!/usr/bin/env python3

"""This module implements finding of Radio Frequency Interference for LOFAR data.

This module is strongly based on pyCRtools findrfi.py by Arthur Corstanje.
However, it has been heavily modified for use with LOFAR-LIM by Brian Hare
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import gaussian

from LoLIM.signal_processing import half_hann_window, num_double_zeros


def FindRFI(TBB_in_file, block_size, initial_block, num_blocks, max_blocks=None, verbose=False, figure_location=None, lower_frequency=10E6, upper_frequency=90E6):
    """ use phase-variance to find RFI in data. TBB_in_file should be a MultiFile_Dal1, encompassing the data for one station. block_size should be around 65536 (2^16). 
    num_blocks should be at least 20. Sometimes a block needs to be skipped, so max_blocks shows the maximum number of blocks used (after initial block) used to find num_blocks 
    number of good blocks. initial block should be such that there is no lightning in the max_blocks number of blocks. If max_blocks is None (default), it is set to num_blocks
    figure_location should be a folder to save relavent figures, default is None (do not save figures).
    
    returns a dictionary with the following key-value pairs:
        "ave_spectrum_magnitude":a numpy array that contains the average of the magnitude of the frequency spectrum
        "ave_spectrum_phase": a numpy array containing the average of the phase of the frequency spectrum
        "phase_variance":a numpy array containing the phase variance of each frequency channel 
        "dirty_channels": an array of indeces indicating the channels that are contaminated with RFI
    """
    
    if max_blocks is None:
        max_blocks = num_blocks
    
    window_function = half_hann_window(block_size, 0.1)
    num_antennas = len(TBB_in_file.get_antenna_names())
    
    if figure_location is not None:
        max_over_blocks = np.zeros((num_antennas, num_blocks), dtype=np.double)
        
    
    ## filter analysis by frequency ###
    frequencies = np.fft.fftfreq(block_size, 1.0/TBB_in_file.get_sample_frequency())
    lower_frequency_index = np.searchsorted(frequencies[:int(len(frequencies)/2)], lower_frequency)
    upper_frequency_index = np.searchsorted(frequencies[:int(len(frequencies)/2)], upper_frequency)
    
    refant = None # determine reference antenna from median power; do not rely on antenna 0 being alive...
    phase_mean = None
    spectrum_mean = None
    antenna_is_good = np.ones(num_antennas, dtype=bool)
    num_analyzed_blocks = 0
    #### first step is to find average spectrum phase and magnitude for each antenna and each channel####
    for block_i in range(max_blocks):
    # accumulate list of arrays of phases, from spectrum of all antennas of every block
        if verbose:
            print( 'Doing block %d of %d' % (block_i, num_blocks) )
        block = block_i + initial_block

        ## get data from every antenna, check antennas are good
        data = np.empty( (num_antennas, block_size), dtype=np.double )
        for ant_i in range(num_antennas):
            
            data[ant_i] = TBB_in_file.get_data( block_size*block, block_size, antenna_index=ant_i )
            
            if num_double_zeros(data[ant_i])>(0.5*block_size):
                antenna_is_good[ant_i] = False
            else:
                antenna_is_good[ant_i] = True
            
            if figure_location is not None:
                max_over_blocks[ant_i, num_analyzed_blocks] = np.max( data[ant_i] )
        
        num_good = np.sum(antenna_is_good)
        if verbose:
            print("  ",num_good, "good antennas out of", num_antennas)
        if num_good<0.5*num_antennas:
            if verbose:
                print("   skipping block")
            continue
            
        ##window the data
        # Note: No hanning window if we want to measure power accurately from spectrum
        # in the same units as power from timeseries. Applying a window gives (at least) a scale factor
        # difference!
        # But no window makes the cleaning less effective... :(
        data[:,...] *= window_function
        
        ### get FFT
        fft_data = np.fft.fft( data, axis=1 )
        
        mag_spectrum = np.abs(fft_data)
        phase = np.array(fft_data)
        phase /= (mag_spectrum + 1.0E-15)
        
        mag_spectrum *= mag_spectrum
        
        if refant is None:
            channel_power = np.sum( mag_spectrum, axis=1 )
            antenna_is_good = np.logical_and( antenna_is_good, channel_power>1)
            sorted_by_power = np.argsort( channel_power )[antenna_is_good]
            refant = sorted_by_power[ int(len(sorted_by_power)/2) ]
            if verbose:
                print( 'Taking channel %d as reference antenna' % refant)
            del channel_power ## obsesed about memory use
        elif not antenna_is_good[refant]:
            if verbose:
                print("   skipping block")
            continue
        
        num_analyzed_blocks += 1
        
        ### increment data ###
        phase[antenna_is_good,2:] /= phase[refant,2:]
        
        if phase_mean is None:
            phase_mean = np.zeros(   (fft_data.shape[0], upper_frequency_index-lower_frequency_index), dtype=complex)
            spectrum_mean = np.zeros((fft_data.shape[0], upper_frequency_index-lower_frequency_index), dtype=np.double)
            
        phase_mean[antenna_is_good,:]    +=        phase[antenna_is_good,lower_frequency_index:upper_frequency_index]
        spectrum_mean[antenna_is_good,:] += mag_spectrum[antenna_is_good,lower_frequency_index:upper_frequency_index]
        
        if num_analyzed_blocks==num_blocks:
            break
    
    if verbose:
        print(num_analyzed_blocks, "analyzed blocks, out of", block_i+1)
    ### get mean and phase stability ###
    spectrum_mean /= num_analyzed_blocks
    
    phase_stability = np.abs(phase_mean)
    phase_stability *= -1.0/num_analyzed_blocks
    phase_stability += 1.0
    
        
    #### get median of stability by channel, across each antenna ###
    median_phase_spread_byChannel = np.median(phase_stability[antenna_is_good], axis=0)
    #### get median across all chanells
    median_spread = np.median( median_phase_spread_byChannel )
    #### create a noise cuttoff###
    sorted_phase_spreads = np.sort( median_phase_spread_byChannel )
    N = len(median_phase_spread_byChannel)
    noise = sorted_phase_spreads[int(N*0.95)] - sorted_phase_spreads[int(N/2)]
    
    #### get channels contaminated by RFI, where phase stability is smaller than noise ###
    dirty_channels = np.where( median_phase_spread_byChannel < (median_spread-3*noise))[0]
    
    ### extend dirty channels by some size, in order to account for shoulders ####
    extend_dirty_channels = np.zeros(N, dtype=bool)
    half_flagwidth = int(block_size/8192)
    for i in dirty_channels:
        flag_min = i-half_flagwidth
        flag_max = i+half_flagwidth
        if flag_min<0:
            flag_min=0
        if flag_max>= N:
            flag_max=N-1
        extend_dirty_channels[flag_min:flag_max] = True
    
    dirty_channels = np.where( extend_dirty_channels )
    
    #### plot and return data ####
    if figure_location is not None:
        frequencies = frequencies[lower_frequency_index:upper_frequency_index]*1.0E-6 
        
        plt.figure()
        plt.plot(frequencies, median_phase_spread_byChannel)
        plt.axhline( median_spread-3*noise, color='r')
        plt.title("Phase spread vs frequency. Red horizontal line shows cuttoff.")
        plt.ylabel("Spread value")
        plt.xlabel("Frequency [MHz]")
        plt.savefig(figure_location+'/phase_spreads.png')
        plt.close()
        
        plt.figure()
        plt.plot(frequencies, spectrum_mean[refant])
        plt.plot(frequencies[dirty_channels], spectrum_mean[refant][dirty_channels], 'ro')
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("magnitude")
        plt.yscale('log', nonposy='clip')
        plt.savefig(figure_location+'/magnitude.png')
        plt.close()
        
        plt.figure()
        for maxes, ant_name in zip(max_over_blocks, TBB_in_file.get_antenna_names()):
            plt.plot(maxes, label=ant_name)
        plt.ylabel("maximum")
        plt.xlabel("block index")
        plt.legend()
        plt.savefig(figure_location+'/max_over_blocks.png')
        plt.close()
        
        
    output_dict = {}
    output_dict["ave_spectrum_magnitude"] = spectrum_mean
    output_dict["ave_spectrum_phase"] = np.angle(phase_mean, deg=False)
    output_dict["phase_variance"] = phase_stability
    output_dict["dirty_channels"] = dirty_channels + lower_frequency_index
    output_dict["blocksize"] = block_size
   
    return output_dict

class window_and_filter:
    def __init__(self, blocksize, find_RFI=None, lower_filter=30.0E6, upper_filter=80.0E6, half_window_percent=0.1, time_per_sample=5.0E-9, filter_roll_width = 2.5E6):
        self.lower_filter = lower_filter
        self.upper_filter = upper_filter
        self.RFI_data = find_RFI
        
        if self.RFI_data is not None:
            if self.RFI_data['blocksize'] != blocksize:
                print("blocksize and findRFI blocksize must match")
                quit()
                
        self.half_hann_window = half_hann_window(blocksize, half_window_percent)

        FFT_frequencies = np.fft.fftfreq(blocksize, d=time_per_sample)
        self.bandpass_filter = np.zeros( len(FFT_frequencies), dtype=complex)
        self.bandpass_filter[ np.logical_and( FFT_frequencies>=lower_filter, FFT_frequencies<=upper_filter  ) ] = 1.0
        gaussian_weights = gaussian(len(FFT_frequencies), int( round(filter_roll_width/(FFT_frequencies[1]-FFT_frequencies[0]) ) ) ) 
        self.bandpass_filter = np.convolve(self.bandpass_filter, gaussian_weights, mode='same' )
        self.bandpass_filter /= np.max(self.bandpass_filter) ##convolution changes the peak value
        
        ## completly reject low-frequency bits
        self.bandpass_filter[0] = 0.0
        self.bandpass_filter[1] = 0.0
        
        ##reject RFI 
        if self.RFI_data is not None:
            self.bandpass_filter[ self.RFI_data["dirty_channels"] ] = 0.0
        
    def get_frequency_response(self):
        return self.bandpass_filter
        
        
    def filter(self, data, additional_filter=None):
        data[...,:] *= self.half_hann_window
        FFT_data = np.fft.fft( data, axis=-1 )
        
        FFT_data[...,:] *= self.bandpass_filter ## note that this implicitly makes a hilbert transform! (negative frequencies set to zero)
        if additional_filter:
            FFT_data[...,:] *= additional_filter
            
        return np.fft.ifft(FFT_data, axis=-1)

    def filter_FFT(self, data, additional_filter=None):
        data[...,:] *= self.half_hann_window
        FFT_data = np.fft.fft( data, axis=-1 )
        
        FFT_data[...,:] *= self.bandpass_filter ## note that this implicitly makes a hilbert transform! (negative frequencies set to zero)
        if additional_filter:
            FFT_data[...,:] *= additional_filter
            
        return FFT_data

        
#    def filter(self, data):
#        
#        data[...,:] *= self.half_hann_window
#        FFT_data = np.fft.fft( data, axis=-1 )
#        
#        FFT_data[...,:] *= self.bandpass_filter ## note that this implicitly makes a hilbert transform! (negative frequencies set to zero)
#        # Reject DC component
#        FFT_data[..., 0] = 0.0
#        # Also reject 1st harmonic (gives a lot of spurious power with Hanning window)
#        FFT_data[..., 1] = 0.0
#        
##       remove RFI
#        if self.RFI_data is not None:
#            FFT_data[..., self.RFI_data["dirty_channels"]] = 0
#            
#
#        return np.fft.ifft(FFT_data, axis=-1)
        
        
        
        
        
        
        
        
        
        