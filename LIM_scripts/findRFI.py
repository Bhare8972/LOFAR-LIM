#!/usr/bin/env python3

"""This module implements finding of Radio Frequency Interference for LOFAR data.

This module is strongly based on pyCRtools findrfi.py by Arthur Corstanje.
However, it has been heavily modified for use with LOFAR-LIM by Brian Hare
"""


from pickle import load

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import gaussian

from LoLIM.utilities import processed_data_dir
from LoLIM.signal_processing import half_hann_window, num_double_zeros

def median_sorted_by_power(psort):

    lpsort = len(psort)
    index = 0
    if lpsort % 2 == 0:
       index = int(lpsort/2)-1
    else:
       index = int(lpsort/2)

    modifier = 0
    out_psort = []
    start_index = index
    for i in range(0,lpsort):
        out_psort.append(psort[index])
        if modifier == 0:
           modifier = 1
        elif modifier > 0:
           modifier = -modifier
        elif  modifier < 0:
           modifier = -(modifier - 1)
        else:
           print("Your head a splode")
        index = start_index + modifier

    return out_psort


def FindRFI(TBB_in_file, block_size, initial_block, num_blocks, max_blocks=None, verbose=False, figure_location=None, lower_frequency=10E6, upper_frequency=90E6, num_dbl_z=100):
    """ use phase-variance to find RFI in data. TBB_in_file should be a MultiFile_Dal1, encompassing the data for one station. block_size should be around 65536 (2^16).
    num_blocks should be at least 20. Sometimes a block needs to be skipped, so max_blocks shows the maximum number of blocks used (after initial block) used to find num_blocks
    number of good blocks. initial block should be such that there is no lightning in the max_blocks number of blocks. If max_blocks is None (default), it is set to num_blocks
    figure_location should be a folder to save relavent figures, default is None (do not save figures). num_dbl_z is number of double zeros allowed in a block, if there are too
    many, then there could be data loss.

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
        max_over_blocks = np.zeros((num_antennas, max_blocks), dtype=np.double)




    #### step one: find which blocks are good, and find average power ####
    oneAnt_data = np.empty( block_size, dtype=np.double )

    if verbose:
        print( 'finding good blocks' )
    blocks_good = np.zeros((num_antennas, max_blocks), dtype=bool)
    num_good_blocks = np.zeros(num_antennas, dtype=int)
    average_power = np.zeros(num_antennas, dtype=np.double)
    for block_i in range(max_blocks):
        block = block_i + initial_block

        for ant_i in range(num_antennas):
            oneAnt_data[:] = TBB_in_file.get_data( block_size*block, block_size, antenna_index=ant_i )

            if num_double_zeros( oneAnt_data ) < num_dbl_z: ## this antenna on this block is good
                blocks_good[ ant_i, block_i ] = True
                num_good_blocks[ ant_i ] += 1

                oneAnt_data *= window_function

                FFT_data = np.fft.fft( oneAnt_data )
                np.abs(FFT_data, out=FFT_data)
                magnitude = FFT_data
                magnitude *= magnitude
                average_power[ ant_i ] += np.real( np.sum( magnitude ) )
#            else:
#                print(block_i, ant_i, num_double_zeros( oneAnt_data ))
#                plt.plot(oneAnt_data)
#                plt.show()

            if figure_location is not None:
                max_over_blocks[ant_i, block_i] = np.max( oneAnt_data[ant_i] )

    average_power[num_good_blocks!=0] /= num_good_blocks[num_good_blocks!=0]



    #### now we try to find the best referance antenna, Require that antenan allows for maximum number of good antennas, and has **best** average recieved power
    allowed_num_antennas = np.empty( num_antennas, dtype=np.int) ## if ant_i is choosen to be your referance antnena, then allowed_num_antennas[ ant_i ] is the number of antennas with num_blocks good blocks
    for ant_i in range(num_antennas):### fill allowed_num_antennas

        blocks_can_use = np.where( blocks_good[ ant_i ] )[0]
        num_good_blocks_per_antenna = np.sum( blocks_good[:,blocks_can_use], axis=1 )
        allowed_num_antennas[ ant_i ] = np.sum( num_good_blocks_per_antenna >= num_blocks )

    max_allowed_antennas = np.max(allowed_num_antennas)

    if max_allowed_antennas < 2:
        print("ERROR: station", TBB_in_file.get_station_name(), "cannot find RFI")
        return


    ## pick ref antenna that allows max number of atnennas, and has most median amount of power
    can_be_ref_antenna = (allowed_num_antennas == max_allowed_antennas )


    sorted_by_power = np.argsort( average_power )
    mps = median_sorted_by_power(sorted_by_power)

    for ant_i in mps:
        if can_be_ref_antenna[ant_i]:
            ref_antenna = ant_i
            break

    if verbose:
        print( 'Taking channel %d as reference antenna' % ref_antenna)

    ## define some helping variables ##
    good_blocks = np.where( blocks_good[ ref_antenna ] )[0]

    num_good_blocks = np.sum( blocks_good[:,good_blocks], axis=1 )
    antenna_is_good = num_good_blocks >= num_blocks

    blocks_good[np.logical_not(antenna_is_good) , : ] = False



    #### process data ####
    num_processed_blocks = np.zeros(num_antennas, dtype=int)
    frequencies = np.fft.fftfreq(block_size, 1.0/TBB_in_file.get_sample_frequency())
    lower_frequency_index = np.searchsorted(frequencies[:int(len(frequencies)/2)], lower_frequency)
    upper_frequency_index = np.searchsorted(frequencies[:int(len(frequencies)/2)], upper_frequency)

    phase_mean =    np.zeros( (num_antennas, upper_frequency_index-lower_frequency_index), dtype=complex  )
    spectrum_mean = np.zeros( (num_antennas, upper_frequency_index-lower_frequency_index), dtype=np.double)

    data = np.empty( (num_antennas, len(frequencies)), dtype=np.complex )
    temp_mag_spectrum = np.empty( (num_antennas, len(frequencies)), dtype=np.double)
    temp_phase_spectrum = np.empty( (num_antennas, len(frequencies)), dtype=np.complex)
    for block_i in good_blocks:
        if verbose:
            print( 'Doing block %d' % block_i )
        block = block_i + initial_block

        for ant_i in range(num_antennas):
            if num_processed_blocks[ant_i] == num_blocks or not blocks_good[ant_i, block_i]:
                continue
            oneAnt_data[:] = TBB_in_file.get_data( block_size*block, block_size, antenna_index=ant_i )

            ##window the data
            # Note: No hanning window if we want to measure power accurately from spectrum
            # in the same units as power from timeseries. Applying a window gives (at least) a scale factor
            # difference!
            # But no window makes the cleaning less effective... :(
            oneAnt_data *= window_function
            data[ant_i] = np.fft.fft( oneAnt_data )


        np.abs( data, out=temp_mag_spectrum )
        temp_phase_spectrum[:] = data
        temp_phase_spectrum /= (temp_mag_spectrum + 1.0E-15)
        temp_phase_spectrum[:,:] /= temp_phase_spectrum[ref_antenna,:]

        temp_mag_spectrum *= temp_mag_spectrum


        for ant_i in range(num_antennas):
            if num_processed_blocks[ant_i] == num_blocks or not blocks_good[ant_i, block_i]:
                continue

            phase_mean[ant_i,:]    +=        temp_phase_spectrum[ant_i][lower_frequency_index:upper_frequency_index]
            spectrum_mean[ant_i,:] += temp_mag_spectrum[ant_i][lower_frequency_index:upper_frequency_index]

            num_processed_blocks[ant_i] += 1

        if np.min(num_processed_blocks[antenna_is_good]) == num_blocks:
            break


    if verbose:
        print(num_blocks, "analyzed blocks", np.sum(antenna_is_good), "analyzed antennas out of", len(antenna_is_good))

    ## get only good antennas
    antenna_is_good[ref_antenna] = False ## we don't want to analyze the phase stability of the referance antenna

    ### get mean and phase stability ###
    spectrum_mean /= num_blocks

    phase_stability = np.abs(phase_mean)
    phase_stability *= -1.0/num_blocks
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
        if flag_min < 0:
            flag_min = 0
        if flag_max >= N:
            flag_max = N-1
        extend_dirty_channels[flag_min:flag_max] = True

    dirty_channels = np.where( extend_dirty_channels )

    antenna_is_good[ref_antenna] = True ## cause'.... ya know.... it is
    #### plot and return data ####
    frequencies = frequencies[lower_frequency_index:upper_frequency_index]
    if figure_location is not None:
        frequencies_MHZ = frequencies*1.0E-6

        plt.figure()
        plt.plot(frequencies_MHZ, median_phase_spread_byChannel)
        plt.axhline( median_spread-3*noise, color='r')
        plt.title("Phase spread vs frequency. Red horizontal line shows cuttoff.")
        plt.ylabel("Spread value")
        plt.xlabel("Frequency [MHz]")
        plt.savefig(figure_location+'/phase_spreads.png')
        plt.close()

        plt.figure()
        plt.plot(frequencies_MHZ, spectrum_mean[ ref_antenna ])
        plt.plot(frequencies_MHZ[dirty_channels], spectrum_mean[ ref_antenna ][dirty_channels], 'ro')
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

    cleaned_spectrum = np.array( spectrum_mean )
    cleaned_spectrum[:, dirty_channels] = 0.0
    output_dict["cleaned_spectrum_magnitude"] = cleaned_spectrum
    output_dict["cleaned_power"] = 2*np.sum( cleaned_spectrum, axis=1 )

    output_dict["antenna_names"] = TBB_in_file.get_antenna_names()
    output_dict["timestamp"] = TBB_in_file.get_timestamp()
    output_dict["antennas_good"] = antenna_is_good
    output_dict["frequency"] = frequencies

    return output_dict

class window_and_filter:
    def __init__(self, blocksize=None, find_RFI=None, timeID=None, sname=None, lower_filter=30.0E6, upper_filter=80.0E6, half_window_percent=0.1, time_per_sample=5.0E-9, filter_roll_width = 2.5E6):
        self.lower_filter = lower_filter
        self.upper_filter = upper_filter

        if timeID is not None:
            if find_RFI is None:
                find_RFI = "/findRFI/findRFI_results"
            find_RFI = processed_data_dir(timeID) + find_RFI

        if isinstance(find_RFI, str): ## load findRFI data from file
            with open( find_RFI, 'rb' ) as fin:
                find_RFI = load(fin)[ sname ]

        self.RFI_data = find_RFI

        if self.RFI_data is not None:
            if blocksize is None:
                blocksize = self.RFI_data['blocksize']
            elif self.RFI_data['blocksize'] != blocksize:
                print("blocksize and findRFI blocksize must match")
                quit()
        elif blocksize is None:
            print("window and filter needs a blocksize")
            ## TODO: check block sizes are consistant
            quit()

        self.blocksize = blocksize
        self.half_window_percent = half_window_percent
        self.half_hann_window = half_hann_window(blocksize, half_window_percent)

        FFT_frequencies = np.fft.fftfreq(blocksize, d=time_per_sample)
        self.bandpass_filter = np.zeros( len(FFT_frequencies), dtype=complex)
        self.bandpass_filter[ np.logical_and( FFT_frequencies>=lower_filter, FFT_frequencies<=upper_filter  ) ] = 1.0
        width = filter_roll_width/(FFT_frequencies[1]-FFT_frequencies[0])
        if width > 1:
            gaussian_weights = gaussian(len(FFT_frequencies), width )
            self.bandpass_filter = np.convolve(self.bandpass_filter, gaussian_weights, mode='same' )
        self.bandpass_filter /= np.max(self.bandpass_filter) ##convolution changes the peak value

        self.FFT_frequencies = FFT_frequencies

        ## completly reject low-frequency bits
        self.bandpass_filter[0] = 0.0
        self.bandpass_filter[1] = 0.0

        ##reject RFI
        if self.RFI_data is not None:
            self.bandpass_filter[ self.RFI_data["dirty_channels"] ] = 0.0

    def get_frequencies(self):
        return self.FFT_frequencies

    def get_frequency_response(self):
        return self.bandpass_filter


    def filter(self, data, additional_filter=None, whiten=False):
        data[...,:] *= self.half_hann_window
        FFT_data = np.fft.fft( data, axis=-1 )

        if whiten:
            FFT_data /= np.abs(FFT_data)

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

if __name__ == "__main__":

    from IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName

    timeID =  "D20170929T202255.000Z"
    station = "CS002"
    antenna_id = 0

    ## these lines are anachronistic and should be fixed at some point
    from LoLIM import utilities
    utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
    utilities.default_processed_data_loc = "/home/brian/processed_files"

    block_size = 2**16
    block_number = 3600

    raw_fpaths = filePaths_by_stationName(timeID)

    data_file = MultiFile_Dal1(raw_fpaths[station])


    ### find the radio stations that make noise ####
    ### this searches the beginning of the data file, before the flash, for noise due to human radio stations ###
    initial_block = 1
    number_blocks = 20
    RFI = FindRFI(data_file, block_size, initial_block, number_blocks, max_blocks=100, verbose=True, figure_location=None) ##set figure location to some folder to see output plots

    quit()

    RFI_filter = window_and_filter(block_size, RFI, lower_filter=30E6, upper_filter=80E6)

#    plt.plot( RFI_filter.bandpass_filter )
#    plt.show()


    ### now open one antenna of data
    data = np.empty((block_size), dtype=np.double)
    data[:] = data_file.get_data(block_size*block_number, block_size, antenna_index=antenna_id) ##get and store the data

    ##filter it
    filtered_data = RFI_filter.filter( data ) ## this works for multiple antennas as well, (like done at bottom of raw_tbb_IO.py)
    ## note that filtering the data turns it into complex numbers. The real component is the value of the signal, the absolute value is an envelope over the data

    plt.plot(np.abs(filtered_data), 'g', linewidth=3 )
    plt.plot(np.real(filtered_data),'r', linewidth=3)

    plt.show()

