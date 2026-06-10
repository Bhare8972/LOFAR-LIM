#!/usr/bin/env python3

"""This module implements finding of Radio Frequency Interference for LOFAR data.

This module is strongly based on pyCRtools findrfi.py by Arthur Corstanje.
However, it has been heavily modified for use with LOFAR-LIM by Brian Hare

This replaces a few functions in findRFI.py. It is very similar, but allows for RFI data to be descrimated by antenna type and polarization. 
"""

from os import path
import json

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal.windows import gaussian

#from LoLIM.utilities import processed_data_dir
from LoLIM import utilities as utils
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

def FindRFI_adv(TBB_in_file, block_size, initial_block, num_blocks, max_blocks=None, verbose=False, figure_location=None, lower_frequency=10E6, upper_frequency=90E6, num_dbl_z=100):
    """ use phase-variance to find RFI in data. TBB_in_file should be a MultiFile_Dal1, encompassing the data for one station. block_size should be around 65536 (2^16).
    num_blocks should be at least 20. Sometimes a block needs to be skipped, so max_blocks shows the maximum number of blocks used (after initial block) used to find num_blocks
    number of good blocks. initial block should be such that there is no lightning in the max_blocks number of blocks. If max_blocks is None (default), it is set to num_blocks
    figure_location should be a folder to save relavent figures, default is None (do not save figures). num_dbl_z is number of double zeros allowed in a block, if there are too
    many, then there could be data loss.

    returns a dictionary with the following key-value pairs:
        "dirty_channels": an array of indeces indicating the channels that are contaminated with RFI

        TODO: fix this
    """

    if max_blocks is None:
        max_blocks = num_blocks

    window_function = half_hann_window(block_size, 0.1)  ## see note below under  "##window the data"

    all_antenna_names = TBB_in_file.get_antenna_names() ## this OUGHT to be garrunteed to be ordered Y,X antennas always
    #num_antennas = len(all_antenna_names)
    num_antenna_pairs = int( len(all_antenna_names)/2 )


    oneAnt_data = np.empty( block_size, dtype=np.double )
    blocks_good = np.zeros((num_antenna_pairs, max_blocks), dtype=bool)
    num_good_blocks = np.zeros(num_antenna_pairs, dtype=int)
    average_power = np.zeros(num_antenna_pairs, dtype=np.double)

    allowed_num_antennas = np.empty( num_antenna_pairs, dtype=int) ## if ant_i is choosen to be your referance antnena, then allowed_num_antennas[ ant_i ] is the number of antennas with num_blocks good blocks

    num_processed_blocks = np.zeros(num_antenna_pairs, dtype=int)

    frequencies = np.fft.fftfreq(block_size, 1.0/TBB_in_file.get_sample_frequency())



## need to be adjusted for higher niyquist zones?
## ARE THESE REALLY NEEDED??
    lower_frequency_index = np.searchsorted(frequencies[:int(len(frequencies)/2)], lower_frequency)
    upper_frequency_index = np.searchsorted(frequencies[:int(len(frequencies)/2)], upper_frequency)

    cut_frequencies = frequencies[lower_frequency_index:upper_frequency_index]
    cut_frequencies_MHZ = cut_frequencies*(1e-6)

    phase_mean =    np.zeros( (num_antenna_pairs, upper_frequency_index-lower_frequency_index), dtype=complex  )

    data = np.empty( (num_antenna_pairs, len(frequencies)), dtype=complex )
    temp_mag_spectrum = np.empty( (num_antenna_pairs, len(frequencies)), dtype=np.double)
    temp_phase_spectrum = np.empty( (num_antenna_pairs, len(frequencies)), dtype=complex)

    RFI_output_list = []
    data_output_list = []

    for polarization in [0,1]:
        blocks_good[:,:] = False
        num_good_blocks[:] = 0
        average_power[:] = 0
        num_processed_blocks[:] = 0
        phase_mean[:,:] = 0


        spectrum_mean = np.zeros( (num_antenna_pairs, upper_frequency_index-lower_frequency_index), dtype=np.double)

        pol_name = ['Y', 'X'][polarization]

        if verbose:
            print('doing polarization:', pol_name)
            print( 'finding good blocks' )

        pol_antenna_names = all_antenna_names[polarization::2]

        #### step one: find which blocks are good, and find average power ####
        
        for block_i in range(max_blocks):
            block = block_i + initial_block

            for pair_i in range(num_antenna_pairs):

                if not TBB_in_file.has_antenna(antenna_name=pol_antenna_names[pair_i]):
                    blocks_good[ pair_i, block_i ] = False
                    continue

                oneAnt_data[:] = TBB_in_file.get_data( block_size*block, block_size, antenna_name=pol_antenna_names[pair_i] )

                if num_double_zeros( oneAnt_data ) < num_dbl_z: ## this antenna on this block is good
                    blocks_good[ pair_i, block_i ] = True
                    num_good_blocks[ pair_i ] += 1

                    oneAnt_data *= window_function

                    FFT_data = np.fft.fft( oneAnt_data )
                    np.abs(FFT_data, out=FFT_data)
                    magnitude = FFT_data
                    magnitude *= magnitude
                    average_power[ pair_i ] += np.real( np.sum( magnitude ) )

        average_power[num_good_blocks!=0] /= num_good_blocks[num_good_blocks!=0]




        #### now we try to find the best referance antenna, Require that antenan allows for maximum number of good antennas, and has **best** average recieved power
        for pair_i in range(num_antenna_pairs):### fill allowed_num_antennas

            blocks_can_use = np.where( blocks_good[ pair_i ] )[0]
            num_good_blocks_per_antenna = np.sum( blocks_good[:,blocks_can_use], axis=1 )
            allowed_num_antennas[ pair_i ] = np.sum( num_good_blocks_per_antenna >= num_blocks )

        max_allowed_antennas = np.max(allowed_num_antennas)

        if max_allowed_antennas < 2:
            print("ERROR: station", TBB_in_file.get_station_name(), "cannot find RFI")
            return


        ## pick ref antenna that allows max number of atnennas, and has most median amount of power
        can_be_ref_antenna = (allowed_num_antennas == max_allowed_antennas )


        sorted_by_power = np.argsort( average_power )
        mps = median_sorted_by_power(sorted_by_power)

        for pair_i in mps:
            if can_be_ref_antenna[pair_i]:
                ref_antenna_pairi = pair_i
                break

        if verbose:
            print( 'Taking channel %d as reference antenna' %(ref_antenna_pairi*2+polarization), ':', pol_antenna_names[ref_antenna_pairi])

        ## define some helping variables ##
        good_blocks = np.where( blocks_good[ ref_antenna_pairi ] )[0]

        num_good_blocks = np.sum( blocks_good[:,good_blocks], axis=1 )
        antenna_is_good = num_good_blocks >= num_blocks

        blocks_good[np.logical_not(antenna_is_good) , : ] = False



        #### process data ####
        for block_i in good_blocks:
            block = block_i + initial_block
            if verbose:
                print( 'Doing block %d' % block )

            for pair_i in range(num_antenna_pairs):
                if (num_processed_blocks[pair_i] == num_blocks and not pair_i==ref_antenna_pairi) or not blocks_good[pair_i, block_i]:
                    continue
                oneAnt_data[:] = TBB_in_file.get_data( block_size*block, block_size, antenna_name=pol_antenna_names[pair_i]  )


                ##window the data
                # Note: No hanning window if we want to measure power accurately from spectrum
                # in the same units as power from timeseries. Applying a window gives (at least) a scale factor
                # difference!
                # But no window makes the cleaning less effective... :(
                oneAnt_data *= window_function

                data[pair_i] = np.fft.fft( oneAnt_data )

            data /= block_size

            np.abs( data, out=temp_mag_spectrum )

            temp_phase_spectrum[:] = data
            temp_phase_spectrum /= (temp_mag_spectrum + 1.0E-15)

            temp_phase_spectrum[:,:] /= temp_phase_spectrum[ref_antenna_pairi,:]

            temp_mag_spectrum *= temp_mag_spectrum ## square


            for pair_i in range(num_antenna_pairs):
                if (num_processed_blocks[pair_i] == num_blocks and not pair_i==ref_antenna_pairi) or not blocks_good[pair_i, block_i]:
                    continue

                phase_mean[pair_i,:]    += temp_phase_spectrum[pair_i][lower_frequency_index:upper_frequency_index]
                spectrum_mean[pair_i,:] += temp_mag_spectrum[pair_i][lower_frequency_index:upper_frequency_index]

                num_processed_blocks[pair_i] += 1

            if np.min(num_processed_blocks[antenna_is_good]) == num_blocks:
                break


        if verbose:
            print(num_blocks, "analyzed blocks", np.sum(antenna_is_good), "analyzed antennas out of", len(antenna_is_good))

        ## get only good antennas
        antenna_is_good[ref_antenna_pairi] = False ## we don't want to analyze the phase stability of the referance antenna

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

        dirty_channels = np.where( extend_dirty_channels )[0]

        antenna_is_good[ref_antenna_pairi] = True ## cause'.... ya know.... it is
        #### plot and return data ####
        if figure_location is not None:

            plt.figure()
            plt.plot(cut_frequencies_MHZ, median_phase_spread_byChannel)
            plt.axhline( median_spread-3*noise, color='r')
            plt.title("Phase spread vs frequency. Red horizontal line shows cuttoff.\nPolarization "+pol_name)
            plt.ylabel("Spread value")
            plt.xlabel("Frequency [MHz]")
            # plt.legend()
            if figure_location == "show":
                plt.show()
            else:
                plt.savefig(figure_location+'/phase_spreads_Pol'+pol_name+'.png')
                plt.close()

            plt.figure()
            plt.plot(cut_frequencies_MHZ,                 spectrum_mean[ ref_antenna_pairi ])
            plt.plot(cut_frequencies_MHZ[dirty_channels], spectrum_mean[ ref_antenna_pairi ][dirty_channels], 'ro')
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("magnitude")
            plt.yscale('log')#, nonposy='clip')
            # plt.legend()
            if figure_location == "show":
                plt.show()
            else:
                plt.savefig(figure_location+'/magnitude_Pol'+pol_name+'.png')
                plt.close()

            #plt.figure()
            #for maxes, ant_name in zip(max_over_blocks, TBB_in_file.get_antenna_names()):
            #    plt.plot(maxes, label=ant_name)
            #plt.ylabel("maximum")
            #plt.xlabel("block index")
            #plt.legend()
            #if figure_location == "show":
            #    plt.show()
            #else:
             #   plt.savefig(figure_location+'/max_over_blocks.png')
             #   plt.close()


        ## NEED: median_phase_spread_byChannel, spectrum_mean[ref_antenna_pairi],  


        RFI_output_dict = {}
        RFI_output_dict["blocksize"] = block_size
        RFI_output_dict["initial_block"] = initial_block
        RFI_output_dict["num_blocks"] = num_blocks
        RFI_output_dict["max_blocks"] = max_blocks

        RFI_output_dict["dirty_channels"] = list(dirty_channels + lower_frequency_index)

        cleaned_spectrum = np.array( spectrum_mean )
        cleaned_spectrum[:, dirty_channels] = 0.0
        RFI_output_dict["cleaned_power"] = list( 2*np.sum( cleaned_spectrum, axis=1 ) )

        RFI_output_dict["median_phase_spread_byChannel"] = median_phase_spread_byChannel
        RFI_output_dict["reference_spectrum"] = spectrum_mean[ref_antenna_pairi]

        RFI_output_dict["antenna_names"] = list(pol_antenna_names)
        RFI_output_dict["antennas_good"] = list( antenna_is_good )
        

        RFI_output_dict["antenna_set"] = TBB_in_file.get_antenna_set()
        RFI_output_dict["filter_selection"] = TBB_in_file.get_filter_selection()
        RFI_output_dict["antenna_polarization"] = pol_name

        RFI_output_dict["timestamp"] = TBB_in_file.get_timestamp()
        RFI_output_dict["lower_frequency_index"] = lower_frequency_index
        RFI_output_dict["upper_frequency_index"] = upper_frequency_index

        RFI_output_list.append( RFI_output_dict )

        # data_output_dict = {}
        # data_output_dict["blocksize"] = block_size
        # data_output_dict["dirty_channels"] = list(dirty_channels + lower_frequency_index)
        # data_output_dict["antenna_set"] = TBB_in_file.get_antenna_set()
        # data_output_dict["filter_selection"] = TBB_in_file.get_filter_selection()
        # data_output_dict["antenna_polarization"] = pol_name

        # data_output_dict["ave_spectrum_magnitude"] = spectrum_mean
        # data_output_dict["ave_spectrum_phase"] = np.angle(phase_mean, deg=False)
        # data_output_dict["phase_variance"] = phase_stability

        # cleaned_spectrum = np.array( spectrum_mean )
        # cleaned_spectrum[:, dirty_channels] = 0.0
        # data_output_dict["cleaned_spectrum_magnitude"] = cleaned_spectrum
        # data_output_dict["cleaned_power"] = 2*np.sum( cleaned_spectrum, axis=1 )

        # #data_output_dict["antenna_names"] = TBB_in_file.get_antenna_names()
        # data_output_dict["timestamp"] = TBB_in_file.get_timestamp()
        # #data_output_dict["antennas_good"] = antenna_is_good
        # data_output_dict["frequency"] = frequencies

        # data_output_dict["lower_frequency_index"] = lower_frequency_index
        # data_output_dict["upper_frequency_index"] = upper_frequency_index

        # data_output_dict["antenna_set"] = TBB_in_file.get_antenna_set()
        # data_output_dict["filter_selection"] = TBB_in_file.get_filter_selection()
        # data_output_dict["antenna_polarization"] = pol_name


        if verbose:
            print( 'cleaned power:', RFI_output_dict["cleaned_power"]  )
            print('uncleaned power', 2*np.sum( spectrum_mean, axis=1 ) )
            print('num dirty channels:', len(dirty_channels))


        #data_output_list

    return { TBB_in_file.get_station_name():RFI_output_list }

def fold_advFindRFI_resultsTogether(results_A, results_B):
    """given two inputs, each an output of: this function, open_advFindRFI, or FindRFI_adv. Create a return a new dictionary that contains the information of the two input dictionaries. 
    Note that results_A has priority over results_B if there is a collision in station/antenna modes."""


    def helperFunction_itemInList(item, item_list):
        AS = item["antenna_set"]
        FS = item["filter_selection"]
        AP = item["antenna_polarization"]

        for il in item_list:
            if (il["antenna_set"]==AS) and (il["filter_selection"]==FS) and (il["antenna_polarization"]==AP):
                return True

        return False


    output = {}

    for sname in results_A.keys():

        station_data = [ i for i in results_A[sname] ]

        if sname in results_B:
            for itemB in results_B[sname]:
                if not helperFunction_itemInList(itemB, station_data):
                    station_data.append( itemB )

        output[sname] = station_data

    for sname in results_B.keys():
        if sname not in output:
            output[sname] = [ i for i in results_B[sname] ]

    return output





def save_advFindRFI( data, timeID=None, folder=None, fname=None ):

    if fname is None:
        fname = 'advFindRFI_results.json'

    if not (timeID is None):
        if (folder is None):
            folder = "findRFI"

        if folder[0]=='/':# python join will not work right in this case...
            folder = folder[1:]
        folder = path.join( utils.processed_data_dir(timeID), folder )

    if not (folder is None):
        fullFname = path.join(folder, fname)

    else:
        fullFname = fname

    json.dump( data, fp=open(fullFname, 'w'),  cls=utils.JSON_CustomEncoder) 

def open_advFindRFI( timeID=None, folder=None, fname=None  ):

    if fname is None:
        fname = 'advFindRFI_results.json'

        if (folder is None) and (timeID is None):
            print('ERROR in open_advFindRFI: fname, folder, and timeID cannot all be None')
            quit()

    if not (timeID is None):
        if (folder is None):
            folder = "findRFI"

        if folder[0]=='/':# python join will not work right in this case...
            folder = folder[1:]
        folder = path.join( utils.processed_data_dir(timeID), folder )

    if not (folder is None):
        fullFname = path.join(folder, fname)
    else:
        fullFname = fname

    return json.load( fp=open(fullFname, 'r'), cls=utils.JSON_CustomDecoder_maker() )
