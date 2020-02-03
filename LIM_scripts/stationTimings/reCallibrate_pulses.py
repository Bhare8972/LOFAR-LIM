#!/usr/bin/env python3

"""Use this if the antenna timeing callibration changes"""

import numpy as np
from scipy.signal import resample

import h5py
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName, read_antenna_pol_flips, read_bad_antennas
from LoLIM.utilities import processed_data_dir, even_antName_to_odd
from LoLIM.signal_processing import parabolic_fit

from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"



def recalibrate_pulse(timeID, input_fname, output_fname, set_polarization_delay=True, upsample_factor=4, polarization_flips="polarization_flips.txt"):

    processed_data_folder = processed_data_dir(timeID)
    raw_fpaths = filePaths_by_stationName(timeID)
    polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    
    input_file = h5py.File(processed_data_folder+'/'+input_fname, "r")
    
    try:
        output_file = h5py.File(processed_data_folder+'/'+output_fname, "r+")
    except:
        output_file = h5py.File(processed_data_folder+'/'+output_fname, "w")
        
    sample_time = 5.0e-9
    if upsample_factor>1:
        sample_time /= upsample_factor
    
    for sname, h5_statGroup in input_file.items():
        out_statGroup = output_file.require_group(sname)
        print()
        print()
        print(sname)
        
        datafile = MultiFile_Dal1(raw_fpaths[sname], polarization_flips=polarization_flips)
        
        if set_polarization_delay:
            datafile.find_and_set_polarization_delay()
            
        antenna_calibrations = { antname:calibration for antname,calibration in zip(datafile.get_antenna_names(), datafile.get_total_delays()) }
        
        for antname, in_antData in h5_statGroup.items():
            out_statGroup.copy(in_antData,  out_statGroup, name= antname)
            out_antData = out_statGroup[antname]
            
            old_even =  out_antData.attrs['PolE_timeOffset_CS']
            old_odd = out_antData.attrs['PolO_timeOffset_CS']
            
            new_even_delay = antenna_calibrations[antname]
            new_odd_delay = antenna_calibrations[ even_antName_to_odd(antname) ]
            
            out_antData.attrs['PolE_timeOffset'] = -new_even_delay ## NOTE: due to historical reasons, there is a sign flip here
            out_antData.attrs['PolO_timeOffset'] = -new_odd_delay ## NOTE: due to historical reasons, there is a sign flip here
            out_antData.attrs['PolE_timeOffset_CS'] = new_even_delay
            out_antData.attrs['PolO_timeOffset_CS'] = new_odd_delay
            
            print(antname, old_even-new_even_delay, old_odd-new_odd_delay)
            
            starting_index = out_antData.attrs['starting_index']
            
            if np.isfinite( out_antData.attrs['PolE_peakTime'] ):
                
                polE_HE = out_antData[1]
                
                if upsample_factor>1:
                    polE_HE = resample(polE_HE, len(polE_HE)*upsample_factor )
            
                PolE_peak_finder = parabolic_fit( polE_HE  )
                out_antData.attrs['PolE_peakTime'] =  starting_index*5.0E-9 - new_even_delay + PolE_peak_finder.peak_index*sample_time
            
            if np.isfinite( out_antData.attrs['PolO_peakTime'] ):
                
                polO_HE = out_antData[3]
                
                if upsample_factor>1:
                    polO_HE = resample(polO_HE, len(polO_HE)*upsample_factor )
            
                PolO_peak_finder = parabolic_fit( polO_HE  )
                out_antData.attrs['PolO_peakTime'] =  starting_index*5.0E-9 - new_odd_delay + PolO_peak_finder.peak_index*sample_time
        
            
            