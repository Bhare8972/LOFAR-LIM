#!/usr/bin/env python3

import numpy as np

from LoLIM.utilities import processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName, read_antenna_pol_flips, read_bad_antennas
from LoLIM.findRFI import window_and_filter
from LoLIM.signal_processing import num_double_zeros

def get_noise_std(timeID, initial_block, max_num_blocks, max_double_zeros=100, stations=None, polarization_flips="polarization_flips.txt", bad_antennas="bad_antennas.txt"):
   
    processed_data_folder = processed_data_dir(timeID)
    if isinstance(polarization_flips, str):
        polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    if isinstance(processed_data_folder, str):
        bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
        
    half_window_percent = 0.1
    half_window_percent *= 1.1 ## just to be sure we are away from edge
        
    raw_fpaths = filePaths_by_stationName(timeID)
    if stations is None:
        stations = raw_fpaths.keys()
    
    out_dict = {}
    for sname in stations:
        print(sname)
        
        TBB_data = MultiFile_Dal1( raw_fpaths[sname], polarization_flips=polarization_flips, bad_antennas=bad_antennas)
        RFI_filter = window_and_filter(timeID=timeID, sname=sname)
        block_size = RFI_filter.blocksize
        edge_size = int( half_window_percent*block_size )
        
        antenna_names = TBB_data.get_antenna_names()
        measured_std = np.zeros( len(antenna_names) )
        measured_std[:] = -1
        
        for block_i in range(initial_block, initial_block+max_num_blocks):
            for ant_i in range(len(antenna_names)):
                if measured_std[ant_i] > 0:
                    continue ## we got a measurment, so skip it
                
                data = np.array( TBB_data.get_data( block_i*block_size, block_size, antenna_index=ant_i ), dtype=np.double)
                
                if num_double_zeros( data ) > max_double_zeros:
                    continue ## bad block, skip
                
                filtered_data = np.real( RFI_filter.filter( data ) )
                filtered_data = filtered_data[edge_size:-edge_size] ##avoid the edge
                measured_std[ant_i] = np.std(filtered_data)
                
            if np.all( measured_std > 0 ): ## we can break early
                break
                
        for ant_name, std in zip(antenna_names,measured_std):
            out_dict[ant_name] = std
            
        filter = measured_std > 0
        print("   ave std:", np.average(measured_std[filter]) )
        print("   total ant:", len(filter), "good ant:", np.sum(filter))
        print()
            
    return out_dict

def to_file(noise_std_dict, folder):
    with open( folder + '/noise_std.txt',  'w') as fout:
        for ant, std in noise_std_dict.items():
            fout.write(ant)
            fout.write(' ')
            fout.write(str(std))
            fout.write('\n')
            
def from_file(folder):
    ret = {}
    with open( folder + '/noise_std.txt',  'r') as fin:
        for line in fin:
            ant, std = line.split()[:2]
            ret[ant] = float(std)
    return ret
                
                

                
                
        
        
        
        
    