#!/usr/bin/env python3

"""
This is the top-level module for reading tbuf and tbb data. It concatenates imports from LOFAR1_tbbIO,  LOFAR2_tbufIO, and calFiles. Largely to both maintain backwards compatibility while also not havting too much in one python file.
This module also has some helper functions for opening data files
"""




import os
import datetime
import json

import numpy as np
import h5py

import LoLIM.IO.metadata as md
import LoLIM.utilities as util
import LoLIM.atmosphere as atmo


from LoLIM.IO.LOFAR1_tbbIO import LOFAR1_tbb_reader
from LoLIM.IO.LOFAR2_tbufIO import LOFAR2_tbuf_reader  ## note for LOFAR1 it is tbb, for LOFAR2.0 it is tbuf

MultiFile_Dal1 = LOFAR1_tbb_reader ## renaming is for backwards compatibility

from LoLIM.IO.calFiles import total_cal_object, read_cal_file,  read_antenna_delays, read_bad_antennas, read_antenna_pol_flips, read_station_delays, read_olaf_file

#### helper functions ####

def filePaths_by_stationName(timeID, raw_data_loc=None):
    """Given a timeID, and a location of raw data (default set in utilities.py), return a dictionary.
    The keys of the dictionary are antenna names, the values are lists of file paths to data files that contain that station."""
    
    data_file_path = util.raw_data_dir(timeID, raw_data_loc)
    h5_files = [f for f in os.listdir(data_file_path) if f[-6:] == 'tbb.h5']
    
    ret = {}
    for fname in h5_files:
        Fpath = data_file_path + '/' + fname
        junk, sname, junk, junk = util.Fname_data(Fpath)

        if sname not in ret:
            ret[sname] = []
            
        ret[sname].append( Fpath )
        
    return ret

def eventData_filePaths(timeID, raw_data_loc=None):
    """Given a timeID, and a location of raw data (default set in utilities.py), return a list of file paths of data files"""
    data_file_path = util.raw_data_dir(timeID, raw_data_loc)
    return [f for f in os.listdir(data_file_path) if f[-6:] == 'tbb.h5']



# L1 argumetns:
#filename_list, force_metadata_ant_pos=True, total_cal=None,
#                 polarization_flips=None, bad_antennas=[], additional_ant_delays=None, station_delay=0.0, 
#                 only_complete_pairs=True, pol_flips_are_bad=False

# L2 argumetns:
#file_list,  total_cal=None,
#                 only_complete_pairs=True, pol_flips_are_bad=False,
#                 antenna_mode='all'
def open_L1_or_L2_data(file_list, total_cal=None, force_metadata_ant_pos=True,  only_complete_pairs=True, pol_flips_are_bad=False, antenna_mode='all'):
    """Given a list of files (for one station), and a series of possible arguments, return either a LOFAR1 data file or a LOFAR2.0 data file. Error if they are not all L1 or L2"""

    mode = 0
    ## 0 is undecided.  1 is that all have been L1 thus far. 2 is all have been L2 so far.  3 is error

    openedH5_files = []
    for filename in file_list:
        file = h5py.File(filename, "r")
        openedH5_files.append( file )

        isL1 = True ## if False than is L2
        if 'TELESCOPE_VERSION' in file.attrs:
            if file.attrs['TELESCOPE_VERSION'] == '2.0':
                isL1 = False 

        if mode == 0: 
            if isL1:
                mode = 1
            else:
                mode = 2
        elif mode == 1:
            if isL1:
                continue
            else:
                mode = 3
                print('ERROR: mix of L1 and L2 style files')
                quit()
        elif mode == 2:
            if not isL1:
                continue
            else:
                mode = 3
                print('ERROR: mix of L1 and L2 style files')
                quit()

    if mode == 1:

        ## lofar 1
        return LOFAR1_tbb_reader(file_list, force_metadata_ant_pos=force_metadata_ant_pos, total_cal=total_cal,
                 only_complete_pairs=only_complete_pairs, pol_flips_are_bad=pol_flips_are_bad)

    elif mode == 2:
        ## lfoar 2

        return LOFAR2_tbuf_reader(openedH5_files,  total_cal=total_cal, only_complete_pairs=only_complete_pairs, pol_flips_are_bad=pol_flips_are_bad,antenna_mode=antenna_mode)








if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    timeID =  "D20170929T202255.000Z"
    station = "RS406"
    antenna_id = 0
    
    block_size = 2**16
    block_number = 30#3900
    
    raw_fpaths = filePaths_by_stationName(timeID)
    
    infile = MultiFile_Dal1(raw_fpaths[station])
#    infile = MultiFile_Dal1(["./new_file.h5"])
    
    print( infile.get_LOFAR_centered_positions() )
    
    data = infile.get_data(block_number*block_size, block_size, antenna_index=antenna_id)
    
    plt.plot(data)
    plt.show()
#        
        
        
        
        
        
        
        
        
        
        
        
        
    