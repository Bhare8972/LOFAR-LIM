#!/usr/bin/env python3

"""this is a short script that runs findRFI and saves the results to a python pickle file"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from utilities import processed_data_dir, log
from raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from findRFI import FindRFI

from os import mkdir
from os.path import isdir
from pickle import dump

if __name__ == "__main__":
    timeID = "D20170929T202255.000Z"
    output_folder = "/findRFI"
    out_fname = "/findRFI_results"
    block_size = 2**16
    initial_block = 5
    num_blocks = 20
    max_blocks = 100
    
    skip_stations = ["CS028"]
    
    
    processed_data_dir = processed_data_dir(timeID)
    
    output_fpath = processed_data_dir + output_folder
    if not isdir(output_fpath):
        mkdir(output_fpath)
        
    log.set(output_fpath+'/log.txt', True)
    log.take_stdout()
        
    
    #### get paths to raw data by station ####
    raw_fpaths = filePaths_by_stationName(timeID)
    
    output = {}
    
    for station in raw_fpaths.keys():
        if station in skip_stations:
            continue
        print("station", station)
        
        path = output_fpath + '/' + station
        if not isdir(path):
            mkdir(path)
            
        TBB_data = MultiFile_Dal1( raw_fpaths[station] )
        out = FindRFI(TBB_data, block_size, initial_block, num_blocks, max_blocks, verbose=True, figure_location=path)
        output[station] = out
        
    with open(output_fpath+out_fname, 'wb') as fout:
        dump(output, fout)