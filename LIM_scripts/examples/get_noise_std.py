#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from os import mkdir
from os.path import isdir

from LoLIM.utilities import processed_data_dir, logger
from LoLIM.noise_analysis import get_noise_std, to_file

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__=="__main__":
    timeID = "D20180813T153001.413Z"
    out_folder = 'noise_std'
    
    processed_data_folder = processed_data_dir(timeID)
    output_fpath = processed_data_folder + '/' + out_folder
    if not isdir(output_fpath):
        mkdir(output_fpath)
        
    log = logger()
    log.set(output_fpath+'/log.txt')
    log.take_stdout()
    
    result_dict = get_noise_std(
         timeID =   timeID,
         initial_block = 5500,
         max_num_blocks = 500
            )
    
    to_file(result_dict, output_fpath)