#!/usr/bin/env python3

"""Use this to actually run the iterative mapper. But first make sure you made a header file with make_header.py"""

import time

from LoLIM.utilities import logger
from LoLIM.iterativeMapper.iterative_mapper import read_header, iterative_mapper

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

out_folder = 'iterMapper_50_CS003'
timeID = 'D20170929T202255.000Z'

final_datapoint = 5500*(2**16) ## this may be exceeded by upto one block size

    
inHeader = read_header(out_folder, timeID)
log_fname = inHeader.next_log_file()

logger_function = logger()
logger_function.set( log_fname, True )
logger_function.take_stdout()
logger_function.take_stderr()

logger_function("date and time run:", time.strftime("%c") )

mapper = iterative_mapper(inHeader, logger_function)

current_datapoint = 0
current_block = 0
while current_datapoint < final_datapoint:
    
    mapper.process_block( current_block, logger_function )
    
    current_datapoint = inHeader.initial_datapoint + (current_block+1)*mapper.usable_block_length
        
    current_block += 1
    

logger_function("done!" )

    
print("all done!")