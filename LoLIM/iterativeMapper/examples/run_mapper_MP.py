#!/usr/bin/env python3

"""Use this to actually run the iterative mapper. But first make sure you made a header file with make_header.py"""

import time
from multiprocessing import Process, Lock

from LoLIM.utilities import logger
from LoLIM.iterativeMapper.iterative_mapper import read_header, iterative_mapper

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

out_folder = 'iterMapper_9Aug2023Cal_evenAnts'
timeID = 'D20190424T194432.504Z'

first_block = 0
final_datapoint = 6000*(2**16) ## this may be exceeded by upto one block size
num_processes = 4
## the mapper is most efficent when it proceses consecutive blocks, and does not jump around
## but we don't want one process to get the majority of the flash, and anouther to get all noise
## therefore, each process images a number of consecutive blocks, then leap-frogs forward.
num_consecutive_blocks = 100


print_lock = Lock()
def run_process(process_i):
    
    inHeader = read_header(out_folder, timeID)
    log_fname = inHeader.next_log_file()
    
    logger_function = logger()
    logger_function.set( log_fname, False )
    # logger_function.take_stdout()
    logger_function.take_stderr()
    
    logger_function("process", process_i)
    logger_function("date and time run:", time.strftime("%c") )

    print_lock.acquire()
    print('loading process', process_i)
    print_lock.release()
    
    mapper = iterative_mapper(inHeader, logger_function)
    
    current_datapoint = 0
    current_run = 0
    while current_datapoint < final_datapoint:
        
        for blockJ in range(num_consecutive_blocks):
            block = first_block + blockJ + process_i*num_consecutive_blocks + current_run*num_consecutive_blocks*num_processes

            print_lock.acquire()
            print('process', process_i, 'on block', block)
            print_lock.release()
            
            mapper.process_block( block, logger_function )
            
            current_datapoint = inHeader.initial_datapoint + (block+1)*mapper.usable_block_length
            if current_datapoint > final_datapoint:
                break
            
        current_run += 1
        
    
    logger_function("done!" )
        
        
processes = []

for run_i in range(num_processes):
    p = Process(target=run_process, args=(run_i,) )
    p.start()
    processes.append(p)
    time.sleep(1)
    
for p in processes:
    p.join()
    time.sleep(60)
    
print("all done!")