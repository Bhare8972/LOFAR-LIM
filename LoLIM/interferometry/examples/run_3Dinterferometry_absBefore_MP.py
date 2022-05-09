#!/usr/bin/env python3

from LoLIM.interferometry.interferometry_absBefore import interferometric_locator
from multiprocessing import Process
from copy import copy
from time import sleep


## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

def run(imager, out_folder, initial_datapoint, start_block, num_blocks, run_number):
    imager.run_multiple_blocks(output_folder = out_folder, 
                                   initial_datapoint = initial_datapoint, 
                                   start_block = start_block, 
                                   blocks_per_run = num_blocks, 
                                   run_number = run_number, 
                                   skip_blocks_done = True,
                                   print_to_screen=False
                                   )

if __name__=="__main__":
    
    imager_utility = interferometric_locator(timeID = "D20180921T194259.023Z",   
                                             station_delays_fname = "station_delays.txt", 
                                             additional_antenna_delays_fname = "ant_delays.txt", 
                                             bad_antennas_fname = "bad_antennas.txt", 
                                             pol_flips_fname = "polarization_flips.txt",
                                             bounding_box = [[-43000, -33000], [-15000, -5000], [0, 9000]],  ## bounds around flash in min max, X, Y, Z
                                             pulse_length = 50, 
                                             num_antennas_per_station = 6,
                                             )
    
    imager_utility.stations_to_exclude = [] 
    imager_utility.block_size = 2**16
    
    imager_utility.prefered_station = None
    imager_utility.use_core_stations_S1 = True
    imager_utility.use_core_stations_S2 = False
    
    imager_utility.do_RFI_filtering = True
    imager_utility.use_saved_RFI_info = True
    imager_utility.initial_RFI_block= None
    imager_utility.RFI_num_blocks = None
    imager_utility.RFI_max_blocks = None
    
    imager_utility.upsample_factor = 8
    imager_utility.max_events_perBlock = 10# 100
    
    imager_utility.stage_1_converg_num = 100
    imager_utility.stage_1_max_itters = 1500
    
    imager_utility.erase_pulses = True
    
    imager_utility.remove_saturation = True
    
    imager_utility.min_pref_ant_amplitude = 10 ## normally 10
    
    
    
    output_folder = "interferometry_out"
    initial_datapoint = 940*2**16
    start_block = 0
    num_total_blocks = 2860
    num_processes = 5
    
    blocks_per_process = int(num_total_blocks/num_processes) + 1
    
    processes = []
    
    for run_i in range(num_processes):
        p = Process(target=run, args=(copy(imager_utility), output_folder, initial_datapoint, start_block, blocks_per_process, run_i))
        p.start()
        processes.append(p)
        sleep(1)
        
    for p in processes:
        p.join()
        sleep(60)
        
    print("all done!")
    

    