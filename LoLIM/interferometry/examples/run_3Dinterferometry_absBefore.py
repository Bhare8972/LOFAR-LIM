#!/usr/bin/env python3

from LoLIM.interferometry.interferometry_absBefore import interferometric_locator


## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__=="__main__":
    
    imager_utility = interferometric_locator(timeID = "D20180813T153001.413Z",   
                                             station_delays = "station_delays.txt", 
                                             additional_antenna_delays = "ant_delays.txt", 
                                             bad_antennas = "bad_antennas.txt", 
                                             polarization_flips = "polarization_flips.txt",
                                             bounding_box = [[-7000, 3000], [-55000, -45000,], [0, 8000]],  ## bounds around flash in min max, X, Y, Z
                                             pulse_length = 50, 
                                             num_antennas_per_station = 6,
                                             )
    
    imager_utility.stations_to_exclude = ['CS031', 'CS401', "RS210", 'RS305', "RS310"] 
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
    imager_utility.max_events_perBlock = 10
    
    imager_utility.stage_1_converg_num = 100
    imager_utility.stage_1_max_itters = 1500
    
    imager_utility.erase_pulses = True
    imager_utility.remove_saturation = True
    
    
    imager_utility.run_multiple_blocks(output_folder = "interferometry_out_fastTest", 
                                       initial_datapoint = 2290*2**16, 
                                       start_block = 0, 
                                       blocks_per_run = 602, 
                                       run_number = 0, 
                                       skip_blocks_done = True,
                                       )
                                        ## this will process blocks starting from data point:  initial_datapoint + start_block*block_size + run_number*blocks_per_run*block_size
                                        ## and will process blocks_per_run number of blocks, where block_size accounts for overlap between blocks. 
    
    