#!/usr/bin/env python3

from os import mkdir
from os.path import isdir

from LoLIM.stationTimings.find_pulses_GUI import plot_stations
from LoLIM.utilities import processed_data_dir

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = "D20180921T194259.023Z"
    initial_block = 3000 #int(   1.1600369/((2**16)*(5.0E-9))  )#3600
    block_size = 2**16
    
    guess_flash_location =  None #[2.43390916381, -50825.1364969, 0.0, 1.56124908739] ## use this if time delays are real and not apparent. Set to None when delays above are apparent delays, not real delays
    
    guess_station_delays = {

'CS002' : 0.0 ,
'CS001' : 1.49644698976e-06 ,
'CS003' : 1.40455863627e-06 ,
'CS004' : 4.30900574133e-07 ,
'CS006' : 8.94409445451e-07 ,
'CS007' : 7.46300933127e-07 ,
'CS011' : 8.76569170358e-08 ,
'CS013' : -1.81465773882e-06 ,
'CS017' : -6.57572451796e-06 ,
'CS024' : 3.06975229646e-06 ,
'CS026' : -6.3484676649e-06 ,
'CS030' : -3.77062002272e-06 ,
'CS032' : -3.73079751276e-06 ,
'CS101' : -4.75385038898e-06 ,
'CS103' : -2.22828292027e-05 ,
'CS201' : -7.49992400687e-06 ,
'CS301' : -1.21776623579e-06 ,
'CS302' : -1.03989730381e-05 ,
'CS501' : -8.84919256839e-06 ,
'RS205' : 9.3374132956e-06 ,
'RS208' : 2.48511924021e-05 ,
'RS210' : 9.64331237672e-05 ,
'RS305' : -1.53867461004e-05 ,
'RS306' : -2.23519432346e-05 ,
'RS307' : -3.80406730138e-05 ,
'RS406' : -1.87082907205e-06 ,
'RS407' : 2.10027144301e-05 ,
'RS503' : 5.92470734832e-06 ,
'RS508' : 8.55157344771e-05 ,
}
            
    referance_station = "CS002" ## only needed if using real delays, via the location on next line
    
    processed_data_folder = processed_data_dir(timeID)
    working_folder = processed_data_folder + "/pulse_finding"
    
    if not isdir(working_folder):
        mkdir(working_folder)
    
    plot_stations(timeID, 
                guess_delays = guess_station_delays,
                block_size = block_size,
                initial_block = initial_block, 
                max_num_stations = 10,
                num_blocks = 5,
                guess_location = guess_flash_location,
                bad_stations=["CS005", "CS021", "CS401", "CS031", "RS106"], 
                polarization_flips="polarization_flips.txt", 
                bad_antennas = "bad_antennas.txt", 
                additional_antenna_delays = "ant_delays.txt", 
                do_remove_saturation = True, 
                do_remove_RFI = True, 
                positive_saturation = 2046, 
                negative_saturation = -2047, 
                saturation_post_removal_length = 50, 
                saturation_half_hann_length = 5, 
                referance_station = "CS002",
                working_folder = working_folder,
                upsample_factor = 4) 
    
    
    