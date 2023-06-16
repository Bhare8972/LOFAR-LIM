#!/usr/bin/env python3

from os import mkdir
from os.path import isdir

from LoLIM.stationTimings.find_pulses_GUI import plot_stations
from LoLIM.utilities import processed_data_dir

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = "D20190424T194432.504Z"
    initial_block = 2506 #int(   1.1600369/((2**16)*(5.0E-9))  )#3600
    block_size = 2**16
    
    
    guess_station_delays = {
# 'CS002' : 0.0 ,
# 'CS001' : 2.226416856775675e-06 ,
# 'CS003' : 1.404712383412741e-06 ,
# 'CS004' : 4.3059884041135274e-07 ,
# 'CS005' : -2.1954797396747427e-07 ,
# 'CS006' : 4.3353708727264124e-07 ,
# 'CS007' : 4.0043261466513844e-07 ,
# 'CS011' : -5.859209592592854e-07 ,
# 'CS013' : -1.8136231233690343e-06 ,
# 'CS017' : -8.439147765446338e-06 ,
# 'CS021' : 9.243058763382305e-07 ,
# 'CS024' : 2.319278597444794e-06 ,
# 'CS026' : -9.234212450770167e-06 ,
# 'CS028' : -9.980370761864838e-06 ,
# 'CS030' : -2.7425177419647835e-06 ,
# 'CS032' : -1.5752455139193158e-06 ,
# 'CS103' : -2.8518911636790604e-05 ,
# 'CS301' : -7.211071085051455e-07 ,
# 'CS302' : -5.357115679322127e-06 ,
# 'CS401' : -9.528508187493772e-07 ,
# 'CS501' : -9.608613551216523e-06 ,
# 'RS106' : 3.3048264122311586e-05 ,
# 'RS205' : 6.984779609437443e-06 ,
# 'RS208' : 3.192288127977451e-05 ,
# 'RS306' : -2.6049586251454802e-05 ,
# 'RS307' : 6.9061424108347545e-06 ,
# 'RS310' : -3.221353235706229e-05 ,
# 'RS406' : -6.042696208711229e-06 ,
# 'RS407' : 1.7085706411113185e-05 ,
# 'RS409' : -5.9982518968538116e-05 ,
# 'RS503' : 6.952949659265937e-06 ,
# 'RS508' : 8.017457923389316e-05 ,
# 'RS210' : 0.00012349431726093842 ,
# 'RS305' : -1.6135095748652226e-05 ,
# 'CS031' : 0.0 ,
# 'CS101' : 0.0 ,
# 'RS509' : 0.00012422785137311543 ,
 
}
            
    referance_station = "CS002" ## only needed if using real delays, via the location on next line
    guess_flash_location = None ## use this if time delays are real and not apparent. Set to None when delays above are apparent delays, not real delays
    
    processed_data_folder = processed_data_dir(timeID)
    working_folder = processed_data_folder + "/pulse_finding_recal"
    
    if not isdir(working_folder):
        mkdir(working_folder)
    
    plot_stations(timeID, 
                guess_delays = guess_station_delays,
                block_size = block_size,
                initial_block = initial_block, 
                max_num_stations = 10,
                num_blocks = 5,
                guess_location = guess_flash_location,
                bad_stations=['CS201'], ## NOTE: if a station is not in this list, but also is not in guess_delays, it's guess delay is assumed to be 0
                total_cal_file = 'TotalCal_6Jan2023.txt',  ##NOTE: ignores station delay info in total cal.
                additional_antenna_delays = None, 
                do_remove_saturation = True, 
                do_remove_RFI = True, 
                positive_saturation = 2046, 
                negative_saturation = -2047, 
                saturation_post_removal_length = 50, 
                saturation_half_hann_length = 5, 
                referance_station = referance_station,
                working_folder = working_folder,
                upsample_factor = 4) 
    
    
## some brief instructions
#elif event.key == 'h':
#    print("how to use:")
#    print(" press 'w' and 'x' to change group of stations. Referance station always stays at bottom")
#    print(" press 'a' and 'd' to view later or earlier blocks")
#    print(" 'p' prints current delays")
#    print(" 'b' prints current block")
#    print(" 'o' prints current pulse info")
#    print(" hold middle mouse button and drag to translate view")
#    print(" j resets view (zoom and translate) back to default.")
#    print()
#    print(" there are multiple modes, that define what happens when main mouse button is dragged")
#    print("   '+' and '-' increase and decrease the pulse number that is being saved to.")
#    print("   'n' will set the pulse number to a new pulse.") ## TODO
#    print("   'z' and 'c' enter zoom-in and zoom-out modes")
#    print("   '1' and '2' shift stations right or left, and adjust timing accordingly. (they shift the station below the mouse, so draw a horizontal line).")
#    print("   '3' selects pulses for all stations that the box crosses")
#    print("   'e' erases pulses, for the station immediatly below the mouse")
#    print("   '0' is 'off', nothing happens if you drag mouse")
#    print()
#    print(" every selected pulse gets a red line, and an annotation on the referance station.")
#    print(" pulses from the other_folder get a black line")
#    print(" areas of the traces affected by saturation are covered by a red box")
#    print()
#    print(" in order to search for a particular pulse, enter 'input mode' by pressing 'i'")
#    print("    then type the index of the event you want to search for")
#    print("    press 'i' again to finish and center on the event")
#    print(" in order to jump to a different block, enter 'input mode' by pressing 'i'")
#    print("    then type the number of the block you want to jump to")
#    print("    then press 'j' to do the jump")
#    
#else:
#    print("pressed:", event.key, ". press 'h' for help")    
