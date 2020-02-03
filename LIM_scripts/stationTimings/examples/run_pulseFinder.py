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
    timeID = "D20180809T141413.250Z"
    initial_block = 4700 #int(   1.1600369/((2**16)*(5.0E-9))  )#3600
    block_size = 2**16
    
    
    guess_station_delays = {
'CS002' : 0.0,
'CS001' : 2.226416856775675e-06 , ## diff to guess: -4.0339220456486204e-10
'CS003' : 1.404712383412741e-06 , ## diff to guess: 2.0552045986067333e-10
'CS004' : 4.3059884041135274e-07 , ## diff to guess: -3.186722669881656e-11
'CS005' : -2.1954797396747427e-07 , ## diff to guess: -1.4122302840502756e-10
'CS006' : 4.3353708727264124e-07 , ## diff to guess: -1.1478028379716546e-10
'CS007' : 4.0043261466513844e-07 , ## diff to guess: 1.3637878299761718e-11
'CS011' : -5.859209592592854e-07 , ## diff to guess: -3.626114960740899e-10
'CS013' : -1.8136231233690343e-06 , ## diff to guess: 6.041022951968016e-10
'CS017' : -8.439147765446338e-06 , ## diff to guess: -2.674936549033755e-10
'CS021' : 9.243058763382305e-07 , ## diff to guess: 7.468842551128533e-10
'CS024' : 2.319278597444794e-06 , ## diff to guess: -1.2773834197680403e-09
'CS026' : -9.234212450770167e-06 , ## diff to guess: -5.293367864818893e-10
'CS028' : -9.980370761864838e-06 , ## diff to guess: 1.3065681625206295e-09
'CS030' : -2.7425177419647835e-06 , ## diff to guess: 1.4979455923512797e-09
'CS032' : -1.5752455139193158e-06 , ## diff to guess: 1.179183751818042e-10
'CS103' : -2.8518911636790604e-05 , ## diff to guess: -1.4511584556655777e-09
'CS201' : -1.0485460372186196e-05 , ## diff to guess: -1.06862147724999e-09
'CS301' : -7.211071085051455e-07 , ## diff to guess: -1.2649010193104883e-09
'CS302' : -5.357115679322127e-06 , ## diff to guess: -9.239451329301756e-10
'CS401' : -9.528508187493772e-07 , ## diff to guess: 6.108623726943449e-10
'CS501' : -9.608613551216523e-06 , ## diff to guess: 1.859615717897346e-09
'RS106' : 7.000488834504146e-06 , ## diff to guess: -1.5488953359982387e-08
'RS205' : 6.984779609437443e-06 , ## diff to guess: -1.1386767735308848e-08
'RS208' : 6.876943502917069e-06 , ## diff to guess: -6.698833678714051e-08
'RS306' : 7.011051614112542e-06 , ## diff to guess: 2.2214063295285274e-10
'RS307' : 6.9061424108347545e-06 , ## diff to guess: -1.7072411579076818e-08
'RS310' : 6.858130574870916e-06 , ## diff to guess: -5.3341260461447606e-08
'RS406' : 6.9811914351924905e-06 , ## diff to guess: 1.8154277731400394e-08
'RS407' : 7.067331300281373e-06 , ## diff to guess: 2.4376947371655664e-08
'RS409' : 6.8760465555923804e-06 , ## diff to guess: 9.972843231846224e-09
'RS503' : 6.952949659265937e-06 , ## diff to guess: 5.29706721931147e-09
'RS508' : 7.040440925442649e-06 , ## diff to guess: 2.6117420976061195e-08
'RS210' : 6.279328465360863e-06 , ## diff to guess: -4.7446115106133076e-07
'RS305' : 6.907167006083306e-06 , ## diff to guess: 1.35156747776032e-09  
}
            
    referance_station = "CS002" ## only needed if using real delays, via the location on next line
    guess_flash_location = [-29681 , -6075 , 4920] ## use this if time delays are real and not apparent. Set to None when delays above are apparent delays, not real delays
    
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
                bad_stations=['RS509'], ## NOTE: if a station is not in this list, but also is not in guess_delays, it's guess delay is assumed to be 0
                polarization_flips="polarization_flips.txt", 
                bad_antennas = "bad_antennas.txt", 
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
