#!/usr/bin/env python3

#### just plots a block of data on all antennas
#### usefull for health checks

import numpy as np
import matplotlib.pyplot as plt

from LoLIM.utilities import processed_data_dir, v_air
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.signal_processing import remove_saturation
from LoLIM.findRFI import window_and_filter
from LoLIM.read_pulse_data import  read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
from matplotlib.transforms import blended_transform_factory

from scipy.signal import hilbert



if __name__ == "__main__":
    timeID = "D20180813T153001.413Z"
    points = [ int(i*(2**16)) for i in range(3500,3515) ]
    block_size = 2**16
    
    polarization_flips = "polarization_flips.txt"
    bad_antennas = "bad_antennas.txt"
    additional_antenna_delays = "ant_delays.txt"
    
    bad_stations = ["CS401"]
    
    do_remove_saturation = True
    do_remove_RFI = True
    
    positive_saturation = 2046
    negative_saturation = -2047
    saturation_post_removal_length = 50
    saturation_half_hann_length = 50
    
    guess_station_delays = {
"CS002": 0.0,
            
            "RS208":  6.97951240511e-06 , ##??
            "RS210": 0.0, ##??
            
'RS305' :  7.1934989871e-06 , ## diff to guess: 1.37699440122e-09 ## station has issues
            
'CS001' :  2.24401172494e-06 , ## diff to guess: -1.97549250145e-12
'CS003' :  1.4070479021e-06 , ## diff to guess: -2.43902860006e-11
'CS004' :  4.40075444977e-07 , ## diff to guess: -2.42217738996e-11
'CS005' :  -2.17678756484e-07 , ## diff to guess: -2.21583592931e-11
'CS006' :  4.30647454293e-07 , ## diff to guess: -1.65542771957e-11
'CS007' :  3.99377465507e-07 , ## diff to guess: -2.02129216007e-11
'CS011' :  -5.8955385854e-07 , ## diff to guess: -1.23462674702e-12
'CS013' :  -1.81293364466e-06 , ## diff to guess: -2.14074126082e-11
'CS017' :  -8.4551582793e-06 , ## diff to guess: 4.31470974773e-11
'CS021' :  9.23663075135e-07 , ## diff to guess: 0.0
'CS024' :  2.33045304324e-06 , ## diff to guess: 8.82976926023e-12
'CS026' :  -9.26723147045e-06 , ## diff to guess: -9.26723147045e-06
'CS030' :  -2.73965173619e-06 , ## diff to guess: 2.41776414432e-11
'CS031' :  6.28836357272e-07 , ## diff to guess: 6.28836357272e-07
'CS032' :  -1.54223879178e-06 , ## diff to guess: -2.35580980099e-06
'CS101' :  -8.21535661203e-06 , ## diff to guess: 1.21635324342e-10
'CS103' :  -2.85878989548e-05 , ## diff to guess: 7.47105624697e-10
'CS201' :  -1.0510320435e-05 , ## diff to guess: 1.70364032269e-10
'CS301' :  -6.87236667019e-07 , ## diff to guess: -2.36495962731e-11
'CS302' :  -5.26832572532e-06 , ## diff to guess: 3.69871327386e-10
'CS501' :  -9.63031916796e-06 , ## diff to guess: -1.77630731779e-11
'RS106' :  6.78633242596e-06 , ## diff to guess: 1.575478568e-08
'RS205' :  7.09137015323e-06 , ## diff to guess: 9.84835165259e-10
'RS306' :  7.41317947913e-06 , ## diff to guess: 1.6935786672e-08
'RS307' :  7.80951637749e-06 , ## diff to guess: 5.67451737617e-08
'RS310' :  7.96312229684e-06 , ## diff to guess: 4.82651956563e-07
'RS406' :  6.96541956339e-06 , ## diff to guess: 8.9825482633e-09
'RS407' :  6.83033534647e-06 , ## diff to guess: 3.5607628373e-09
'RS409' :  7.4346353712e-06 , ## diff to guess: 1.55561380515e-07
'RS503' :  6.9238247372e-06 , ## diff to guess: 2.51105167164e-10
'RS508' :  6.36331650134e-06 , ## diff to guess: 2.40127126525e-09

            }
    
    referance_station = "CS002" ## only needed if using real delays, via the location on next line
    guess_flash_location = [ 2.43390916381 , -50825.1364969 , 0.0 , 1.56124908739 ] ## use this if time delays are real and not apparent. Set to None when delays above are apparent delays, not real delays
    
    
    processed_data_folder = processed_data_dir(timeID)
    
    polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
    additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
    
        
    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(raw_fpaths[sname], force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays) \
                      for sname in raw_fpaths.keys() if sname not in bad_stations}
    
    if guess_flash_location is not None:
        guess_flash_location = np.array(guess_flash_location)
        
        ref_stat_file = raw_data_files[ referance_station ]
        ant_loc = ref_stat_file.get_LOFAR_centered_positions()[0]
        ref_delay = np.linalg.norm(ant_loc-guess_flash_location[:3])/v_air - ref_stat_file.get_nominal_sample_number()*5.0E-9
        
        for sname, data_file in raw_data_files.items():
            if sname not in guess_station_delays:
                guess_station_delays[sname] = 0.0
            
            data_file = raw_data_files[sname]
            ant_loc = data_file.get_LOFAR_centered_positions()[0]
            guess_station_delays[sname] += (np.linalg.norm(ant_loc-guess_flash_location[:3])/v_air - ref_delay) 
            guess_station_delays[sname] -= data_file.get_nominal_sample_number()*5.0E-9
    
    RFI_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in raw_fpaths.keys() if sname not in bad_stations}
    
    data = np.empty(block_size, dtype=np.double)
    
    height = 0
    t0 = np.arange(block_size)*5.0E-9
    transform = blended_transform_factory(plt.gca().transAxes, plt.gca().transData)
    sname_X_loc = 0.0
    for sname, data_file in raw_data_files.items():
        print(sname)
        
        station_delay = 0.0
        if sname in guess_station_delays:
            station_delay = guess_station_delays[sname] 
        
        station_delay_points = int(station_delay/5.0E-9)
            
        RFI_filter = RFI_filters[sname]
        ant_names = data_file.get_antenna_names()
        
        num_antenna_pairs = int( len( ant_names )/2 )
        peak_height = 0.0
        for point in points:
            T = t0 + point*5.0E-9
            for pair in range(num_antenna_pairs):
                data[:] = data_file.get_data(point+station_delay_points, block_size, antenna_index=pair*2)
                
                if do_remove_saturation:
                    remove_saturation(data, positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length)
                if do_remove_RFI:
                    filtered_data = RFI_filter.filter( data )
                else:
                    filtered_data = hilbert(data)
                even_HE = np.abs(filtered_data)
                
                data[:] = data_file.get_data(point+station_delay_points, block_size, antenna_index=pair*2+1)
                if do_remove_saturation:
                    remove_saturation(data, positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length)
                if do_remove_RFI:
                    filtered_data = RFI_filter.filter( data )
                else:
                    filtered_data = hilbert(data)
                odd_HE = np.abs(filtered_data)
                
                plt.plot(T, even_HE + height, 'r')
                plt.plot(T, odd_HE + height, 'g' )
                
                max_even = np.max(even_HE)
                if max_even > peak_height:
                    peak_height = max_even
                max_odd = np.max(odd_HE)
                if max_odd > peak_height:
                    peak_height = max_odd
                    
#        plt.annotate(sname, (points[-1]*5.0E-9+t0[-1], height))
        plt.annotate(sname, (sname_X_loc, height), textcoords=transform, xycoords=transform)
        height += 2*peak_height
            
    plt.show()