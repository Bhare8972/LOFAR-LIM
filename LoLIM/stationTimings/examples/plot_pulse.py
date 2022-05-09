#!/usr/bin/env python3

from run_Fitter4 import *
from LoLIM.stationTimings.timingInspector_4 import plot_all_stations, plot_station

#plot_all_stations(40,
#        
#        timeID = "D20180809T141413.250Z", 
#                        output_folder = "Callibration_1", 
#                        pulse_input_folders = ['pulse_finding'], 
#                        guess_timings = guess_timings, 
#                        sources_to_fit = known_sources, ## NOTE: that the index here is file_index + source_index*10 
#                        guess_source_locations = guess_source_locations,
#                   source_polarizations = known_polarizations, ## NOTE: 0 is even, 1 is odd, 2 is both
#                   source_stations_to_exclude = stations_to_exclude, 
#                   source_antennas_to_exclude = antennas_to_exclude, 
#                   bad_ants = bad_antennas,
#                   antennas_to_recalibrate = antennas_to_recallibrate,
#                   min_ant_amplitude = 10,
#                   ref_station = "CS002")

plot_station(110,
             'CS002',
             True,
        
        timeID = "D20180809T141413.250Z", 
                        output_folder = "Callibration_1", 
                        pulse_input_folders = ['pulse_finding'], 
                        guess_timings = guess_timings, 
                        sources_to_fit = known_sources, ## NOTE: that the index here is file_index + source_index*10 
                        guess_source_locations = guess_source_locations,
                   source_polarizations = known_polarizations, ## NOTE: 0 is even, 1 is odd, 2 is both
                   source_stations_to_exclude = stations_to_exclude, 
                   source_antennas_to_exclude = antennas_to_exclude, 
                   bad_ants = bad_antennas,
                   antennas_to_recalibrate = antennas_to_recallibrate,
                   min_ant_amplitude = 10,
                   ref_station = "CS002")
