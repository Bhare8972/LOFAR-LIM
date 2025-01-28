# #!/usr/bin/env python3

# """goal of this file is to provide functions to chnage between different calibration file formats"""

This file is kept for historical purpouses

# from LoLIM.IO import raw_tbb_IO as tbb
# import LoLIM.utilities as utils

# def OlafCal_to_LoLIMCal(cal_file_loc, out_file_loc, timeID):
#     """convert Olaf format to new total format"""
    
#     fpaths = tbb.filePaths_by_stationName( timeID )
    
#     data_files = {sname:tbb.MultiFile_Dal1(paths, force_metadata_ant_pos=True) for sname,paths in fpaths.items() }
#     known_tbb_timing_cals = {} ## used to cache results so don't have to re-extract every time
    
#     resulting_antenna_calibrations = {} ## a nested dictionary/ First key is station name. second key is antenna_name, value is extracted total delay
#     resulting_station_cal = {}

#     mode = 1 ## 1 means reading antenna delays, 2 means reading station delays
#     with open(cal_file_loc, 'r') as fin:
#         for line in fin:
#             data = line.split()
#             if data[0][0] == '=':
#                 mode = 2
#                 continue
            
#             if mode == 1:
#                 ant_name, calibration = data
                
#                 ## find station
#                 sname = utils.SId_to_Sname[ int(ant_name[:-3]) ]
#                 if sname not in data_files:
#                     # print('warning: station given, but has no data file:', sname)
#                     continue
                
#                 data_file = data_files[sname]
#                 if sname not in resulting_antenna_calibrations:
#                     resulting_antenna_calibrations[sname] = {}
#                     if not data_file.needs_metadata():
#                         known_tbb_timing_cals[sname] = data_file.get_timing_callibration_delays( force_file_delays=True )
#                 ant_names = data_file.get_antenna_names()
                
#                 ## find antenna
#                 last_three_digits = ant_name[-3:]
#                 found_ant_i = None
#                 for ant_i, ant_name_to_check in enumerate( ant_names ):
#                     if ant_name_to_check[-3:] == last_three_digits:
#                         found_ant_i = ant_i
#                         full_ant_name = ant_name_to_check
#                         break
#                 if found_ant_i is None:
#                     # print('cannot find antenna:', ant_name)
#                     # print( '  known:' )
#                     # print( ant_names )
#                     continue
                
                
                
#                 ## record cal value
#                 known_cal = 0
#                 if not data_file.needs_metadata():
#                     known_cal = known_tbb_timing_cals[sname][ found_ant_i ]
#                 cal_value = float(calibration)*5.0e-9 + known_cal
#                 resulting_antenna_calibrations[sname][full_ant_name] = cal_value
                
#             if mode == 2:
#                 sname, scal = data
#                 resulting_station_cal[sname] = float(scal)*5.0e-9
                
#     ### now we output
#     with open(out_file_loc, 'w') as fout:
#         fout.write('v1\n')
#         fout.write('# converted from Olaf Cal\n')
#         fout.write('# timeID ')
#         fout.write(timeID)
#         fout.write('\n')
        
#         ## record antenna delays
#         fout.write('antenna_delays\n')
#         for sdata in resulting_antenna_calibrations.values():
#             for ant_name, ant_cal in sdata.items():
#                 fout.write(ant_name)
#                 fout.write(' ')
#                 fout.write(str(ant_cal))
#                 fout.write('\n')
                
#         ## find 'bad' antennas
#         ## assume if not in olaf cal then they are bad
#         fout.write('bad_antennas\n')
#         for sname, sdata in data_files.items():
#             all_station_ants = sdata.get_antenna_names()
#             for ant_name in all_station_ants:
#                 found = False
#                 for olaf_included_ant_name in resulting_antenna_calibrations[sname].keys():
#                     if olaf_included_ant_name==ant_name:
#                         found = True
#                         break
#                 if not found:
#                     ## not in olaf data, thus is bad!
#                     ## this is not at ALL a good garuntee, as Olaf includes MANY antennas!
#                     fout.write(ant_name)
#                     fout.write('\n')
                    
#         ## assume no pol flips
        
#         ## station delays!!
#         center_delay = resulting_station_cal['CS002']
#         fout.write('station_delays\n')
#         for sname, delay in resulting_station_cal.items():
#             fout.write(sname)
#             fout.write(' ')
#             fout.write(str(-(delay-center_delay)))
#             fout.write('\n')
            
        
                
        
        
                
                
                
#                 