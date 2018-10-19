#!/usr/bin/env python3

""" This file takes the antenna delays (from raw_tbb_IO.py), and saves them to a series of text files. This is an anachronism, in order to support some older code"""

from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.utilities import processed_data_dir

if __name__=="__main__":
    timeID = "D20180809T141413.250Z"
    output_folder = "cal_tables/TxtAntDelay" ## we assume this exists 
    
    stations_to_skip = []
    
    
    output_dir = processed_data_dir( timeID )
    raw_fpaths = filePaths_by_stationName(timeID)
    
    for station, fpaths in raw_fpaths.items():
        if station in stations_to_skip:
            continue
        
        print("processing station", station)
        
        infile = MultiFile_Dal1(raw_fpaths[station])
        antenna_names = infile.get_antenna_names()
        ant_cal_delays = infile.get_timing_callibration_delays()
        
        with open(output_dir+'/'+output_folder+'/'+station+'.txt', 'w') as fout:
            for ant_name, delay in zip(antenna_names, ant_cal_delays):
                name = str( ant_name[-3:] )
                fout.write(name+' '+str(delay)+'\n')