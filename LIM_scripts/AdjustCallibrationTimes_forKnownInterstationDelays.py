#!/usr/bin/env python3
"""This file takes the callibrated station delays, and subtracts off (or adds depending on definition) the known time delay due to cables between stations"""

from LoLIM.utilities import processed_data_dir
from LoLIM.read_pulse_data import  read_station_delays
from LoLIM.IO.metadata import getClockCorrections

if __name__ == "__main__":
    timeID = "D20170929T202255.000Z"
    station_delay_file = "station_delays2.txt"
    
    sign = +1 ## substract or add delays
    
    zero_station = "CS002"
    
    processed_data_folder = processed_data_dir(timeID)
    station_timing_offsets = read_station_delays( processed_data_folder+'/'+station_delay_file )
    
    known_clock_corrections = getClockCorrections()
    
    zero_station_offset = station_timing_offsets[zero_station]
    if zero_station in known_clock_corrections:
        zero_station_offset += sign*known_clock_corrections[zero_station]
        
    for sname, offset in station_timing_offsets.items():
        addition = 0.0
        if sname in known_clock_corrections:
            addition = known_clock_corrections[sname]
        
        print(sname, "  ", offset + sign*addition - zero_station_offset)
        
        