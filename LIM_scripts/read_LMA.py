#!/usr/bin/env python3
import os
import numpy as np

from LoLIM.IO.metadata import geoditic_to_ITRF, convertITRFToLocal

""" this is a set of code designed to read LMA data """

class LMA_header:
    """LMA header data. Doesn't do much yet"""
    def __init__(self):
        pass

class LMA_source:
    def __init__(self, header):
        self.header = header
        
        self.time_of_day = None
        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.red_chi_squared = None
        self.power = None
        self.mask = None
        self.local_XYZ = None
        
    def get_XYZ(self):
        
        if self.local_XYZ is None:
            ITRF = geoditic_to_ITRF( [self.latitude, self.longitude, self.altitude] )
            self.local_XYZ = convertITRFToLocal( np.array( [ITRF] ) )[0]
        
        
        return self.local_XYZ
        

def read_LMA_file_data(fname):
    header = LMA_header()
    source_list = []
    
    status = 0 ## 0 means read header, 1 means read source
    with open(fname, 'r') as file:
        for line_number, line in enumerate(file):
            if line[:12] == "*** data ***":
                status = 1
            elif status==0:
                pass ## nothing here yet
            elif status==1:
                new_LMA_source = LMA_source(header)
                
                time, lat, lon, alt, fit, power, mask = line.split()
                new_LMA_source.time_of_day = float(time)
                new_LMA_source.latitude = float(lat)
                new_LMA_source.longitude = float(lon)
                new_LMA_source.altitude = float(alt)
                new_LMA_source.red_chi_squared = float(fit)
                new_LMA_source.power = float(power)
                new_LMA_source.mask = mask
                
                source_list.append(new_LMA_source)
                
    return header, source_list
            

def read_LMA_folder_data(folder):
    LMA_files = [fname for fname in os. listdir(folder) if fname[-4:]==".dat"]
    
    ret = []
    for file in LMA_files:
        header, sources = read_LMA_file_data(folder+'/'+file)
        ret += sources
        
    return ret
    