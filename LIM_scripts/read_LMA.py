#!/usr/bin/env python3
import os
import numpy as np
import gzip

from LoLIM.IO.metadata import geoditic_to_ITRF, convertITRFToLocal

""" this is a set of code designed to read LMA data """

class LMA_header:
    """LMA header data. Doesn't do much yet"""
    
    class ant_info:
        def __init__(self, line):
            data = line.split()
            self.id = data[1]
            self.name = data[2]
            self.lat = float(data[3])
            self.lon = float(data[4])
            self.alt = float(data[5])
            self.antenna_delay = float(data[6])
            self.board_rev = int(data[7])
            self.rec_ch = int(data[8])
    
    def __init__(self, fin):
        self.file_name = fin.name
        
        self.antenna_info_list = [] ##NOTE: order is critical!
        
        for line in fin:
            line = line.decode()
            
            if len(line)>17 and line[:17]=='Number of events:':
                self.number_events = int( line[17:] )
                return
            elif len(line)>9 and line[:9]=="Sta_info:":
                new_antenna = LMA_header.ant_info( line )
                self.antenna_info_list.append( new_antenna )
                
    def read_aux_file(self, fname=None):
        
        class aux_data:
            def __init__(self, peak_times, raw_powers, above_thresholds, upper_covariance_tri):
                self.peak_times = peak_times
                self.raw_powers = raw_powers
                self.above_thresholds = above_thresholds
                self.upper_covariance_tri = upper_covariance_tri
        
        if fname is not None:
            new_fname = fname
        else:
            new_fname = self.file_name.replace('dat', 'aux')
        
        if new_fname[-3:] =='.gz':
            func = gzip.open
            symbol = 'rb'
        else:
            func = open
            symbol = 'r'
            
        return_data = []
            
        with func(new_fname, symbol) as file:
            
            for line in file:
                line = line.decode()
                
                if len(line)>=12 and line[:12] == "*** data ***":
                    break
                
            for source_i in range(self.number_events):
                peak_times = file.readline().decode().split()
                raw_powers = file.readline().decode().split()
                above_threshold = file.readline().decode().split()
                upper_tri_covariance = file.readline().decode().split()
                
                N = len(peak_times)
                if N != len(self.antenna_info_list):
                    print("ERROR A")
                    quit()
                if N != len(raw_powers):
                    print("ERROR B")
                    quit()
                if N != len(above_threshold):
                    print("ERROR C")
                    quit()
                if len(upper_tri_covariance) != 10:
                    print("ERROR D")
                    quit()
                    
                peak_times = np.array([float(d) for d in peak_times])
                raw_powers = np.array([int(d) for d in raw_powers])
                above_threshold = np.array([int(d) for d in above_threshold])
                upper_tri_covariance = np.array([float(d) for d in upper_tri_covariance])
                    
                new_data = aux_data(peak_times, raw_powers, above_threshold, upper_tri_covariance)
                return_data.append( new_data )
                
        return return_data
                    
            
            
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
        
def LMA_fname_info(fname):
    info = fname.split('_')
    array_name = info[0]
    date = info[1]
    time = info[2]
    end_stuff = info[3].split('.')
    seconds_processed = end_stuff[0]
    is_gzip = end_stuff[-1]=='gz'
    return array_name, date, time, int(seconds_processed), is_gzip
    
def read_LMA_file_data(fname):
    source_list = []
    
    is_gzip = fname[-3:] =='.gz'
    if is_gzip:
        func = gzip.open
        symbol = 'rb'
    else:
        func = open
        symbol = 'r'
    
    status = 0 ## 0 means read header, 1 means read source
    with func(fname, symbol) as file:
        
        header = LMA_header(file)
        
        for line in file:
            line = line.decode()
            
            if len(line)>=12 and line[:12] == "*** data ***":
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
                
#                if new_LMA_source.red_chi_squared < 1:
                source_list.append(new_LMA_source)
                
    return header, source_list

def read_LMA_multiple_files(LMA_files):
    
    ret = []
    for file in LMA_files:
        header, sources = read_LMA_file_data(file)
        print(file, len(sources))
        ret += sources
        
    return ret
            

def read_LMA_folder_data(folder, date=None, min_time=None, max_time=None):
    LMA_fnames = (fname for fname in os. listdir(folder) if fname[-4:]==".dat" or fname[-3:]=='.gz')
    
    if (date is not None):
        new_LMA_fnames = []
        for fname in LMA_fnames:
            throw, LMA_date, time, throw, throw = LMA_fname_info(fname)
            if date==LMA_date and (min_time is None or min_time<=time) and (max_time is None or time<max_time):
                new_LMA_fnames.append( fname )
        LMA_fnames = new_LMA_fnames
            
    if folder[-1] != '/':
        folder = folder + '/'
    LMA_fnames = [folder + fname for fname in LMA_fnames]
      
    print(LMA_fnames)
    
    return read_LMA_multiple_files(LMA_fnames)




