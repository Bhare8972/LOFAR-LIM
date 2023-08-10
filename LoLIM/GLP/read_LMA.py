#!/usr/bin/env python3
import os
import numpy as np
import gzip


import datetime

""" this is a set of code designed to read LMA data """


### some initial functions and info
ITRFCS002 = np.array([3826577.06611,   461022.947639,   5064892.786   ]) ## CS002 location in ITRF coordinates
latlonCS002 = np.array([52.91512249, 6.869837540]) ## lattitude and longitude of CS002 in degrees


def geoditic_to_ITRF(latLonAlt):
    """for a latLonAlt in degrees, (can be list of three numpy arrays), convert to ITRF coordinates. Using information at: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Geodetic_to/from_ENU_coordinates and 
    http://itrf.ensg.ign.fr/faq.php?type=answer"""
    
    a = 6378137.0  #### semi-major axis, m
    e2 = 0.00669438002290 ## eccentricity squared
    
    def N(lat):
        return a/np.sqrt( 1 - e2*(np.sin(lat)**2) )
    
    lat = latLonAlt[0]/RTD
    lon = latLonAlt[1]/RTD
    
    X = ( N(lat) + latLonAlt[2] ) *np.cos(lat) *np.cos(lon)
    Y = ( N(lat) + latLonAlt[2] ) *np.cos(lat) *np.sin(lon)

    b2_a2 =  1-e2
    Z = ( b2_a2*N(lat) + latLonAlt[2] ) *np.sin(lat)
    
    return np.array( [X,Y,Z] )

def convertITRFToLocal(itrfpos, phase_center=ITRFCS002, reflatlon=latlonCS002, out=None):
    """
    ================== ==============================================
    Argument           Description
    ================== ==============================================
    *itrfpos*          an ITRF position as 1D numpy array, or list of positions as a 2D array
    *phase_center*     the origin of the coordinate system, in ITRF. Default is CS002.
    *reflatlon*        the rotation of the coordinate system. Is the [lat, lon] (in degrees) on the Earth which defines "UP"
    
    Function returns a 2D numpy array (even if input is 1D).
    Out cannot be same array as itrfpos
    """
    if out is itrfpos:
        print("out cannot be same as itrfpos in convertITRFToLocal. TODO: make this a real error")
        quit()
    
    lat = reflatlon[0]/RTD
    lon = reflatlon[1]/RTD
    arg0 = np.array([-np.sin(lon),   -np.sin(lat) * np.cos(lon),   np.cos(lat) * np.cos(lon)])
    arg1 = np.array([np.cos(lon) ,   -np.sin(lat) * np.sin(lon),   np.cos(lat) * np.sin(lon)])
    arg2 = np.array([         0.0,    np.cos(lat),                 np.sin(lat)])
    
    if out is None:
        ret = np.empty(itrfpos.shape, dtype=np.double )
    else:
        ret = out
    
    ret[:]  = np.outer(itrfpos[...,0]-phase_center[0], arg0 )
    ret += np.outer(itrfpos[...,1]-phase_center[1], arg1 )
    ret += np.outer(itrfpos[...,2]-phase_center[2], arg2 )
    
    return ret
    



### following two classes are not used directly ###

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
            
        def add_station_data(self, line):
            data = line.split()
            if data[1] != self.id:
                print("PROBLEM IN LMA HEADER 1")
                quit()
                
            if len(data) != 10:
                print("PROBLEM IN LMA HEADER 2")
                quit()
                
            self.window = int( data[3] )
            self.data_ver = int( data[4] )
            self.RMS_error = float( data[5] )
            self.sources = int( data[6] )
            self.percent = float( data[7] )
            self.power_fraction = float( data[8] )
            self.active = data[9]
            
        def get_XYZ(self, center='LOFAR'):
            ## TODO: use python package instead.
            
            ITRF = geoditic_to_ITRF( [self.lat, self.lon, self.alt] )
            
            if center=='LOFAR':
                phase_center = ITRFCS002
                reflatlon = latlonCS002
            elif center=="LMA":
                array_latlonalt = np.array([ self.array_lat, self.array_lon, self.array_alt ])
                reflatlon = array_latlonalt[:2]
                phase_center = geoditic_to_ITRF( array_latlonalt )
            else:
                ## assume center is lat lon alt
                array_latlonalt = np.array( center )
                reflatlon = array_latlonalt[:2]
                phase_center = geoditic_to_ITRF( array_latlonalt )
            
            return convertITRFToLocal( np.array( [ITRF] ), phase_center, reflatlon )[0]
                
    def __init__(self, fin, decode_func):
        self.file_name = fin.name
        
        self.antenna_info_list = [] ##NOTE: order is critical!
        self.num_stations = 0
        
        sta_data_index = 0
         
        for line in fin:
            line = decode_func(line)
            
            if len(line)>17 and line[:17]=='Number of events:':
                self.number_events = int( line[17:] )
                return
            
            elif len(line)>17 and line[:16] == "Data start time:":
                line_data = line.split()
                self.start_date = line_data[-2]
                self.start_time = line_data[-1]
            
            elif len(line)>9 and line[:9]=="Sta_info:":
                new_antenna = LMA_header.ant_info( line )
                self.antenna_info_list.append( new_antenna )
                self.num_stations += 1
                
            elif len(line)>17 and line[:17] == "Coordinate center":
                lat,lon,alt = line.split()[-3:]
                self.array_lat = float(lat)
                self.array_lon = float(lon)
                self.array_alt = float(alt)
                
            elif len(line)>9 and line[:9]=="Sta_data:":
                self.antenna_info_list[ sta_data_index ].add_station_data( line )
                sta_data_index += 1
                
    def midnight_datetime(self):
        month,day,year = self.start_date.split('/')
        date = datetime.date( day=int(day), month=int(month), year=int(year) )
        
        return datetime.datetime.combine(date,  datetime.time(0), tzinfo= datetime.timezone.utc )
                
    def read_aux_file(self, fname=None):
        
        class aux_data:
            def __init__(self, peak_times, raw_powers, above_thresholds, upper_covariance_tri):
                self.peak_times = peak_times
                self.raw_powers = raw_powers
                self.above_thresholds = above_thresholds
                self.upper_covariance_tri = upper_covariance_tri
                
            def get_covariance_matrix(self):
                covariance_matrix = np.empty( (4,4) )
                covariance_matrix[0,0] = self.upper_covariance_tri[0]
                covariance_matrix[0,1] = self.upper_covariance_tri[1]
                covariance_matrix[0,2] = self.upper_covariance_tri[2]
                covariance_matrix[0,3] = self.upper_covariance_tri[3]
                
                covariance_matrix[1,1] = self.upper_covariance_tri[4]
                covariance_matrix[1,2] = self.upper_covariance_tri[5]
                covariance_matrix[1,3] = self.upper_covariance_tri[6]
                
                covariance_matrix[2,2] = self.upper_covariance_tri[7]
                covariance_matrix[2,3] = self.upper_covariance_tri[8]
                
                covariance_matrix[3,3] = self.upper_covariance_tri[9]
                
                covariance_matrix[1,0] = covariance_matrix[0,1]
                covariance_matrix[2,0] = covariance_matrix[0,2]
                covariance_matrix[3,0] = covariance_matrix[0,3] 
                covariance_matrix[2,1] = covariance_matrix[1,2]
                covariance_matrix[3,1] = covariance_matrix[1,3]
                covariance_matrix[3,2] = covariance_matrix[2,3]
                
                return covariance_matrix
        
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
        
    def get_XYZ(self, center='LOFAR'):
        
        ## TODO: use python package instead.
        
        if self.local_XYZ is None:
            ITRF = geoditic_to_ITRF( [self.latitude, self.longitude, self.altitude] )
            
            if center=='LOFAR':
                phase_center = ITRFCS002
                reflatlon = latlonCS002
            elif center=="LMA":
                array_latlonalt = np.array([ self.header.array_lat, self.header.array_lon, self.header.array_alt ])
                reflatlon = array_latlonalt[:2]
                phase_center = geoditic_to_ITRF( array_latlonalt )
            else:
                ## assume center is lat lon alt
                array_latlonalt = np.array( center )
                reflatlon = array_latlonalt[:2]
                phase_center = geoditic_to_ITRF( array_latlonalt )
            
            self.local_XYZ = convertITRFToLocal( np.array( [ITRF] ), phase_center, reflatlon )[0]
        
        
        return self.local_XYZ
    
    def in_XYZ_bounds(self, bounds, center='LOFAR'):
        XYZ = self.get_XYZ(center)
        in_X = ( bounds[0][0] <= XYZ[0] <= bounds[0][1] )
        in_Y = ( bounds[1][0] <= XYZ[1] <= bounds[1][1] )
        in_Z = ( bounds[2][0] <= XYZ[2] <= bounds[2][1] )
        return in_X and in_Y and in_Z
        
    
    def get_number_stations(self):
        mask_str = '{0:0'+str(self.header.num_stations)+'d}'
        mask2 = mask_str.format(int(bin(int(self.mask, 16))[2:]))
        self.num_stations = mask2.count('1')
        return self.num_stations
    
    def time_as_datetime(self, midnight_datetime = None):
        """returns datetime, accurate to microsecond, and excess time beyond that (in units of seconds)"""
        TD = datetime.timedelta(seconds = self.time_of_day)
        excess = self.time_of_day - TD.total_seconds() 
        
        if midnight_datetime is None:
            midnight_datetime = self.header.midnight_datetime()
        
        return midnight_datetime + TD, excess
        


### These functions are used by end-user ###
        
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
    """This is the default function to use. Give it an LMA file name, it will open it and return a header and list of sources"""

    source_list = []
    
    is_gzip = fname[-3:] =='.gz'
    if is_gzip:
        func = gzip.open
        symbol = 'rb'
        decode_func = lambda X: X.decode()
    else:
        func = open
        symbol = 'r'
        decode_func = lambda X: X
    
    status = 0 ## 0 means read header, 1 means read source
    with func(fname, symbol) as file:
        
        header = LMA_header(file, decode_func)
        
        for line in file:
            line = decode_func( line )
            
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




