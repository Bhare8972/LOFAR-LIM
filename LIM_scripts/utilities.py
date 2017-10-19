#!/usr/bin/env python3

##ON APP MACHINE

import sys

from os import listdir, mkdir
from os.path import isdir

import weakref

from scipy import fftpack
import numpy as np

default_raw_data_loc = "/home/brian/raw_data"
default_processed_data_loc = "/home/brian/processed_files"

#### constants
C = 299792458.0
RTD = 180.0/3.1415926 ##radians to degrees
n_air = 1.000293
v_air = C/n_air

#### log data to screen and to a file

class logger(object):
    class std_writer(object):
        def __init__(self, logger):
            self.logger_ref = weakref.ref(logger)
            
        def write(self, msg):
            logger=self.logger_ref()
            logger.out_file.write(msg)
            if logger.to_screen:
                logger.old_stdout.write(msg)
            
        def flush(self):
            logger=self.logger_ref()
            logger.out_file.flush()
    
    
    def __init__(self):
        self.set("out_log")
        
    def set(self, fname, to_screen=True):
        self.out_file = open(fname, 'w')
        self.has_stderr = False
        self.has_stdout = False
        
        self.old_stderr = sys.stderr
        self.old_stdout = sys.stdout
        
        self.to_screen = to_screen
        
        
    def __call__(self, *args):
        for a in args:
            if self.to_screen:
                self.old_stdout.write(str(a))
                self.old_stdout.write(" ")
                
            self.out_file.write(str(a))
            self.out_file.write(" ")
            
        self.out_file.write("\n")
        if self.to_screen:
            self.old_stdout.write("\n")
            
        self.out_file.flush()
        self.old_stdout.flush()
        
    def set_to_screen(self, to_screen=True):
        self.to_screen = to_screen
        
    def take_stdout(self):
        
        if not self.has_stdout:
            sys.stdout = self.std_writer(self)
            self.has_stdout = True
							
    def take_stderr(self):
        
        if not self.has_stderr:
            sys.stderr = self.std_writer(self)
            self.has_stderr = True
            
    def restore_stdout(self):
        if self.has_stdout:
            sys.stdout = self.old_stdout
            self.has_stdout = False
            
    def restore_stderr(self):
        if self.has_stderr:
            sys.stderr = self.old_stderr
            self.has_stderr = False
            
    def flush(self):
        self.out_file.flush()
            
    def __del__(self):
        self.restore_stderr()
        self.restore_stdout()
log = logger()
        
def iterate_pairs(list_one, list_two, list_one_avoid=[], list_two_avoid=[]):
    """returns an iterator that loops over all pairs of the two lists"""
    for item_one in list_one:
        if item_one in list_one_avoid:
            continue
        for item_two in list_two:
            if item_two in list_two_avoid:
                continue
            yield (item_one, item_two)
        
        
        
#### some file utils

def Fname_data(Fpath):
    """ takes both pulse data file names and h5 file names and returns UTC_time, station_name, Fpath"""
    Fname = Fpath.split('/')[-1]
    data = Fname.split('_')
    timeID = data[1]
    station_name = data[2]
    
    file_number = 0 ## FIX THIS!
    
    return timeID, station_name, Fpath, file_number


##note that timeID is a string representing the datetime of a LOFAR trigger. such as:   D20130619T094846.507Z
## the timeID is used to uniquely identify triggers

def get_timeID(fname):
    data=fname.split("_")
    return data[1]

def year_from_timeID(timeID):
    return timeID[1:5]

def raw_data_dir(timeID, data_loc=None):
    """gives path to the raw data folder for a particular timeID, given location of data structure. Defaults to "/home/brian/raw_data/" """
    
    if data_loc is None:
        data_loc = default_raw_data_loc
    
    path = data_loc + '/' + year_from_timeID(timeID)+"/"+timeID
    if not isdir(path):
        mkdir(path)
    return path

def processed_data_dir(timeID, data_loc=None):
    """gives path to the analysis folders for a particular timeID, given location of data structure. Defaults to "/home/brian/processed_files/" """
    
    if data_loc is None:
        data_loc = default_processed_data_loc
    
    path=data_loc + "/" + year_from_timeID(timeID)+"/"+timeID
    if not isdir(path):
        mkdir(path)
    return path


#### data processing functions ####

def upsample_N_envelope(timeseries_data, upsample_factor):
    
    x = timeseries_data
    Nx = len(x)
    num = int( Nx*upsample_factor )
    
    X = fftpack.fft(x)

    sl = [slice(None)]
    newshape = list(x.shape)
    newshape[-1] = num
    Y = np.zeros(newshape, 'D')
    sl[-1] = slice(0, (Nx + 1) // 2)
    Y[sl] = X[sl]
    sl[-1] = slice(-(Nx - 1) // 2, None) ##probably don't need this line and next lint
    Y[sl] = X[sl]

    if Nx % 2 == 0:  # special treatment if low number of points is even. So far we have set Y[-N/2]=X[-N/2]
        # upsampling
        sl[-1] = slice(num-Nx//2,num-Nx//2+1,None)  # select the component at frequency -Nx/2
        Y[sl] /= 2  # halve the component at -N/2
        temp = Y[sl]
        sl[-1] = slice(Nx//2,Nx//2+1,None)  # select the component at +Nx/2
        Y[sl] = temp  # set that equal to the component at -Nx/2
        
    h = np.zeros(num)
    if num % 2 == 0:
        h[0] = h[num // 2] = 1
        h[1:num // 2] = 2
    else:
        h[0] = 1
        h[1:(num + 1) // 2] = 2

    y = fftpack.ifft(Y*h, axis=-1) * (float(num) / float(Nx))

    new_data = y.real**2
    new_data += y.imag**2

    return np.sqrt(y.real**2 + y.imag**2)









class parabolic_fit:
    def __init__(self, data, index, n_points):
        
        self.index = index
        self.n_points = n_points
        
        self.matrix, self.error_ratio_A, self.error_ratio_B, self.error_ratio_C = self.get_fitting_matrix(n_points)
        
        half_n_points = int( (n_points-1)/2 )
        points = data[ index-half_n_points :  index+1+half_n_points ]
        
        self.A, self.B, self.C = np.dot( self.matrix, points )
        
        self.peak_relative_index = -self.B/(2.0*self.A)
        self.peak_index = self.peak_relative_index + (index-half_n_points)
        
        self.peak_index_error_ratio = self.peak_relative_index*np.sqrt( (self.error_ratio_A/self.A)**2 + (self.error_ratio_B/self.B)**2 )
        
        X = np.arange(n_points)
        residuals = (self.A*(X*X) + self.B*X * C) - points
        self.RMS_fit = np.sqrt( np.sum(residuals*residuals)/n_points )
        
        self.amplitude = self.C - (self.B**2)/(4.0*self.A)
         
    def get_fitting_matrix(self, n_points):
        if "fit_matrix" not in parabolic_fit.__dict__:
            parabolic_fit.fit_matrix = {}
            
        if n_points not in parabolic_fit.fit_matrix:
        
            tmp_matrix = np.zeros((n_points,3), dtype=np.double)
            for n_i in range(n_points):
                tmp_matrix[n_i, 0] = n_i**2
                tmp_matrix[n_i, 1] = n_i
                tmp_matrix[n_i, 2] = 1.0
            peak_time_matrix = np.linalg.pinv( tmp_matrix ) #### this matrix, when multilied by vector of data points, will give a parabolic fit(A,B,C): A*x*x + B*x + C (with x in units of index)
                           
            hessian = np.zeros((3,3))
            for n_i in range(n_points):
                hessian[0,0] += n_i**4
                hessian[0,1] += n_i**3
                hessian[0,2] += n_i**2
                
                hessian[1,1] += n_i**2
                hessian[1,2] += n_i
                
                hessian[2,2] +=1
                        
            hessian[1,0] = hessian[0,1]
            hessian[2,0] = hessian[0,2]
            hessian[2,1] = hessian[1,2]
                
            inverse = np.linalg.inv(hessian)
            #### these error ratios, for each value in the parabolic fit, give teh std of the error in that paramter when mupltipled by the std of the noise of the data ####
            error_ratio_A = np.sqrt(inverse[0,0])
            error_ratio_B = np.sqrt(inverse[1,1])
            error_ratio_C = np.sqrt(inverse[2,2])
    
            parabolic_fit.fit_matrix[n_points] = [peak_time_matrix, error_ratio_A, error_ratio_B, error_ratio_C]
        
        
        return parabolic_fit.fit_matrix[n_points]
    
    
## a dictionary where the keys are the number of a station and the values are the station name
SNum_to_SName_dict = {
        '001':"CS001",
        '002':"CS002",
        '004':"CS004",
        "006":"CS006",
        "011":"CS011",
        "013":"CS013",
        "021":"CS021",
        "026":"CS026",
        "028":"CS028",
        "030":"CS030",
        "031":"CS031",
        "032":"CS032",
        "142":"CS302",
        "106":"RS106",
        "125":"RS205",
        "128":"RS208",
        "145":"RS305",
        "146":"RS306",
        "147":"RS307",
        "166":"RS406",
        "167":"RS407",
        "183":"RS503",
        "188":"RS508",
        "189":"RS509"
        } 




















    
    
    
    
    
    
    
