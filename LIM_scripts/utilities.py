#!/usr/bin/env python3

##ON APP MACHINE

import sys

from os import listdir, mkdir
from os.path import isdir, dirname, abspath

import weakref

from scipy import fftpack
import numpy as np

default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
default_processed_data_loc = "/home/brian/processed_files"

MetaData_directory =  dirname(abspath(__file__)) + '/data' ## change this if antenna_response_model is in a folder different from this module

#### constants
C = 299792458.0
RTD = 180.0/3.1415926 ##radians to degrees
n_air = 1.000293
v_air = C/n_air

latlonCS002 = np.array([52.91512249, 6.869837540]) ## lattitude and longitude of CS002 in degrees

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
    
    if len(data[3][1:])==0:
        file_number = 0
    else:
        file_number = int(data[3][1:])
    
    return timeID, station_name, Fpath, file_number


##note that timeID is a string representing the datetime of a LOFAR trigger. such as:   D20130619T094846.507Z
## the timeID is used to uniquely identify triggers

def get_timeID(fname):
    data=fname.split("_")
    return data[1]

def year_from_timeID(timeID):
    return timeID[1:5]

def raw_data_dir(timeID, data_loc=None):
    """gives path to the raw data folder for a particular timeID, given location of data structure. Defaults to  default_raw_data_loc"""
    
    if data_loc is None:
        data_loc = default_raw_data_loc
    
    path = data_loc + '/' + year_from_timeID(timeID)+"/"+timeID
    return path

def processed_data_dir(timeID, data_loc=None):
    """gives path to the analysis folders for a particular timeID, given location of data structure. Defaults to  default_processed_data_loc
    makes the directory if it doesn't exist"""
    
    if data_loc is None:
        data_loc = default_processed_data_loc
    
    path=data_loc + "/" + year_from_timeID(timeID)+"/"+timeID
    if not isdir(path):
        mkdir(path)
    return path


#### data processing functions ####
### TODO: move these to signal_processing

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
    
## a python list where the keys are the number of a station and the values are the station name
SId_to_Sname = [None]*209 #just to pre-initilize list, so syntax below is possible
SId_to_Sname[1] = "CS001"
SId_to_Sname[2] = "CS002"
SId_to_Sname[3] = "CS003"
SId_to_Sname[4] = "CS004"
SId_to_Sname[5] = "CS005"
SId_to_Sname[6] = "CS006"
SId_to_Sname[7] = "CS007"
#SId_to_Sname[8] = "CS008"
#SId_to_Sname[9] = "CS009"
#SId_to_Sname[10] = "CS010"
SId_to_Sname[11] = "CS011"
#SId_to_Sname[12] = "CS012"
SId_to_Sname[13] = "CS013"
#SId_to_Sname[14] = "CS014"
#SId_to_Sname[15] = "CS015"
#SId_to_Sname[16] = "CS016"
SId_to_Sname[17] = "CS017"
#SId_to_Sname[18] = "CS018"
#SId_to_Sname[19] = "CS019"
#SId_to_Sname[20] = "CS020"
SId_to_Sname[21] = "CS021"
#SId_to_Sname[22] = "CS022"
#SId_to_Sname[23] = "CS023"
SId_to_Sname[24] = "CS024"
#SId_to_Sname[25] = "CS025"
SId_to_Sname[26] = "CS026"
#SId_to_Sname[27] = "CS027"
SId_to_Sname[28] = "CS028"
#SId_to_Sname[29] = "CS029"
SId_to_Sname[30] = "CS030"
SId_to_Sname[31] = "CS031"
SId_to_Sname[32] = "CS032"
SId_to_Sname[101] = "CS101"
#SId_to_Sname[102] = "CS102"
SId_to_Sname[103] = "CS103"
SId_to_Sname[121] = "CS201"
SId_to_Sname[141] = "CS301"
SId_to_Sname[142] = "CS302"
SId_to_Sname[161] = "CS401"
SId_to_Sname[181] = "CS501"

#SId_to_Sname[104] = "RS104"
#SId_to_Sname[105] = "RS105"
SId_to_Sname[106] = "RS106"
#SId_to_Sname[107] = "RS107"
#SId_to_Sname[108] = "RS108"
#SId_to_Sname[109] = "RS109"
#SId_to_Sname[122] = "RS202"
#SId_to_Sname[123] = "RS203"
#SId_to_Sname[124] = "RS204"
SId_to_Sname[125] = "RS205"
#SId_to_Sname[126] = "RS206"
#SId_to_Sname[127] = "RS207"
SId_to_Sname[128] = "RS208"
#SId_to_Sname[129] = "RS209"
SId_to_Sname[130] = "RS210"
#SId_to_Sname[143] = "RS303"
#SId_to_Sname[144] = "RS304"
SId_to_Sname[145] = "RS305"
SId_to_Sname[146] = "RS306"
SId_to_Sname[147] = "RS307"
#SId_to_Sname[148] = "RS308"
#SId_to_Sname[149] = "RS309"
SId_to_Sname[150] = "RS310"
SId_to_Sname[166] = "RS406"
SId_to_Sname[167] = "RS407"
SId_to_Sname[169] = "RS409"
SId_to_Sname[183] = "RS503"
SId_to_Sname[188] = "RS508"
SId_to_Sname[189] = "RS509"

SId_to_Sname[201] = "DE601"
SId_to_Sname[202] = "DE602"
SId_to_Sname[203] = "DE603"
SId_to_Sname[204] = "DE604"
SId_to_Sname[205] = "DE605"
SId_to_Sname[206] = "FR606"
SId_to_Sname[207] = "SE607"
SId_to_Sname[208] = "UK608"

## this just "inverts" the previous list, discarding unused values
Sname_to_SId_dict = {name:ID for ID,name in enumerate(SId_to_Sname) if name is not None}

def even_antName_to_odd(even_ant_name):
    even_num = int(even_ant_name)
    odd_num = even_num + 1
    return str( odd_num ).zfill( 9 )

def antName_is_even(ant_name):
    return not int(ant_name)%2

def odd_antName_to_even(odd_ant_name):
    odd_num = int(odd_ant_name)
    even_num = odd_num + 1
    return str( even_num ).zfill( 9 )
    

#### plotting utilities ####
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    
    

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    
