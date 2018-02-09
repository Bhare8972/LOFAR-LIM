#!/usr/bin/env python2

##python
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


from os import mkdir
from os.path import isdir

##external
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, brute
from scipy.signal import hilbert

##mine
from utilities import log, processed_data_dir
from antenna_responce import ant_calibrator

if __name__ == "__main__":
    timeID = "D20160712T173455.100Z"
    fname = "L526419_D20160712T173455.100Z_CS001_R000_tbb.caltst"
    ant_name = "001000000"
    output_folder = "calibration_test"
    
    azimuth_from_north = 20.0
    elivation = 20.0
    
    azimuth = (azimuth_from_north+90)*np.pi/180.0
    zenith = (90.0-elivation)*np.pi/180.0
    
        #### setup directory variables ####
    processed_data_dir = processed_data_dir(timeID)
    
    data_dir = processed_data_dir + "/" + output_folder
    if not isdir(data_dir):
        mkdir(data_dir)
        
    logging_folder = data_dir + '/logs_and_plots'
    if not isdir(logging_folder):
        mkdir(logging_folder)

    #Setup logger and open initial data set
    log.set(logging_folder + "/log_out.txt") ## TODo: save all output to a specific output folder
    log.take_stderr()
    log.take_stdout()
    
    
    with open(data_dir+'/'+fname, 'rb') as fin:
        data = np.load(fin)
        original_timeseries = data["arr_0"]
        calibrated_timeseries = data["arr_1"]
        
    plt.plot(original_timeseries[0])
    plt.plot(original_timeseries[1])
    plt.show()
        
    plt.plot(calibrated_timeseries[0])
    plt.plot(calibrated_timeseries[1])
    plt.show()
    
    
    calibrator = ant_calibrator(timeID)
    calibrator.FFT_prep(ant_name, original_timeseries[0], original_timeseries[1])
    correction_factors = calibrator.apply_GalaxyCal()
    print("correction factors:", correction_factors)
                
    calibrator.unravelAntennaResponce(azimuth, zenith)
    zenith_data, azimuth_data = calibrator.getResult()
    
    plt.plot( zenith_data, 'r')
    plt.plot( calibrated_timeseries[0], 'bo' )
    plt.show()
    
    plt.plot( azimuth_data, 'r')
    plt.plot( calibrated_timeseries[1], 'bo' )
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    