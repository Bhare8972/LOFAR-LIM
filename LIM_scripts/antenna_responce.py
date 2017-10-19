#!/usr/bin/env python2

##internal
import os
import glob

##external 
import numpy as np
from scipy.interpolate import interp1d

##pycrtools
import pycrtools as cr

##mine
from utilities import processed_data_dir

#### This module is for un-raveling the antenna responce function and applying the galaxy calibration curve
#### Currently only works in Python 2 (depends on PyCRtools). Needs to be re-written to work in Python 3

class ant_calibrator:
    
    def __init__(self, timeID):
        
        ### Galaxy callibration data ###
        self.calibration_factors = {}
        galcal_data_loc = processed_data_dir(timeID) + '/cal_tables/galaxy_cal'
        galcal_fpaths = glob.glob(galcal_data_loc + '/*.gcal')
        for fpath in galcal_fpaths:
            with open(fpath, 'rb') as fin:
                data = np.load(fin)
                ant_names = data["arr_0"]
                factors = data["arr_1"]
            ant_i = 0
            while ant_i<len(ant_names):
                self.calibration_factors[ant_names[ant_i]] = [factors[ant_i], factors[ant_i+1]]
                ant_i += 2
            
            
        
        ### antenna responce data ###
        self.vt = np.loadtxt(os.environ["LOFARSOFT"] + "/data/lofar/antenna_response_model/LBA_Vout_theta.txt", skiprows=1)
        self.vp = np.loadtxt(os.environ["LOFARSOFT"] + "/data/lofar/antenna_response_model/LBA_Vout_phi.txt", skiprows=1)
        
        self.cvt = cr.hArray(self.vt[:, 3] + 1j * self.vt[:, 4])
        self.cvp = cr.hArray(self.vp[:, 3] + 1j * self.vp[:, 4])
    
        self.fstart = 10.0 * 1.e6
        self.fstep = 1.0 * 1.e6
        self.fn = 101
        self.tstart = 0.0
        self.tstep = 5.0
        self.tn = 19
        self.pstart = 0.0
        self.pstep = 10.0
        self.pn = 37
        
        
        
    def FFT_prep(self, even_ant_name, even_pol_data, odd_pol_data):
        """ prepare to apply callibrations to a pair of dipoles. Essentually just takes FFT. Assume (hope) a hanning window is not needed."""
        
        if len(even_pol_data) != len(odd_pol_data):
            raise ValueError('even and odd polarization data need to be same length')
            
        self.even_ant_name = even_ant_name
        self.N_points = len(even_pol_data)
        
        ##FFT
        self.even_pol_FFT = np.fft.rfft(even_pol_data)
        self.odd_pol_FFT =  np.fft.rfft(odd_pol_data)
        
        ##Frequencies
        self.frequencies = np.fft.rfftfreq(self.N_points, 5.0E-9)
        
        
#        self.even_pol_data = cr.hArray(float, even_pol_data )
#        self.odd_pol_data  = cr.hArray(float, odd_pol_data )
#        
#        # Calculate sample frequency in Hz
#        self.frequencies = cr.hArray(float, self.N_points / 2 + 1)
#        cr.hFFTFrequencies(self.frequencies, 1.0/(5.0E-9), 1)
#        
#        #### take FFT ####
#        self.even_pol_FFT = cr.hArray(complex, self.N_points / 2 + 1)
#        self.odd_pol_FFT = cr.hArray(complex, self.N_points / 2 + 1)
#        
##        FFT_plan = cr.FFTWPlanManyDftR2c(self.N_points, 1, 1, 1, 1, 1, cr.fftw_flags.ESTIMATE) ##?????s elf.__blocksize, self.__file.nofSelectedDatasets(), 1, self.__blocksize, 1, self.__blocksize / 2 + 1
#        
#        FFT_plan = cr.FFTWPlanManyDftR2c(self.N_points, 1, 1, self.N_points, 1, self.N_points/2+1, cr.fftw_flags.ESTIMATE) 
#        
#        cr.hFFTWExecutePlan(self.even_pol_FFT, self.even_pol_data, FFT_plan)
#        cr.hFFTWExecutePlan(self.odd_pol_FFT,  self.odd_pol_data, FFT_plan)
#        # Swap Nyquist zone if needed ????
#        self.even_pol_FFT.nyquistswap(1)###???
#        self.odd_pol_FFT.nyquistswap(1)###???
#    

        
    def apply_GalaxyCal(self):
        """applies the galaxy calibration to this data. Needs to be done BEFORE unravelAntennaResponce"""
        
        ### first we apply the correction factors ###
        PolE_factor, PolO_factor = self.calibration_factors[self.even_ant_name]

        self.even_pol_FFT *= PolE_factor
        self.odd_pol_FFT  *= PolO_factor
        
        
        #### now we apply a calibration curve
        Calibration_curve = np.zeros(101)
        Calibration_curve[29:82] = np.array([0,  1.09124663e-06,   1.11049910e-06,   1.11101995e-06,
                 1.14234774e-06,   1.15149299e-06,   1.17121699e-06,
                 1.18578121e-06,   1.19696124e-06,   1.20458122e-06,
                 1.24675978e-06,   1.27966600e-06,   1.32418333e-06,
                 1.32115453e-06,   1.33871075e-06,   1.34295545e-06,
                 1.34157430e-06,   1.37660390e-06,   1.39226359e-06,
                 1.39827006e-06,   1.51409426e-06,   1.61610247e-06,
                 1.74643510e-06,   1.74588169e-06,   1.73061463e-06,
                 1.69229172e-06,   1.64633321e-06,   1.60982965e-06,
                 1.59572009e-06,   1.64618678e-06,   1.81628916e-06,
                 2.09520281e-06,   2.17610590e-06,   2.20907337e-06,
                 2.12050148e-06,   2.04923844e-06,   2.06549879e-06,
                 2.24906987e-06,   2.40356459e-06,   2.52199062e-06,
                 2.48380048e-06,   2.40835417e-06,   2.38248922e-06,
                 2.48599834e-06,   2.60617662e-06,   2.66466169e-06,
                 2.78010597e-06,   2.90548503e-06,   3.08686745e-06,
                 3.26101312e-06,   3.50261561e-06,   3.74739666e-06, 0])  
                ## 30 - 80 MHz, derived from average galaxy model + electronics

        Calibration_curve_interp = interp1d(np.linspace(0.e6,100e6,101), Calibration_curve, kind='linear', bounds_error=False, fill_value=0.0)
        Calibration_curve_interp = Calibration_curve_interp(self.frequencies)
        
        self.even_pol_FFT *= Calibration_curve_interp
        self.odd_pol_FFT  *= Calibration_curve_interp

        return [PolE_factor, PolO_factor]
        
    def unravelAntennaResponce(self, azimuth, zenith):
        """given a direction to source (azimuth off X and zenith from Z, in radius ), remove antenna responce curve."""
        

        ## get angles in the coordinate system handeld by pyCRtools
        azimuth_FN  = ( azimuth - np.pi/2.0)*180.0/np.pi
        inclination = ( np.pi/2.0 - zenith)*180.0/np.pi
        
        
        ### make the jones matrix
        jones_matrix = cr.hArray(complex, dimensions=(self.frequencies.shape[0], 2, 2) )
        
        for i, f in enumerate(self.frequencies): ### this function is the ONLY reason we need pycrtools....
            cr.hGetJonesMatrix(jones_matrix[i], f, azimuth_FN, inclination, self.cvt, self.cvp, self.fstart, self.fstep, self.fn, self.tstart, self.tstep, self.tn, self.pstart, self.pstep, self.pn)
    
        inverse_jones_matrix = cr.hArray(complex, dimensions=(self.frequencies.shape[0], 2, 2) )
        cr.hInvertComplexMatrix2D(inverse_jones_matrix, jones_matrix) ### can probably replace this with pure analytic function...
        
        
        ### apply the Jones matrix.  Note that the polarities (even and odd) need to be flipped)
    
        inverse_jones_matrix = inverse_jones_matrix.toNumpy()
        zenith_component = self.odd_pol_FFT*inverse_jones_matrix[:, 0,0] +  self.even_pol_FFT*inverse_jones_matrix[:, 0,1]
        azimuth_component = self.odd_pol_FFT*inverse_jones_matrix[:, 1,0] +  self.even_pol_FFT*inverse_jones_matrix[:, 1,1]
#        
        self.even_pol_FFT = zenith_component
        self.odd_pol_FFT = azimuth_component
    
#        zenith_component = self.odd_pol_FFT.new()
#        azimuth_component = self.even_pol_FFT.new()
#        zenith_component.copy(self.odd_pol_FFT)
#        azimuth_component.copy(self.even_pol_FFT)
#        cr.hMatrixMix(zenith_component, azimuth_component, inverse_jones_matrix)
        
    def getResult(self):
        """get the results of analysis. Essentially preforms inverse FFT. 
        If unravelAntennaResponce was called, first return is zenith component, second is azimuthal. Else first return is even polarization, second is odd"""
        
#        ifftwplan = cr.FFTWPlanManyDftC2r(self.N_points, 1, 1, 1, 1, 1, cr.fftw_flags.ESTIMATE)
#        
#        A = self.even_pol_data.new()
#        B = self.even_pol_data.new()
#        
#        cr.hFFTWExecutePlan(A, self.even_pol_FFT, ifftwplan)
#        cr.hFFTWExecutePlan(B, self.odd_pol_FFT, ifftwplan)
        
        return np.fft.irfft(self.even_pol_FFT),    np.fft.irfft(self.odd_pol_FFT)
        
        
        
        
        
        
        
        

