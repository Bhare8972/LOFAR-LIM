#!/usr/bin/env python3

import numpy as np
from scipy.signal import gaussian
from scipy.signal import hann
from scipy import fftpack

from matplotlib import pyplot as plt

### need to move these functions here
from LoLIM.utilities import upsample_N_envelope

def simple_bandpass(frequencies, lower_freq=30.0E6, upper_freq=80.0E6, roll_width = 2.5E6):
    """create a simple bandpass filter between two frequencies. Sets all negative frequencies to zero, returns frequency response at 'frequencies'"""
   
    bandpass_filter = np.zeros( len(frequencies), dtype=complex)
    bandpass_filter[ np.logical_and( frequencies>=lower_freq, frequencies<=upper_freq  ) ] = 1.0
    
    gaussian_weights = gaussian(len(frequencies), int( round(roll_width/(frequencies[1]-frequencies[0]) ) ) )
    
    bandpass_filter = np.convolve(bandpass_filter, gaussian_weights, mode='same' )
    
    bandpass_filter /= np.max(bandpass_filter) ##convolution changes the peak value
    
    return bandpass_filter

def block_bandpass(frequencies, lower_freq=30.0E6, upper_freq=80.0E6):
    
    bandpass_filter = np.zeros( len(frequencies), dtype=complex)
    bandpass_filter[ np.logical_and( frequencies>=lower_freq, frequencies<=upper_freq  ) ] = 1.0
    
    return bandpass_filter


    
def half_hann_window(length, half_percent=None, hann_window_length=None):
    """produce a half-hann window. Note that this is different than a Hamming window."""
    if half_percent is not None:
        hann_window_length = int(length*half_percent)
    hann_window = hann(2*hann_window_length)
    
    half_hann_widow = np.ones(length, dtype=np.double)
    half_hann_widow[:hann_window_length] = hann_window[:hann_window_length]
    half_hann_widow[-hann_window_length:] = hann_window[hann_window_length:]
    
    return half_hann_widow


class upsample_and_correlate:
    def __init__(self, data_length, upsample_factor):
        self.input_length = data_length
        self.out_length = 2*data_length*upsample_factor
        
        self.workspace_1 = np.empty(2*data_length, dtype=np.complex)
        self.output = np.empty(self.out_length, dtype=np.complex)
        
        self.slice_A = slice(0, (2*self.input_length + 1) // 2)
        self.slice_B = slice((2*self.input_length + 1) // 2, -(2*self.input_length - 1) // 2)
        self.slice_C = slice(-(2*self.input_length - 1) // 2, None)
        
        self.slice_D = slice(self.out_length-self.input_length, self.out_length-self.input_length+1,None)
        self.slice_E = slice(self.input_length,self.input_length+1,None)
        
    def run(self, A, B):
        self.workspace_1[:] = 0
        self.workspace_1[:min(self.input_length,len(B))] = B
        fftpack.fft(self.workspace_1, overwrite_x = True)
        
        np.conjugate(self.workspace_1[self.slice_A], out=self.output[self.slice_A])
        self.output[self.slice_B] = 0.0
        np.conjugate(self.workspace_1[self.slice_C], out=self.output[self.slice_C])
        
        
        
        self.workspace_1[:] = 0
        self.workspace_1[:min(self.input_length,len(A))] = A
        fftpack.fft(self.workspace_1, overwrite_x = True)
        
        self.output[self.slice_A] *= self.workspace_1[self.slice_A]
        self.output[self.slice_C] *= self.workspace_1[self.slice_C]
        
        
        # special treatment if low number of points is even. So far we have set Y[-N/2]=X[-N/2]
        self.output[self.slice_D] /= 2  # halve the component at -N/2
        temp = self.output[self.slice_D]
        self.output[self.slice_E] = temp  # set that equal to the component at -Nx/2
        
        
        fftpack.ifft(self.output, overwrite_x=True)
        return self.output
        
def correlateFFT(FFTdata_i, FFTdata_j, out):
    """given two FFT data arrays, upsample, correlate, and inverse fourier transform."""
        #### upsample, and multiply A*conj(B), using as little memory as possible
        
    in_len = FFTdata_i.shape[0]
    out_len = out.shape[0]
    
    A = 0
    B = (in_len + 1) // 2
    
    ##set lower positive
    np.conjugate(FFTdata_j[A:B],    out=out[A:B])
    out[A:B] *= FFTdata_i[A:B]
    
    ### all upper frequencies
    A = (in_len + 1) // 2
    B = out_len - (in_len - 1) // 2
    out[A:B] = 0.0

    ## set lower negative
    A = out_len - (in_len - 1) // 2
    B = in_len - (in_len - 1) // 2
    np.conjugate(FFTdata_j[B:],    out=out[A:])
    out[A:] *= FFTdata_i[B:]
    
     # special treatment if low number of points is even. So far we have set Y[-N/2]=X[-N/2]
    out[out_len-in_len//2] *= 0.5  # halve the component at -N/2
    temp = out[out_len-in_len//2]
    out[in_len//2:in_len//2+1] = temp  # set that equal to the component at -Nx/2
    
    ## ifft in place
    fftpack.ifft(out, n=out_len, overwrite_x=True)
    
    
class parabolic_fit:
    def __init__(self, data, index=None, n_points=5):
        """a class that fits a parabola to find the peak. data should be 1D array. index should be the index of the peak to fit in data, if None:
            index = np.argmax(data). n_points is the total number of points to fit, should be odd, normally 5.  parabolic_fit.peak_index is
            the fractional location of the peak"""
        
        if index is None:
            index = np.argmax( data )
        
        self.index = index
        self.n_points = n_points
        
        self.matrix, self.error_ratio_A, self.error_ratio_B, self.error_ratio_C = self.get_fitting_matrix(n_points)
        
        half_n_points = int( (n_points-1)/2 )
        points = data.take(np.arange(n_points)+index-half_n_points, axis=0, mode="wrap" )  #data[ index-half_n_points :  index+1+half_n_points ]
        
        self.A, self.B, self.C = np.dot( self.matrix, points )
        
        self.peak_relative_index = -self.B/(2.0*self.A)
        self.peak_index = self.peak_relative_index + (index-half_n_points)
        
        self.peak_index_error_ratio = self.peak_relative_index*np.sqrt( (self.error_ratio_A/self.A)**2 + (self.error_ratio_B/self.B)**2 )
        
        X = np.arange(n_points)
        
        residuals = (self.A*(X*X) + self.B*X + self.C) - points
        self.RMS_fit = np.sqrt( np.sum(residuals*residuals)/n_points )
        
        self.amplitude = self.C - (self.B**2)/(4.0*self.A)
        
    def plot(self,c,N):
        X = np.arange(N)*self.n_points/(N-1)
        
        curve = (self.A*(X*X) + self.B*X + self.C)
        
        half_n_points = int( (self.n_points-1)/2 )
        plt.plot(X+(self.index-half_n_points), curve, c)
         
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


def remove_saturation(data, positive_saturation, negative_saturation, post_removal_length=50, half_hann_length=50):
    """given some data, as a 1-D numpy array, remove areas where the signal saturates by multiplying with a half-hann filter. 
    Operates on input data. 
    positive saturation and negative saturation are the values that above and below which the data is considered to be in saturation
    
    post_remove_length is the number of data points to remove after the data comes out of saturation
    
    half_hann_length is the size of the hann-wings on either side of the zeroed data"""
    
    ## find where saturated
    is_saturated = np.logical_or(data>positive_saturation, data<negative_saturation)
    leaveSaturation_indexes = np.where(np.logical_and( is_saturated[:-1]==1, is_saturated[1:]==0)  ) [0]
    
    ## extend saturation areas by some length
    for i in leaveSaturation_indexes:
        is_saturated[i+1:i+1+post_removal_length] = 1
        
    ## make the hann window
    hann_window = 1.0 - hann(2*half_hann_length)
    pre_window = hann_window[:half_hann_length]
    post_window = hann_window[half_hann_length:]
    
    ## find the indeces where we need to remove data
    window_starts = np.where(np.logical_and( is_saturated[:-1]==0, is_saturated[1:]==1)  )[0]
    window_ends = np.where(np.logical_and( is_saturated[:-1]==1, is_saturated[1:]==0)  )[0] ##note that this is INCLUSIVE!
    
    ## remove the data!
    start_i = 0
    end_i = 0
    
    data_cut = []
    
    if len(window_ends)>0 and ( len(window_starts)==0 or window_ends[0]<window_starts[0]):
        ## the data starts saturatated
        win_end = window_ends[ end_i ]
        
        data[:end_i+1] = 0.0
        
        
        ## take care if we are near end of file
        N = half_hann_length
        end_i_end = end_i+1+half_hann_length
        if end_i_end >= len(data):
            end_i_end = len(data)
            N = end_i_end - (end_i+1)
        
        data[end_i+1 : end_i_end] *= post_window[:N]
        
        end_i += 1
        
        data_cut.append([0,end_i_end])
     
    for x in range(len(window_ends)-end_i):
        win_start = window_starts[ start_i ]
        win_end = window_ends[ end_i ]
        
        data[win_start:win_end+1] = 0.0
        
        ## take care at start
        win_start_start = win_start-half_hann_length
        N = half_hann_length
        if win_start_start<0:
            win_start_start = 0
            N = win_start - win_start_start
      
        data[win_start_start : win_start] *= pre_window[half_hann_length-N:]
        
        
        ## take care if we are near end of file
        N = half_hann_length
        end_i_end = end_i+1+half_hann_length
        if end_i_end >= len(data):
            end_i_end = len(data)
            N = end_i_end - (end_i+1)
        
        data[end_i+1 : end_i_end] *= post_window[:N]
        
        
        start_i += 1
        end_i += 1
        data_cut.append([win_start_start,end_i_end])
        
    if start_i == len(window_starts)-1:
        ## the data ends saturated
        win_start = window_starts[ start_i ]
        
        data[win_start:] = 0.0
        
        ## take care at start
        win_start_start = win_start-half_hann_length
        N = half_hann_length
        if win_start_start<0:
            win_start_start = 0
            N = win_start - win_start_start
      
        data[win_start_start : win_start] *= pre_window[half_hann_length-N:]
        data_cut.append([win_start_start,len(data)])
        
    return data_cut

def data_cut_at_index(data_cuts, index):
    """given data_cuts, which is the return value of remove_saturation, and index, return True if index is in a region that 
    has been cut due to saturation, false otherwise"""
    
    for start,stop in data_cuts:
        if start <= index < stop:
            return True
    return False
    
    
def num_double_zeros(data):
    """if data is a numpy array, give number of points that have  zero preceded by a zero"""
    is_zero = (data==0)
    
    bad = np.logical_and( is_zero[:-1], is_zero[1:] )
    return np.sum(bad)

def FFT_time_shift(frequencies, FFT_data, dt):
    """given some frequency dependent data, apply a positive time-shift dt. Operates
    on the data in-place """
    FFT_data *= np.exp( frequencies*(-1j*2*np.pi*dt) )
    

