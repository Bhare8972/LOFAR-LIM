#!/usr/bin/env python3

import numpy as np
from scipy.signal import gaussian
from scipy.signal import hann
from scipy import fftpack

from matplotlib import pyplot as plt

### need to move these functions here
#from LoLIM.utilities import upsample_N_envelope

def simple_bandpass(frequencies, lower_freq=30.0E6, upper_freq=80.0E6, roll_width = 2.5E6):
    """create a simple bandpass filter between two frequencies. Sets all negative frequencies to zero, returns frequency response at 'frequencies'
    Essentually block filter with smothed edges"""
   
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
    
### this class is being phased-out, use fitter below
class parabolic_fit:
    def __init__(self, data, index=None, n_points=5, data_mode='wrap'):
        """a class that fits a parabola to find the peak. data should be 1D array. index should be the index of the peak to fit in data, if None:
            index = np.argmax(data). n_points is the total number of points to fit, should be odd, normally 5.  parabolic_fit.peak_index is
            the fractional location of the peak. data_mode is passed to the 'mode' parameter of data.take"""
        
        if index is None:
            index = np.argmax( data )
        
        self.index = index
        self.n_points = n_points
        
        self.matrix, self.error_ratio_A, self.error_ratio_B, self.error_ratio_C = self.get_fitting_matrix(n_points)
        
        half_n_points = int( (n_points-1)/2 )
        points = data.take(np.arange(n_points)+index-half_n_points, axis=0, mode=data_mode )  #data[ index-half_n_points :  index+1+half_n_points ]
        
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
    
class parabolic_fitter:
    def __init__(self, n_points=5):
        self.n_points = n_points
        self.half_n_points = int( (n_points-1)/2 )
        
        self.matrix, self.error_ratio_A, self.error_ratio_B, self.error_ratio_C = self.get_fitting_matrix(n_points)
        self.ABC_tmp = np.empty(3, dtype=np.double) ## Ax^2 + Bx + C
        
    def fit(self, data, index=None):
        """return fractional index of peak location"""
        
        if index is None:
            index = np.argmax( data )
        self.index = index
            
        self.ABC_tmp[:] = 0.0
        for i in range(self.n_points):
            data_i = index + i - self.half_n_points
            if data_i >= len(data):
                data_i -= len(data)
            self.ABC_tmp += self.matrix[:, i]*data[ data_i ]
            
        return -self.ABC_tmp[1]/(2.0*self.ABC_tmp[0]) + index - self.half_n_points
    
    ### TODO:: add a more spohesitated fucntion, that mimics all the fidly bits of paraborlic_fit?

    def get_amplitude(self):
        """return Y-value of parabola peak"""
        return  self.ABC_tmp[2] - (self.ABC_tmp[1]**2)/(4.0*self.ABC_tmp[0])

    def X_at_peak(self, x_values):
        """given the x-values of previous data points, return fitted x-value of the peak. Assumes points are regularly spaced."""
        I = -self.ABC_tmp[1]/(2.0*self.ABC_tmp[0]) + self.index - self.half_n_points
        int_index = int(I)
        if int_index<0 or int_index>=len(x_values):
            return np.nan

        fractional_index = I - int_index
        delta = x_values[int_index+1] - x_values[int_index]
        return x_values[int_index] + delta*fractional_index



    def second_derivative(self):
        """givin a previous fit, return the second derivative"""
        return 2.0*self.ABC_tmp[0]
         
    def get_fitting_matrix(self, n_points):
        if "fit_matrix" not in parabolic_fitter.__dict__:
            parabolic_fitter.fit_matrix = {}
            
        if n_points not in parabolic_fitter.fit_matrix:
        
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
    
            parabolic_fitter.fit_matrix[n_points] = [peak_time_matrix, error_ratio_A, error_ratio_B, error_ratio_C]
        
        
        return parabolic_fitter.fit_matrix[n_points]
    
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
        
        data[:win_end+1] = 0.0
        
        
        ## take care if we are near end of file
        N = half_hann_length
        end_i_end = win_end+1+half_hann_length
        if end_i_end >= len(data):
            end_i_end = len(data)
            N = end_i_end - (win_end+1)
        
        data[win_end+1 : end_i_end] *= post_window[:N]
        
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
        end_i_end = win_end+1+half_hann_length
        if end_i_end >= len(data):
            end_i_end = len(data)
            N = end_i_end - (win_end+1)
        
        data[win_end+1 : end_i_end] *= post_window[:N]
        
        
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
    
def num_double_zeros(data, threshold=None, ave_shift=False):
    """if data is a numpy array, give number of points that have  zero preceded by a zero"""
    
    if ave_shift:
        data = data - np.average(data)
    
    if threshold is None:
        is_zero = (data==0)
    else:
        is_zero = np.abs(data) < threshold
    
    bad = np.logical_and( is_zero[:-1], is_zero[1:] )
    return np.sum(bad)

def locate_data_loss(data, num_zeros):
    spans = []
    total = 0
    
#    mode = 0 ## 0 means off
    last_start = None
    dz_i = 0 ## num double zeros counted
    for i, d in enumerate(data):
        if d == 0: ## 
            dz_i += 1
            if last_start is None:
                last_start = i
                
        elif last_start is not None:
            if dz_i >= num_zeros:
                spans.append( [last_start, i] )
                total += i-last_start
            dz_i = 0
            last_start = None
            
        
    if dz_i >= num_zeros:
        spans.append( [last_start, i] )
        total += i-last_start
            
    return spans, total

def data_cut_at_index(data_cuts, index):
    """given data_cuts, which is the return value of remove_saturation, and index, return True if index is in a region that 
    has been cut due to saturation, false otherwise."""
    
    for start,stop in data_cuts:
        if start <= index < stop:
            return True
    return False

def data_cut_inspan(data_cuts, span_start, span_stop):
    """given data_cuts, which is the return value of remove_saturation, and index, return True if index is in a region that 
    has been cut due to saturation, false otherwise."""
    
    for start,stop in data_cuts:
        if (start < span_stop) and (stop>span_start):
            return True
    return False

def add_spans(span_list_A, span_list_B):
    """givin two lists of spans, return anouther list where the spans are combined. Essentially A or B."""
    
    ## math notes: algorithm is symetric between two lists. Adding only produces one or two spans
    
    current_spans = span_list_A + span_list_B
    good_spans = []
    for span_i, span in enumerate(current_spans):
        spans_to_check = current_spans[span_i+1:]
        for span_j, check_span in enumerate(spans_to_check):
            if check_span[0]==check_span[1]:
                continue
            
            if (span[1] < check_span[0]) or (check_span[1] < span[0]):
                continue
            elif (check_span[0] <= span[0]) and (check_span[1] >= span[1]):
                ## throw span
                span = [0,0]
                break
            elif (span[0] <= check_span[0]) and (span[1] >= check_span[1]):
                ## flip data and throw span
                current_spans[span_i+1+span_j] = span
                span = [0,0]
                break
            elif (check_span[0] < span[0]) and ( span[0] <= check_span[1] < span[1]):
                current_spans[span_i+1+span_j] = [check_span[0], span[1]]
                span = [0,0]
                break
            elif (span[0] < check_span[0]) and ( check_span[0] <= span[1] < check_span[1]):
#                spans_to_check[span_j] = [span[0], check_span[1]]
                current_spans[span_i+1+span_j] = [span[0], check_span[1]]
                span = [0,0]
                break
            else:
                raise Exception("algorithm error. This shouldn't be reached")
            
            if span[0]==span[1]:
                break
        if span[0]!=span[1]:
            good_spans.append(span)
    return good_spans

def subtract_spans(span_list_A, span_list_B):
    """give spans that are in A, but not B. UNTESTED"""
    
    ## math notes: algorithm is NOT symetric between two lists. 
    # Subtracting produces up to three spans
    
    good_spans = []
    spans_to_test = span_list_A
    next_spans_to_test = []
    while len(spans_to_test) != 0:
        
        for span in spans_to_test:
            if span[0]==span[1]:
                continue
            
            print('tst:', span)
            
            for Bspan in span_list_B:
                if Bspan[0]==Bspan[1]:
                    continue
                print('  against:', Bspan)
                
                if (span[1] <= Bspan[0]) or (Bspan[1] <= span[0]):
                    print('  1')
                    continue
                elif (Bspan[0] <= span[0]) and (Bspan[1] >= span[1]):
                    print('  2')
                    ## no span!
                    span = [0,0]
                    break
                elif (span[0] <= Bspan[0]) and (span[1] >= Bspan[1]):
                    next_spans_to_test.append( [Bspan[1], span[1]] )
                    span[1] = Bspan[0]
                    print('  3', 'new span:', span, 'future span:', next_spans_to_test[-1])
                elif (Bspan[0] < span[0]) and ( span[0] < Bspan[1] < span[1]):
                    print('  4')
                    span = [Bspan[1], span[1]]
                elif (span[0] < Bspan[0]) and ( Bspan[0] < span[1] < Bspan[1]):
                    print('  5')
                    span = [span[0], Bspan[0]]
                    
                if span[0]==span[1]:
                    break
            
            if span[0]!=span[1]:
                print('returning:', span)
                good_spans.append([span[0],span[1]])
        
        spans_to_test = next_spans_to_test
        next_spans_to_test = []
        
    return good_spans
            
def union_spans(span_list_A, span_list_B):
    """give spans that are in both A and B."""
    
    ## math notes: very similar to A-B, but we return the bit that is removed.
    ## despite simularities to subtract above, it is symetric
    
    good_spans = []
    spans_to_test = span_list_A
    next_spans_to_test = []
    while len(spans_to_test) != 0:
        
        for span in spans_to_test:
            if span[0]==span[1]:
                continue
            
            for Bspan in span_list_B:
                if Bspan[0]==Bspan[1]:
                    continue
                
                if (span[1] <= Bspan[0]) or (Bspan[1] <= span[0]):
                    continue
                elif (Bspan[0] <= span[0]) and (Bspan[1] >= span[1]):
                    span = [0,0]
                    good_spans.append( [Bspan[0],Bspan[1]] )
                    break
                elif (span[0] <= Bspan[0]) and (span[1] >= Bspan[1]):
                    span[1] = Bspan[0]
                    good_spans.append( [Bspan[0],Bspan[1]] )
                    next_spans_to_test.append( [Bspan[1], span[1]] )
                elif (Bspan[0] < span[0]) and ( span[0] < Bspan[1] < span[1]):
                    span = [Bspan[1], span[1]]
                    good_spans.append( [span[0],Bspan[1]] )
                elif (span[0] < Bspan[0]) and ( Bspan[0] < span[1] < Bspan[1]):
                    span = [span[0], Bspan[0]]
                    good_spans.append( [Bspan[0],span[1]] )
                    
                if span[0]==span[1]:
                    break
        
        spans_to_test = next_spans_to_test
        next_spans_to_test = []
        
    return good_spans

def plot_spans(spans, Y=0, T_array=None, color=None):
    for span in spans:
        T0 = span[0]
        T1 = span[1]
        if T_array is not None:
            T0 = T_array[T0]
            T1 = T_array[T1]
        print('plotting', T0,T1)
        plt.plot([T0,T1],[Y,Y],c=color)
    
            

def FFT_time_shift(frequencies, FFT_data, dt, out=None):
    """given some frequency dependent data, apply a positive time-shift dt. Operates
    on the data in-place """
    if out is None:
        A = frequencies*(-2j*np.pi*dt)
    else:
        A[:] = frequencies
        A *= (-2j*np.pi*dt)

    FFT_data *= np.exp( A, out=A )
    
def make_stokes(X, Y):
    """given complex electric fields in X and Y direction (can be numpy arrays), return I, Q, U, and V"""
    
    TMP = np.empty(len(X), dtype=np.double)
    
    I = np.empty(len(X), dtype=np.double)
    Q = np.empty(len(X), dtype=np.double)
    U = np.empty(len(X), dtype=np.double)
    V = np.empty(len(X), dtype=np.double)
    
    ### calc I and Q ####
    I[:] = X.real
    I *= I
    Q[:] = I
    
    TMP[:] = X.imag
    TMP *= TMP
    I += TMP
    Q += TMP
    
    TMP[:] = Y.real
    TMP *= TMP
    I += TMP
    Q -= TMP
    
    TMP[:] = Y.imag
    TMP *= TMP
    I += TMP
    Q -= TMP
    
    ### calc U
    U[:] = X.real
    U[:] *= Y.real

    TMP[:] = X.imag
    TMP[:] *= Y.imag
    U += TMP
    U *= 2.0
    
    ### calc V
    V[:] = X.imag
    V[:] *= Y.real
    
    TMP[:] = X.real
    TMP[:] *=Y.imag
    V -= TMP
    V *= -2
    
    return I,Q,U,V
    
    
from scipy.stats import poisson
from scipy.optimize import root_scalar
def poisson_error_bars(N):
    """given an input number of counts (integer), returns 32% error bars. Equal to 1 standard devation for normal distribution.
    Returns two numbers. First is with lower error bar, second is upper error bar"""

    p=0.32

    ## some utility functions
    def CDF(L):
        return poisson.cdf(N, L)
    
    def LA(L):
        return CDF(L) - p
    def LB(L):
        return (1.0-CDF(L)) - p
    
    def LA_prime(L):
        L_low = L-0.001
        L_high = L+0.001
        if L_low<0:
            L_low=0
        return (LA(L_high)-LA(L_low))/(L_high-L_low)
    
    def LB_prime(L):
        L_low = L-0.001
        L_high = L+0.001
        if L_low<0:
            L_low=0
        return (LB(L_high)-LB(L_low))/(L_high-L_low)




    if N >= 100:
        ## large number limit
        lower = N-np.sqrt(N)/2
        upper = N+np.sqrt(N)/2
    else:
        lower = root_scalar(LB, x0=N, fprime=LB_prime, method='newton').root
        upper = root_scalar(LA, x0=N, fprime=LA_prime, method='newton').root

    if N==0 or N==1:
        lower = 0

    return lower, upper



