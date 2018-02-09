#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

from LoLIM.utilities import v_air, RTD, processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.findRFI import FindRFI, window_and_filter
from LoLIM.interferometry.imaging_2D import image2D
from LoLIM.signal_processing import half_hann_window, remove_saturation

def plot_antenna_data(data, antenna_delays, use_seconds, M=None):
    if M is None:
        M = max( np.max(data), np.abs(np.min(data)) )
    T = np.arange(data.shape[1], dtype=np.double)
    if use_seconds:
        T *= 5.0E-9
    for i,(trace, delay) in enumerate(zip(data, antenna_delays)):
        D = delay
        if not use_seconds:
            D /= 5.0E-9
        plt.plot(T-D, trace + 2*i*M)
        
    return M
        
        

if __name__ == "__main__":
    block_size = 2**16
    
    timeID = "D20160712T173455.100Z"
    station = "RS508"
    
    data_index = int( 11500*block_size )
    
    
    
#    timeID = "D20170929T202255.000Z"
#    station = "RS406"
    
#    data_index = int( 3905*block_size )
    
    
    
    findRFI_initial_block = 5
    findRFI_num_blocks = 20
    findRFI_max_blocks = 100
    
    start = 42250
    print(data_index + start)
    
    n_points = int( 1.0E-6/5.0E-9)
    n_points = 2**( int(np.log2(n_points))+1 )
    print(n_points)
    
    
    #### open data and find RFI ####
    raw_fpaths = filePaths_by_stationName(timeID)
#    print(raw_fpaths[station])
    TBB_data = MultiFile_Dal1( raw_fpaths[station], force_metadata_ant_pos=True )
    
    findRFI_result = FindRFI(TBB_data, block_size, findRFI_initial_block, findRFI_num_blocks, findRFI_max_blocks, verbose=True, figure_location=None)
    data_filter = window_and_filter(block_size, find_RFI=findRFI_result)
    
    data_filter = window_and_filter(block_size, find_RFI=None)
    
    num_antennas = len(TBB_data.get_antenna_names())
    antenna_delays = TBB_data.get_timing_callibration_delays()
    
    
    #### read, filiter, and plot ####
    raw_data = np.zeros((num_antennas, block_size), dtype=np.double)
    plot_i = None
    for ant_i in range(0,num_antennas):
        raw_data[ant_i] = TBB_data.get_data(data_index, block_size, antenna_index=ant_i)
        
        if np.any(raw_data[ant_i]>2000.0) or np.any(raw_data[ant_i]<-2000.0):
            print("sturation on antena", ant_i)
            plot_i = ant_i
            
    if plot_i is not None:
        plt.plot(raw_data[plot_i])
        plt.show()
        
#        remove_saturation( raw_data[ant_i] )
        
        
    filtered_data = data_filter.filter( raw_data )
    raw_data = None
        
    M = plot_antenna_data( np.real(filtered_data), antenna_delays, False)
    plot_antenna_data( np.abs(filtered_data), antenna_delays, False, M)
    plt.show()
    
    
    #### select the data, mulitply by window, plot again ####
    antenna_delays = antenna_delays[::2]
    filtered_data = filtered_data[::2]
    antenna_positions = TBB_data.get_LOFAR_centered_positions()[::2]
    
    imager_data = filtered_data[:,start:start+n_points]
    imager_data *= half_hann_window(n_points,0.1)
    
    M=plot_antenna_data( np.real(imager_data), antenna_delays, False)
    plot_antenna_data( np.abs(imager_data), antenna_delays, False, M)
    plt.show()
    
    
#    imager_data /= M
    
   
    
    ### FFT and image!!
    n_points = 500
    bbox = np.array( [[-1.1,1.1],[-1.1,1.1]] )
    imager_data = fftpack.fft(imager_data, n=imager_data.shape[1]*2, axis=-1)
    speed_of_light = -1#290798684
    ant_del = antenna_delays #np.random.normal(scale=1.0E-9, size=len(antenna_delays))
    image = image2D(imager_data, antenna_positions, ant_del, bbox=bbox, num_points=n_points, upsample=2, speed_light=speed_of_light)




    bbox_info = np.array([(bbox[0,1]-bbox[0,0])/n_points,  (bbox[1,1]-bbox[1,0])/n_points,  bbox[0,0],  bbox[1,0]])
    cos_alpha_index, cos_beta_index = np.unravel_index(image.argmax(), image.shape)
    cos_alpha = bbox_info[0]*cos_alpha_index + bbox_info[2]
    cos_beta = bbox_info[1]*cos_beta_index + bbox_info[3]
    
    offset = 0.0
    if cos_beta<0.0:
        if cos_alpha < 0.0:
            offset = - 90.0
        else:
            offset += 90.0
    
    sin_ze_sq = cos_alpha*cos_alpha + cos_beta*cos_beta
    sin_ze = np.sqrt( sin_ze_sq )
    sin_az = cos_alpha/sin_ze
    
    print("Ze", np.arcsin(sin_ze)*RTD, "Az", np.arcsin(sin_az)*RTD+offset)
    
#    R = np.array([27000.0, 32000.0, 3000.0]) - antenna_positions[0]
#    distance = np.sqrt( np.sum( R*R ) )
#    Ze_real = np.arccos(R[2]/distance)
#    Az_real = np.arctan2(R[1],R[0])
#    print(distance/1000.0, Ze_real*RTD, Az_real*RTD)
#    
#    cos_alpha_real = np.sin(Az_real)*np.sin(Ze_real)
#    cos_beta_real = np.cos(Az_real)*np.sin(Ze_real)
#    
#    print()
#    print(cos_alpha_real-cos_alpha, bbox_info[0])
#    print(cos_beta_real-cos_beta, bbox_info[1])
#    
#    print()
#    rho = distance = np.sqrt( np.sum( R[:2]*R[:2] ) )
#    print("X", rho*np.cos(Az)+antenna_positions[0,0],  'Y:',rho*np.sin(Az)+antenna_positions[0,1], 'Z:',distance*np.cos(Ze)+antenna_positions[0,2])
    
    
    print("arg")
    plt.imshow( image, origin='lower' )
    plt.colorbar()
    plt.show()
    

#mega = 10.0**6
#micro = 10.0**(-6)
#nano = 10.0**(-9)
#
#frequency = 60.0*mega
#width = 10*nano
#
#sampling_frequency = 5*nano
#
#noise_amplitude = None
#
#upsample=2**3
#
### list of sources. each source has an amplitude, zenith, azimith (in radians), and delay (ns)
#sources = [
#        [1.0, 45.0/RTD, 0.0/RTD, -200.0],
#        #[5.0, 45.0/RTD, 90.0/RTD, 200]
#        ]
#
#antenna_positions = np.array([
# [  5562.6334907,   36178.7231285,    -120.13021202],
#   #[  5562.6334907,   36178.7231285,    -120.13021202],
# [  5546.1705939,   36155.8844127,    -119.99846893],
#   #[  5546.1705939,   36155.8844127,    -119.99846893],
# [  5587.4104329,   36158.52725628,   -119.9799359],
#   #[  5587.4104329,   36158.52725628,   -119.9799359],
# [  5541.91829985,  36187.84130857,   -120.20579218],
#   #[  5541.91829985,  36187.84130857,   -120.20579218],
# [  5578.06779154,  36129.82244824,   -119.80497083],
#   #[  5578.06779154,  36129.82244824,   -119.80497083],
# [  5558.33773452,  36202.24046988,   -120.28399756],
#   #[  5558.33773452,  36202.24046988,   -120.28399756]
# ])
#
#
##antenna_positions[:,0] -= np.average( antenna_positions[:,0] )
##antenna_positions[:,1] -= np.average( antenna_positions[:,1] )
##antenna_positions[:,2] -= np.average( antenna_positions[:,2] )
#
#X_ave = np.average( antenna_positions[:,0] )
#Y_ave = np.average( antenna_positions[:,1] )
#Z_ave = np.average( antenna_positions[:,2] )
#
#def get_timeDelays(zenith, azimuth, output=None):
#    sin_z = np.sin(zenith)
#    cos_z = np.cos(zenith)
#    sin_a = np.sin(azimuth)
#    cos_a = np.cos(azimuth)
#    
#    if output is None:
#        ret = np.zeros(len(antenna_positions))
#    else:
#        ret[:] = 0
#    ret += (antenna_positions[:,0]-X_ave)*(sin_z*cos_a)
#    ret += (antenna_positions[:,1]-Y_ave)*(sin_z*sin_a)
#    ret += (antenna_positions[:,2]-Z_ave)*cos_z
#    
#    ret *= -1.0/v_air
#    
#    return ret
#    
#    
#def signal(T, t_delay, width, frequency, phase=0):
#    shifted = T-t_delay
#    return np.exp( -0.5*(shifted/width)**2 )*np.sin( (2*np.pi*frequency)*shifted + phase)
#
#n_points = int( 1.0*micro/sampling_frequency)
#n_points = 2**( int(np.log2(n_points))+1 )
#print(n_points, "points", n_points*sampling_frequency/micro, "microseconds" )
#
#time_array = np.arange(n_points)*sampling_frequency - n_points*sampling_frequency/2.0
#
#antenna_signals = np.zeros( (len(antenna_positions), n_points) )
#
#for source_i, (source_amp, source_zenith, source_azimuth, delay) in enumerate(sources):
#    print("making source", source_i)
#    print("   cos alpha", np.sin(source_azimuth)*np.sin(source_zenith), "cos beta", np.cos(source_azimuth)*np.sin(source_zenith))
#    time_delays = get_timeDelays(source_zenith, source_azimuth)
#    
#    for antenna_i, (t_delay,trace) in enumerate(zip(time_delays, antenna_signals)):
#        trace += signal(time_array, t_delay+delay*nano, width, frequency)*source_amp
#        
##### add noise here if desired ####
#if noise_amplitude is not None:
#    for trace in antenna_signals:
#        trace += np.random.normal(scale=noise_amplitude, size = n_points)
#        
##### plot the signals ####
#max_amp = 0
#for trace in antenna_signals:
#    max_amp = max(max_amp, np.max(trace))
#    
##for i,trace in enumerate(antenna_signals):
##    plt.plot(time_array, trace + i*max_amp*1.5)
##plt.show()
#
#zero_delays = np.zeros(len(antenna_positions))
#plot_antenna_data(antenna_signals, zero_delays, True)
#plt.show()
#
#### take FFT
#antenna_signals = fftpack.fft(antenna_signals, n=antenna_signals.shape[1]*2, axis=-1)
#
#image = image2D(antenna_signals, antenna_positions, np.zeros(len(antenna_positions)), upsample=upsample)
#    
#print("arg")
#plt.imshow( image, origin='lower' )
#plt.colorbar()
#plt.show()
