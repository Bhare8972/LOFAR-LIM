#cython: language_level=3 
#cython: cdivision=True
#cython: boundscheck=False
# cython: profile=False

cimport numpy as np
import numpy as np
from scipy import fftpack
from libc.math cimport sqrt,M_PI

cdef extern from "complex.h":
    complex conj (complex z)
    complex cexp (complex z)
    double creal (complex z)

cdef double inv_v_air = 1.0/(299792458.0/1.000293)
    

cdef void zero_a_slice(complex[:] IN):
    cdef int i
    for i in range(IN.shape[0]):
        IN[i] = 0.0
        
cdef double square_N_sum(complex[:] IN):
    cdef double out = 0.0
    
    cdef int i
    for i in range(IN.shape[0]):
        out += creal( i*conj(i) )
        
    return out

cdef double time_delay(double[:,:] antenna_locs, int ant_i, double X, double Y, double Z):
    
    cdef double ant_X = antenna_locs[ant_i,0]
    cdef double ant_Y = antenna_locs[ant_i,1]
    cdef double ant_Z = antenna_locs[ant_i,2]
    
    ant_X -= X
    ant_Y -= Y
    ant_Z -= Z
    
    ant_X *= ant_X
    ant_Y *= ant_Y
    ant_Z *= ant_Z

    return sqrt( ant_X + ant_Y + ant_Z )*inv_v_air

cdef void delayNsum_subloop(complex[:] image, double X, double Y, double Z, complex[:,:] antenna_FFT_data, double[:,:] antenna_locations, double[:] antenna_delays, double[:] frequencies):
    
    cdef double dt
    cdef complex shift_number
    
    cdef int ant_i
    cdef int freq_i
    for ant_i in range(antenna_FFT_data.shape[0]):
        dt = time_delay(antenna_locations,ant_i, X,Y,Z) - antenna_delays[ant_i]
        shift_number = -1j*2*M_PI*dt/5.0E-9
        
        for freq_i in range(image.shape[0]):
            image[freq_i] += antenna_FFT_data[ant_i,freq_i]*cexp( shift_number*frequencies[freq_i] )
    


def beamform_3D(np.ndarray[complex, ndim=2] antenna_FFT_data, np.ndarray[double, ndim=2] antenna_locations, np.ndarray[double, ndim=1] antenna_delays, 
            np.ndarray[double, ndim=2] bbox, np.ndarray[int, ndim=1] num_points):
    
    cdef np.ndarray[double, ndim=3] image = np.zeros((num_points[0], num_points[1], num_points[2]), dtype=np.double)
    
    cdef complex[:,:] FFT_data_slice = antenna_FFT_data
    cdef double[:,:] antenna_locs_slice = antenna_locations
    cdef double[:] antenna_delays_slice = antenna_delays
    
    cdef complex[:] TMP_image_array = np.zeros(antenna_FFT_data.shape[1], dtype=complex)
    cdef complex[:] TMP_time_image
    cdef double[:] frequencies = fftpack.fftfreq(antenna_FFT_data.shape[1])
    
    
    cdef double delta_X = (bbox[0,1]-bbox[0,0])/num_points[0]
    cdef double delta_Y = (bbox[1,1]-bbox[1,0])/num_points[1]
    cdef double delta_Z = (bbox[2,1]-bbox[2,0])/num_points[2]
    
    cdef double X0 = bbox[0,0]
    cdef double Y0 = bbox[1,0]
    cdef double Z0 = bbox[2,0]
    
    cdef double X
    cdef double Y
    cdef double Z
    
    cdef int X_index
    cdef int Y_index
    cdef int Z_index
    for X_index in range(num_points[0]):
        X = X_index*delta_X + X0
        
        for Y_index in range(num_points[1]):
            Y = Y_index*delta_Y + Y0
            print(X_index, Y_index)
        
            for Z_index in range(num_points[2]):
                Z = Z_index*delta_Z + Z0
                
                zero_a_slice(TMP_image_array)
    
                delayNsum_subloop(TMP_image_array, X,Y,Z,   FFT_data_slice, antenna_locs_slice, antenna_delays_slice, frequencies)
                
                TMP_time_image = fftpack.ifft(TMP_image_array)
                
                image[X_index, Y_index, Z_index] = square_N_sum(TMP_time_image)
                
    return image
    
    
    
    
    
    
    