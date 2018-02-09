#cython: language_level=3 
#cython: cdivision=True
# cython: boundscheck=True
cimport numpy as np
import numpy as np
from scipy import fftpack
from libc.math cimport sin, cos, M_PI, sqrt
from libc.stdlib cimport malloc, free

cdef extern from "complex.h":
    pass
   # complex conj(complex z)

cdef double inv_v_air = 1.0/(299792458.0/1.000293)

cdef void multiply_inplace(complex[:] A, complex[:] B, int Astart, int Bstart, int num):
    """ perform A *= B """
    cdef int i
    for i in range(num):
        A[Astart+i] = A[Astart+i]*B[Bstart+i]
    

cdef double time_delay(double[:] delta_ant_pos, double cos_alpha, double cos_beta):
    """given the locations of two antennas, and a location in the sky (in cosine projection), return the timedelay between the two antennas"""
    cdef double sin_ze_sq = cos_alpha*cos_alpha + cos_beta*cos_beta
    
    cdef double sin_ze = sqrt( sin_ze_sq )
    cdef double cos_ze = sqrt(1.0-sin_ze_sq)
    
    cdef double sin_az = cos_alpha/sin_ze
    cdef double cos_az = cos_beta/sin_ze
    if sin_ze==0.0:
        sin_az = 0.0
        cos_az = 0.0
    
    cdef double output = 0
    output += delta_ant_pos[0]*sin_ze*cos_az
    output += delta_ant_pos[1]*sin_ze*sin_az
    output += delta_ant_pos[2]*cos_ze
    
    output *= inv_v_air
    
    return output

cdef void correlate(complex[:,:] FFT_data, int ant_i, int ant_j, np.ndarray[complex, ndim=1] out):
    """given two FFT data arrays, upsample, correlate, and inverse fourier transform"""
        #### upsample, and multiply A*conj(B), using as little memory as possible
        
    cdef int in_len = FFT_data.shape[1]
    cdef int out_len = out.shape[0]
    
    cdef int A = 0
    cdef int B = (in_len + 1) // 2
    
    ##set lower positive
    np.conjugate(FFT_data[ant_j, A:B],    out=out[A:B])
    multiply_inplace( out, FFT_data[ant_i], A, A, B-A )
    
    ### all upper frequencies
    A = (in_len + 1) // 2
    B = out_len - (in_len - 1) // 2
    out[A:B] = 0.0

    ## set lower negative
    A = out_len - (in_len - 1) // 2
    B = in_len - (in_len - 1) // 2
    np.conjugate(FFT_data[ant_j, B:],    out=out[A:])
    multiply_inplace( out, FFT_data[ant_i], A, B, (in_len - 1) // 2 )
    
     # special treatment if low number of points is even. So far we have set Y[-N/2]=X[-N/2]
    out[out_len-in_len//2] *= 0.5  # halve the component at -N/2
    cdef complex temp = out[out_len-in_len//2]
    out[in_len//2:in_len//2+1] = temp  # set that equal to the component at -Nx/2
    
    ## ifft in place
    fftpack.ifft(out, n=out_len, overwrite_x=True)

cdef double quad_interp( double Yn1, double Y0, double Y1, double Y2, double i ):
    """given four equaly spaced points, and i between 0 and 1. Use quadratic interpolation to return the point that is the fraction of i between Y0 and Y1"""
    cdef double A = Y0
    cdef double C = (Y1 -2*Y0 +Yn1)*0.5
    cdef double D = (Y2 + 3*Y0 - Yn1 - 3*Y1)/6.0
    cdef double B = Y0 - Yn1 + C - D
    
    cdef double tot = A
   
    return ((D*i + C)*i + B)*i + A

cdef void imager_subloop(double[:,:] image, complex[:] cross_correlation, double[:] delta_ant_pos, double[:] bbox_data, 
                         double ant_delay, double upsampledtime, int num_angles):
    """given the cross correlation between two antennas, loop over all angles and add the result to the image."""
    
    cdef int CC_size = cross_correlation.shape[0]
    
    cdef double cos_alpha = 0.0
    cdef double cos_beta = 0.0
    cdef double dt = 0.0
    
    cdef int A = 0
    cdef int B = 0
    cdef int C = 0
    cdef int D = 0
    cdef double i_frac = 0.0
    
    cdef complex tmpcomlex = 0
    cdef double Ya = 0
    cdef double Yb = 0
    cdef double Yc = 0
    cdef double Yd = 0
    
    cdef int cos_alpha_k
    cdef int cos_beta_l
    for cos_alpha_k in range(num_angles): ##want to multi-thread this
        for cos_beta_l in range(num_angles):
            
            cos_alpha = bbox_data[0]*cos_alpha_k + bbox_data[2]
            cos_beta  = bbox_data[1]*cos_beta_l + bbox_data[3]
            
            if cos_alpha*cos_alpha + cos_beta*cos_beta < 1.0:

                dt = time_delay(delta_ant_pos, cos_alpha, cos_beta  ) - ant_delay
                
                dt /= upsampledtime
                        
                B = <int>dt
                i_frac = dt-B
                        
                A = B-1
                C = B+1
                D = C+2
                        
                ### wrap A, B, C, and D
                while A<0:
                    A += CC_size
                while B<0:
                    B += CC_size
                while C<0:
                    C += CC_size
                while D<0:
                    D += CC_size
                    
                tmpcomlex = cross_correlation[A]
                Ya = tmpcomlex.real
                
                tmpcomlex = cross_correlation[B]
                Yb = tmpcomlex.real
                
                tmpcomlex = cross_correlation[C]
                Yc = tmpcomlex.real
                
                tmpcomlex = cross_correlation[D]
                Yd = tmpcomlex.real
                
                image[cos_alpha_k, cos_beta_l] += quad_interp(Ya, Yb, Yc, Yd, i_frac)
        

def image2D(np.ndarray[complex, ndim=2] antenna_FFT_data, np.ndarray[double, ndim=2] antenna_locations, np.ndarray[double, ndim=1] antenna_delays, 
            np.ndarray[double, ndim=2] bbox = np.array( [[-1.1,1.1],[-1.1,1.1]], dtype=np.double), int num_points=200, int upsample=1, double sampletime=5.0E-9,
            double speed_light=-1):
    
    global inv_v_air
    if speed_light>0:
        inv_v_air = 1.0/speed_light
    
    cdef np.ndarray[double, ndim=2] image = np.zeros((num_points, num_points), dtype=np.double)
    cdef int num_ant = antenna_FFT_data.shape[0]
    
    cdef np.ndarray[double, ndim=1] bbox_info = np.array([(bbox[0,1]-bbox[0,0])/num_points,  (bbox[1,1]-bbox[1,0])/num_points,  bbox[0,0],  bbox[1,0]])
    cdef double upsampletime = sampletime/upsample
    
    cdef int CC_size = antenna_FFT_data.shape[1]*upsample
    cdef np.ndarray[complex, ndim=1] cross_correlation = np.empty( CC_size, dtype=np.complex )
    
    ## memslices for easy passing of data
    cdef complex[:,:] FFT_data_slice = antenna_FFT_data
    cdef double[:,:] image_slice = image
    cdef complex[:] CC_slice = cross_correlation
    cdef double[:] delta_ant_pos= np.zeros(3)
    cdef double [:] BBox_info_slice = bbox_info
    
    cdef int ant_i
    cdef int ant_j
    for ant_i in range(num_ant):
        
        for ant_j in range(ant_i+1,num_ant):
            
            correlate(FFT_data_slice, ant_i, ant_j, cross_correlation)
 
            delta_ant_pos = antenna_locations[ant_j] - antenna_locations[ant_i]
            
            imager_subloop( image_slice, CC_slice, delta_ant_pos, BBox_info_slice, (antenna_delays[ant_j] - antenna_delays[ant_i]), upsampletime, num_points )

    return image
    
    
    
    
    
    