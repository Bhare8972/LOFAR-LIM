#cython: language_level=3 
#cython: cdivision=True
#cython: boundscheck=False
# cython: linetrace=False
# cython: binding=False
# cython: profile=True
cimport numpy as np
import numpy as np
from scipy import fftpack
from libc.math cimport sqrt

cdef extern from "complex.h":
    pass

cdef double inv_v_air = 1.0/(299792458.0/1.000293)
    
cdef void multiply_inplace( double complex[:] A, double complex[:] B, int Astart, int Bstart, int num):
    """ perform A *= B """
    cdef int i
    for i in range(num):
        A[Astart+i] = A[Astart+i]*B[Bstart+i]
    
cdef void correlate(double complex[:,:] FFT_data, int ant_i, int ant_j, np.ndarray[double complex, ndim=1] out):
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
    cdef double complex temp = out[out_len-in_len//2]
    out[in_len//2:in_len//2+1] = temp  # set that equal to the component at -Nx/2
    
    ## ifft in place
    fftpack.ifft(out, n=out_len, overwrite_x=True)

cdef inline double quad_interp( double Yn1, double Y0, double Y1, double Y2, double i ):
    """given four equaly spaced points, and i between 0 and 1. Use quadratic interpolation to return the point that is the fraction of i between Y0 and Y1"""
    cdef double A = Y0
    cdef double C = (Y1 -2*Y0 +Yn1)*0.5
    cdef double D = (Y2 + 3*Y0 - Yn1 - 3*Y1)/6.0
    cdef double B = Y0 - Yn1 + C - D
   
    return ((D*i + C)*i + B)*i + A

cdef inline double linear_interp( double Y0, double Y1, double i ):
    """given two points, and i between 0 and 1. Use quadratic interpolation to return the point that is the fraction of i between Y0 and Y1"""
   
    return (Y1-Y0)*i + Y0

cdef inline double time_delay(double[:] ant_i_loc, double[:] ant_j_loc, double X, double Y, double Z):
    
    cdef double ds_i = sqrt( (ant_i_loc[0]-X)*(ant_i_loc[0]-X) + (ant_i_loc[1]-Y)*(ant_i_loc[1]-Y) + (ant_i_loc[2]-Z)*(ant_i_loc[2]-Z) )
    cdef double ds_j = sqrt( (ant_j_loc[0]-X)*(ant_j_loc[0]-X) + (ant_j_loc[1]-Y)*(ant_j_loc[1]-Y) + (ant_j_loc[2]-Z)*(ant_j_loc[2]-Z) )
    
    return (ds_j - ds_i)*inv_v_air

cdef void imager_subloop(double[:,:,:] image, double complex[:] cross_correlation, double[:] ant_i_loc, double[:] ant_j_loc, double[:] bbox_data, 
                    int[:] num_points, double ant_delay, double upsampledtime):
    
    cdef int CC_size = cross_correlation.shape[0]
    
    cdef int A = 0
    cdef int B = 0
    cdef int C = 0
    cdef int D = 0
    cdef double i_frac = 0.0
    
    cdef double complex tmpcomlex = 0
    cdef double Ya = 0
    cdef double Yb = 0
    cdef double Yc = 0
    cdef double Yd = 0
    
    cdef double X
    cdef double Y
    cdef double Z
    cdef double dt
    
    cdef int X_pos_i
    cdef int Y_pos_j
    cdef int Z_pos_k
    for X_pos_i in range(num_points[0]): ##want to multi-thread this
        for Y_pos_j in range(num_points[1]):
            for Z_pos_k in range(num_points[2]):
                
                X = bbox_data[0]*X_pos_i + bbox_data[1]
                Y = bbox_data[2]*Y_pos_j + bbox_data[3]
                Z = bbox_data[4]*Z_pos_k + bbox_data[5]
                
                dt = -time_delay(ant_i_loc, ant_j_loc, X, Y, Z ) + ant_delay
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
                
                image[X_pos_i, Y_pos_j, Z_pos_k] += quad_interp(Ya, Yb, Yc, Yd, i_frac)
#                image[X_pos_i, Y_pos_j, Z_pos_k] += linear_interp(Yb, Yc, i_frac)
    

def imager_3D(np.ndarray[double complex, ndim=2] antenna_FFT_data, np.ndarray[double, ndim=2] antenna_locations, np.ndarray[double, ndim=1] antenna_delays, 
            np.ndarray[double, ndim=2] bbox, np.ndarray[int, ndim=1] num_points, int upsample=1, double max_baseline=np.inf, double sampletime=5.0E-9):
    
    cdef np.ndarray[double, ndim=3] image = np.zeros((num_points[0], num_points[1], num_points[2]), dtype=np.double)
    cdef int num_ant = antenna_FFT_data.shape[0]
    
    cdef np.ndarray[double, ndim=1] bbox_info = np.array([(bbox[0,1]-bbox[0,0])/num_points[0], bbox[0,0],
                                                          (bbox[1,1]-bbox[1,0])/num_points[1], bbox[1,0],
                                                          (bbox[2,1]-bbox[2,0])/num_points[2], bbox[2,0] ])
    
    cdef double upsampletime = sampletime/upsample
    cdef int CC_size = antenna_FFT_data.shape[1]*upsample
    cdef np.ndarray[double complex, ndim=1] cross_correlation = np.empty( CC_size, dtype=np.complex )
    
    ## memslices for easy passing of data
    cdef double complex[:,:] FFT_data_slice = antenna_FFT_data
    cdef double[:,:,:] image_slice = image
    cdef double complex[:] CC_slice = cross_correlation
    cdef double [:] BBox_info_slice = bbox_info
    cdef int[:] num_points_slice = num_points
    
    cdef double [:] ant_i_loc
    cdef double [:] ant_j_loc
    cdef double del_X
    cdef double del_Y
    cdef double del_Z
    cdef double max_baseline_sq = max_baseline*max_baseline
    
    cdef double ant_delay
    
    cdef int ant_i
    cdef int ant_j
    for ant_i in range(num_ant):
        ant_i_loc = antenna_locations[ant_i]
            
        print("imaging", ant_i, "out of", num_ant)
        
        for ant_j in range(ant_i+1,num_ant):
            ant_j_loc = antenna_locations[ant_j]
            
            del_X = ant_i_loc[0] - ant_j_loc[0]
            del_Y = ant_i_loc[1] - ant_j_loc[1]
            del_Z = ant_i_loc[2] - ant_j_loc[2]
            if del_X*del_X + del_Y*del_Y + del_Z*del_Z > max_baseline_sq:
                continue
            
            
            correlate(FFT_data_slice, ant_i, ant_j, cross_correlation)
 
            ant_delay = antenna_delays[ant_i] - antenna_delays[ant_j]
            
            imager_subloop( image_slice, CC_slice, ant_i_loc, ant_j_loc, BBox_info_slice, num_points_slice, ant_delay, upsampletime)
            
    return image
    
def imager_3D_bypairs(np.ndarray[double complex, ndim=2] antenna_FFT_data, np.ndarray[double, ndim=2] antenna_locations, np.ndarray[double, ndim=1] antenna_delays, 
            np.ndarray[long, ndim=2] pairs, np.ndarray[double, ndim=2] bbox, np.ndarray[int, ndim=1] num_points, int upsample=1, double sampletime=5.0E-9, int report_spacing=50):
    
    cdef np.ndarray[double, ndim=3] image = np.zeros((num_points[0], num_points[1], num_points[2]), dtype=np.double)
    cdef int num_ant = antenna_FFT_data.shape[0]
    
    cdef np.ndarray[double, ndim=1] bbox_info = np.array([(bbox[0,1]-bbox[0,0])/num_points[0], bbox[0,0],
                                                          (bbox[1,1]-bbox[1,0])/num_points[1], bbox[1,0],
                                                          (bbox[2,1]-bbox[2,0])/num_points[2], bbox[2,0] ])
    
    cdef double upsampletime = sampletime/upsample
    cdef int CC_size = antenna_FFT_data.shape[1]*upsample
    cdef np.ndarray[double complex, ndim=1] cross_correlation = np.empty( CC_size, dtype=np.complex )
    
    ## memslices for easy passing of data
    cdef double complex[:,:] FFT_data_slice = antenna_FFT_data
    cdef double[:,:,:] image_slice = image
    cdef double complex[:] CC_slice = cross_correlation
    cdef double [:] BBox_info_slice = bbox_info
    cdef int[:] num_points_slice = num_points
    
    cdef double [:] ant_i_loc
    cdef double [:] ant_j_loc
    
    cdef double ant_delay
    
    cdef int ant_i
    cdef int ant_j
    cdef int pair_i
    cdef long[:,:] pair_slice = pairs
    for pair_i in range(pair_slice.shape[0]):
        ant_i = pair_slice[pair_i, 0]
        ant_j = pair_slice[pair_i, 1]
    
        ant_i_loc = antenna_locations[ant_i]
        ant_j_loc = antenna_locations[ant_j]
        
        if (pair_i%report_spacing)==0:
            print("imaging pair", pair_i, "out of", pair_slice.shape[0])
            
        correlate(FFT_data_slice, ant_i, ant_j, cross_correlation)
 
        ant_delay = antenna_delays[ant_i] - antenna_delays[ant_j]
        
        imager_subloop( image_slice, CC_slice, ant_i_loc, ant_j_loc, BBox_info_slice, num_points_slice, ant_delay, upsampletime)
            
    return image

    