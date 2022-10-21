#!/usr/bin/env python3

#cython: language_level=3 
#cython: cdivision=False
#cython: boundscheck=True

#cython: linetrace=False
#cython: binding=False
#cython: profile=False

cimport numpy as np
import numpy as np
cimport cython
# from libc.math cimport sqrt, log, sin, cos, acos, atan2, M_PI
# from libc.stdlib cimport malloc, free
#cimport scipy.linalg.cython_lapack as lp
#cimport scipy.linalg.cython_blas as bs

# from libc.stdlib cimport malloc, free

# cdef extern from "complex.h" nogil:
#     double complex cexp(double complex)
#     double cabs(double complex z)
#     double cimag(double complex)
#     double creal(double complex)
#     double carg(double complex arg) ## returns phase in radians
#     double complex conj( double complex z )
    
    
    
cdef extern from "gsl/gsl_fft_complex.h" nogil:
    ctypedef struct gsl_fft_complex_wavetable:
        size_t n
        size_t nf
        size_t factor[64]
        complex* trig
        complex* twiddle[64]


    gsl_fft_complex_wavetable* gsl_fft_complex_wavetable_alloc(size_t n)
    void gsl_fft_complex_wavetable_free(gsl_fft_complex_wavetable* wavetable)
    void* gsl_fft_complex_workspace_alloc(size_t n)
    void gsl_fft_complex_workspace_free(void* workspace)
    int gsl_fft_complex_forward(double* data, size_t stride, size_t n, const gsl_fft_complex_wavetable* wavetable, void* work)
    int gsl_fft_complex_inverse(double* data, size_t stride, size_t n, const gsl_fft_complex_wavetable* wavetable, void* work)



cdef struct GSL_complexFFT_struct:
    gsl_fft_complex_wavetable* FFT_wavetable
    void* FFT_workspace

cdef int initialize_GSL_complexFFT_struct(GSL_complexFFT_struct* S, int N) nogil:
    if N < 0:
        return -1

    S.FFT_wavetable = gsl_fft_complex_wavetable_alloc( N )
    S.FFT_workspace = gsl_fft_complex_workspace_alloc( N )

    if not S.FFT_wavetable:
        return 1
    if not S.FFT_workspace:
        return 2

    return 0

cdef void dealloc_GSL_complexFFT_struct(GSL_complexFFT_struct* S):
    if S.FFT_wavetable:
        gsl_fft_complex_wavetable_free( S.FFT_wavetable )
    if S.FFT_workspace:
        gsl_fft_complex_workspace_free( S.FFT_workspace )

cdef int num_factors(GSL_complexFFT_struct* S) nogil:
    return S.FFT_wavetable.nf

cdef int GSL_FFT(GSL_complexFFT_struct* S, double* data, int stride) nogil:
    return gsl_fft_complex_forward(data, stride,  S.FFT_wavetable.n,  S.FFT_wavetable, S.FFT_workspace)

cdef int GSL_IFFT(GSL_complexFFT_struct* S, double* data, int stride) nogil:
    return gsl_fft_complex_inverse(data, stride,  S.FFT_wavetable.n,  S.FFT_wavetable, S.FFT_workspace)




## PYTHON INTERFACE
cdef class GSL_complex_FFT:
    cdef GSL_complexFFT_struct data_struct

    def __init__(self, int N):
        initialize_GSL_complexFFT_struct( &self.data_struct, N )

    def __dealloc__(self):
        dealloc_GSL_complexFFT_struct( &self.data_struct )

    def get_num_factors(self):
        return num_factors( &self.data_struct )

    def do_FFT(self, np.ndarray[double complex, ndim=1] data  ):
        return GSL_FFT( &self.data_struct, <double*>&data[0], int( data.strides[0]/data.itemsize ) )

    def do_IFFT(self, np.ndarray[double complex, ndim=1] data  ):
        return GSL_IFFT( &self.data_struct, <double*>&data[0], int( data.strides[0]/data.itemsize ) )



    