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
    
    
def rawDataStatistics(np.ndarray[np.int16_t, ndim=1] data, int satMax=0, int satMin=0):
    """returns maximum of data, number of saturated samples, and number samples that are zero and followed by a zero"""

    cdef np.int16_t[:] dataMemoryview = data
    cdef int length = len(data)

    cdef np.int16_t current_max = 0
    cdef int num_saturation = 0
    cdef int num_dbl_zeros = 0

    cdef int previousIsZero = 0

    cdef int i
    cdef np.int16_t value
    for i in range(length):
        value = dataMemoryview[i]

        if value == 0:
            if previousIsZero == 1:
                num_dbl_zeros += 1
            previousIsZero = 1

        else:
            if (value >= satMax) or (value<=satMin):
                num_saturation += 1

            if value > current_max:
                current_max = value
            elif (-value ) > current_max:
                current_max = -value

            previousIsZero = 0

    return num_saturation, num_dbl_zeros, current_max



    