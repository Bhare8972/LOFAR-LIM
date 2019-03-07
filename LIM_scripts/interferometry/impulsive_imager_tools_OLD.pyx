#cython: language_level=3 
#cython: cdivision=True
#cython: boundscheck=False
#cython: linetrace=False
#cython: binding=False
#cython: profile=False


cimport numpy as np
import numpy as np
from scipy import fftpack
from libc.math cimport sqrt,log
from libc.stdlib cimport malloc, free

from cython.parallel import prange


cdef extern from "complex.h" nogil:
    double creal(double complex z)
    double cabs( double complex z )
    double cimag( double complex z )

cdef extern from "gsl/gsl_math.h" nogil:
    int gsl_finite(const double x)
    double gsl_hypot(const double x, const double y)
    
cdef extern from "gsl/gsl_vector.h" nogil:
    
    void* gsl_vector_alloc(size_t n)
    void gsl_vector_free(void* v)
    
    double gsl_vector_get(const void* v, const size_t i)
    void gsl_vector_set(void* v, const size_t i, double x)
    double* gsl_vector_ptr(void* v, size_t i)
    
    void gsl_vector_set_all(void* v, double x)
    

cdef extern from "gsl/gsl_fft_complex.h" nogil:
    int gsl_fft_complex_radix2_forward(double* data, size_t stride, size_t n)
    int gsl_fft_complex_radix2_inverse(double* data, size_t stride, size_t n)

cdef extern from "gsl/gsl_spline.h" nogil:
    cdef void* gsl_interp_cspline_periodic
    cdef void* gsl_interp_linear
    void* gsl_spline_alloc( void* interp_type, size_t size)
    void gsl_spline_free(void* spline)
    
    int gsl_spline_init(void* spline, double* xa, double* ya, size_t size)
    double gsl_spline_eval(void* spline, double x, void* acc) 
    double gsl_spline_eval_deriv(void* spline, double x, void* acc)
    
#cdef extern from "gsl/gsl_multimin.h" nogil:
#    int GSL_CONTINUE
#    int GSL_SUCCESS
#    int GSL_ENOPROG
#    
#    ctypedef struct gsl_multimin_function:
#        double (* f) (void* x, void* params)
#        size_t n
#        void * params
#        
#    ctypedef struct gsl_multimin_fminimizer:
#        pass
#    
#    ctypedef struct gsl_multimin_fminimizer_type:
#        pass
#    
#    gsl_multimin_fminimizer_type* gsl_multimin_fminimizer_nmsimplex2
#        
#    gsl_multimin_fminimizer* gsl_multimin_fminimizer_alloc(const gsl_multimin_fminimizer_type* T, size_t n)
#    void gsl_multimin_fminimizer_free(gsl_multimin_fminimizer*  s)
#    
#    int gsl_multimin_fminimizer_set(gsl_multimin_fminimizer* s, gsl_multimin_function* f, void* x, void* step_size)
#    int gsl_multimin_fminimizer_iterate(gsl_multimin_fminimizer* s)
#    
#    void* gsl_multimin_fminimizer_x(const gsl_multimin_fminimizer* s)
#    double gsl_multimin_fminimizer_minimum(const gsl_multimin_fminimizer* s)
#    double gsl_multimin_fminimizer_size(const gsl_multimin_fminimizer* s)
#    
#    int gsl_multimin_test_size(const double size, double epsabs)
    

#cdef extern from "gsl/gsl_siman.h" nogil:
#
#    ctypedef struct gsl_siman_params_t:
#        int n_tries ## points to try for each step
#        int iters_fixed_T ## num itterations at each temp
#        double step_size ## max step size in random walk
#        double k, t_initial, mu_t, t_min ##boltzman distribution params
#    
#    ##energy of a location
#    ctypedef double (*gsl_siman_Efunc_t) (void *xp)  
#    
#    ##change location randomly up to a step size
#    ctypedef void (*gsl_siman_step_t) (const gsl_rng *r, void *xp, double step_size) 
#    
#    ##distance between two configurations
#    ctypedef double (*gsl_siman_metric_t) (void *xp, void *yp)
#    
#    ##print configuration. Null if no printing wanted
#    ctypedef void (*gsl_siman_print_t) (void *xp)
#    
#    ## copy source to destination Null if size of xp is constant
#    ctypedef void (*gsl_siman_copy_t) (void *source, void *dest)
#    
#    ## make new copy of xpNull if size of xp is constant
#    ctypedef void * (*gsl_siman_copy_construct_t) (void *xp)
#    
#    ## dealoc xp. Null if size of xp is constant
#    ctypedef void (*gsl_siman_destroy_t) (void *xp)
#    
#    void gsl_siman_solve(const gsl_rng * r, 
#                     void *x0_p, gsl_siman_Efunc_t Ef,
#                     gsl_siman_step_t take_step,
#                     gsl_siman_metric_t distance,
#                     gsl_siman_print_t print_position,
#                     gsl_siman_copy_t copyfunc,
#                     gsl_siman_copy_construct_t copy_constructor,
#                     gsl_siman_destroy_t destructor,
#                     size_t element_size,
#                     gsl_siman_params_t params)
#    
    

cdef double c_air_inverse = 1.000293/299792458.0

cdef double vec_3_DiffNorm(double* XYZ1, double* XYZ2) nogil:
    cdef double dif1 = XYZ1[0] - XYZ2[0]
    cdef double dif2 = XYZ1[1] - XYZ2[1]
    cdef double dif3 = XYZ1[2] - XYZ2[2]
    return sqrt( dif1*dif1 + dif2*dif2 + dif3*dif3 )

cdef inline double real_max(double complex *A, int N) nogil:
    cdef int i
    cdef double cur_max = creal(A[0])
    cdef double tmp
    for i in range(N):
        tmp = creal(A[i])
        if tmp > cur_max:
            cur_max = tmp
            
    return cur_max
    

##### info to store image, C-style! #####
    
cdef struct image_data_struct:
    ## universal
    int num_antennas
    int FFT_length
    int correlation_length
    int upsample_factor
    double[:,:] antenna_locations
    double[:] antenna_delays
    
    void** splines
    double complex[:,:] data_FFT
    
    double[:] interpolation_time_space
    
    ##stage 2
    double [:] additional_delays

#    ##test
#    bint have_error
#    double X
#    double Y
#    double Z
    
cdef void cross_correlate(image_data_struct* image_data, int ant_i, int ant_j, np.ndarray[double complex, ndim=1] out):
    
    cdef int out_len = out.shape[0]
    
    cdef double complex[:] FFTdata_i = image_data.data_FFT[ant_i]
    cdef double complex[:] FFTdata_j = image_data.data_FFT[ant_j]
    
    cdef int A = 0
    cdef int B = (image_data.FFT_length + 1) // 2
    
    ##set lower positive
    np.conjugate(FFTdata_j[A:B],    out=out[A:B])
    out[A:B] *= FFTdata_i[A:B]
    
    ### all upper frequencies
    A = (image_data.FFT_length + 1) // 2
    B = image_data.correlation_length - (image_data.FFT_length - 1) // 2
    out[A:B] = 0.0

    ## set lower negative
    A = image_data.correlation_length - (image_data.FFT_length - 1) // 2
    B = image_data.FFT_length - (image_data.FFT_length - 1) // 2
    np.conjugate(FFTdata_j[B:],    out=out[A:])
    out[A:] *= FFTdata_i[B:]
    
     # special treatment if low number of points is even. So far we have set Y[-N/2]=X[-N/2]
#        out[self.correlation_length-self.FFT_length//2] *= 0.5  # halve the component at -N/2
    out[image_data.correlation_length-image_data.FFT_length//2] =  out[image_data.correlation_length-image_data.FFT_length//2]*0.5  # halve the component at -N/2
    temp = out[image_data.correlation_length-image_data.FFT_length//2]
    out[image_data.FFT_length//2:image_data.FFT_length//2+1] = temp  # set that equal to the component at -Nx/2
    
    ## ifft in place
    gsl_fft_complex_radix2_inverse(<double*>&out[0], 1, image_data.correlation_length)
    
cdef double stage1_image_intensity(image_data_struct* image_data, double* XYZ, int prefered_antenna) nogil:
    cdef int ant_i
    cdef double ret = 0.0
    cdef double modeled_dt
    for ant_i in range( image_data.num_antennas ):
        if ant_i == prefered_antenna:
            continue            
        
        modeled_dt = vec_3_DiffNorm(&image_data.antenna_locations[ prefered_antenna,0], XYZ)
        modeled_dt -= vec_3_DiffNorm(&image_data.antenna_locations[ant_i,0], XYZ)
        modeled_dt *= c_air_inverse
        modeled_dt += image_data.antenna_delays[prefered_antenna] - image_data.antenna_delays[ant_i]
        modeled_dt *= image_data.upsample_factor/5.0E-9
        
        while modeled_dt < 0:
            modeled_dt += image_data.correlation_length
        while modeled_dt >= image_data.correlation_length:
            modeled_dt -= image_data.correlation_length
            
#        if modeled_dt > (image_data.correlation_length-1):
#            modeled_dt = 0
            
        ret += gsl_spline_eval(image_data.splines[ant_i], modeled_dt, NULL) 
        
    return -ret
    
#cdef double stage2_image_intensity(image_data_struct* image_data, double* XYZ) nogil:
#    cdef int ant_i
#    cdef int ant_j
#    cdef int corr_i=0
#    cdef double ret = 0.0
#    cdef double modeled_dt
#    for ant_i in range( image_data.num_antennas ):
#        for ant_j in range(ant_i+1, image_data.num_antennas ):
#        
#            modeled_dt = vec_3_DiffNorm(&image_data.antenna_locations[ ant_i,0], XYZ)
#            modeled_dt -= vec_3_DiffNorm(&image_data.antenna_locations[ant_j,0], XYZ)
#            modeled_dt *= c_air_inverse
#            modeled_dt += image_data.antenna_delays[ ant_i ] - image_data.antenna_delays[ant_j]
#            modeled_dt += image_data.additional_delays[ ant_i ] - image_data.additional_delays[ant_j]
#            modeled_dt *= image_data.upsample_factor/5.0E-9
#            
#            while modeled_dt < 0:
#                modeled_dt += image_data.correlation_length
#            while modeled_dt >= image_data.correlation_length:
#                modeled_dt -= image_data.correlation_length
#            
#            ret += gsl_spline_eval(image_data.splines[corr_i], modeled_dt, NULL) 
#            corr_i += 1
#        
#    return -ret   
 
cdef double stage2_image_intensity_absAfter(image_data_struct* image_data, double* XYZ) nogil:
    cdef int ant_i
    cdef int ant_j
    cdef int corr_i=0
    cdef double ret_real = 0.0
    cdef double ret_imag = 0.0
    cdef double modeled_dt
    for ant_i in range( image_data.num_antennas ):
        for ant_j in range(ant_i+1, image_data.num_antennas ):
        
            modeled_dt = vec_3_DiffNorm(&image_data.antenna_locations[ ant_i,0], XYZ)
            modeled_dt -= vec_3_DiffNorm(&image_data.antenna_locations[ant_j,0], XYZ)
            modeled_dt *= c_air_inverse
            modeled_dt += image_data.antenna_delays[ ant_i ] - image_data.antenna_delays[ant_j]
            modeled_dt += image_data.additional_delays[ ant_i ] - image_data.additional_delays[ant_j]
            modeled_dt *= image_data.upsample_factor/5.0E-9
            
            while modeled_dt < 0:
                modeled_dt += image_data.correlation_length
            while modeled_dt >= image_data.correlation_length:
                modeled_dt -= image_data.correlation_length
            
            ret_real += gsl_spline_eval(image_data.splines[corr_i*2], modeled_dt, NULL) 
            ret_imag += gsl_spline_eval(image_data.splines[corr_i*2+1], modeled_dt, NULL) 
            corr_i += 1
        
    return - gsl_hypot(ret_real, ret_imag)
 
cdef double stage2_image_intensity_absBefore(image_data_struct* image_data, double* XYZ) nogil:
    cdef int ant_i
    cdef int ant_j
    cdef int corr_i=0
    cdef double ret = 0.0
    cdef double tmp_real
    cdef double tmp_imag
    cdef double modeled_dt
    for ant_i in range( image_data.num_antennas ):
        for ant_j in range(ant_i+1, image_data.num_antennas ):
        
            modeled_dt = vec_3_DiffNorm(&image_data.antenna_locations[ ant_i,0], XYZ)
            modeled_dt -= vec_3_DiffNorm(&image_data.antenna_locations[ant_j,0], XYZ)
            modeled_dt *= c_air_inverse
            modeled_dt += image_data.antenna_delays[ ant_i ] - image_data.antenna_delays[ant_j]
            modeled_dt += image_data.additional_delays[ ant_i ] - image_data.additional_delays[ant_j]
            modeled_dt *= image_data.upsample_factor/5.0E-9
            
            while modeled_dt < 0:
                modeled_dt += image_data.correlation_length
            while modeled_dt >= image_data.correlation_length:
                modeled_dt -= image_data.correlation_length
            
            tmp_real = gsl_spline_eval(image_data.splines[corr_i*2], modeled_dt, NULL) 
            tmp_imag = gsl_spline_eval(image_data.splines[corr_i*2+1], modeled_dt, NULL)
            ret += gsl_hypot(tmp_real, tmp_imag)
            
            corr_i += 1
        
    return -ret
 
cdef double stage2_image_intensity_real(image_data_struct* image_data, double* XYZ) nogil:
    cdef int ant_i
    cdef int ant_j
    cdef int corr_i=0
    cdef double ret = 0.0
    cdef double modeled_dt
    for ant_i in range( image_data.num_antennas ):
        for ant_j in range(ant_i+1, image_data.num_antennas ):
        
            modeled_dt = vec_3_DiffNorm(&image_data.antenna_locations[ ant_i,0], XYZ)
            modeled_dt -= vec_3_DiffNorm(&image_data.antenna_locations[ant_j,0], XYZ)
            modeled_dt *= c_air_inverse
            modeled_dt += image_data.antenna_delays[ ant_i ] - image_data.antenna_delays[ant_j]
            modeled_dt += image_data.additional_delays[ ant_i ] - image_data.additional_delays[ant_j]
            modeled_dt *= image_data.upsample_factor/5.0E-9
            
            while modeled_dt < 0:
                modeled_dt += image_data.correlation_length
            while modeled_dt >= image_data.correlation_length:
                modeled_dt -= image_data.correlation_length
            
            ret += gsl_spline_eval(image_data.splines[corr_i*2], modeled_dt, NULL) 
            
            corr_i += 1
        
    return -ret


    
#cdef void stage1_image_gradient(image_data_struct* image_data, double* XYZ, double* gradient_out):
#    gradient_out[0] = 0
#    gradient_out[1] = 0
#    gradient_out[2] = 0
#    cdef int ant_i
#    cdef double dsdt
#    cdef double norm_1
#    cdef double norm_2
#    cdef double del_1
#    cdef double del_2
#    cdef double modeled_dt
#    cdef double Tratio = image_data.upsample_factor/5.0E-9
#    for ant_i in range( image_data.num_antennas ):
#        if ant_i == image_data.prefered_antenna:
#            continue
#        
#        norm_1 = vec_3_DiffNorm(&image_data.antenna_locations[ image_data.prefered_antenna,0], XYZ)
#        norm_2 = vec_3_DiffNorm(&image_data.antenna_locations[ant_i,0], XYZ)
#        
#        modeled_dt = norm_1 - norm_2
#        modeled_dt *= c_air_inverse
#        modeled_dt += image_data.antenna_delays[image_data.prefered_antenna] - image_data.antenna_delays[ant_i]
#        modeled_dt *= Tratio
#        
#        while modeled_dt < 0:
#            modeled_dt += image_data.correlation_length
#        while modeled_dt >= image_data.correlation_length:
#            modeled_dt -= image_data.correlation_length
#            
##        if modeled_dt > (image_data.correlation_length-1):
##            modeled_dt = 0
#        
#        dsdt = gsl_spline_eval_deriv(image_data.splines[ant_i], modeled_dt, NULL) 
#        
#        del_1 = XYZ[0]-image_data.antenna_locations[ image_data.prefered_antenna,0]
#        del_2 = XYZ[0]-image_data.antenna_locations[ant_i,0]
#        gradient_out[0] -= dsdt*Tratio*c_air_inverse*( ((del_1)/norm_1) - ((del_2)/norm_2)   ) ##note negative sign, becouse we are minimizing
#        
#        del_1 = XYZ[1]-image_data.antenna_locations[ image_data.prefered_antenna,1]
#        del_2 = XYZ[1]-image_data.antenna_locations[ant_i,1]
#        gradient_out[1] -= dsdt*Tratio*c_air_inverse*( ((del_1)/norm_1) - ((del_2)/norm_2)   )
#        
#        del_1 = XYZ[2]-image_data.antenna_locations[ image_data.prefered_antenna,2]
#        del_2 = XYZ[2]-image_data.antenna_locations[ant_i,2]
#        gradient_out[2] -= dsdt*Tratio*c_air_inverse*( ((del_1)/norm_1) - ((del_2)/norm_2)   )
        
    
cdef class image_data_wrapper:
    cdef image_data_struct image_data
    
    
#cdef class image_data_stage1(image_data_wrapper):
#    
#    cdef np.ndarray  interpolation_workpace
#    cdef np.ndarray correlation_workspace
#    cdef int prefered_antenna
#    
#    def __init__(self, np.ndarray[double, ndim=2] _antenna_locations, np.ndarray[double, ndim=1] _antenna_delays, int num_data_points, int _upsample_factor):
#        
#        self.image_data.num_antennas = _antenna_locations.shape[0]
#        self.image_data.FFT_length = num_data_points*2
#        self.image_data.correlation_length = _upsample_factor*self.image_data.FFT_length
#        self.image_data.upsample_factor = _upsample_factor
#        self.image_data.antenna_locations = _antenna_locations
#        self.image_data.antenna_delays = _antenna_delays
#        
#        ##allocate memory for FFT data
#        self.image_data.data_FFT = np.empty([self.image_data.num_antennas, self.image_data.FFT_length], dtype=np.complex)
#        
#        ##allocate memory for the correlation workspace
#        self.correlation_workspace = np.empty(self.image_data.correlation_length, dtype=np.complex)
#        self.interpolation_workpace = np.empty(self.image_data.correlation_length+1, dtype=np.double)
#        self.image_data.interpolation_time_space = np.arange(self.image_data.correlation_length+1, dtype=np.double)
##        self.interpolation_workpace = np.empty(self.image_data.correlation_length, dtype=np.double)
##        self.image_data.interpolation_time_space = np.arange(self.image_data.correlation_length, dtype=np.double)
#        
#        ## allocate memory for the splines
#        self.image_data.splines = <void**>malloc(sizeof(void*)*self.image_data.num_antennas)
#        cdef int ant_i
#        for ant_i in range(self.image_data.num_antennas):
#            self.image_data.splines[ant_i] = gsl_spline_alloc(gsl_interp_cspline_periodic, self.image_data.correlation_length+1)
##            self.image_data.splines[ant_i] = gsl_spline_alloc(gsl_interp_linear, self.image_data.correlation_length)
#          
#            
#    def __dealloc__(self):
#        
#        cdef int ant_i
#        for ant_i in range(self.image_data.num_antennas):
#            gsl_spline_free( self.image_data.splines[ant_i] )
#            
#        free(<void*>self.image_data.splines)
#        
#    def intensity(self, np.ndarray[double, ndim=1] XYZ):
#        return stage1_image_intensity(&self.image_data, &XYZ[0], self.prefered_antenna)
#    
#    def intensity_multiprocessed(self, np.ndarray[double, ndim=2] XYZs, np.ndarray[double, ndim=1] image_out, int num_threads):
#        cdef int i
#        cdef double[:,:] XYZs_view = XYZs
#        cdef double[:] image_out_view = image_out
#        for i in prange(XYZs.shape[0], nogil=True, num_threads=num_threads):
##        for i in range(XYZs.shape[0]):
#            image_out_view[i] = stage1_image_intensity(&self.image_data, &XYZs_view[i, 0], self.prefered_antenna)
#    
##    def gradient(self, np.ndarray[double, ndim=1] XYZ):
##        cdef np.ndarray[double, ndim=1] out = np.empty(3, dtype=np.double)
##        stage1_image_gradient(&self.image_data, &XYZ[0], &out[0])
##        return out
#            
#    def set_data(self, np.ndarray[double complex, ndim=1] data, int ant_i):
#        
#        #### copy data ####
#        A = self.image_data.data_FFT[ant_i,:data.shape[0]]
#        A[:] = data
#        
#        A = self.image_data.data_FFT[ant_i,data.shape[0]:] 
#        A[:] = 0.0
#        
#    def prepare_image(self, prefered_antenna, whitener=None, min_rel_amp=1.0E-10):
#        self.prefered_antenna = prefered_antenna
#        #### FFT of all data ####
#        cdef int ant_i
#        cdef int point_i = 0
#        for ant_i in range( self.image_data.num_antennas ):
#            gsl_fft_complex_radix2_forward(<double*>&self.image_data.data_FFT[ant_i,0], 1, self.image_data.data_FFT.shape[1])
#            
#            if whitener is not None:
#                ABS_spectra =  np.abs(self.image_data.data_FFT[ant_i])##slow, makes new memory
#                thresh = np.max( ABS_spectra )*min_rel_amp
#                       
#                for point_i in range(self.image_data.FFT_length):
#                    if ABS_spectra[point_i] > thresh:
#                        self.image_data.data_FFT[ant_i,point_i] = whitener[point_i]/ABS_spectra[point_i] 
#            
#        #### cross correlate and find splines ####
#        cdef np.ndarray[double, ndim=1] interp_workspace = self.interpolation_workpace ##clumsy...
#        for ant_i in range( self.image_data.num_antennas ):
#            if ant_i == self.prefered_antenna:
#                continue
#            
#            ## cross correlate ##
#            cross_correlate(&self.image_data, self.prefered_antenna, ant_i, self.correlation_workspace)
#            
#            for point_i in range(self.image_data.correlation_length):
#                    interp_workspace[point_i] = cabs( self.correlation_workspace[point_i] )
#            
#            interp_workspace[self.image_data.correlation_length] = interp_workspace[0] ## to complete circular symetry
#            
#            ## now interpolate!
#            gsl_spline_init(self.image_data.splines[ant_i], &self.image_data.interpolation_time_space[0], &interp_workspace[0], self.image_data.correlation_length+1)
##            gsl_spline_init(self.image_data.splines[ant_i], &self.image_data.interpolation_time_space[0], &interp_workspace[0], self.image_data.correlation_length)
#            
        
    
cdef class image_data_stage1(image_data_wrapper):
    
    cdef np.ndarray  interpolation_workpace
    cdef np.ndarray correlation_workspace
    cdef int prefered_antenna
    cdef double normalization_factor
    
    def __init__(self, np.ndarray[double, ndim=2] _antenna_locations, np.ndarray[double, ndim=1] _antenna_delays, int num_data_points, int _upsample_factor):
        
        self.image_data.num_antennas = _antenna_locations.shape[0]
        self.image_data.FFT_length = num_data_points*2
        self.image_data.correlation_length = _upsample_factor*self.image_data.FFT_length
        self.image_data.upsample_factor = _upsample_factor
        self.image_data.antenna_locations = _antenna_locations
        self.image_data.antenna_delays = _antenna_delays
        
        ##allocate memory for FFT data
        self.image_data.data_FFT = np.empty([self.image_data.num_antennas, self.image_data.FFT_length], dtype=np.complex)
        
        ##allocate memory for the correlation workspace
        self.correlation_workspace = np.empty(self.image_data.correlation_length, dtype=np.complex)
        self.interpolation_workpace = np.empty(self.image_data.correlation_length+1, dtype=np.double)
        self.image_data.interpolation_time_space = np.arange(self.image_data.correlation_length+1, dtype=np.double)
#        self.interpolation_workpace = np.empty(self.image_data.correlation_length, dtype=np.double)
#        self.image_data.interpolation_time_space = np.arange(self.image_data.correlation_length, dtype=np.double)
        
        self.normalization_factor = 1.0
        
        ## allocate memory for the splines
        self.image_data.splines = <void**>malloc(sizeof(void*)*self.image_data.num_antennas)
        cdef int ant_i
        for ant_i in range(self.image_data.num_antennas):
            self.image_data.splines[ant_i] = gsl_spline_alloc(gsl_interp_cspline_periodic, self.image_data.correlation_length+1)
#            self.image_data.splines[ant_i] = gsl_spline_alloc(gsl_interp_linear, self.image_data.correlation_length)
          
            
    def __dealloc__(self):
        
        cdef int ant_i
        for ant_i in range(self.image_data.num_antennas):
            gsl_spline_free( self.image_data.splines[ant_i] )
            
        free(<void*>self.image_data.splines)
        
    def intensity(self, np.ndarray[double, ndim=1] XYZ):
        return stage1_image_intensity(&self.image_data, &XYZ[0], self.prefered_antenna)*self.normalization_factor
    
    def intensity_multiprocessed(self, np.ndarray[double, ndim=2] XYZs, np.ndarray[double, ndim=1] image_out, int num_threads):
        cdef int i
        cdef double[:,:] XYZs_view = XYZs
        cdef double[:] image_out_view = image_out
        for i in prange(XYZs.shape[0], nogil=True, num_threads=num_threads):
#        for i in range(XYZs.shape[0]):
            image_out_view[i] = stage1_image_intensity(&self.image_data, &XYZs_view[i, 0], self.prefered_antenna)*self.normalization_factor
    
            
    def set_data(self, np.ndarray[double complex, ndim=1] data, int ant_i):
        
        #### copy data ####
        A = self.image_data.data_FFT[ant_i,:data.shape[0]]
        A[:] = data
        
        A = self.image_data.data_FFT[ant_i,data.shape[0]:] 
        A[:] = 0.0
        
    def prepare_image(self, prefered_antenna,  double min_amp=2):
        self.prefered_antenna = prefered_antenna
        #### FFT of all data ####
        cdef int ant_i
        for ant_i in range( self.image_data.num_antennas ):
            
            if real_max( &self.image_data.data_FFT[ant_i,0], self.image_data.data_FFT.shape[1] ) > min_amp:
                gsl_fft_complex_radix2_forward(<double*>&self.image_data.data_FFT[ant_i,0], 1, self.image_data.data_FFT.shape[1])
            else:
                self.image_data.data_FFT[ant_i,:] = 0.0
            
        #### cross correlate and find splines ####
        cdef int point_i
        cdef double correlation_sum
        cdef double correlation_maximum
        self.normalization_factor = 0.0
#        cdef np.ndarray[double, ndim=1] interp_workspace = self.interpolation_workpace ##why isn't this a memory-view?
        cdef double[:] interp_workspace = self.interpolation_workpace ##clumsy..
        for ant_i in range( self.image_data.num_antennas ):
            if ant_i == self.prefered_antenna:
                continue
            
            ## cross correlate ##
            cross_correlate(&self.image_data, self.prefered_antenna, ant_i, self.correlation_workspace)
            
            
            ### get correct componentn and calculate normalization factors
            correlation_sum = 0.0
            correlation_maximum = 0.0
            for point_i in range(self.image_data.correlation_length):
                    interp_workspace[point_i] = cabs( self.correlation_workspace[point_i] )
                    correlation_sum += interp_workspace[point_i]
                    
                    if interp_workspace[point_i] > correlation_maximum:
                        correlation_maximum = interp_workspace[point_i]
            
            interp_workspace[self.image_data.correlation_length] = interp_workspace[0] ## to complete circular symetry
            
                
            ### apply normalization
            correlation_sum = 1.0/correlation_sum
            if correlation_sum < np.inf:
                for point_i in range(self.image_data.correlation_length+1):
                    interp_workspace[point_i] *= correlation_sum
                self.normalization_factor +=  correlation_maximum*correlation_sum
            
            
            ## now interpolate!
            gsl_spline_init(self.image_data.splines[ant_i], &self.image_data.interpolation_time_space[0], &interp_workspace[0], self.image_data.correlation_length+1)
#            gsl_spline_init(self.image_data.splines[ant_i], &self.image_data.interpolation_time_space[0], &interp_workspace[0], self.image_data.correlation_length)
            
        print()
        self.normalization_factor = 1.0/self.normalization_factor
        
    def get_normalization(self):
        return self.normalization_factor
    
  
cdef class image_data_stage2(image_data_wrapper):
    
    cdef int num_correlations
    cdef np.ndarray interpolation_workpace
    cdef np.ndarray correlation_workspace
    cdef np.ndarray interferometry_peak_times
#    cdef np.ndarray usage_mask
    cdef np.ndarray RMS_mask
    
    cdef double normalization_factor
    
    def __init__(self, np.ndarray[double, ndim=2] _antenna_locations, np.ndarray[double, ndim=1] _antenna_delays, int num_data_points, int _upsample_factor):
        
        self.image_data.num_antennas = _antenna_locations.shape[0]
        self.image_data.FFT_length = num_data_points*2
        self.image_data.correlation_length = _upsample_factor*self.image_data.FFT_length
        self.image_data.upsample_factor = _upsample_factor
        self.image_data.antenna_locations = _antenna_locations
        self.image_data.antenna_delays = _antenna_delays
        self.image_data.additional_delays = np.empty(self.image_data.num_antennas, dtype=np.double)
                
        self.normalization_factor = 1.0
        self.RMS_mask = np.ones(self.image_data.num_antennas,  dtype=np.bool) ## track if antennas are used, based on amplitude
        
        ##allocate memory for FFT data
        self.image_data.data_FFT = np.empty([self.image_data.num_antennas, self.image_data.FFT_length], dtype=np.complex)
        
        ##allocate memory for the correlation workspace
        self.correlation_workspace = np.empty(self.image_data.correlation_length, dtype=np.complex)
        self.interpolation_workpace = np.empty( (self.image_data.correlation_length+1)*2, dtype=np.double) # times 3 cous' we have real and imaginary components
        self.image_data.interpolation_time_space = np.arange(self.image_data.correlation_length+1, dtype=np.double)
        
        ## allocate memory for the splines
        self.num_correlations = int( self.image_data.num_antennas*(self.image_data.num_antennas-1)/2 )*2 # times 3 cous' we have real and imaginary components
        self.image_data.splines = <void**>malloc(sizeof(void*)*self.num_correlations)
        self.interferometry_peak_times = np.empty(self.num_correlations, dtype=np.double)
        cdef int cor_i
        for cor_i in range( self.num_correlations ):
            self.image_data.splines[cor_i] = gsl_spline_alloc(gsl_interp_cspline_periodic, self.image_data.correlation_length+1)
          
            
    def __dealloc__(self):
        
        cdef int cor_i
        for cor_i in range( self.num_correlations ):
            gsl_spline_free( self.image_data.splines[cor_i] )
            
        free(<void*>self.image_data.splines)
        
        
    def intensity_ABSafter(self, np.ndarray[double, ndim=1] XYZ):
        return stage2_image_intensity_absAfter(&self.image_data, &XYZ[0])*self.normalization_factor
        
    def intensity_multiprocessed_ABSafter(self, np.ndarray[double, ndim=2] XYZs, np.ndarray[double, ndim=1] image_out, int num_threads):
        cdef int i
        cdef double[:,:] XYZs_view = XYZs
        cdef double[:] image_out_view = image_out
        for i in prange(XYZs.shape[0], nogil=True, num_threads=num_threads):
#        for i in range(XYZs.shape[0]):
            image_out_view[i] = stage2_image_intensity_absAfter(&self.image_data, &XYZs_view[i, 0])*self.normalization_factor
            
            
            
    def intensity_ABSbefore(self, np.ndarray[double, ndim=1] XYZ):
        return stage2_image_intensity_absBefore(&self.image_data, &XYZ[0])*self.normalization_factor
        
    def intensity_multiprocessed_ABSbefore(self, np.ndarray[double, ndim=2] XYZs, np.ndarray[double, ndim=1] image_out, int num_threads):
        cdef int i
        cdef double[:,:] XYZs_view = XYZs
        cdef double[:] image_out_view = image_out
        for i in prange(XYZs.shape[0], nogil=True, num_threads=num_threads):
#        for i in range(XYZs.shape[0]):
            image_out_view[i] = stage2_image_intensity_absBefore(&self.image_data, &XYZs_view[i, 0])*self.normalization_factor
         
            
            
    def intensity_real(self, np.ndarray[double, ndim=1] XYZ):
        return stage2_image_intensity_real(&self.image_data, &XYZ[0])*self.normalization_factor
        
    def intensity_multiprocessed_real(self, np.ndarray[double, ndim=2] XYZs, np.ndarray[double, ndim=1] image_out, int num_threads):
        cdef int i
        cdef double[:,:] XYZs_view = XYZs
        cdef double[:] image_out_view = image_out
        for i in prange(XYZs.shape[0], nogil=True, num_threads=num_threads):
#        for i in range(XYZs.shape[0]):
            image_out_view[i] = stage2_image_intensity_real(&self.image_data, &XYZs_view[i, 0])*self.normalization_factor
         
            
            
    def set_data(self, np.ndarray[double complex, ndim=1] data, int ant_i, double additional_delay):
        
        #### copy data ####
        A = self.image_data.data_FFT[ant_i,:data.shape[0]]
        A[:] = data
        
        A = self.image_data.data_FFT[ant_i,data.shape[0]:] 
        A[:] = 0.0
        
        self.image_data.additional_delays[ant_i] = additional_delay
       
    def prepare_image(self, double min_amp=2, double RMS_min_amp=10):
        #### FFT of all data ####
        cdef int ant_i
        cdef double tmp_max
        for ant_i in range( self.image_data.num_antennas ):
            tmp_max = real_max( &self.image_data.data_FFT[ant_i,0], self.image_data.data_FFT.shape[1] )
            if  tmp_max > min_amp:
                gsl_fft_complex_radix2_forward(<double*>&self.image_data.data_FFT[ant_i,0], 1, self.image_data.data_FFT.shape[1])
            else:
                self.image_data.data_FFT[ant_i,:] = 0.0
                
            if  tmp_max > RMS_min_amp:
                self.RMS_mask[ant_i] = 1
            else:
                self.RMS_mask[ant_i] = 0
            
        #### cross correlate and find splines ####
        cdef double[:] interp_workspace = self.interpolation_workpace ##clumsy...
        cdef double[:] peak_time_workspace = self.interferometry_peak_times
        cdef int ant_j
        cdef int point_i
        cdef int corr_i = 0
        cdef double correlation_sum
        cdef double correlation_maximum
        cdef int correlation_peak_loc=0
        cdef double abs_point
        self.normalization_factor = 0.0
        for ant_i in range( self.image_data.num_antennas ):
            for ant_j in range(ant_i+1, self.image_data.num_antennas ):
                
                ## cross correlate ##
                cross_correlate(&self.image_data, ant_i, ant_j, self.correlation_workspace)
                
                
                ### get correct componentn and calculate normalization factors
                correlation_sum = 0.0
                correlation_maximum = 0.0
                for point_i in range(self.image_data.correlation_length):
                    interp_workspace[point_i] = creal( self.correlation_workspace[point_i] )
                    interp_workspace[point_i+self.image_data.correlation_length+1] = cimag( self.correlation_workspace[point_i] )
                    
                    abs_point = cabs( self.correlation_workspace[point_i] )
                    correlation_sum += abs_point
                    
                    if abs_point > correlation_maximum:
                        correlation_maximum = abs_point
                        correlation_peak_loc = point_i
                        
                if self.RMS_mask[ant_i] and self.RMS_mask[ant_j]:
                    peak_time_workspace[corr_i] = correlation_peak_loc
                else:
                    peak_time_workspace[corr_i] = np.nan
                    
                
                interp_workspace[self.image_data.correlation_length] = interp_workspace[0] ## to complete circular symetry
                interp_workspace[self.image_data.correlation_length*2+1] = interp_workspace[self.image_data.correlation_length+1]
                
                
                ### apply normalization
                correlation_sum = 1.0/correlation_sum
                if correlation_sum < np.inf:
                    for point_i in range(self.image_data.correlation_length+1):
                        interp_workspace[point_i] *= correlation_sum
                        interp_workspace[point_i+self.image_data.correlation_length+1] *= correlation_sum
                        
                    self.normalization_factor +=  correlation_maximum*correlation_sum
                    
                
                ## now interpolate!
                gsl_spline_init(self.image_data.splines[corr_i*2],   &self.image_data.interpolation_time_space[0], &interp_workspace[0],                                   self.image_data.correlation_length+1) ## real component
                gsl_spline_init(self.image_data.splines[corr_i*2+1], &self.image_data.interpolation_time_space[0], &interp_workspace[self.image_data.correlation_length+1], self.image_data.correlation_length+1) ## imag component

                corr_i += 1
                
        self.normalization_factor = 1.0/self.normalization_factor
        
    def get_normalization(self):
        return self.normalization_factor
    
    def get_RMS(self, np.ndarray[double, ndim=1] XYZ):
        cdef int ant_i
        cdef int ant_j
        cdef int corr_i=0
        cdef int num_measure=0
        cdef double ret = 0.0
        cdef double tmp_dt = 0.0
        cdef double modeled_dt
        cdef double *XYZ_pointer = &XYZ[0]
        for ant_i in range( self.image_data.num_antennas ):
            for ant_j in range(ant_i+1, self.image_data.num_antennas ):
            
                modeled_dt = vec_3_DiffNorm(&self.image_data.antenna_locations[ ant_i,0], XYZ_pointer)
                modeled_dt -= vec_3_DiffNorm(&self.image_data.antenna_locations[ant_j,0], XYZ_pointer)
                modeled_dt *= c_air_inverse
                modeled_dt += self.image_data.antenna_delays[ ant_i ] - self.image_data.antenna_delays[ant_j]
                modeled_dt += self.image_data.additional_delays[ ant_i ] - self.image_data.additional_delays[ant_j]
                
                if gsl_finite( self.interferometry_peak_times[corr_i] ):
                    if self.interferometry_peak_times[corr_i] < self.image_data.correlation_length*0.5:
                        tmp_dt = ( modeled_dt - self.interferometry_peak_times[corr_i]*5.0E-9/self.image_data.upsample_factor )
#                        print(ant_i, ant_j, modeled_dt, self.interferometry_peak_times[corr_i]*5.0E-9/self.image_data.upsample_factor)
                    else:
                        tmp_dt = ( modeled_dt - (self.interferometry_peak_times[corr_i]-self.image_data.correlation_length )*5.0E-9/self.image_data.upsample_factor )
#                        print(ant_i, ant_j, modeled_dt, (self.interferometry_peak_times[corr_i]-self.image_data.correlation_length )*5.0E-9/self.image_data.upsample_factor)
                        
                    ret += tmp_dt**2
                    num_measure += 1
                    
                
                corr_i += 1
                
                
            
        return sqrt( ret/num_measure )
    
    
    
cdef double stage2_image_intensity_absBefore2(image_data_struct* image_data, double* XYZ) nogil:
    cdef int ant_i
    cdef int ant_j
    cdef int corr_i=0
    cdef double ret = 0.0
    cdef double modeled_dt
    for ant_i in range( image_data.num_antennas ):
        for ant_j in range(ant_i+1, image_data.num_antennas ):
        
            modeled_dt = vec_3_DiffNorm(&image_data.antenna_locations[ ant_i,0], XYZ)
            modeled_dt -= vec_3_DiffNorm(&image_data.antenna_locations[ant_j,0], XYZ)
            modeled_dt *= c_air_inverse
            modeled_dt += image_data.antenna_delays[ ant_i ] - image_data.antenna_delays[ant_j]
            modeled_dt += image_data.additional_delays[ ant_i ] - image_data.additional_delays[ant_j]
            modeled_dt *= image_data.upsample_factor/5.0E-9
            
            while modeled_dt < 0:
                modeled_dt += image_data.correlation_length
            while modeled_dt >= image_data.correlation_length:
                modeled_dt -= image_data.correlation_length
                
            ret += gsl_spline_eval(image_data.splines[corr_i], modeled_dt, NULL)
            
            corr_i += 1
        
    return -ret
    
    
cdef class image_data_stage2_absBefore(image_data_wrapper):
    
    cdef int num_correlations
    cdef np.ndarray interpolation_workpace
    cdef np.ndarray correlation_workspace
    
    cdef double normalization_factor
    
    def __init__(self, np.ndarray[double, ndim=2] _antenna_locations, np.ndarray[double, ndim=1] _antenna_delays, int num_data_points, int _upsample_factor):
        
        self.image_data.num_antennas = _antenna_locations.shape[0]
        self.image_data.FFT_length = num_data_points*2
        self.image_data.correlation_length = _upsample_factor*self.image_data.FFT_length
        self.image_data.upsample_factor = _upsample_factor
        self.image_data.antenna_locations = _antenna_locations
        self.image_data.antenna_delays = _antenna_delays
        self.image_data.additional_delays = np.empty(self.image_data.num_antennas, dtype=np.double)
                
        self.normalization_factor = 1.0
        
        ##allocate memory for FFT data
        self.image_data.data_FFT = np.empty([self.image_data.num_antennas, self.image_data.FFT_length], dtype=np.complex)
        
        ##allocate memory for the correlation workspace
        self.correlation_workspace = np.empty(self.image_data.correlation_length, dtype=np.complex)
        self.interpolation_workpace = np.empty( (self.image_data.correlation_length+1), dtype=np.double)
        self.image_data.interpolation_time_space = np.arange(self.image_data.correlation_length+1, dtype=np.double)
        
        ## allocate memory for the splines
        self.num_correlations = int( self.image_data.num_antennas*(self.image_data.num_antennas-1)/2 )
        self.image_data.splines = <void**>malloc(sizeof(void*)*self.num_correlations)
        cdef int cor_i
        for cor_i in range( self.num_correlations ):
            self.image_data.splines[cor_i] = gsl_spline_alloc(gsl_interp_cspline_periodic, self.image_data.correlation_length+1)
          
            
    def __dealloc__(self):
        
        cdef int cor_i
        for cor_i in range( self.num_correlations ):
            gsl_spline_free( self.image_data.splines[cor_i] )
            
        free(<void*>self.image_data.splines)

            
    def intensity_ABSbefore(self, np.ndarray[double, ndim=1] XYZ):
        return stage2_image_intensity_absBefore2(&self.image_data, &XYZ[0])*self.normalization_factor
        
    def intensity_multiprocessed_ABSbefore(self, np.ndarray[double, ndim=2] XYZs, np.ndarray[double, ndim=1] image_out, int num_threads):
        cdef int i
        cdef double[:,:] XYZs_view = XYZs
        cdef double[:] image_out_view = image_out
        for i in prange(XYZs.shape[0], nogil=True, num_threads=num_threads):
#        for i in range(XYZs.shape[0]):
            image_out_view[i] = stage2_image_intensity_absBefore2(&self.image_data, &XYZs_view[i, 0])*self.normalization_factor
        
            
    def set_data(self, np.ndarray[double complex, ndim=1] data, int ant_i, double additional_delay):
        
        #### copy data ####
        A = self.image_data.data_FFT[ant_i,:data.shape[0]]
        A[:] = data
        
        A = self.image_data.data_FFT[ant_i,data.shape[0]:] 
        A[:] = 0.0
        
        self.image_data.additional_delays[ant_i] = additional_delay
       
    def prepare_image(self, double min_amp=2):
        #### FFT of all data ####
        cdef int ant_i
        cdef double tmp_max
        for ant_i in range( self.image_data.num_antennas ):
            tmp_max = real_max( &self.image_data.data_FFT[ant_i,0], self.image_data.data_FFT.shape[1] )
            if  tmp_max > min_amp:
                gsl_fft_complex_radix2_forward(<double*>&self.image_data.data_FFT[ant_i,0], 1, self.image_data.data_FFT.shape[1])
            else:
                self.image_data.data_FFT[ant_i,:] = 0.0
            
        #### cross correlate and find splines ####
        cdef double[:] interp_workspace = self.interpolation_workpace ##clumsy...
        cdef int ant_j
        cdef int point_i
        cdef int corr_i = 0
        cdef double correlation_sum
        cdef double correlation_maximum
        cdef double abs_point
        self.normalization_factor = 0.0
        for ant_i in range( self.image_data.num_antennas ):
            for ant_j in range(ant_i+1, self.image_data.num_antennas ):
                
                ## cross correlate ##
                cross_correlate(&self.image_data, ant_i, ant_j, self.correlation_workspace)
                
                
                ### get correct componentn and calculate normalization factors
                correlation_sum = 0.0
                correlation_maximum = 0.0
                for point_i in range(self.image_data.correlation_length):
                    
                    abs_point = cabs( self.correlation_workspace[point_i] )
                    correlation_sum += abs_point
                    interp_workspace[point_i] = abs_point
                    
                    if abs_point > correlation_maximum:
                        correlation_maximum = abs_point
                    
                interp_workspace[self.image_data.correlation_length] = interp_workspace[0] ## to complete circular symetry
                
                ### apply normalization
                correlation_sum = 1.0/correlation_sum
                if correlation_sum < np.inf:
                    for point_i in range(self.image_data.correlation_length+1):
                        interp_workspace[point_i] *= correlation_sum
                        
                    self.normalization_factor +=  correlation_maximum*correlation_sum
                    
                
                ## now interpolate!
                gsl_spline_init(self.image_data.splines[corr_i],   &self.image_data.interpolation_time_space[0], &interp_workspace[0],  self.image_data.correlation_length+1)

                corr_i += 1
                
        self.normalization_factor = 1.0/self.normalization_factor
        

cdef double stage2_image_intensity_sumLog(image_data_struct* image_data, double* XYZ) nogil:
    cdef int ant_i
    cdef int ant_j
    cdef int corr_i=0
    cdef double ret = 0.0
    cdef double modeled_dt
    cdef double tmp
    for ant_i in range( image_data.num_antennas ):
        for ant_j in range(ant_i+1, image_data.num_antennas ):
        
            modeled_dt = vec_3_DiffNorm(&image_data.antenna_locations[ ant_i,0], XYZ)
            modeled_dt -= vec_3_DiffNorm(&image_data.antenna_locations[ant_j,0], XYZ)
            modeled_dt *= c_air_inverse
            modeled_dt += image_data.antenna_delays[ ant_i ] - image_data.antenna_delays[ant_j]
            modeled_dt += image_data.additional_delays[ ant_i ] - image_data.additional_delays[ant_j]
            modeled_dt *= image_data.upsample_factor/5.0E-9
            
            while modeled_dt < 0:
                modeled_dt += image_data.correlation_length
            while modeled_dt >= image_data.correlation_length:
                modeled_dt -= image_data.correlation_length
            
            tmp = log( gsl_spline_eval(image_data.splines[corr_i], modeled_dt, NULL) )
            if gsl_finite( tmp ):
                ret += tmp
            
            corr_i += 1
        
    return -ret

cdef class image_data_sumLog(image_data_wrapper):
    
    cdef int num_correlations
    cdef np.ndarray interpolation_workpace
    cdef np.ndarray correlation_workspace
    cdef np.ndarray interferometry_peak_times
    cdef np.ndarray usage_mask
    
    def __init__(self, np.ndarray[double, ndim=2] _antenna_locations, np.ndarray[double, ndim=1] _antenna_delays, int num_data_points, int _upsample_factor):
        
        self.image_data.num_antennas = _antenna_locations.shape[0]
        self.image_data.FFT_length = num_data_points*2
        self.image_data.correlation_length = _upsample_factor*self.image_data.FFT_length
        self.image_data.upsample_factor = _upsample_factor
        self.image_data.antenna_locations = _antenna_locations
        self.image_data.antenna_delays = _antenna_delays
        self.image_data.additional_delays = np.empty(self.image_data.num_antennas, dtype=np.double)
                
        self.usage_mask = np.ones(self.image_data.num_antennas,  dtype=np.bool) ## track if antennas are used, based on amplitude
        ### TODO: improve use of mask
        
        ##allocate memory for FFT data
        self.image_data.data_FFT = np.empty([self.image_data.num_antennas, self.image_data.FFT_length], dtype=np.complex)
        
        ##allocate memory for the correlation workspace
        self.correlation_workspace = np.empty(self.image_data.correlation_length, dtype=np.complex)
        self.interpolation_workpace = np.empty( (self.image_data.correlation_length+1), dtype=np.double)
        self.image_data.interpolation_time_space = np.arange(self.image_data.correlation_length+1, dtype=np.double)
        
        ## allocate memory for the splines
        self.num_correlations = int( self.image_data.num_antennas*(self.image_data.num_antennas-1)/2 )
        self.image_data.splines = <void**>malloc(sizeof(void*)*self.num_correlations)
        self.interferometry_peak_times = np.empty(self.num_correlations, dtype=np.double)
        cdef int cor_i
        for cor_i in range( self.num_correlations ):
            self.image_data.splines[cor_i] = gsl_spline_alloc(gsl_interp_cspline_periodic, self.image_data.correlation_length+1)
          
            
    def __dealloc__(self):
        
        cdef int cor_i
        for cor_i in range( self.num_correlations ):
            gsl_spline_free( self.image_data.splines[cor_i] )
            
        free(<void*>self.image_data.splines)
        
        
    def intensity_sumLog(self, np.ndarray[double, ndim=1] XYZ):
        cdef double R = stage2_image_intensity_sumLog(&self.image_data, &XYZ[0])
        if not gsl_finite( R ):
            print("NOT FINITE!", R)
        return R
        
    def intensity_multiprocessed_sumLog(self, np.ndarray[double, ndim=2] XYZs, np.ndarray[double, ndim=1] image_out, int num_threads):
        cdef int i
        cdef double[:,:] XYZs_view = XYZs
        cdef double[:] image_out_view = image_out
        for i in prange(XYZs.shape[0], nogil=True, num_threads=num_threads):
#        for i in range(XYZs.shape[0]):
            image_out_view[i] = stage2_image_intensity_sumLog(&self.image_data, &XYZs_view[i, 0])
            
            
            
            
    def set_data(self, np.ndarray[double complex, ndim=1] data, int ant_i, double additional_delay):
        
        #### copy data ####
        A = self.image_data.data_FFT[ant_i,:data.shape[0]]
        A[:] = data
        
        A = self.image_data.data_FFT[ant_i,data.shape[0]:] 
        A[:] = 0.0
        
        self.image_data.additional_delays[ant_i] = additional_delay
       
    def prepare_image(self, double min_amp=2):
        #### FFT of all data ####
        cdef int ant_i
        for ant_i in range( self.image_data.num_antennas ):
            if real_max( &self.image_data.data_FFT[ant_i,0], self.image_data.data_FFT.shape[1] ) > min_amp:
                gsl_fft_complex_radix2_forward(<double*>&self.image_data.data_FFT[ant_i,0], 1, self.image_data.data_FFT.shape[1])
                self.usage_mask[ant_i] = 1
            else:
                self.image_data.data_FFT[ant_i,:] = 0.0
                self.usage_mask[ant_i] = 0
            
        #### cross correlate and find splines ####
        cdef double[:] interp_workspace = self.interpolation_workpace ##clumsy...
        cdef double[:] peak_time_workspace = self.interferometry_peak_times
        cdef int ant_j
        cdef int point_i
        cdef int corr_i = 0
        cdef double correlation_maximum
        cdef int correlation_peak_loc=0
        cdef double abs_point
        for ant_i in range( self.image_data.num_antennas ):
            for ant_j in range(ant_i+1, self.image_data.num_antennas ):
                
                ## cross correlate ##
                cross_correlate(&self.image_data, ant_i, ant_j, self.correlation_workspace)
                
                
                ### get correct componentn and calculate normalization factors
                correlation_maximum = 0.0
                for point_i in range(self.image_data.correlation_length):
                    abs_point = cabs( self.correlation_workspace[point_i] )
                    interp_workspace[point_i] = abs_point
                    
                    if abs_point > correlation_maximum:
                        correlation_maximum = abs_point
                        correlation_peak_loc = point_i
                        
                if self.usage_mask[ant_i] and self.usage_mask[ant_j]:
                    peak_time_workspace[corr_i] = correlation_peak_loc
                else:
                    peak_time_workspace[corr_i] = np.nan
                    
                
                interp_workspace[self.image_data.correlation_length] = interp_workspace[0] ## to complete circular symetry
                
                
                ### apply normalization
                correlation_maximum = 1.0/correlation_maximum
                if correlation_maximum < np.inf:
                    for point_i in range(self.image_data.correlation_length+1):
                        interp_workspace[point_i] *= correlation_maximum
                    
                
                ## now interpolate!
                gsl_spline_init(self.image_data.splines[corr_i],   &self.image_data.interpolation_time_space[0],  &interp_workspace[0],  self.image_data.correlation_length+1) 
                corr_i += 1
    
    def get_RMS(self, np.ndarray[double, ndim=1] XYZ):
        cdef int ant_i
        cdef int ant_j
        cdef int corr_i=0
        cdef int num_measure=0
        cdef double ret = 0.0
        cdef double tmp_dt = 0.0
        cdef double modeled_dt
        cdef double *XYZ_pointer = &XYZ[0]
        for ant_i in range( self.image_data.num_antennas ):
            for ant_j in range(ant_i+1, self.image_data.num_antennas ):
            
                modeled_dt = vec_3_DiffNorm(&self.image_data.antenna_locations[ ant_i,0], XYZ_pointer)
                modeled_dt -= vec_3_DiffNorm(&self.image_data.antenna_locations[ant_j,0], XYZ_pointer)
                modeled_dt *= c_air_inverse
                modeled_dt += self.image_data.antenna_delays[ ant_i ] - self.image_data.antenna_delays[ant_j]
                modeled_dt += self.image_data.additional_delays[ ant_i ] - self.image_data.additional_delays[ant_j]
                
                if gsl_finite( self.interferometry_peak_times[corr_i] ):
                    if self.interferometry_peak_times[corr_i] < self.image_data.correlation_length*0.5:
                        tmp_dt = ( modeled_dt - self.interferometry_peak_times[corr_i]*5.0E-9/self.image_data.upsample_factor )
#                        print(ant_i, ant_j, modeled_dt, self.interferometry_peak_times[corr_i]*5.0E-9/self.image_data.upsample_factor)
                    else:
                        tmp_dt = ( modeled_dt - (self.interferometry_peak_times[corr_i]-self.image_data.correlation_length )*5.0E-9/self.image_data.upsample_factor )
#                        print(ant_i, ant_j, modeled_dt, (self.interferometry_peak_times[corr_i]-self.image_data.correlation_length )*5.0E-9/self.image_data.upsample_factor)
                    
                    ret += tmp_dt**2
                    num_measure += 1
                    
                    
                
                corr_i += 1
                
                
            
        return sqrt( ret/num_measure )
                
                
                
                

#cdef double stage_1_minimize(void* x, void* params):
#    cdef double* XYZ = gsl_vector_ptr(x, 0) 
#    return stage1_image_intensity( <image_data_struct*>params,  XYZ)
#
#cdef class local_image_minimizer:
#    cdef image_data_struct* image_obj
#    cdef int stage
#    cdef gsl_multimin_fminimizer* minimizer_space
#    cdef void* XYZ_loc
#    cdef void* step_size
#    cdef gsl_multimin_function minimizer_func
#    
#    cdef int num_itters
#    cdef double function_value
#    cdef bint converged
#    
#    def __init__(self, image_data_wrapper imager, int stage):
#        self.image_obj = &imager.image_data
#        self.minimizer_space = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2, 3)
#        self.XYZ_loc = gsl_vector_alloc(3)
#        self.step_size = gsl_vector_alloc(3)
#        
#        self.stage = stage
#        self.minimizer_func.n = 3
#        self.minimizer_func.params = <void*>self.image_obj
#        if self.stage==1:
#            self.minimizer_func.f = &stage_1_minimize
#        
#    def __dealloc__(self):
#        gsl_vector_free(self.XYZ_loc)
#        gsl_vector_free(self.step_size)
#        gsl_multimin_fminimizer_free(self.minimizer_space)
#        
#    cdef void* get_current_loc_ref(self):
#        return self.XYZ_loc
#    
#    cdef void set_position(self):
#        gsl_vector_set(self.XYZ_loc,  0,  gsl_vector_get(gsl_multimin_fminimizer_x(self.minimizer_space), 0))
#        gsl_vector_set(self.XYZ_loc,  1,  gsl_vector_get(gsl_multimin_fminimizer_x(self.minimizer_space), 1))
#        gsl_vector_set(self.XYZ_loc,  2,  gsl_vector_get(gsl_multimin_fminimizer_x(self.minimizer_space), 2))
#        self.function_value = gsl_multimin_fminimizer_minimum(self.minimizer_space)
#    
#    cdef bint run_minimizer(self, double tol=0.01, int max_itter=500, double initial_step_size=10):
#        
#        gsl_vector_set_all(self.step_size, initial_step_size)
#    
#        gsl_multimin_fminimizer_set(self.minimizer_space, &self.minimizer_func, self.XYZ_loc, self.step_size)
#        
#        cdef int iter_i = 0 
#        cdef int status
#        for iter_i in range(max_itter):
#            status = gsl_multimin_fminimizer_iterate(self.minimizer_space)
#            if status == GSL_ENOPROG:
#                self.set_position()
#                self.num_itters = iter_i
#                self.converged = False
#                return self.converged ## minimizer has stalled
#            
#            status = gsl_multimin_test_size(gsl_multimin_fminimizer_size(self.minimizer_space), tol);
#            if status == GSL_SUCCESS:
#                self.set_position()
#                self.num_itters = iter_i
#                self.converged = True
#                return self.converged ## we have converged!
#                 
#        self.set_position()
#        self.num_itters = iter_i
#        self.converged = False
#        return self.converged## we have run out of itterations
#        
#    def run(self, np.ndarray[double, ndim=1] X0, double tol=0.01, int max_itter=500, double initial_step_size=10):
#        gsl_vector_set(self.XYZ_loc,  0, X0[0])
#        gsl_vector_set(self.XYZ_loc,  0, X0[1])
#        gsl_vector_set(self.XYZ_loc,  0, X0[2])
#        
#        A = self.run_minimizer(tol, max_itter, initial_step_size)
#        
#        return A
#    
#    def get_current_loc(self):
#        cdef np.ndarray[double, ndim=1] ret = np.empty(3)
#        ret[0] = gsl_vector_get(self.XYZ_loc, 0)
#        ret[1] = gsl_vector_get(self.XYZ_loc, 1)
#        ret[2] = gsl_vector_get(self.XYZ_loc, 2)
#        return ret
#    
#    def is_converged(self):
#        return self.converged
#    
#    def func(self):
#        return self.function_value
#    
#    def get_num_itters(self):
#        return self.num_itters



