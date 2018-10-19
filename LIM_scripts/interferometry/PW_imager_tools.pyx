#cython: language_level=3 
#cython: cdivision=True
#cython: boundscheck=False
#cython: linetrace=False
#cython: binding=False
#cython: profile=False

#### this is a set of tools to image a lightning flash in 2D using short baselines (ie a planewave model)

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
    #double [:] additional_delays
    
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

cdef double image_intensity_absAfter(image_data_struct* image_data, double* cosAlphacosBeta, int prefered_antenna) nogil:
    cdef int ant_i
    cdef int corr_i=0
    cdef double ret_real = 0.0
    cdef double ret_imag = 0.0
    cdef double modeled_dt
    cdef double cosAlpha = cosAlphacosBeta[0]
    cdef double cosBeta = cosAlphacosBeta[1]
    
    
    cdef double Rz = cosAlpha*cosAlpha + cosBeta*cosBeta
    if RV >= 1.0:
        return 0.0
    
    Rz = sqrt(1.0 - Rz)
    
    for ant_i in range( image_data.num_antennas ):
        if ant_i == prefered_antenna:
            continue  
        
        modeled_dt = (image_data.antenna_locations[ prefered_antenna,0] - image_data.antenna_locations[ ant_i,0])*cosBeta
        modeled_dt += (image_data.antenna_locations[ prefered_antenna,1] - image_data.antenna_locations[ ant_i,1])*cosAlpha
        modeled_dt += (image_data.antenna_locations[ prefered_antenna,2] - image_data.antenna_locations[ ant_i,2])*Rz
        
        modeled_dt *= c_air_inverse
        modeled_dt += image_data.antenna_delays[ ant_i ] - image_data.antenna_delays[ant_j]
        #modeled_dt += image_data.additional_delays[ ant_i ] - image_data.additional_delays[ant_j]
        modeled_dt *= image_data.upsample_factor/5.0E-9
        
        while modeled_dt < 0:
            modeled_dt += image_data.correlation_length
        while modeled_dt >= image_data.correlation_length:
            modeled_dt -= image_data.correlation_length
        
        ret_real += gsl_spline_eval(image_data.splines[corr_i*2], modeled_dt, NULL) 
        ret_imag += gsl_spline_eval(image_data.splines[corr_i*2+1], modeled_dt, NULL) 
        corr_i += 1
        
    return - gsl_hypot(ret_real, ret_imag)
        
    
cdef class image_data_wrapper:
    cdef image_data_struct image_data
    
        
    
cdef class image_data(image_data_wrapper):
    
    cdef np.ndarray interpolation_workpace
    cdef np.ndarray correlation_workspace
    cdef np.ndarray interferometry_peak_times
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
        self.interpolation_workpace = np.empty( (self.image_data.correlation_length+1)*2, dtype=np.double) # times 2 cous' we have real and imaginary components
        self.image_data.interpolation_time_space = np.arange(self.image_data.correlation_length+1, dtype=np.double)
        
        self.normalization_factor = 1.0
        
        ## allocate memory for the splines
        self.interferometry_peak_times = np.empty(self.image_data.num_antennas, dtype=np.double)
        self.image_data.splines = <void**>malloc(sizeof(void*)*self.image_data.num_antennas*2)# times 2 cous' we have real and imaginary components
        cdef int ant_i
        for ant_i in range(self.image_data.num_antennas*2):
            self.image_data.splines[ant_i] = gsl_spline_alloc(gsl_interp_cspline_periodic, self.image_data.correlation_length+1)
          
            
    def __dealloc__(self):
        
        cdef int ant_i
        for ant_i in range(self.image_data.num_antennas*2):
            gsl_spline_free( self.image_data.splines[ant_i] )
            
        free(<void*>self.image_data.splines)
        
    def intensity_absAfter(self, np.ndarray[double, ndim=1] cosAlphacosBeta):
        return image_intensity_absAfter(&self.image_data, &cosAlphacosBeta[0], self.prefered_antenna)*self.normalization_factor
    
    def intensity_multiprocessed_absAfter(self, np.ndarray[double, ndim=2] cosAlphacosBetas, np.ndarray[double, ndim=1] image_out, int num_threads):
        cdef int i
        cdef double[:,:] cosAlphacosBetas_view = cosAlphacosBetas
        cdef double[:] image_out_view = image_out
        for i in prange(XYZs.shape[0], nogil=True, num_threads=num_threads):
#        for i in range(XYZs.shape[0]):
            image_out_view[i] = stage1_image_intensity(&self.image_data, &cosAlphacosBetas_view[i, 0], self.prefered_antenna)*self.normalization_factor
    
            
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
        cdef double correlation_maximum
        cdef int correlation_peak_loc=0
        self.normalization_factor = 0.0
        
        cdef double[:] interp_workspace = self.interpolation_workpace ##clumsy..
        cdef double[:] peak_time_workspace = self.interferometry_peak_times
        
        for ant_i in range( self.image_data.num_antennas ):
            if ant_i == self.prefered_antenna:
                continue
            
            ## cross correlate ##
            cross_correlate(&self.image_data, self.prefered_antenna, ant_i, self.correlation_workspace)
            
            
            ### get correct componentn and calculate normalization factors
            correlation_maximum = 0.0
            for point_i in range(self.image_data.correlation_length):
                interp_workspace[point_i] = creal( self.correlation_workspace[point_i] )
                interp_workspace[point_i+self.image_data.correlation_length+1] = cimag( self.correlation_workspace[point_i] )
                
                abs_point = cabs( self.correlation_workspace[point_i] )
                
                if abs_point > correlation_maximum:
                    correlation_maximum = abs_point
                    correlation_peak_loc = point_i
            
            interp_workspace[self.image_data.correlation_length] = interp_workspace[0] ## to complete circular symetry
            interp_workspace[self.image_data.correlation_length*2+1] = interp_workspace[self.image_data.correlation_length+1]
                
            ### apply normalization
            correlation_maximum = 1.0/correlation_maximum
            if correlation_maximum < np.inf:
                for point_i in range(self.image_data.correlation_length+1):
                    interp_workspace[point_i] *= correlation_maximum
                    interp_workspace[point_i+self.image_data.correlation_length+1] *= correlation_maximum
                
                
                peak_time_workspace[ant_i] = correlation_peak_loc
                self.normalization_factor +=  1.0
            else:
                peak_time_workspace[ant_i] = np.nan
            
            
            ## now interpolate!
            gsl_spline_init(self.image_data.splines[ant_i*2],   &self.image_data.interpolation_time_space[0], &interp_workspace[0],                                   self.image_data.correlation_length+1) ## real component
            gsl_spline_init(self.image_data.splines[ant_i*2+1], &self.image_data.interpolation_time_space[0], &interp_workspace[self.image_data.correlation_length+1], self.image_data.correlation_length+1) ## imag component

            
        self.normalization_factor = 1.0/self.normalization_factor
        
    def get_normalization(self):
        return self.normalization_factor
    
    def get_RMS(self, np.ndarray[double, ndim=1] cosAlphacosBeta):
        cdef int ant_i
        cdef int num_measure=0
        cdef double ret = 0.0
        cdef double tmp_dt = 0.0
        cdef double modeled_dt
        cdef double *cosAlphacosBeta_pointer = &cosAlphacosBeta[0]
        for ant_i in range( self.image_data.num_antennas ):
            if ant_i == prefered_antenna:
                continue  
            
            modeled_dt = (image_data.antenna_locations[ self.prefered_antenna,0] - image_data.antenna_locations[ ant_i,0])*cosBeta
            modeled_dt += (image_data.antenna_locations[ self.prefered_antenna,1] - image_data.antenna_locations[ ant_i,1])*cosAlpha
            modeled_dt += (image_data.antenna_locations[ self.prefered_antenna,2] - image_data.antenna_locations[ ant_i,2])*Rz
        
            modeled_dt *= c_air_inverse
            #modeled_dt += self.image_data.antenna_delays[ ant_i ] - self.image_data.antenna_delays[ant_j]
            modeled_dt += self.image_data.additional_delays[ ant_i ] - self.image_data.additional_delays[ant_j]
            
            if gsl_finite( self.interferometry_peak_times[ant_i] ):
                if self.interferometry_peak_times[ant_i] < self.image_data.correlation_length*0.5:
                    tmp_dt = ( modeled_dt - self.interferometry_peak_times[corr_i]*5.0E-9/self.image_data.upsample_factor )
#                        print(ant_i, ant_j, modeled_dt, self.interferometry_peak_times[corr_i]*5.0E-9/self.image_data.upsample_factor)
                else:
                    tmp_dt = ( modeled_dt - (self.interferometry_peak_times[ant_i]-self.image_data.correlation_length )*5.0E-9/self.image_data.upsample_factor )
#                        print(ant_i, ant_j, modeled_dt, (self.interferometry_peak_times[corr_i]-self.image_data.correlation_length )*5.0E-9/self.image_data.upsample_factor)
                    
                ret += tmp_dt**2
                num_measure += 1
                
        return sqrt( ret/num_measure )

