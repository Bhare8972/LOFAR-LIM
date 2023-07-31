#cython: language_level=3 
#cython: cdivision=True
#cython: boundscheck=True
#cython: linetrace=False
#cython: binding=False
#cython: profile=False


cimport numpy as np

import numpy as np
#from scipy import fftpack
from libc.math cimport sqrt,log, sin, cos
from libc.stdlib cimport malloc, free

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
    
cdef extern from "gsl/gsl_matrix.h" nogil:
    
    void* gsl_matrix_alloc(size_t n1, size_t n2)
    void gsl_matrix_free(void* m)
    
    double gsl_matrix_get(const void * m, const size_t i, const size_t j)
    void gsl_matrix_set(void* m, const size_t i, const size_t j, double x)
    
    void gsl_matrix_set_all(void* m, double x)
    
cdef extern from "gsl/gsl_fft_complex.h" nogil:
    int gsl_fft_complex_radix2_forward(double* data, size_t stride, size_t n)
    int gsl_fft_complex_radix2_inverse(double* data, size_t stride, size_t n)
    
cdef extern from "gsl/gsl_multifit_nlinear.h" nogil:
    int GSL_SUCCESS
    
    ctypedef struct gsl_multifit_nlinear_fdf:
        int (* f) ( const void* x, void* params, void* residuals)
        int (* df) ( const void* x, void * params, void* jaccobian)
        int (* fvv) (const void* x, const void* v, void* params, void* fvv)
        size_t n # number residuals, returned by f
        size_t p # number of variables to fit, length of x
        void* params
        size_t nevalf # counts num function evalutions, set by init
        size_t nevaldf # counts df evaluations, set by init
        size_t nevalfvv # counts fvv evaluations, set by init
        
        
    ctypedef struct gsl_multifit_nlinear_parameters:
        pass
    
    gsl_multifit_nlinear_parameters gsl_multifit_nlinear_default_parameters()
 
    void* gsl_multifit_nlinear_trust ## type of minimizer
    
    void* gsl_multifit_nlinear_alloc(const void* min_type, const gsl_multifit_nlinear_parameters* params, const size_t n, const size_t p)
    
    int gsl_multifit_nlinear_init(const void* initial_x, gsl_multifit_nlinear_fdf* fdf, void* workspace)
    
    void gsl_multifit_nlinear_free(void* workspace)
    
    void* gsl_multifit_nlinear_position(const void* workspace)
    void* gsl_multifit_nlinear_residual(const void* workspace)
    
    int gsl_multifit_nlinear_driver(const size_t maxiter, const double xtol, const double gtol, const double ftol, void* callback, void* callback_params, int* info, void* workspace)
    
    size_t gsl_multifit_nlinear_niter(const void* workspace)

cdef double c_air_inverse = 1.000293/299792458.0

def set_c_air_inverse(double IN):
    """Set the internal variable: 1/(speed of light in air)"""
    global c_air_inverse
    c_air_inverse = IN

#### helper functions
cdef int argmax(double* data, int first, int last):
    """return index of maximum in data between first and last, not including last. if max is at first, will return 0, etc.."""
    cdef double current_best = data[first]
    cdef int current_index = 0
    cdef int i = first
    for i in range(first, last):
        if data[i] > current_best:
            current_best = data[i]
            current_index = i-first
    return current_index

#cdef double mean_square(double* data, int length):
#    """return the mean of the squares"""
#    cdef double ret = 0
#    cdef int i=0
#    for i in range(length):
#        ret += data[i]*data[i]
#    return ret/length
    

#### parabola fitter #####
cdef struct parabolic_fitter_struct:
    int num_points_to_fit
    int half_num_points
    double* fit_matrix ## 2D 3 X num_points matrix
    
cdef double fit_parabola( double* input_data, int peak_index, int data_length, parabolic_fitter_struct* data_struct):

    cdef double A=0
    cdef double B=0
    cdef double C=0
    
    cdef int i
    cdef double value
    cdef int data_i   
    for i in range( data_struct.num_points_to_fit ):
        data_i = peak_index + i - data_struct.half_num_points
        if data_i < 0:
            data_i += data_length
        elif data_i >= data_length:
            data_i -= data_length
            
        value = input_data[ data_i ]
        A += data_struct.fit_matrix[ i ]*value
        B += data_struct.fit_matrix[ i + data_struct.num_points_to_fit]*value
        C += data_struct.fit_matrix[ i + 2*data_struct.num_points_to_fit]*value
        
    return -B/(2.0*A) + peak_index - data_struct.half_num_points
    

cdef class parabolic_fitter:
    cdef parabolic_fitter_struct fitter_data
    
    cdef double[:,:] fit_matrix
    
    def __init__(self, int num_data_points=5):
        
        tmp_matrix = np.zeros((num_data_points,3), dtype=np.double)
        for n_i in range(num_data_points):
            tmp_matrix[n_i, 0] = n_i**2
            tmp_matrix[n_i, 1] = n_i
            tmp_matrix[n_i, 2] = 1.0
        self.fit_matrix = np.linalg.pinv( tmp_matrix )
        
        self.fitter_data.num_points_to_fit = num_data_points
        self.fitter_data.half_num_points = int( (num_data_points-1)/2 )
        self.fitter_data.fit_matrix = &self.fit_matrix[0,0]
        
    def fit(self, np.ndarray[double, ndim=1] data, index=None):
        cdef int peak_index
        if index is None:
            peak_index = np.argmax( data )
        else:
            peak_index = index
            
        cdef int L = len(data)
        
        return fit_parabola( &data[0], peak_index, L, &self.fitter_data )
        










#### upsample and correlate ######
        
def next_power_of_two(X):
    return int( 2**(np.ceil((np.log2(X)))) )
    
cdef class autoadjusting_upsample_and_correlate:
    
    cdef long upsample_factor
    cdef long current_input_length
    cdef long output_length
    
    cdef long ref_length
    
    cdef np.ndarray workspace_input
    cdef np.ndarray workspace_ref
    cdef np.ndarray output
    
    cdef complex[:] workspace_ref_tmp
    
#    cdef complex[:] workspace_input_array
#    cdef complex[:] workspace_ref_array
#    cdef complex[:] workspace_ref_tmp_array
#    cdef complex[:] output_array
    
    def __init__(self, long min_data_length, long upsample_factor):
        self.upsample_factor = upsample_factor
        self.current_input_length = 0
        self.set( min_data_length )
        
    def set(self, long data_length, np.ndarray[complex, ndim=1] old_input=None, np.ndarray[complex, ndim=1] old_ref=None, np.ndarray[complex, ndim=1] old_ref_tmp= None):
        
        cdef long input_length
        
        if data_length > self.current_input_length:
        
            input_length = next_power_of_two(data_length)
            self.current_input_length = input_length
            self.output_length = 2*input_length*self.upsample_factor
            
            self.workspace_input = np.zeros(2*input_length, dtype=np.complex)
            if old_input is not None:
                self.workspace_input[:len(old_input)]  = old_input
                
            self.workspace_ref = np.zeros(2*input_length, dtype=np.complex)
            if old_ref is not None:
                self.workspace_ref[:len(old_ref)]  = old_ref
                
            self.workspace_ref_tmp = np.zeros(2*input_length, dtype=np.complex)
            if old_ref_tmp is not None:
                self.workspace_ref_tmp[:len(old_ref_tmp)]  = old_ref_tmp
            
            self.output = np.empty(self.output_length, dtype=np.complex)
        else:
            
            if old_input is not None:
                self.workspace_input[:len(old_input)]  = old_input
                self.workspace_input[len(old_input):]  = 0.0
            else:
                self.workspace_input[:] = 0.0
                
            if old_ref is not None:
                self.workspace_ref[:len(old_ref)]  = old_ref
                self.workspace_ref[len(old_ref):]  = 0.0
            else:
                self.workspace_ref[:] = 0.0
                
            if old_ref_tmp is not None:
                self.workspace_ref_tmp[:len(old_ref_tmp)]  = old_ref_tmp
                self.workspace_ref_tmp[len(old_ref_tmp):]  = 0.0
            else:
                self.workspace_ref_tmp[:] = 0.0
    
#        self.workspace_input_array   = workspace_input
#        self.workspace_ref_array     = workspace_ref
#        self.workspace_ref_tmp_array = workspace_ref_tmp
#        self.output_array            = output
                
    def get_current_output_length(self):
        return self.output_length
                
    def set_referance(self, data):
        self.ref_length = len(data)
        self.set( self.ref_length, old_ref=data )
        
    def correlate(self, data):
        cdef long data_length = len(data)
        self.set(data_length, old_input=data, old_ref=self.workspace_ref ) ## data into workspace_input, ref in workspace_ref
        
        cdef long input_length = next_power_of_two( max( data_length, self.ref_length ) )
        cdef long out_length = 2*input_length*self.upsample_factor
        
        
        #### move some memory around 
        cdef complex[:] input_slice = self.workspace_input[:2*input_length]
        gsl_fft_complex_radix2_forward(<double*>&input_slice[0], 1, 2*input_length)
        
#        print(2*input_length)
#        self.workspace_ref[0:2*input_length]
#        self.workspace_ref_tmp[0:2*input_length]
#        AM HERE, LINE BELOW AINT WORKIN
        cdef complex[:] TMP = self.workspace_ref[0:2*input_length]
        self.workspace_ref_tmp[0:2*input_length] = TMP
        gsl_fft_complex_radix2_forward(<double*>&self.workspace_ref_tmp[0], 1, 2*input_length)
        
        cdef np.ndarray[complex, ndim=1] output_slice = self.output[:out_length]
        
        
        cdef long A
        cdef long B
        
        #### set lower positive
        A = 0
        B = (2*input_length + 1) // 2
        np.conjugate(self.workspace_ref_tmp[A:B], out=output_slice[A:B])
        output_slice[A:B] *= input_slice[A:B]
        
        ## set upper frequencies to zero (upsample)
        A = (2*input_length + 1) // 2
        B = -(2*input_length - 1) // 2
        output_slice[A:B] = 0.0
        
        ## set lower negative
        A = out_length - (2*input_length - 1) // 2
        B = 2*input_length - (2*input_length - 1) // 2
        np.conjugate(self.workspace_ref_tmp[B:2*input_length], out=output_slice[A:])
        output_slice[A:] *= input_slice[B:]
        
        # special treatment if low number of points is even. So far we have set Y[-N/2]=X[-N/2]
        output_slice[out_length-input_length] =  output_slice[out_length-input_length]*0.5  # halve the component at -N/2
#        cdef complex temp = 
        output_slice[input_length:input_length+1] = output_slice[out_length-input_length]  # set that equal to the component at -Nx/2
    
        
        gsl_fft_complex_radix2_inverse(<double*>&output_slice[0], 1, out_length)
        
        return output_slice

        
        
        
        
        
        

#### planewave fitter ####

cdef struct planewave_locate_struct:
    long num_antennas ## not including referance antenna!!!
    long total_cross_correlation_length
    double CC_sample_time ## to be 5.0E-9/upsample_factor
#    double half_window_length
    
    double* cross_correlation_data## 2D antennas X cross-correlation, assume is HE 
    double* relative_antennas_locations    ## 2D antennas X (xyz) (relative to referance antenna)
    double* antenna_delays        ## 1D antennas, relative to referance antenna
    long* mask                     ## 1D antennas
    long* used_CC_length                     ## 1D antennas
    
    double* current_workspace     ## 1D antennas
    double* measured_dt           ## 1D antennas
    
    parabolic_fitter_struct* parabolic_fitter
    
cdef void set_planewave_model(double Zenith, double Azimuth, planewave_locate_struct* data_struct) nogil:
    """calculates the arrival dt for each antenna, stores result in data_struct.current_workspace """
    cdef double cos_Ze = cos( Zenith )
    cdef double sin_Ze = sin( Zenith )
    cdef double cos_Az = cos( Azimuth )
    cdef double sin_Az = sin( Azimuth )
    
    cdef long ant_i = 0
    for ant_i in range(data_struct.num_antennas):
        data_struct.current_workspace[ ant_i ] = cos_Ze*data_struct.relative_antennas_locations[ ant_i*3 + 2 ]
        data_struct.current_workspace[ ant_i ] += sin_Ze*cos_Az*data_struct.relative_antennas_locations[ ant_i*3 + 0 ]
        data_struct.current_workspace[ ant_i ] += sin_Ze*sin_Az*data_struct.relative_antennas_locations[ ant_i*3 + 1 ]
        data_struct.current_workspace[ ant_i ] *= -c_air_inverse
        
cdef double image(double Zenith, double Azimuth, planewave_locate_struct* data_struct):
    """returns sum of hilbert envelopes of cross corelation. (envelope becouse less sensative)"""
    
    set_planewave_model( Zenith, Azimuth, data_struct ) ## load model into current_workspace
        
    
    cdef long ant_i
    cdef double predicted_dt
    cdef long predict_i
    cdef double* CC
    
    cdef double data=0
    for ant_i in range( data_struct.num_antennas ):
        if data_struct.mask[ant_i] == 0:
            data_struct.measured_dt[ant_i] = 0
            data_struct.current_workspace[ant_i] = 0
            continue
        
        predicted_dt = data_struct.current_workspace[ant_i] - data_struct.antenna_delays[ant_i]
        predict_i = long( predicted_dt / data_struct.CC_sample_time )
        
        if predict_i < 0:
            predict_i += data_struct.used_CC_length[ant_i]
        
        
        CC = &data_struct.cross_correlation_data[ data_struct.total_cross_correlation_length*ant_i ]
        data += CC[ predict_i ]
        
    return data

cdef void measure_close_dt(double Zenith, double Azimuth, planewave_locate_struct* data_struct):
    """find the T of the peak closest to a predicted arrival time. places the residual in current_workspace."""
        
    set_planewave_model( Zenith, Azimuth, data_struct ) ## load model into current_workspace
        
    cdef long ant_i
    cdef double predicted_dt
    cdef double* CC
    
    cdef long current_i
    cdef double current_amp
    cdef long previous_i
    cdef double previous_amp
    cdef long next_i
    cdef double next_amp
    
    cdef double peak_location
    
    cdef int looping
    for ant_i in range( data_struct.num_antennas ):
        if data_struct.mask[ant_i] == 0:
            data_struct.measured_dt[ant_i] = 0
            data_struct.current_workspace[ant_i] = 0
            continue
        
        predicted_dt = data_struct.current_workspace[ant_i] - data_struct.antenna_delays[ant_i]
        current_i = long( predicted_dt / data_struct.CC_sample_time )
        CC = &data_struct.cross_correlation_data[ data_struct.total_cross_correlation_length*ant_i ]
        
        looping = True
        while looping:
            if current_i < 0:
                current_i += data_struct.used_CC_length[ant_i]
            if current_i >= data_struct.used_CC_length[ant_i]:
                current_i -= data_struct.used_CC_length[ant_i]
                
            current_amp = CC[ current_i ]
            
            previous_i = current_i-1
            if previous_i<0:
                previous_i += data_struct.used_CC_length[ant_i]
            previous_amp = CC[ previous_i ]
                
            next_i = current_i+1
            if next_i >= data_struct.used_CC_length[ant_i]:
                next_i -= data_struct.used_CC_length[ant_i]
            next_amp = CC[ next_i ]
            
            if (current_amp>=previous_amp) and (current_amp>=next_amp):
                looping = False ## done!
            elif (current_amp>=previous_amp) and (current_amp<next_amp):
                current_i += 1 ## move forward
            elif (current_amp<previous_amp) and (current_amp>=next_amp):
                current_i -= 1 ## move backward
            elif (current_amp<previous_amp) and (current_amp<next_amp):
                ## saddle!!
                if previous_amp >= next_amp:
                    current_i -= 1 ## move backward
                elif previous_amp < next_amp:
                    current_i += 1 ## move forward
                    
        peak_location = fit_parabola( CC, current_i, data_struct.used_CC_length[ant_i], data_struct.parabolic_fitter )
        
        if peak_location >  data_struct.used_CC_length[ant_i]/2:
            peak_location -= data_struct.used_CC_length[ant_i]
            
        peak_location *= data_struct.CC_sample_time
        peak_location += data_struct.antenna_delays[ant_i]
        data_struct.measured_dt[ant_i] = peak_location
        data_struct.current_workspace[ant_i] -= peak_location
        
cdef void measure_peak_dt(planewave_locate_struct* data_struct):
    """find the T of the highest peak. places the residual in current_workspace."""
        
    cdef long ant_i
    cdef double* CC
    cdef long CC_length
    
    cdef double current_amp
    cdef long best_i
    cdef long current_i
    
    cdef double peak_location
    
    for ant_i in range( data_struct.num_antennas ):
        if data_struct.mask[ant_i] == 0:
            data_struct.measured_dt[ant_i] = 0
            data_struct.current_workspace[ant_i] = 0
            continue
        
        CC = &data_struct.cross_correlation_data[ data_struct.total_cross_correlation_length*ant_i ]
        CC_length = data_struct.used_CC_length[ant_i]
        
        current_amp = CC[0]
        best_i = 0
        
        for current_i in range(CC_length):
            if CC[current_i] > current_amp:
                current_amp = CC[current_i]
                best_i = current_i
                    
        peak_location = fit_parabola( CC, best_i, CC_length, data_struct.parabolic_fitter )
        
        if peak_location >  data_struct.used_CC_length[ant_i]/2:
            peak_location -= data_struct.used_CC_length[ant_i]
            
        peak_location *= data_struct.CC_sample_time
        peak_location += data_struct.antenna_delays[ant_i]
        data_struct.measured_dt[ant_i] = peak_location
        data_struct.current_workspace[ant_i] -= peak_location 

cdef int planewave_residuals(const void* ZeAz, void* data_struct_void, void* out_residuals) nogil:
    
    cdef planewave_locate_struct* data_struct = <planewave_locate_struct*>data_struct_void
    
    set_planewave_model( gsl_vector_get(ZeAz, 0), gsl_vector_get(ZeAz, 1), data_struct ) ## load model into current_workspace
    
    cdef int ant_i
    for ant_i in range( data_struct.num_antennas ):
        if data_struct.mask[ant_i] == 0:
            gsl_vector_set(out_residuals,ant_i, 0)
            continue
        
        gsl_vector_set(out_residuals,ant_i, data_struct.current_workspace[ ant_i ] - data_struct.measured_dt[ant_i])
        
    return GSL_SUCCESS
        
cdef class planewave_locator_helper:
    
    cdef planewave_locate_struct planewave_data

    
    cdef parabolic_fitter parabolic_peak_fitter
    cdef double upsample_factor
    cdef double[:,:] ZeAz_test_points
    
    ## non-linear fitting params
    cdef gsl_multifit_nlinear_fdf fit_function
    cdef gsl_multifit_nlinear_parameters fit_parameters
    cdef void* fit_workspace
    cdef void* vector_ZeAz
    cdef void* vector_residuals
    
    def __init__(self, double upsample_factor, long num_antennas, long max_CC_length, int axis_itters):
        self.parabolic_peak_fitter = parabolic_fitter()
        self.upsample_factor = upsample_factor
        
        self.planewave_data.total_cross_correlation_length = max_CC_length
        self.planewave_data.num_antennas = num_antennas
        self.planewave_data.CC_sample_time = 5.0E-9/upsample_factor
        
        self.planewave_data.parabolic_fitter = &self.parabolic_peak_fitter.fitter_data
        
        
        #### arange itters equally about the sky,=
        TMP = []
        space = np.linspace(-1, 1, num=axis_itters)
        for alpha in space:
            for beta in space:
                if alpha*alpha + beta*beta < 1: ## we are on the sky
                    Az = np.arctan2(alpha, beta)
                    Ze = np.arcsin( alpha/np.sin(Az) )
                    
                    TMP.append( [Ze,Az] )
        self.ZeAz_test_points = np.array(TMP)
        
        ### set non-linear things
        self.fit_parameters = gsl_multifit_nlinear_default_parameters()
        
        self.fit_function.f = &planewave_residuals
        self.fit_function.df = NULL
        self.fit_function.fvv = NULL
        self.fit_function.n = num_antennas
        self.fit_function.p = 2
        self.fit_function.params = <void*>&self.planewave_data
        
        self.fit_workspace = gsl_multifit_nlinear_alloc(gsl_multifit_nlinear_trust, &self.fit_parameters,
                                                       num_antennas, self.fit_function.p)
        
        self.vector_ZeAz = gsl_multifit_nlinear_position( self.fit_workspace )
        self.vector_residuals = gsl_multifit_nlinear_residual( self.fit_workspace )
        
    def __dealloc__(self):
        gsl_multifit_nlinear_free( self.fit_workspace )
        
    def get_test_points(self):
        return np.array(self.ZeAz_test_points)
        
    def set_memory(self, np.ndarray[double, ndim=2] cross_correlations, np.ndarray[double, ndim=2] relative_antenna_locs, np.ndarray[double, ndim=1] antenna_delays,
                   np.ndarray[double, ndim=1] workspace, np.ndarray[double, ndim=1] measured_dt, np.ndarray[long, ndim=1] mask, np.ndarray[long, ndim=1] CC_lengths):
        
        self.planewave_data.cross_correlation_data = &cross_correlations[0,0]
        self.planewave_data.relative_antennas_locations = &relative_antenna_locs[0,0]
        self.planewave_data.antenna_delays = &antenna_delays[0]
        self.planewave_data.mask = &mask[0]
        self.planewave_data.used_CC_length = &CC_lengths[0]
        self.planewave_data.current_workspace = &workspace[0]
        self.planewave_data.measured_dt = &measured_dt[0] 
        
    def run_brute(self, np.ndarray[double, ndim=1] out_ZeAz):
        
        cdef double best_image = 0.0
        
        cdef long i
        cdef double current_image
        for i in range(len(self.ZeAz_test_points)):
            current_image = image(self.ZeAz_test_points[i,0], self.ZeAz_test_points[i,1], &self.planewave_data)
            
            if current_image > best_image:
                
                best_image = current_image
                out_ZeAz[0] = self.ZeAz_test_points[i,0]
                out_ZeAz[1] = self.ZeAz_test_points[i,1]
                
#        if out_ZeAz[0] <= 0:
#            out_ZeAz[0] = 0.0001
#        if out_ZeAz[0] >= np.pi/2:
#            out_ZeAz[0] = (np.pi/2)*0.99
#        if out_ZeAz[1] <= 0:
#            out_ZeAz[1] = 0.0001
#        if out_ZeAz[1] >= np.pi*2:
#            out_ZeAz[1] = (np.pi*2)*0.99
            
    def image_value(self, ZeAz):
        return image(ZeAz[0], ZeAz[1], &self.planewave_data)
    
    def run_minimizer(self, np.ndarray[double, ndim=1] guess_ZeAz, int max_itters, double xtol, double gtol, double ftol):
        """ tries to fit the given info. returns True if it has converged, False otherwise. Also returns final ZeAz and RMS"""
        
        measure_close_dt(guess_ZeAz[0], guess_ZeAz[1], &self.planewave_data) ## sets the measured data
#        measure_peak_dt(&self.planewave_data) ## sets the measured data
        
        gsl_vector_set(self.vector_ZeAz, 0, guess_ZeAz[0])
        gsl_vector_set(self.vector_ZeAz, 1, guess_ZeAz[1])
        
        gsl_multifit_nlinear_init(self.vector_ZeAz, &self.fit_function, self.fit_workspace)
        
        cdef int info
        cdef int ret
        ret = gsl_multifit_nlinear_driver(max_itters, xtol, gtol, ftol, 
                                          NULL,NULL , &info, self.fit_workspace)
        
        guess_ZeAz[0] = gsl_vector_get(self.vector_ZeAz, 0)
        guess_ZeAz[1] = gsl_vector_get(self.vector_ZeAz, 1)
        
        planewave_residuals(self.vector_ZeAz, <void*>&self.planewave_data, self.vector_residuals )
        
        cdef int N = 0
        cdef double total = 0
        cdef double temp
        cdef int i =0
        for i in range( self.planewave_data.num_antennas ):
            if self.planewave_data.mask[i] == 1:
                N += 1
                temp = gsl_vector_get(self.vector_residuals, i)
                total += temp*temp
        
        return ret==GSL_SUCCESS, sqrt( total/N )
    
    def get_num_iters(self):
        return gsl_multifit_nlinear_niter( self.fit_workspace )
    
    def get_model_dt(self, ZeAz):
        """sets planewave model to the input workspace. Note: other functions could overwrite this data!"""
        set_planewave_model(ZeAz[0], ZeAz[1], &self.planewave_data)
        
        
#### pointsource fitter ####
cdef struct pointsource_data_struct:
    long num_antennas

    long ref_ant_i
    double* antennas_locations    ## 2D antennas X (xyz)
    double* measured_times        ## 1D antennas, relative to referance antenna
    double* weights              ## 1D antennas
    long* mask                    ## 1D antennas
    
cdef double relative_arrival_time(long ant_i, double* XYZc, pointsource_data_struct* data_struct) nogil:
    """return arrival time on antenna i, if source is at XYZc, relative to the referance antenna"""
    cdef long ref_i = data_struct.ref_ant_i
    cdef double* ref_XYZ = &data_struct.antennas_locations[3*ref_i]
    cdef double* ant_XYZ = &data_struct.antennas_locations[3*ant_i]
    
    cdef double dX = XYZc[0] - ant_XYZ[0]
    cdef double dY = XYZc[1] - ant_XYZ[1]
    cdef double dZ = XYZc[2] - ant_XYZ[2]
    cdef double dt = sqrt( dX*dX + dY*dY + dZ*dZ )
    
    ## this isn't necisary. Could do this before hand, would save computation power
    dX = XYZc[0] - ref_XYZ[0]
    dY = XYZc[1] - ref_XYZ[1]
    dZ = XYZc[2] - ref_XYZ[2]
    dt -= sqrt( dX*dX + dY*dY + dZ*dZ )
    
    return dt*c_air_inverse + XYZc[3]

cdef int pointsource_residuals(const void* XYZc, void* data_struct_void, void* out_residuals) nogil:
    
    cdef pointsource_data_struct* data_struct = <pointsource_data_struct*>data_struct_void
    cdef double* XYZc_p = gsl_vector_ptr( XYZc, 0 )
    
    cdef double TMP
    cdef int ant_i
    for ant_i in range( data_struct.num_antennas ):
        if data_struct.mask[ant_i] == 0:
            gsl_vector_set(out_residuals, ant_i, 0)
            continue
        
        TMP = relative_arrival_time( ant_i, XYZc_p, data_struct) - data_struct.measured_times[ ant_i ]
        gsl_vector_set(out_residuals,ant_i, TMP/data_struct.weights[ ant_i ])
        
    return GSL_SUCCESS

cdef int pointsource_jacobian(const void* XYZc, void* data_struct_void, void* jacobian_matrix) nogil:
    
    cdef pointsource_data_struct* data_struct = <pointsource_data_struct*>data_struct_void
#    cdef double* XYZc_p = gsl_vector_ptr( XYZc, 0 )
    
    cdef long ref_i = data_struct.ref_ant_i
    cdef double refX = gsl_vector_get(XYZc, 0) - data_struct.antennas_locations[3*ref_i + 0]
    cdef double refY = gsl_vector_get(XYZc, 1) - data_struct.antennas_locations[3*ref_i + 1]
    cdef double refZ = gsl_vector_get(XYZc, 2) - data_struct.antennas_locations[3*ref_i + 2]
    cdef double refNorm = sqrt( refX*refX + refY*refY + refZ*refZ )
    refX /= refNorm
    refY /= refNorm
    refZ /= refNorm
    
    cdef double antX
    cdef double antY
    cdef double antZ
    cdef double antNorm
    cdef int ant_i
    for ant_i in range( data_struct.num_antennas ):
        if data_struct.mask[ant_i] == 0:
            gsl_matrix_set(jacobian_matrix, ant_i, 0, 0)
            gsl_matrix_set(jacobian_matrix, ant_i, 1, 0)
            gsl_matrix_set(jacobian_matrix, ant_i, 2, 0)
            gsl_matrix_set(jacobian_matrix, ant_i, 3, 0)
            continue
        
        antX = gsl_vector_get(XYZc, 0) - data_struct.antennas_locations[3*ant_i + 0]
        antY = gsl_vector_get(XYZc, 1) - data_struct.antennas_locations[3*ant_i + 1]
        antZ = gsl_vector_get(XYZc, 2) - data_struct.antennas_locations[3*ant_i + 2]
        antNorm = sqrt( antX*antX + antY*antY + antZ*antZ )
        antX /= antNorm
        antY /= antNorm
        antZ /= antNorm
        
        gsl_matrix_set(jacobian_matrix, ant_i, 0, (antX-refX)*c_air_inverse/data_struct.weights[ ant_i ] )  
        gsl_matrix_set(jacobian_matrix, ant_i, 1, (antY-refY)*c_air_inverse/data_struct.weights[ ant_i ])
        gsl_matrix_set(jacobian_matrix, ant_i, 2, (antZ-refZ)*c_air_inverse/data_struct.weights[ ant_i ])
        gsl_matrix_set(jacobian_matrix, ant_i, 3, 1.0/data_struct.weights[ ant_i ])
        
    return GSL_SUCCESS
    
cdef class pointsource_locator:
    
    cdef pointsource_data_struct data_struct
    
#    long num_antennas
#
#    long ref_ant_i
#    double* antennas_locations    ## 2D antennas X (xyz)
#    double* measured_times        ## 1D antennas, relative to referance antenna
#    double* weights              ## 1D antennas
#    long* mask                    ## 1D antennas
    
    
    
    ## non-linear fitting params
    cdef gsl_multifit_nlinear_fdf fit_function
    cdef gsl_multifit_nlinear_parameters fit_parameters
    cdef void* fit_workspace
    cdef void* vector_XYZc
    cdef void* vector_weighted_residuals
    
    def __init__(self, num_antennas):
        self.data_struct.num_antennas = num_antennas
        
        self.fit_parameters = gsl_multifit_nlinear_default_parameters()
        
        self.fit_function.f = &pointsource_residuals
        self.fit_function.df = NULL #&pointsource_jacobian
        self.fit_function.fvv = NULL
        self.fit_function.n = num_antennas
        self.fit_function.p = 4
        self.fit_function.params = <void*>&self.data_struct
        
        self.fit_workspace = gsl_multifit_nlinear_alloc(gsl_multifit_nlinear_trust, &self.fit_parameters,
                                                       num_antennas, self.fit_function.p)
        
        self.vector_XYZc = gsl_multifit_nlinear_position( self.fit_workspace )
        self.vector_weighted_residuals = gsl_multifit_nlinear_residual( self.fit_workspace )
        
    def __dealloc(self):
        gsl_multifit_nlinear_free( self.fit_workspace )
    
    def set_memory(self, np.ndarray[double, ndim=2] antenna_locs, np.ndarray[double, ndim=1] measured_times, np.ndarray[long, ndim=1] mask, np.ndarray[double, ndim=1] weights ):
        self.data_struct.antennas_locations = &antenna_locs[0,0]
        self.data_struct.measured_times = &measured_times[0]
        self.data_struct.mask = &mask[0]
        self.data_struct.weights = &weights[0]
    
    def set_ref_antenna(self, long ref_ant_i):
        self.data_struct.ref_ant_i = ref_ant_i
    
    def relative_arrival_time(self, long ant_i, np.ndarray[double, ndim=1] XYZc):
        return relative_arrival_time( ant_i, &XYZc[0], &self.data_struct )
    
    def run_minimizer(self, np.ndarray[double, ndim=1] guess_XYZc, int max_itters, double xtol, double gtol, double ftol):
        
        gsl_vector_set(self.vector_XYZc, 0, guess_XYZc[0])
        gsl_vector_set(self.vector_XYZc, 1, guess_XYZc[1])
        gsl_vector_set(self.vector_XYZc, 2, guess_XYZc[2])
        gsl_vector_set(self.vector_XYZc, 3, guess_XYZc[3])
        
        gsl_multifit_nlinear_init(self.vector_XYZc, &self.fit_function, self.fit_workspace)
        
        cdef int info
        cdef int ret
        ret = gsl_multifit_nlinear_driver(max_itters, xtol, gtol, ftol, 
                                          NULL,NULL , &info, self.fit_workspace)
    
        guess_XYZc[0] = gsl_vector_get(self.vector_XYZc, 0)
        guess_XYZc[1] = gsl_vector_get(self.vector_XYZc, 1)
        guess_XYZc[2] = gsl_vector_get(self.vector_XYZc, 2)
        guess_XYZc[3] = gsl_vector_get(self.vector_XYZc, 3)
        
        cdef int N = 0
        cdef double tmp
        cdef double chi_squared = 0.0
        cdef int ant_i = 0
        for ant_i in range( self.data_struct.num_antennas ):
            if self.data_struct.mask[ ant_i ]:
                N += 1
                tmp = gsl_vector_get(self.vector_weighted_residuals, ant_i)
                chi_squared += tmp*tmp
                
        return ret==GSL_SUCCESS, chi_squared/(N-4)
    
    def get_num_iters(self):
        return gsl_multifit_nlinear_niter( self.fit_workspace )
    
    def get_RMS(self):
        
        cdef double* XYZc_p = gsl_vector_ptr( self.vector_XYZc, 0 )
    
        cdef double TMP
        cdef double RMS = 0
        cdef long N = 0
        cdef int ant_i
        for ant_i in range( self.data_struct.num_antennas ):
            if self.data_struct.mask[ant_i] == 0:
                continue
            
            TMP = relative_arrival_time( ant_i, XYZc_p, &self.data_struct) - self.data_struct.measured_times[ ant_i ]
            RMS += TMP*TMP
            N += 1
            
        return sqrt( RMS/N )
    
    def load_covariance_matrix(self, np.ndarray[double, ndim=1] guess_XYZc, np.ndarray[double, ndim=2] covariance_out ):
            

        cdef double inverse_norm
        cdef double inverse_norm_cubed
        
        #### first, ref antenna bit
        cdef double refX = guess_XYZc[0]
        cdef double refY = guess_XYZc[1]
        cdef double refZ = guess_XYZc[2]
        
        refX -= self.data_struct.antennas_locations[ 3*self.data_struct.ref_ant_i + 0 ]
        refY -= self.data_struct.antennas_locations[ 3*self.data_struct.ref_ant_i + 1 ]
        refZ -= self.data_struct.antennas_locations[ 3*self.data_struct.ref_ant_i + 2 ]
        
        cdef double ref_norm = sqrt( refX*refX + refY*refY + refZ*refZ )
        inverse_norm = 1.0/ref_norm
        inverse_norm_cubed = inverse_norm*inverse_norm*inverse_norm
        
        cdef double refHess_00 = -refX*refX*inverse_norm_cubed
        cdef double refHess_01 = -refX*refY*inverse_norm_cubed
        cdef double refHess_02 = -refX*refZ*inverse_norm_cubed
        cdef double refHess_10 = -refY*refX*inverse_norm_cubed
        cdef double refHess_11 = -refY*refY*inverse_norm_cubed
        cdef double refHess_12 = -refY*refZ*inverse_norm_cubed
        cdef double refHess_20 = -refZ*refX*inverse_norm_cubed
        cdef double refHess_21 = -refZ*refY*inverse_norm_cubed
        cdef double refHess_22 = -refZ*refZ*inverse_norm_cubed
        
        refHess_00 += inverse_norm
        refHess_11 += inverse_norm
        refHess_22 += inverse_norm
        
        refX *= inverse_norm
        refY *= inverse_norm
        refZ *= inverse_norm
        
        
        cdef double antX
        cdef double antY
        cdef double antZ
        
        cdef double ant_norm
    
        cdef double antMat_00
        cdef double antMat_01
        cdef double antMat_02
        cdef double antMat_10
        cdef double antMat_11
        cdef double antMat_12
        cdef double antMat_20
        cdef double antMat_21
        cdef double antMat_22
    
        cdef double TOA_measurment
        cdef double weight
        
        cdef double HessOut_00 = 0.0
        cdef double HessOut_01 = 0.0
        cdef double HessOut_02 = 0.0
        cdef double HessOut_10 = 0.0
        cdef double HessOut_11 = 0.0
        cdef double HessOut_12 = 0.0
        cdef double HessOut_20 = 0.0
        cdef double HessOut_21 = 0.0
        cdef double HessOut_22 = 0.0
        
        for ant_i in range(  self.data_struct.num_antennas  ):
            if not self.data_struct.mask[ ant_i ]:
                continue
            
            #### now, the measurment antenna bit
            antX = guess_XYZc[0]
            antY = guess_XYZc[1]
            antZ = guess_XYZc[2]
            
            antX -= self.data_struct.antennas_locations[ 3*ant_i + 0 ]
            antY -= self.data_struct.antennas_locations[ 3*ant_i + 1 ]
            antZ -= self.data_struct.antennas_locations[ 3*ant_i + 2 ]
            
            ant_norm = sqrt( antX*antX + antY*antY + antZ*antZ )
            inverse_norm = 1.0/ant_norm
            inverse_norm_cubed = inverse_norm*inverse_norm*inverse_norm
            
            antMat_00 = -antX*antX*inverse_norm_cubed
            antMat_01 = -antX*antY*inverse_norm_cubed
            antMat_02 = -antX*antZ*inverse_norm_cubed
            antMat_10 = -antY*antX*inverse_norm_cubed
            antMat_11 = -antY*antY*inverse_norm_cubed
            antMat_12 = -antY*antZ*inverse_norm_cubed
            antMat_20 = -antZ*antX*inverse_norm_cubed
            antMat_21 = -antZ*antY*inverse_norm_cubed
            antMat_22 = -antZ*antZ*inverse_norm_cubed
            
            antMat_00 += inverse_norm
            antMat_11 += inverse_norm
            antMat_22 += inverse_norm
            
            
            #### take differance of hessians , multiple, add to total hessian
            
            antMat_00 -= refHess_00
            antMat_01 -= refHess_01
            antMat_02 -= refHess_02
            antMat_10 -= refHess_10
            antMat_11 -= refHess_11
            antMat_12 -= refHess_12
            antMat_20 -= refHess_20
            antMat_21 -= refHess_21
            antMat_22 -= refHess_22
            
            TOA_measurment = (ant_norm - ref_norm)*c_air_inverse            
            TOA_measurment -= self.data_struct.measured_times[ ant_i ] - guess_XYZc[3]
            weight= 1.0/(self.data_struct.weights[ ant_i ]*self.data_struct.weights[ ant_i ])
            
            antMat_00 *= c_air_inverse*TOA_measurment*weight
            antMat_01 *= c_air_inverse*TOA_measurment*weight
            antMat_02 *= c_air_inverse*TOA_measurment*weight
            antMat_10 *= c_air_inverse*TOA_measurment*weight
            antMat_11 *= c_air_inverse*TOA_measurment*weight
            antMat_12 *= c_air_inverse*TOA_measurment*weight
            antMat_20 *= c_air_inverse*TOA_measurment*weight
            antMat_21 *= c_air_inverse*TOA_measurment*weight
            antMat_22 *= c_air_inverse*TOA_measurment*weight
            
            HessOut_00 += antMat_00
            HessOut_01 += antMat_01
            HessOut_02 += antMat_02
            HessOut_10 += antMat_10
            HessOut_11 += antMat_11
            HessOut_12 += antMat_12
            HessOut_20 += antMat_20
            HessOut_21 += antMat_21
            HessOut_22 += antMat_22
            
            
            #### now calculate the single derivative component and add to total hessian

            antX *= inverse_norm
            antY *= inverse_norm
            antZ *= inverse_norm
            antX -= refX
            antY -= refY
            antZ -= refZ
            antX *= c_air_inverse
            antY *= c_air_inverse
            antZ *= c_air_inverse
            
            antMat_00 = antX*antX*weight
            antMat_01 = antX*antY*weight
            antMat_02 = antX*antZ*weight
            antMat_10 = antY*antX*weight
            antMat_11 = antY*antY*weight
            antMat_12 = antY*antZ*weight
            antMat_20 = antZ*antX*weight
            antMat_21 = antZ*antY*weight
            antMat_22 = antZ*antZ*weight
            
            HessOut_00 += antMat_00
            HessOut_01 += antMat_01
            HessOut_02 += antMat_02
            HessOut_10 += antMat_10
            HessOut_11 += antMat_11
            HessOut_12 += antMat_12
            HessOut_20 += antMat_20
            HessOut_21 += antMat_21
            HessOut_22 += antMat_22
            
            
        #### now we invert  ####
        # first we calculate the norm
        cdef double hessNorm = HessOut_00*norm( HessOut_11, HessOut_12,
                                                HessOut_21, HessOut_22 )
        hessNorm -= HessOut_01*norm( HessOut_10, HessOut_12,
                                     HessOut_20, HessOut_22 )
        hessNorm += HessOut_02*norm( HessOut_10, HessOut_11,
                                     HessOut_20, HessOut_21 )
        hessNorm = 1.0/hessNorm
        
        # then the little fiddly bits, using the symetry property
        covariance_out[0,0] = hessNorm*norm( HessOut_11, HessOut_12,
                                             HessOut_21, HessOut_22  )
        covariance_out[0,1] = hessNorm*norm( HessOut_02, HessOut_01,
                                             HessOut_22, HessOut_21  )
        covariance_out[0,2] = hessNorm*norm( HessOut_01, HessOut_02,
                                             HessOut_11, HessOut_12  )
        
        covariance_out[1,0] = covariance_out[0,1]
        
        covariance_out[1,1] = hessNorm*norm( HessOut_00, HessOut_02,
                                             HessOut_20, HessOut_22  )
        covariance_out[1,2] = hessNorm*norm( HessOut_02, HessOut_00,
                                             HessOut_12, HessOut_10  )
        
        covariance_out[2,0] = covariance_out[0,2]
        
        covariance_out[2,1] = covariance_out[1,2]
        
        covariance_out[2,2] = hessNorm*norm( HessOut_00, HessOut_01,
                                             HessOut_10, HessOut_11  )
    
    
cdef inline double norm(double A, double B, double C, double D):
    return A*D - B*C
    
def abs_max( np.ndarray[complex, ndim=1] IN):
    cdef double MAX=0.0
    cdef double TMP
    cdef int i
    for i in range(len(IN)):
        TMP = cabs( IN[i] )
        if TMP > MAX:
            MAX = TMP
    return MAX
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
