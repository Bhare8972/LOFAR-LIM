#cython: language_level=3 
#cython: cdivision=True
#cython: boundscheck=True
#cython: linetrace=False
#cython: binding=False
#cython: profile=False


cimport numpy as np
import numpy as np
from libcpp cimport bool
from libc.math cimport sqrt, fabs, isfinite
from libc.stdlib cimport malloc, free

#cdef extern from "gsl/gsl_math.h" nogil:
#    int gsl_finite(const double x)

#cdef extern from "gsl/gsl_vector.h" nogil:
#    
#    void* gsl_vector_alloc(size_t n)
#    void gsl_vector_free(void* v)
#    
#    double gsl_vector_get(const void* v, const size_t i)
#    void gsl_vector_set(void* v, const size_t i, double x)
#    double* gsl_vector_ptr(void* v, size_t i)
#    
#    void gsl_vector_set_all(void* v, double x)
#    
#
#cdef extern from "gsl/gsl_matrix.h" nogil:
#    void* gsl_matrix_alloc(size_t n1, size_t n2)
#    void gsl_matrix_free(void *m)
#    
#    double gsl_matrix_get(void *m, size_t i, size_t j)
#    void gsl_matrix_set(void *m, size_t i, size_t j, double x)
#    void gsl_matrix_set_all(void *m, double x)
#
#cdef extern from "gsl/gsl_cblas.h" nogil:
#    int gsl_blas_ddot(void *x, void *y, double *result)


#cdef extern from "gsl/gsl_multifit_nlinear.h" nogil:
#    ctypedef struct gsl_multifit_nlinear_fdtype:
#        pass
#    
#    ctypedef struct gsl_multifit_nlinear_parameters:
#        void* trs
#        void* scale
#        void* solver
#        gsl_multifit_nlinear_fdtype fdtype
#        double factor_up
#        double factor_down
#        double avmax
#        double h_df
#        double hfvv
#        
#    ctypedef struct gsl_multifit_nlinear_fdf:
#        int (* f) (const void *x, void *params, void *f)
#        int (* df) (const void *x, void *params, void *J)
#        void *fvv
#        size_t n
#        size_t p
#        void *params
#        size_t nevalf
#        size_t nevaldf
#        size_t nevalfvv
#        
#    int GSL_SUCCESS
#    void* gsl_multifit_nlinear_trust
#    
#    gsl_multifit_nlinear_parameters gsl_multifit_nlinear_default_parameters()
#    void* gsl_multifit_nlinear_alloc(void *T,  gsl_multifit_nlinear_parameters *params, size_t n, size_t p)
#    int gsl_multifit_nlinear_init(void *x, gsl_multifit_nlinear_fdf *fdf, void *w)
#    int gsl_multifit_nlinear_driver( size_t maxiter, double xtol, double gtol, double ftol, void* callback, void *callback_params, int *info, void *w)
#    void* gsl_multifit_nlinear_jac(void *w)
#    int gsl_multifit_nlinear_covar(void *J, double epsrel, void *covar)
#    void* gsl_multifit_nlinear_residual(void *w) 
#    void* gsl_multifit_nlinear_position( void *w)
#    
#    void gsl_multifit_nlinear_free(void *w) 
    
        

cdef double c_air_inverse = 1.000293/299792458.0

cdef struct stationDelay_dataStruct:
    int num_station_delays 
    int num_antennas 
    int num_events 
    
#    np.ndarray[double, ndim=2] antenna_locations
#    np.ndarray[int, ndim=1] station_indexes 
#    np.ndarray[double, ndim=1] measurement_times 
#    np.ndarray[double, ndim=1] measurement_filter
    
    double[:,:] antenna_locations
    long[:] station_indexes 
    double[:] measurement_times 
    np.uint8_t[:] measurement_filter

#cdef int objective_fun(const void *x, void *_data, void *f):
#    cdef stationDelay_dataStruct* data = <stationDelay_dataStruct*> _data
#    
#    cdef double X
#    cdef double Y
#    cdef double Z
#    cdef double T
#    
#    cdef double dx
#    cdef double dy
#    cdef double dz
#    cdef double dt
#    
#    cdef int event_i
#    cdef int antenna_i
#    cdef int output_i = 0
#    cdef np.ndarray[double, ndim=1] measurement_slice
#    cdef np.ndarray[double, ndim=1] filter_slice
#    for event_i in range(data.num_events):
#        measurement_slice = data.measurement_times[event_i*data.num_antennas : (event_i+1)*data.num_antennas]
#        filter_slice = data.measurement_filter[event_i*data.num_antennas : (event_i+1)*data.num_antennas]
#        X = gsl_vector_get(x, data.num_station_delays + event_i*4 + 0)
#        Y = gsl_vector_get(x, data.num_station_delays + event_i*4 + 1)
#        Z = gsl_vector_get(x, data.num_station_delays + event_i*4 + 2)
#        Z = fabs(Z)
#        T = gsl_vector_get(x, data.num_station_delays + event_i*4 + 3)
#        
#        for antenna_i in range(data.num_antennas):
#            if filter_slice[antenna_i]:
#                
#                dx = data.antenna_locations[antenna_i, 0] - X
#                dy = data.antenna_locations[antenna_i, 1] - Y
#                dz = data.antenna_locations[antenna_i, 2] - Z
#                dt = sqrt(dx*dx + dy*dy + dz*dz)*c_air_inverse + T + gsl_vector_get(x, data.station_indexes[ antenna_i ] )
#                dt -= measurement_slice[ antenna_i ]
#                
#                gsl_vector_set( f, output_i, dt )
#                output_i += 1
    
#cdef int objective_jac(const void *x, void *_data, void *jac):
#    cdef stationDelay_dataStruct* data = <stationDelay_dataStruct*> _data
#    
#    gsl_matrix_set_all(jac, 0.0)
#    
#    cdef double X
#    cdef double Y
#    cdef double Z
#    cdef double T
#    
#    cdef double dx
#    cdef double dy
#    cdef double dz
#    cdef double inv_R
#    
#    cdef int event_i
#    cdef int antenna_i
#    cdef int output_i = 0
#    cdef np.ndarray[double, ndim=1] measurement_slice
#    cdef np.ndarray[double, ndim=1] filter_slice
#    for event_i in range(data.num_events):
#        measurement_slice = data.measurement_times[event_i*data.num_antennas : (event_i+1)*data.num_antennas]
#        filter_slice = data.measurement_filter[event_i*data.num_antennas : (event_i+1)*data.num_antennas]
#        X = gsl_vector_get(x, data.num_station_delays + event_i*4 + 0)
#        Y = gsl_vector_get(x, data.num_station_delays + event_i*4 + 1)
#        Z = gsl_vector_get(x, data.num_station_delays + event_i*4 + 2)
#        Z = fabs(Z)
#        T = gsl_vector_get(x, data.num_station_delays + event_i*4 + 3)
#        
#        for antenna_i in range(data.num_antennas):
#            if filter_slice[antenna_i]:
#                
#                dx = X - data.antenna_locations[antenna_i, 0]
#                dy = Y - data.antenna_locations[antenna_i, 1]
#                dz = Z - data.antenna_locations[antenna_i, 2]
#                
#                inv_R = c_air_inverse/sqrt(dx*dx + dy*dy + dz*dy)
#                
#                gsl_matrix_set(jac, output_i, data.num_station_delays + event_i*4 + 0, dx*inv_R)
#                gsl_matrix_set(jac, output_i, data.num_station_delays + event_i*4 + 1, dy*inv_R)
#                gsl_matrix_set(jac, output_i, data.num_station_delays + event_i*4 + 2, dz*inv_R)
#                
#                gsl_matrix_set(jac, output_i, data.num_station_delays + event_i*4 + 3, 1.0)
#                gsl_matrix_set(jac, output_i, data.station_indexes[ antenna_i ], 1.0)
#                
#                output_i += 1

cdef class stationDelay_fitter:
    
    cdef int total_num_measurments
    cdef int next_event_i
    
    
    cdef void *fitter_workspace
    cdef void *covariance_matrix
    cdef void *initial_guess_copy
    
    cdef stationDelay_dataStruct fitting_info
#            cdef int num_station_delays 
#            cdef int num_antennas 
#            cdef int num_events 
#            
#            cdef np.ndarray antenna_locations 
#            cdef np.ndarray station_indexes 
#            cdef np.ndarray measurement_times 
#            cdef np.ndarray measurement_filter 
    
    #### outputs placed in these two arrays
    cdef np.ndarray solution_out
    cdef np.ndarray estimated_error_out
    cdef double RMS_out
    
    def __init__(self, np.ndarray[double, ndim=2] _antenna_locations, np.ndarray[long, ndim=1] _station_indexes, _num_events, _num_station_delays):
        self.fitting_info.antenna_locations = _antenna_locations
        self.fitting_info.num_station_delays = _num_station_delays
        self.fitting_info.num_antennas = _antenna_locations.shape[0]
        self.fitting_info.num_events = _num_events
        self.fitting_info.station_indexes = _station_indexes
        
        self.fitting_info.measurement_times = np.empty( _num_events*self.fitting_info.num_antennas, dtype=np.double )
        self.fitting_info.measurement_filter = np.zeros( _num_events*self.fitting_info.num_antennas, np.uint8 )
        
        self.total_num_measurments = 0
        self.next_event_i = 0
        
    def set_event(self, np.ndarray[double, ndim=1] arrival_times):
        filter_slice = self.fitting_info.measurement_filter[self.next_event_i*self.fitting_info.num_antennas : (self.next_event_i+1)*self.fitting_info.num_antennas]
        measurment_slice = self.fitting_info.measurement_times[self.next_event_i*self.fitting_info.num_antennas : (self.next_event_i+1)*self.fitting_info.num_antennas]
        
        cdef int ant_i
        for ant_i in range(self.fitting_info.num_antennas):
            if isfinite( arrival_times[ant_i] ):
                filter_slice[ant_i] = 1
                self.total_num_measurments += 1
            else:
                filter_slice[ant_i] = 0
                
            measurment_slice[ant_i] = arrival_times[ant_i] 
            
        self.next_event_i += 1
        
    def objective_fun(self, guess):
        cdef np.ndarray[double , ndim=1] ret = np.empty(self.total_num_measurments)
            
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double dt
        
        cdef int event_i
        cdef int antenna_i
        cdef int station_i
        cdef double stat_delay
        cdef int output_i = 0
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        for event_i in range(self.fitting_info.num_events):
            measurement_slice = self.fitting_info.measurement_times[event_i*self.fitting_info.num_antennas : (event_i+1)*self.fitting_info.num_antennas]
            filter_slice = self.fitting_info.measurement_filter[event_i*self.fitting_info.num_antennas : (event_i+1)*self.fitting_info.num_antennas]
            X = guess[self.fitting_info.num_station_delays + event_i*4 + 0]
            Y = guess[self.fitting_info.num_station_delays + event_i*4 + 1]
            Z = guess[self.fitting_info.num_station_delays + event_i*4 + 2]
            Z = fabs(Z)
            T = guess[self.fitting_info.num_station_delays + event_i*4 + 3]
            
            for antenna_i in range(self.fitting_info.num_antennas):
                if filter_slice[antenna_i]:
                    
                    dx = self.fitting_info.antenna_locations[antenna_i, 0] - X
                    dy = self.fitting_info.antenna_locations[antenna_i, 1] - Y
                    dz = self.fitting_info.antenna_locations[antenna_i, 2] - Z
                    stat_delay = 0.0
                    station_i = self.fitting_info.station_indexes[ antenna_i ]
                    if station_i != self.fitting_info.num_station_delays:
                        stat_delay = guess[ station_i ]
                    dt = sqrt(dx*dx + dy*dy + dz*dz)*c_air_inverse + T + stat_delay
                    dt -= measurement_slice[ antenna_i ]
                    
                    ret[output_i] = dt
                    output_i += 1
                    
        return ret
    
    def RMS(self, guess, num_DOF):
        diffs = self.objective_fun( guess )
        diffs *= diffs
        return np.sqrt( np.sum(diffs)/num_DOF )
    
    def objective_jac(self, guess):
        cdef np.ndarray[double , ndim=2] ret = np.zeros((self.total_num_measurments, self.fitting_info.num_station_delays + 4*self.fitting_info.num_events))
        
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double inv_R
        
        cdef int event_i
        cdef int antenna_i
        cdef int station_i
        cdef int output_i = 0
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        for event_i in range(self.fitting_info.num_events):
            measurement_slice = self.fitting_info.measurement_times[event_i*self.fitting_info.num_antennas : (event_i+1)*self.fitting_info.num_antennas]
            filter_slice = self.fitting_info.measurement_filter[event_i*self.fitting_info.num_antennas : (event_i+1)*self.fitting_info.num_antennas]
            X = guess[self.fitting_info.num_station_delays + event_i*4 + 0]
            Y = guess[self.fitting_info.num_station_delays + event_i*4 + 1]
            Z = guess[self.fitting_info.num_station_delays + event_i*4 + 2]
            Z = fabs(Z)
            T = guess[self.fitting_info.num_station_delays + event_i*4 + 3]
            
            for antenna_i in range(self.fitting_info.num_antennas):
                if filter_slice[antenna_i]:
                    
                    dx = X - self.fitting_info.antenna_locations[antenna_i, 0]
                    dy = Y - self.fitting_info.antenna_locations[antenna_i, 1]
                    dz = Z - self.fitting_info.antenna_locations[antenna_i, 2]
                    
                    inv_R = c_air_inverse/sqrt(dx*dx + dy*dy + dz*dy)
                    
                    ret[output_i, self.fitting_info.num_station_delays + event_i*4 + 0] = dx*inv_R
                    ret[output_i, self.fitting_info.num_station_delays + event_i*4 + 1] = dy*inv_R
                    ret[output_i, self.fitting_info.num_station_delays + event_i*4 + 2] = dz*inv_R
                    
                    ret[output_i, self.fitting_info.num_station_delays + event_i*4 + 3] = 1
                    
                    station_i = self.fitting_info.station_indexes[ antenna_i ]
                    if station_i != self.fitting_info.num_station_delays:
                        ret[output_i, station_i] = 1
                    
                    output_i += 1
                    
        return ret
        
        
#    def setup_fitter(self):
#        
#        
#        cdef int num_parameters = self.stationDelay_dataStruct.num_station_delays + 4*self.stationDelay_dataStruct.num_events
#        
#        
#        # allocate workspace with default parameters
#        cdef gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters()
#          
#        cdef void *T = gsl_multifit_nlinear_trust
#        self.fitter_workspace = gsl_multifit_nlinear_alloc(T, &fdf_params, self.total_num_measurments, num_parameters)
#        
#        self.covariance_matrix = gsl_matrix_alloc(num_parameters, num_parameters)
#        
#        self.initial_guess_copy = gsl_vector_alloc( num_parameters )
#        
#        
#    def run_fitter(self, np.ndarray[double, ndim=1] initial_guess, int max_num_iters=100, double xtol=1.0e-8, double gtol = 1.0e-8, double ftol = 0.0):
#        
#        
#        cdef int num_parameters = self.stationDelay_dataStruct.num_station_delays + 4*self.stationDelay_dataStruct.num_events
#        ## check initial guess
#    
#        cdef int i
#        for i in range(num_parameters):
#            gsl_vector_set(self.initial_guess_copy, i,  initial_guess[i])
#        
#        # initialize solver with starting point
#        cdef gsl_multifit_nlinear_fdf fdf
#        fdf.f = objective_fun
#        fdf.df = objective_jac   # set to NULL for finite-difference Jacobian 
#        fdf.fvv = NULL     # not using geodesic acceleration
#        fdf.n = self.total_num_measurments
#        fdf.p = num_parameters
#        fdf.params = <void*> &self.fitting_info
#        gsl_multifit_nlinear_init(self.initial_guess_copy, &fdf, self.fitter_workspace)
#
#        # solve the system
#        cdef int info
#        cdef int status = gsl_multifit_nlinear_driver(max_num_iters, xtol, gtol, ftol, NULL, NULL, &info, self.fitter_workspace)
#
#
#        ## compute covariance of best fit parameters */
#        cdef double dof = self.total_num_measurments - num_parameters
#        cdef double chisq
#        
#        cdef void *residual_vector = gsl_multifit_nlinear_residual( self.fitter_workspace )
#        cdef void *jac_matrix = gsl_multifit_nlinear_jac( self.fitter_workspace )
#        cdef void *result_X = gsl_multifit_nlinear_position( self.fitter_workspace  )
#        
#        gsl_multifit_nlinear_covar(jac_matrix, 0.0, self.covariance_matrix)
#        gsl_blas_ddot(residual_vector, residual_vector, &chisq)
#        
#        
#        ### save solution out
#        self.RMS_out = sqrt(chisq / dof)
#        
#        self.solution_out = np.empty(num_parameters)
#        self.estimated_error_out = np.empty(num_parameters)
#        for i in range(num_parameters):
#            self.solution_out[i] = gsl_vector_get(result_X, i)
#            self.estimated_error_out[i] = self.RMS_out*sqrt( gsl_matrix_get(self.covariance_matrix,i,i) )
#            
#        ### TODO: return status
#            
#            
#    def free_memory(self):
#        gsl_multifit_nlinear_free( self.fitter_workspace )
#        gsl_matrix_free( self.covariance_matrix )
#        gsl_vector_free( self.initial_guess_copy )


        
        

