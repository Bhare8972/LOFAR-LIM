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

cdef double c_air_inverse = 1.000293/299792458.0

#cdef struct stationDelay_dataStruct:
#    int num_station_delays 
#    int num_antennas 
#    int num_events
#    
##    double[:,:] antenna_locations
##    long[:] station_indexes 
##    double[:] measurement_times 
##    double[:] measurement_times_original
##    np.uint8_t[:] measurement_filter
#    
#    double* antenna_locations
#    long* station_indexes 
#    double* measurement_times 
#    #double* measurement_times_original
#    np.uint8_t* measurement_filter


cdef class stationDelay_fitter:
    
    cdef int total_num_measurments
    cdef int next_event_i
    
    cdef void *fitter_workspace
    cdef void *covariance_matrix
    cdef void *initial_guess_copy
    
    #cdef stationDelay_dataStruct fitting_info
    cdef int num_station_delays 
    cdef int num_antennas 
    cdef int num_events
    
    cdef double[:,:] antenna_locations
    cdef long[:] station_indexes
    cdef double[:] measurement_times
    cdef np.ndarray measurement_times_original
    cdef np.uint8_t[:] measurement_filter
    
    #### outputs placed in these two arrays
    cdef np.ndarray solution_out
    cdef np.ndarray estimated_error_out
    cdef double RMS_out
    
    def __init__(self, np.ndarray[double, ndim=2] _antenna_locations, np.ndarray[long, ndim=1] _station_indexes, _num_events, _num_station_delays):
        self.num_antennas = _antenna_locations.shape[0]
        self.num_events = _num_events
        
        self.antenna_locations = _antenna_locations
#        self.fitting_info.antenna_locations = &self.antenna_locations[0,0]
        
        self.num_station_delays = _num_station_delays
#        self.fitting_info.num_station_delays = &self.num_station_delays[0]
        
        self.station_indexes = _station_indexes
#        self.fitting_info.station_indexes = &self.station_indexes[0]
        
        self.measurement_times = np.empty( _num_events*self.num_antennas, dtype=np.double )
#        self.fitting_info.measurement_times = &self.measurement_times[0]
        
        self.measurement_filter = np.zeros( _num_events*self.num_antennas, np.uint8 )
#        self.fitting_info.measurement_filter = &self.measurement_filter[0]
        
        self.total_num_measurments = 0
        self.next_event_i = 0
        
    def set_event(self, np.ndarray[double, ndim=1] arrival_times):
        filter_slice = self.measurement_filter[self.next_event_i*self.num_antennas : (self.next_event_i+1)*self.num_antennas]
        measurment_slice = self.measurement_times[self.next_event_i*self.num_antennas : (self.next_event_i+1)*self.num_antennas]
        
        cdef int ant_i
        for ant_i in range(self.num_antennas):
            if isfinite( arrival_times[ant_i] ):
                filter_slice[ant_i] = 1
                self.total_num_measurments += 1
            else:
                filter_slice[ant_i] = 0
                
            measurment_slice[ant_i] = arrival_times[ant_i] 
            
        self.next_event_i += 1
        
    def prep_for_random_pert(self):
        self.measurement_times_original = np.array( self.measurement_times )
        
    def random_perturbation(self, deviation, antenna_error_deviation):
        self.measurement_times = np.random.normal(loc=self.measurement_times_original, scale=deviation, size=len(self.measurement_times_original))
        
        cdef np.ndarray[double, ndim=1] antenna_rand = np.random.normal(loc=0.0, scale=antenna_error_deviation, size=self.num_antennas)
        cdef int event_i
        cdef int ant_i
        for event_i in range(self.num_events):
            for ant_i in range(self.num_antennas):
                self.measurement_times[event_i*self.num_antennas + ant_i] +=  antenna_rand[ant_i]
            
            
        
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
        for event_i in range(self.num_events):
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            X = guess[self.num_station_delays + event_i*4 + 0]
            Y = guess[self.num_station_delays + event_i*4 + 1]
            Z = guess[self.num_station_delays + event_i*4 + 2]
            Z = fabs(Z)
            T = guess[self.num_station_delays + event_i*4 + 3]
            
            for antenna_i in range(self.num_antennas):
                if filter_slice[antenna_i]:
                    
                    dx = self.antenna_locations[antenna_i, 0] - X
                    dy = self.antenna_locations[antenna_i, 1] - Y
                    dz = self.antenna_locations[antenna_i, 2] - Z
                    stat_delay = 0.0
                    station_i = self.station_indexes[ antenna_i ]
                    if station_i != self.num_station_delays:
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
    
    
    
    
    