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

cdef double c_air = 299792458.0/1.000293
cdef double c_air_inverse = 1.0/c_air

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
    
cdef class delay_fitter:
    
    cdef int total_num_measurments
    cdef int next_event_i
    
    #cdef void *fitter_workspace
    #cdef void *covariance_matrix
    #cdef void *initial_guess_copy
    
    #cdef stationDelay_dataStruct fitting_info
    cdef int num_station_delays 
    cdef int num_antennas 
    cdef int num_events
    cdef int num_antenna_recalibrations
    
    cdef double[:,:] antenna_locations # three nums per antenna
    cdef long[:] station_indexes # one num per antenna
    cdef long[:] antenna_recalibration_indeces # one per antenna. If -1, ignore. Else, adjust calibration that the value according to that index
    cdef double[:] measurement_times # num_events*num_antennas
    #cdef np.ndarray measurement_times_original
    cdef np.uint8_t[:] measurement_filter
    
    #### outputs placed in these two arrays
    #cdef np.ndarray solution_out
   # cdef np.ndarray estimated_error_out
    #cdef double RMS_out
    
    def __init__(self, np.ndarray[double, ndim=2] _antenna_locations, np.ndarray[long, ndim=1] _station_indexes, np.ndarray[long, ndim=1] antennas_to_recalibrate, _num_events, _num_station_delays):
        self.num_antennas = _antenna_locations.shape[0]
        self.num_events = _num_events
        
        self.num_station_delays = _num_station_delays
        
        self.antenna_locations = _antenna_locations
        self.station_indexes = _station_indexes
        
        self.measurement_times = np.empty( _num_events*self.num_antennas, dtype=np.double )
        self.measurement_filter = np.zeros( _num_events*self.num_antennas, np.uint8 )
        
        self.antenna_recalibration_indeces = np.ones(self.num_antennas, dtype=long )
        self.num_antenna_recalibrations = len(antennas_to_recalibrate)
        self.antenna_recalibration_indeces[:] = -1
        cdef int i
        for i, ant_i in enumerate(antennas_to_recalibrate):
            self.antenna_recalibration_indeces[ant_i] = i
        
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
        
    def objective_fun(self, np.ndarray[double , ndim=1] guess):
        cdef np.ndarray[double , ndim=1] ret = np.empty(self.total_num_measurments)
        
        cdef double[:] station_delays = guess[ : self.num_station_delays]
        cdef double[:] antenna_delays = guess[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:] event_XYZTs = guess[self.num_station_delays+self.num_antenna_recalibrations : ]
            
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
        cdef int recalibrate_i
        cdef double total_delay
        cdef int output_i = 0
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        for event_i in range(self.num_events):
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
                        
            X = event_XYZTs[ event_i*4 + 0]
            Y = event_XYZTs[ event_i*4 + 1]
            Z = event_XYZTs[ event_i*4 + 2]
            Z = fabs(Z)
            T = event_XYZTs[ event_i*4 + 3]
            
            for antenna_i in range(self.num_antennas):
                if filter_slice[antenna_i]:
                    
                    dx = self.antenna_locations[antenna_i, 0] - X
                    dy = self.antenna_locations[antenna_i, 1] - Y
                    dz = self.antenna_locations[antenna_i, 2] - Z
                    
                    total_delay = 0.0
                    station_i = self.station_indexes[ antenna_i ]
                    if station_i != self.num_station_delays:
                        total_delay = guess[ station_i ]
                    recalibrate_i =  self.antenna_recalibration_indeces[ antenna_i ]
                    if recalibrate_i != -1:
                        total_delay += antenna_delays[ recalibrate_i ]
                        
                    dt = sqrt(dx*dx + dy*dy + dz*dz)*c_air_inverse + T + total_delay
                    dt -= measurement_slice[ antenna_i ]
                    
                    ret[output_i] = dt
                        
                    output_i += 1
                    
        return ret
    
    def objective_fun_sq(self, np.ndarray[double , ndim=1] guess):
        cdef np.ndarray[double , ndim=1] ret = np.empty(self.total_num_measurments)
        
        cdef double[:] station_delays = guess[ : self.num_station_delays]
        cdef double[:] antenna_delays = guess[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:] event_XYZTs = guess[self.num_station_delays+self.num_antenna_recalibrations : ]
            
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        
        cdef double dx
        cdef double dy
        cdef double dz
#        cdef double dt
        
        cdef int event_i
        cdef int antenna_i
        cdef int station_i
        cdef int recalibrate_i
        cdef double total_delay
        cdef int output_i = 0
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        for event_i in range(self.num_events):
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            
            X = event_XYZTs[ event_i*4 + 0]
            Y = event_XYZTs[ event_i*4 + 1]
            Z = event_XYZTs[ event_i*4 + 2]
            Z = fabs(Z)
            T = event_XYZTs[ event_i*4 + 3]
            
            for antenna_i in range(self.num_antennas):
                if filter_slice[antenna_i]:
                    
                    dx = self.antenna_locations[antenna_i, 0] - X
                    dy = self.antenna_locations[antenna_i, 1] - Y
                    dz = self.antenna_locations[antenna_i, 2] - Z
                    
                    total_delay = 0.0
                    station_i = self.station_indexes[ antenna_i ]
                    if station_i != self.num_station_delays:
                        total_delay = guess[ station_i ]
                    recalibrate_i =  self.antenna_recalibration_indeces[ antenna_i ]
                    if recalibrate_i != -1:
                        total_delay += antenna_delays[ recalibrate_i ]
                    
                    ret[output_i] = T + total_delay - measurement_slice[ antenna_i ]
                    ret[output_i] *= c_air
                    ret[output_i] *= ret[output_i]
                    
                    ret[output_i] -= dx*dx
                    ret[output_i] -= dy*dy
                    ret[output_i] -= dz*dz
                    
                    output_i += 1
                    
        return ret
    
    def RMS(self, guess, num_DOF):
        diffs = self.objective_fun( guess )
        diffs *= diffs
        #print(diffs)
        return np.sqrt( np.sum(diffs)/num_DOF )
    
    def print_antenna_info(self, int antenna_index, np.ndarray[double , ndim=1] guess):
        
        cdef double[:] station_delays = guess[ : self.num_station_delays]
        cdef double[:] antenna_delays = guess[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:] event_XYZTs = guess[self.num_station_delays+self.num_antenna_recalibrations : ]
            
        cdef double antX = self.antenna_locations[antenna_index, 0]
        cdef double antY = self.antenna_locations[antenna_index, 1]
        cdef double antZ = self.antenna_locations[antenna_index, 2]
        print("antenna XYZ")
        print(" ", antX, antY, antZ)
        
        
        cdef double total_delay = 0.0
        cdef int station_i = self.station_indexes[ antenna_index ]
        if station_i != self.num_station_delays:
            total_delay = guess[ station_i ]
        print("station delay:", total_delay)
        cdef int recalibrate_i = self.antenna_recalibration_indeces[ antenna_index ]
        if recalibrate_i != -1:
            print("recalibration with delay:", antenna_delays[ recalibrate_i ])
            total_delay += antenna_delays[ recalibrate_i ]
        print("total delay", total_delay)
        
        
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double dt
        
        cdef int event_i
        
        print()
        
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        for event_i in range(self.num_events):
            print( event_i)
            
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
                        
            X = event_XYZTs[ event_i*4 + 0]
            Y = event_XYZTs[ event_i*4 + 1]
            Z = event_XYZTs[ event_i*4 + 2]
            Z = fabs(Z)
            T = event_XYZTs[ event_i*4 + 3]
            print(" ", X,Y,Z,T)
            
            if filter_slice[antenna_index]:
                
                dx = antX - X
                dy = antY - Y
                dz = antZ - Z
                    
                print('R:', sqrt(dx*dx + dy*dy + dz*dz)*c_air_inverse)
                dt = sqrt(dx*dx + dy*dy + dz*dz)*c_air_inverse + T + total_delay
                print('model:', dt)
                dt -= measurement_slice[ antenna_index ]
                print('error:', dt)
            else:
                print( " antenna not used in event" )
                
            print()
    
    def event_SSqE(self, event_i, guess, ant_range=None):
        if ant_range is None:
            start = 0
            stop = self.num_antennas
        else:
            start = ant_range[0]
            stop = ant_range[1]
            
        cdef double[:] station_delays = guess[ : self.num_station_delays]
        cdef double[:] antenna_delays = guess[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:] event_XYZTs = guess[self.num_station_delays+self.num_antenna_recalibrations : ]
            
        cdef double X = event_XYZTs[ event_i*4 + 0]
        cdef double Y = event_XYZTs[ event_i*4 + 1]
        cdef double Z = event_XYZTs[ event_i*4 + 2]
        Z = fabs(Z)
        cdef double T = event_XYZTs[ event_i*4 + 3]
        
        cdef double[:] measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
        cdef np.uint8_t[:] filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
        
        cdef double SSqE = 0
        cdef int num_ants = 0
        cdef int antenna_i
        for antenna_i in range(start, stop):
            if filter_slice[antenna_i]:
                
                dx = self.antenna_locations[antenna_i, 0] - X
                dy = self.antenna_locations[antenna_i, 1] - Y
                dz = self.antenna_locations[antenna_i, 2] - Z
                
                total_delay = 0.0
                station_i = self.station_indexes[ antenna_i ]
                if station_i != self.num_station_delays:
                    total_delay = guess[ station_i ]
                recalibrate_i =  self.antenna_recalibration_indeces[ antenna_i ]
                if recalibrate_i != -1:
                    total_delay += antenna_delays[ recalibrate_i ]
                    
                    
                dt = sqrt(dx*dx + dy*dy + dz*dz)*c_air_inverse + T + total_delay
                dt -= measurement_slice[ antenna_i ]
                
                SSqE += dt*dt
                num_ants += 1
                
        return SSqE, num_ants
    
    def analytical_covariance_matrix(self, solution, weight=1.0e-9):
        
        cdef double inv_weight_sq = 1.0/(weight*weight)
        
        #cdef double[:] station_delays = solution[ : self.num_station_delays]
        #cdef double[:] antenna_delays = solution[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        #cdef double[:] event_XYZTs    = solution[self.num_station_delays+self.num_antenna_recalibrations : ]
        
        N = len(solution) #- self.num_station_delays - self.num_antenna_recalibrations
        hess = np.zeros( (N, N), dtype=np.double )
        
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double distance
        cdef double inv_D
        cdef double f
        cdef double dfdx
        cdef double dfdy
        cdef double dfdz
        cdef double ddfddx
        cdef double ddfddy
        cdef double ddfddz
        cdef double ddfdxdy
        cdef double ddfdxdz
        cdef double ddfdydz
        cdef int X_i
        cdef int Y_i
        cdef int Z_i
        cdef int T_i
        cdef int stat_cal_i
        cdef int ant_cal_i
        
        cdef int event_i
        cdef int antenna_i
        for event_i in range(self.num_events):
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
                        
            
            X_i = self.num_station_delays+self.num_antenna_recalibrations + event_i*4 + 0
            Y_i = self.num_station_delays+self.num_antenna_recalibrations + event_i*4 + 1
            Z_i = self.num_station_delays+self.num_antenna_recalibrations + event_i*4 + 2
            T_i = self.num_station_delays+self.num_antenna_recalibrations + event_i*4 + 3
            
            X = solution[ X_i ]
            Y = solution[ Y_i ]
            Z = solution[ Z_i ]
            Z = fabs(Z)
            T = solution[ T_i ]
            
#            X_i -= self.num_station_delays + self.num_antenna_recalibrations
#            Y_i -= self.num_station_delays + self.num_antenna_recalibrations
#            Z_i -= self.num_station_delays + self.num_antenna_recalibrations
#            T_i -= self.num_station_delays + self.num_antenna_recalibrations
            
            for antenna_i in range(self.num_antennas):
                if filter_slice[antenna_i]:
                    dx = X - self.antenna_locations[antenna_i, 0]
                    dy = Y - self.antenna_locations[antenna_i, 1]
                    dz = Z - self.antenna_locations[antenna_i, 2]
                    
                    distance = np.sqrt( dx*dx + dy*dy + dz*dz )
                    
                    f = distance*c_air_inverse + T
                    f -= measurement_slice[antenna_i]
                    stat_cal_i = self.station_indexes[ antenna_i ]
                    if stat_cal_i != self.num_station_delays:
                        f += solution[ stat_cal_i ]
                    ant_cal_i =  self.antenna_recalibration_indeces[ antenna_i ]
                    if ant_cal_i != -1:
                        ant_cal_i += self.num_station_delays
                        f += solution[ ant_cal_i ]
                    
                    #### pure spatial component ####
                    inv_D = 1.0/(distance*c_air)
                    dfdx = dx*inv_D
                    dfdy = dy*inv_D
                    dfdz = dz*inv_D
                    
                    inv_D = -1.0/(distance*distance*distance*c_air)
                    ddfddx = dx*dx*inv_D
                    ddfdxdy = dx*dy*inv_D
                    ddfdxdz = dx*dz*inv_D
                    ddfddy = dy*dy*inv_D
                    ddfdydz = dy*dz*inv_D
                    ddfddz = dz*dz*inv_D
                    
                    inv_D = 1.0/(distance*c_air)
                    ddfddx += inv_D
                    ddfddy += inv_D
                    ddfddz += inv_D
                    
                    hess[X_i, X_i] += 2*inv_weight_sq*( dfdx*dfdx + f*ddfddx )
                    hess[Y_i, Y_i] += 2*inv_weight_sq*( dfdy*dfdy + f*ddfddy )
                    hess[Z_i, Z_i] += 2*inv_weight_sq*( dfdz*dfdz + f*ddfddz )
                    
                    hess[X_i, Y_i] += 2*inv_weight_sq*( dfdx*dfdy + f*ddfdxdy )
                    hess[Y_i, X_i] += 2*inv_weight_sq*( dfdx*dfdy + f*ddfdxdy )
                    
                    hess[X_i, Z_i] += 2*inv_weight_sq*( dfdx*dfdz + f*ddfdxdz )
                    hess[Z_i, X_i] += 2*inv_weight_sq*( dfdx*dfdz + f*ddfdxdz )
                    
                    hess[Y_i, Z_i] += 2*inv_weight_sq*( dfdy*dfdz + f*ddfdydz )
                    hess[Z_i, Y_i] += 2*inv_weight_sq*( dfdy*dfdz + f*ddfdydz )
                    
                    #### pure time ####
                    hess[T_i, T_i] += 2*inv_weight_sq # so simple!
                    
                    #### mix time and space ####
                    hess[T_i, X_i] += 2*inv_weight_sq*( dfdx )
                    hess[X_i, T_i] += 2*inv_weight_sq*( dfdx )
                    
                    hess[T_i, Y_i] += 2*inv_weight_sq*( dfdy )
                    hess[Y_i, T_i] += 2*inv_weight_sq*( dfdy )
                    
                    hess[T_i, Z_i] += 2*inv_weight_sq*( dfdz )
                    hess[Z_i, T_i] += 2*inv_weight_sq*( dfdz )
                    
                    #### station cal ####
                    if stat_cal_i != self.num_station_delays:
                        #### pure ####
                        hess[stat_cal_i, stat_cal_i] += 2*inv_weight_sq
                        
                        #### mix with space ####
                        hess[stat_cal_i, X_i] += 2*inv_weight_sq*( dfdx )
                        hess[X_i, stat_cal_i] += 2*inv_weight_sq*( dfdx )
                        
                        hess[stat_cal_i, Y_i] += 2*inv_weight_sq*( dfdy )
                        hess[Y_i, stat_cal_i] += 2*inv_weight_sq*( dfdy )
                        
                        hess[stat_cal_i, Z_i] += 2*inv_weight_sq*( dfdz )
                        hess[Z_i, stat_cal_i] += 2*inv_weight_sq*( dfdz )
                    
                        #### mix with time ####
                        hess[stat_cal_i, T_i] += 2*inv_weight_sq
                        hess[T_i, stat_cal_i] += 2*inv_weight_sq
                        
                    #### finally, antenna calibration ####
                    if ant_cal_i != -1:
                        #### pure ####
                        hess[ant_cal_i, ant_cal_i] += 2*inv_weight_sq
                        
                        #### mix with space ####
                        hess[ant_cal_i, X_i] += 2*inv_weight_sq*( dfdx )
                        hess[X_i, ant_cal_i] += 2*inv_weight_sq*( dfdx )
                        
                        hess[ant_cal_i, Y_i] += 2*inv_weight_sq*( dfdy )
                        hess[Y_i, ant_cal_i] += 2*inv_weight_sq*( dfdy )
                        
                        hess[ant_cal_i, Z_i] += 2*inv_weight_sq*( dfdz )
                        hess[Z_i, ant_cal_i] += 2*inv_weight_sq*( dfdz )
                    
                        #### mix with time ####
                        hess[ant_cal_i, T_i] += 2*inv_weight_sq
                        hess[T_i, ant_cal_i] += 2*inv_weight_sq
                    
                        #### mix with stat. cal ####
                        if stat_cal_i != self.num_station_delays:
                            hess[stat_cal_i, ant_cal_i] += 2*inv_weight_sq
                            hess[ant_cal_i, stat_cal_i] += 2*inv_weight_sq
                    
        cov_mat = np.linalg.inv( hess )      
        cov_mat *= 2
        return cov_mat
    
cdef class delay_fitter_polT:
    
    cdef int total_num_measurments
    cdef int total_num_parameters
    cdef int next_event_i
    
    #cdef void *fitter_workspace
    #cdef void *covariance_matrix
    #cdef void *initial_guess_copy
    
    #cdef stationDelay_dataStruct fitting_info
    cdef int num_station_delays 
    cdef int num_antennas 
    cdef int num_events
    cdef int num_antenna_recalibrations
    
    cdef double[:,:] antenna_locations # three nums per antenna
    cdef long[:] station_indexes # one num per antenna
    cdef long[:] antenna_recalibration_indeces # one per antenna. If -1, ignore. Else, adjust calibration that the value according to that index
    cdef long[:] event_polarizations # one per event. 0 for only even. 1 for only odd. 2 for both, assume fit XYZ the same and T seperately
    cdef double[:] measurement_times # num_events*num_antennas
    #cdef np.ndarray measurement_times_original
    cdef np.uint8_t[:] measurement_filter
    
    #### outputs placed in these two arrays
    #cdef np.ndarray solution_out
   # cdef np.ndarray estimated_error_out
    #cdef double RMS_out
    
    def __init__(self, np.ndarray[double, ndim=2] _antenna_locations, np.ndarray[long, ndim=1] _station_indexes, np.ndarray[long, ndim=1] antennas_to_recalibrate, _num_events, _num_station_delays):
        self.num_antennas = _antenna_locations.shape[0]
        self.num_events = _num_events
        
        self.num_station_delays = _num_station_delays
        
        self.antenna_locations = _antenna_locations
        self.station_indexes = _station_indexes
        
        self.measurement_times = np.empty( _num_events*self.num_antennas, dtype=np.double )
        self.measurement_filter = np.zeros( _num_events*self.num_antennas, np.uint8 )
        
        self.antenna_recalibration_indeces = np.ones(self.num_antennas, dtype=long )
        self.num_antenna_recalibrations = len(antennas_to_recalibrate)
        self.antenna_recalibration_indeces[:] = -1
        cdef int i
        for i, ant_i in enumerate(antennas_to_recalibrate):
            self.antenna_recalibration_indeces[ant_i] = i
        
        self.total_num_measurments = 0
        self.next_event_i = 0
        self.event_polarizations = np.empty( self.num_events, dtype=int )

        self.total_num_parameters = self.num_station_delays + self.num_antenna_recalibrations  ## not done untill all events added!
        
    def set_event(self, np.ndarray[double, ndim=1] arrival_times, int polarization):
        filter_slice = self.measurement_filter[self.next_event_i*self.num_antennas : (self.next_event_i+1)*self.num_antennas]
        measurment_slice = self.measurement_times[self.next_event_i*self.num_antennas : (self.next_event_i+1)*self.num_antennas]        
        
        self.event_polarizations[self.next_event_i] = polarization
        
        cdef int ant_i
        for ant_i in range(self.num_antennas):
            if isfinite( arrival_times[ant_i] ):
                filter_slice[ant_i] = 1
                self.total_num_measurments += 1
            else:
                filter_slice[ant_i] = 0
                
            measurment_slice[ant_i] = arrival_times[ant_i] 
            
        self.next_event_i += 1  


        self.total_num_parameters += 4
        if polarization == 2:
            self.total_num_parameters += 1
        elif polarization == 3:
            self.total_num_parameters += 3


    def get_num_measurments(self):
        return self.total_num_measurments

    def get_num_parameters(self):
        return self.total_num_parameters
        
    def objective_fun(self, np.ndarray[double , ndim=1] guess, np.ndarray[double , ndim=1] ret=None, info=None):
        if ret is None:
            ret = np.empty(self.total_num_measurments)
        
        cdef double[:] station_delays = guess[ : self.num_station_delays]
        cdef double[:] antenna_delays = guess[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:] event_XYZTs = guess[self.num_station_delays+self.num_antenna_recalibrations : ]
            
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        
        cdef double Xodd = 0
        cdef double Yodd = 0
        cdef double Zodd = 0
        cdef double Todd = 0
        
        cdef double X_to_use
        cdef double Y_to_use
        cdef double Z_to_use
        cdef double T_to_use
        
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double dt
        
        cdef int event_i
        cdef int antenna_i
        cdef int station_i
        cdef int recalibrate_i
        cdef double total_delay
        cdef int output_i = 0
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        cdef long current_param_i = 0
        cdef long event_polarization
        cdef int antenna_polarization # 0 for even, 1 for odd
        for event_i in range(self.num_events):
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            
            event_polarization = self.event_polarizations[ event_i ]
                        
            X = event_XYZTs[ current_param_i + 0]
            Y = event_XYZTs[ current_param_i + 1]
            Z = event_XYZTs[ current_param_i + 2]
            Z = fabs(Z)
            T = event_XYZTs[ current_param_i + 3]
            
            current_param_i += 4
            if event_polarization == 2:
                Todd = event_XYZTs[ current_param_i ]
                current_param_i += 1
            elif  event_polarization == 3:
                Xodd = event_XYZTs[ current_param_i + 0]
                Yodd = event_XYZTs[ current_param_i + 1]
                Zodd = event_XYZTs[ current_param_i + 2]
                Zodd = fabs(Zodd)
                Todd = event_XYZTs[ current_param_i + 3]
                current_param_i += 4
            
            antenna_polarization = -1
            for antenna_i in range(self.num_antennas):
                antenna_polarization += 1
                if antenna_polarization == 2:
                    antenna_polarization = 0
                
                if filter_slice[antenna_i] and (event_polarization==2 or event_polarization==3 or (event_polarization==0 and antenna_polarization==0) or (event_polarization==1 and antenna_polarization==1)):
                    
                    X_to_use = X
                    Y_to_use = Y
                    Z_to_use = Z
                    T_to_use = T
                    if event_polarization==2 and antenna_polarization==1:
                        T_to_use = Todd
                    elif event_polarization==3 and antenna_polarization==1:
                        X_to_use = Xodd
                        Y_to_use = Yodd
                        Z_to_use = Zodd
                        T_to_use = Todd
                    
                    dx = self.antenna_locations[antenna_i, 0] - X_to_use
                    dy = self.antenna_locations[antenna_i, 1] - Y_to_use
                    dz = self.antenna_locations[antenna_i, 2] - Z_to_use
                    
                    total_delay = 0.0
                    station_i = self.station_indexes[ antenna_i ]
                    if station_i != self.num_station_delays:
                        total_delay = guess[ station_i ]
                    recalibrate_i =  self.antenna_recalibration_indeces[ antenna_i ]
                    if recalibrate_i != -1:
                        total_delay += antenna_delays[ recalibrate_i ]
                        
                    dt = sqrt(dx*dx + dy*dy + dz*dz)*c_air_inverse + T_to_use + total_delay
                    dt -= measurement_slice[ antenna_i ]
                    
                    ret[output_i] = dt
                        
                    output_i += 1

        return ret

    def objective_fun_jacobian(self, np.ndarray[double , ndim=1] guess, np.ndarray[double , ndim=2] jacobian_ret=None, info=None):
        if jacobian_ret is None:
            jacobian_ret = np.zeros( (self.total_num_measurments, self.total_num_parameters), dtype=np.double )
        
        cdef double[:] station_delays = guess[ : self.num_station_delays]
        cdef double[:] antenna_delays = guess[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:] event_XYZTs = guess[self.num_station_delays+self.num_antenna_recalibrations : ]


        cdef double[:,:] station_jacobian = jacobian_ret[:, :self.num_station_delays]
        cdef double[:,:] antenna_jacobian = jacobian_ret[:, self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:,:] event_L_jacobian = jacobian_ret[:, self.num_station_delays+self.num_antenna_recalibrations :]
            
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        
        cdef double Xodd = 0
        cdef double Yodd = 0
        cdef double Zodd = 0
        cdef double Todd = 0
        
        cdef double X_to_use
        cdef double Y_to_use
        cdef double Z_to_use
        cdef double T_to_use
        
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double dt
        cdef double R
        
        cdef int event_i
        cdef int antenna_i
        cdef int station_i
        cdef int recalibrate_i
        cdef double total_delay
        cdef int output_i = 0
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        cdef long current_param_i = 0
        cdef long next_param_i
        cdef long event_polarization
        cdef int antenna_polarization # 0 for even, 1 for odd
        for event_i in range(self.num_events):
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            
            event_polarization = self.event_polarizations[ event_i ]
                        
            X = event_XYZTs[ current_param_i + 0]
            Y = event_XYZTs[ current_param_i + 1]
            Z = event_XYZTs[ current_param_i + 2]
            Z = fabs(Z)
            T = event_XYZTs[ current_param_i + 3]
            
            next_param_i = current_param_i + 4
            if event_polarization == 2:
                Todd = event_XYZTs[ current_param_i+4 ]
                next_param_i += 1
            elif  event_polarization == 3:
                Xodd = event_XYZTs[ current_param_i + 4+ 0]
                Yodd = event_XYZTs[ current_param_i + 4+ 1]
                Zodd = event_XYZTs[ current_param_i + 4+ 2]
                Zodd = fabs(Zodd)
                Todd = event_XYZTs[ current_param_i + 4+ 3]
                next_param_i += 4
            
            antenna_polarization = -1
            for antenna_i in range(self.num_antennas):
                antenna_polarization += 1
                if antenna_polarization == 2:
                    antenna_polarization = 0
                
                if filter_slice[antenna_i] and (event_polarization==2 or event_polarization==3 or (event_polarization==0 and antenna_polarization==0) or (event_polarization==1 and antenna_polarization==1)):
                    
                    X_to_use = X
                    Y_to_use = Y
                    Z_to_use = Z
                    T_to_use = T
                    if event_polarization==2 and antenna_polarization==1:
                        T_to_use = Todd
                    elif event_polarization==3 and antenna_polarization==1:
                        X_to_use = Xodd
                        Y_to_use = Yodd
                        Z_to_use = Zodd
                        T_to_use = Todd
                    
                    dx = X_to_use - self.antenna_locations[antenna_i, 0]
                    dy = Y_to_use - self.antenna_locations[antenna_i, 1]
                    dz = Z_to_use - self.antenna_locations[antenna_i, 2]
                    R = sqrt(dx*dx + dy*dy + dz*dz)

                    dx *= c_air_inverse/R
                    dy *= c_air_inverse/R
                    dz *= c_air_inverse/R

                    station_i = self.station_indexes[ antenna_i ]
                    if station_i != self.num_station_delays:
                        station_jacobian[output_i, station_i] = 1

                    recalibrate_i = self.antenna_recalibration_indeces[ antenna_i ]
                    if recalibrate_i != -1:
                        antenna_jacobian[output_i, recalibrate_i] = 1

                    if event_polarization==0 or event_polarization==1:
                        event_L_jacobian[output_i, current_param_i + 0] = dx
                        event_L_jacobian[output_i, current_param_i + 1] = dy
                        event_L_jacobian[output_i, current_param_i + 2] = dz
                        event_L_jacobian[output_i, current_param_i + 3] = 1

                    elif event_polarization==2 and antenna_polarization==1:
                        event_L_jacobian[output_i, current_param_i + 0] = dx
                        event_L_jacobian[output_i, current_param_i + 1] = dy
                        event_L_jacobian[output_i, current_param_i + 2] = dz
                        event_L_jacobian[output_i, current_param_i + 4] = 1

                    elif event_polarization==3 and antenna_polarization==1:
                        event_L_jacobian[output_i, current_param_i + 4+ 0] = dx
                        event_L_jacobian[output_i, current_param_i + 4+ 1] = dy
                        event_L_jacobian[output_i, current_param_i + 4+ 2] = dz
                        event_L_jacobian[output_i, current_param_i + 4+ 3] = 1
                        
                    output_i += 1

            current_param_i = next_param_i

        return jacobian_ret
        
    def objective_fun_sq(self, np.ndarray[double , ndim=1] guess, np.ndarray[double , ndim=1] ret=None, info=None):
        if ret is None:
            ret = np.empty(self.total_num_measurments)
        
        cdef double[:] station_delays = guess[ : self.num_station_delays]
        cdef double[:] antenna_delays = guess[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:] event_XYZTs = guess[self.num_station_delays+self.num_antenna_recalibrations : ]
            
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        
        cdef double Xodd = 0
        cdef double Yodd = 0
        cdef double Zodd = 0
        cdef double Todd = 0
        
        cdef double X_to_use
        cdef double Y_to_use
        cdef double Z_to_use
        cdef double T_to_use
        
        cdef double dx
        cdef double dy
        cdef double dz
        #cdef double dt
        
        cdef int event_i
        cdef int antenna_i
        cdef int station_i
        cdef int recalibrate_i
        cdef double total_delay
        cdef int output_i = 0
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        cdef long current_param_i = 0
        cdef long event_polarization
        cdef int antenna_polarization # 0 for even, 1 for odd
        for event_i in range(self.num_events):
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            
            event_polarization = self.event_polarizations[ event_i ]
                        
            X = event_XYZTs[ current_param_i + 0]
            Y = event_XYZTs[ current_param_i + 1]
            Z = event_XYZTs[ current_param_i + 2]
            Z = fabs(Z)
            T = event_XYZTs[ current_param_i + 3]
            
            current_param_i += 4
            if event_polarization == 2:
                Todd = event_XYZTs[ current_param_i ]
                current_param_i += 1
            elif  event_polarization == 3:
                Xodd = event_XYZTs[ current_param_i + 0]
                Yodd = event_XYZTs[ current_param_i + 1]
                Zodd = event_XYZTs[ current_param_i + 2]
                Zodd = fabs(Zodd)
                Todd = event_XYZTs[ current_param_i + 3]
                current_param_i += 4
            
            antenna_polarization = -1
            for antenna_i in range(self.num_antennas):
                antenna_polarization += 1
                if antenna_polarization == 2:
                    antenna_polarization = 0
                
                if filter_slice[antenna_i] and (event_polarization==2 or event_polarization==3 or (event_polarization==0 and antenna_polarization==0) or (event_polarization==1 and antenna_polarization==1)):
                    
                    X_to_use = X
                    Y_to_use = Y
                    Z_to_use = Z
                    T_to_use = T
                    if event_polarization==2 and antenna_polarization==1:
                        T_to_use = Todd
                    elif event_polarization==3 and antenna_polarization==1:
                        X_to_use = Xodd
                        Y_to_use = Yodd
                        Z_to_use = Zodd
                        T_to_use = Todd
                    
                    dx = self.antenna_locations[antenna_i, 0] - X_to_use
                    dy = self.antenna_locations[antenna_i, 1] - Y_to_use
                    dz = self.antenna_locations[antenna_i, 2] - Z_to_use
                    
                    total_delay = 0.0
                    station_i = self.station_indexes[ antenna_i ]
                    if station_i != self.num_station_delays:
                        total_delay = guess[ station_i ]
                    recalibrate_i =  self.antenna_recalibration_indeces[ antenna_i ]
                    if recalibrate_i != -1:
                        total_delay += antenna_delays[ recalibrate_i ]
                    
                    ret[output_i] = T_to_use + total_delay - measurement_slice[ antenna_i ]
                    ret[output_i] *= c_air
                    ret[output_i] *= ret[output_i]
                    
                    ret[output_i] -= dx*dx
                    ret[output_i] -= dy*dy
                    ret[output_i] -= dz*dz
                        
                    output_i += 1
        
        return ret
        
    def objective_fun_sq_jacobian(self, np.ndarray[double , ndim=1] guess, np.ndarray[double , ndim=2] jacobian_ret=None, info=None):
        if jacobian_ret is None:
            jacobian_ret = np.zeros( (self.total_num_measurments, self.total_num_parameters), dtype=np.double )
        
        cdef double[:] station_delays = guess[ : self.num_station_delays]
        cdef double[:] antenna_delays = guess[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:] event_XYZTs = guess[self.num_station_delays+self.num_antenna_recalibrations : ]


        cdef double[:,:] station_jacobian = jacobian_ret[:, :self.num_station_delays]
        cdef double[:,:] antenna_jacobian = jacobian_ret[:, self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:,:] event_L_jacobian = jacobian_ret[:, self.num_station_delays+self.num_antenna_recalibrations :]
            
            
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        
        cdef double Xodd = 0
        cdef double Yodd = 0
        cdef double Zodd = 0
        cdef double Todd = 0
        
        cdef double X_to_use
        cdef double Y_to_use
        cdef double Z_to_use
        cdef double T_to_use
        
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double dt
        
        cdef int event_i
        cdef int antenna_i
        cdef int station_i
        cdef int recalibrate_i
        cdef double total_delay
        cdef int output_i = 0
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        cdef long current_param_i = 0
        cdef long next_param_i
        cdef long event_polarization
        cdef int antenna_polarization # 0 for even, 1 for odd
        for event_i in range(self.num_events):
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            
            event_polarization = self.event_polarizations[ event_i ]
                        
            X = event_XYZTs[ current_param_i + 0]
            Y = event_XYZTs[ current_param_i + 1]
            Z = event_XYZTs[ current_param_i + 2]
            Z = fabs(Z)
            T = event_XYZTs[ current_param_i + 3]
            
            next_param_i = current_param_i + 4
            if event_polarization == 2:
                Todd = event_XYZTs[ current_param_i ]
                next_param_i += 1
            elif  event_polarization == 3:
                Xodd = event_XYZTs[ current_param_i + 0]
                Yodd = event_XYZTs[ current_param_i + 1]
                Zodd = event_XYZTs[ current_param_i + 2]
                Zodd = fabs(Zodd)
                Todd = event_XYZTs[ current_param_i + 3]
                next_param_i += 4
            
            antenna_polarization = -1
            for antenna_i in range(self.num_antennas):
                antenna_polarization += 1
                if antenna_polarization == 2:
                    antenna_polarization = 0
                
                if filter_slice[antenna_i] and (event_polarization==2 or event_polarization==3 or (event_polarization==0 and antenna_polarization==0) or (event_polarization==1 and antenna_polarization==1)):
                    
                    X_to_use = X
                    Y_to_use = Y
                    Z_to_use = Z
                    T_to_use = T
                    if event_polarization==2 and antenna_polarization==1:
                        T_to_use = Todd
                    elif event_polarization==3 and antenna_polarization==1:
                        X_to_use = Xodd
                        Y_to_use = Yodd
                        Z_to_use = Zodd
                        T_to_use = Todd
                    
                    dx = self.antenna_locations[antenna_i, 0] - X_to_use
                    dy = self.antenna_locations[antenna_i, 1] - Y_to_use
                    dz = self.antenna_locations[antenna_i, 2] - Z_to_use
                    
                    total_delay = 0.0

                    station_i = self.station_indexes[ antenna_i ]
                    if station_i != self.num_station_delays:
                        total_delay = guess[ station_i ]

                    recalibrate_i =  self.antenna_recalibration_indeces[ antenna_i ]
                    if recalibrate_i != -1:
                        total_delay += antenna_delays[ recalibrate_i ]

                    dt = 2*c_air*c_air*( T_to_use + total_delay - measurement_slice[ antenna_i ] )


                    if station_i != self.num_station_delays:
                        station_jacobian[output_i, station_i] = dt

                    if recalibrate_i != -1:
                        antenna_jacobian[output_i, recalibrate_i] = dt

                    if event_polarization==0 or event_polarization==1:
                        event_L_jacobian[output_i, current_param_i + 0] = -dx
                        event_L_jacobian[output_i, current_param_i + 1] = -dy
                        event_L_jacobian[output_i, current_param_i + 2] = -dz
                        event_L_jacobian[output_i, current_param_i + 3] = dt

                    elif event_polarization==2 and antenna_polarization==1:
                        event_L_jacobian[output_i, current_param_i + 0] = -dx
                        event_L_jacobian[output_i, current_param_i + 1] = -dy
                        event_L_jacobian[output_i, current_param_i + 2] = -dz
                        event_L_jacobian[output_i, current_param_i + 4] = dt

                    elif event_polarization==3 and antenna_polarization==1:
                        event_L_jacobian[output_i, current_param_i + 4+ 0] = -dx
                        event_L_jacobian[output_i, current_param_i + 4+ 1] = -dy
                        event_L_jacobian[output_i, current_param_i + 4+ 2] = -dz
                        event_L_jacobian[output_i, current_param_i + 4+ 3] = dt

                    output_i += 1
            current_param_i = next_param_i
        
        return jacobian_ret
    
    def RMS(self, guess, num_DOF):
        diffs = self.objective_fun( guess )
        diffs *= diffs
        #print(diffs)
        return np.sqrt( np.sum(diffs)/num_DOF )
    
    def event_SSqE(self, event_i, guess, ant_range=None):
        if ant_range is None:
            start = 0
            stop = self.num_antennas
        else:
            start = ant_range[0]
            stop = ant_range[1]
            
        cdef double[:] station_delays = guess[ : self.num_station_delays]
        cdef double[:] antenna_delays = guess[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:] event_XYZTs = guess[self.num_station_delays+self.num_antenna_recalibrations : ]
            
        cdef int offset=0
        cdef int i
        for i in range( event_i ):
            offset += 4
            if self.event_polarizations[ i ] == 2:
                offset += 1
            if self.event_polarizations[ i ] == 3:
                offset += 4
        
        cdef long event_polarization = self.event_polarizations[ event_i ]
        
        cdef double X = event_XYZTs[ offset + 0]
        cdef double Y = event_XYZTs[ offset + 1]
        cdef double Z = event_XYZTs[ offset + 2]
        Z = fabs(Z)
        cdef double T = event_XYZTs[ offset + 3]
        
        cdef double Xodd = 0
        cdef double Yodd = 0
        cdef double Zodd = 0
        cdef double Todd = 0
        
        cdef double X_to_use = 0
        cdef double Y_to_use = 0
        cdef double Z_to_use = 0
        cdef double T_to_use = 0
        
        if event_polarization == 2:
            Todd = event_XYZTs[ offset + 4]
        elif  event_polarization == 3:
            Xodd = event_XYZTs[ offset + 4]
            Yodd = event_XYZTs[ offset + 5]
            Zodd = event_XYZTs[ offset + 6]
            Zodd = fabs(Zodd)
            Todd = event_XYZTs[ offset + 7]
        
        cdef double[:] measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
        cdef np.uint8_t[:] filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
        
        cdef double SSqE = 0
        cdef int num_ants = 0
        cdef int antenna_i
        cdef int antenna_polarization = -1
        for antenna_i in range(start, stop):
            antenna_polarization += 1
            if antenna_polarization == 2:
                antenna_polarization = 0
                    
            if filter_slice[antenna_i] and (event_polarization==2 or event_polarization==3 or (event_polarization==0 and antenna_polarization==0) or (event_polarization==1 and antenna_polarization==1)):
                
                X_to_use = X
                Y_to_use = Y
                Z_to_use = Z
                T_to_use = T
                if event_polarization==2 and antenna_polarization==1:
                    T_to_use = Todd
                elif event_polarization==3 and antenna_polarization==1:
                    X_to_use = Xodd
                    Y_to_use = Yodd
                    Z_to_use = Zodd
                    T_to_use = Todd
                
                dx = self.antenna_locations[antenna_i, 0] - X_to_use
                dy = self.antenna_locations[antenna_i, 1] - Y_to_use
                dz = self.antenna_locations[antenna_i, 2] - Z_to_use
                
                total_delay = 0.0
                station_i = self.station_indexes[ antenna_i ]
                if station_i != self.num_station_delays:
                    total_delay = guess[ station_i ]
                recalibrate_i =  self.antenna_recalibration_indeces[ antenna_i ]
                if recalibrate_i != -1:
                    total_delay += antenna_delays[ recalibrate_i ]
                    
                dt = sqrt(dx*dx + dy*dy + dz*dz)*c_air_inverse + T_to_use + total_delay
                dt -= measurement_slice[ antenna_i ]
                
                SSqE += dt*dt
                num_ants += 1
                
        return SSqE, num_ants
    
    def print_antenna_info(self, int antenna_index, np.ndarray[double , ndim=1] guess):
        
        cdef double[:] station_delays = guess[ : self.num_station_delays]
        cdef double[:] antenna_delays = guess[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:] event_XYZTs = guess[self.num_station_delays+self.num_antenna_recalibrations : ]
            
        cdef double antX = self.antenna_locations[antenna_index, 0]
        cdef double antY = self.antenna_locations[antenna_index, 1]
        cdef double antZ = self.antenna_locations[antenna_index, 2]
        print("antenna XYZ")
        print(" ", antX, antY, antZ)
        
        
        cdef int antenna_polarization = 0 # just to guess
        
        
        cdef double total_delay = 0.0
        cdef int station_i = self.station_indexes[ antenna_index ]
        if station_i != self.num_station_delays:
            total_delay = guess[ station_i ]
        print("station delay:", total_delay)
        cdef int recalibrate_i = self.antenna_recalibration_indeces[ antenna_index ]
        if recalibrate_i != -1:
            print("recalibration with delay:", antenna_delays[ recalibrate_i ])
            total_delay += antenna_delays[ recalibrate_i ]
        print("total delay", total_delay)
        
        
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        
        cdef double Xodd = 0
        cdef double Yodd = 0
        cdef double Zodd = 0
        cdef double Todd = 0
        
        cdef double X_to_use
        cdef double Y_to_use
        cdef double Z_to_use
        cdef double T_to_use
        
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double dt
        
        cdef int event_i
        cdef long event_polarization
        cdef long current_param_i = 0
        
        print()
        
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        for event_i in range(self.num_events):
            print('event', event_i)
            
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
                    
            event_polarization = self.event_polarizations[ event_i ]
            
            X = event_XYZTs[ event_i*4 + 0]
            Y = event_XYZTs[ event_i*4 + 1]
            Z = event_XYZTs[ event_i*4 + 2]
            Z = fabs(Z)
            T = event_XYZTs[ event_i*4 + 3]
            
            current_param_i += 4
            if event_polarization == 2:
                Todd = event_XYZTs[ current_param_i ]
                current_param_i += 1
            elif  event_polarization == 3:
                Xodd = event_XYZTs[ current_param_i + 0]
                Yodd = event_XYZTs[ current_param_i + 1]
                Zodd = event_XYZTs[ current_param_i + 2]
                Zodd = fabs(Zodd)
                Todd = event_XYZTs[ current_param_i + 3]
                current_param_i += 4
            
            print(" ", X,Y,Z,T)
            
            if filter_slice[antenna_index]:
                
                X_to_use = X
                Y_to_use = Y
                Z_to_use = Z
                T_to_use = T
                if event_polarization==2 and antenna_polarization==1:
                    T_to_use = Todd
                elif event_polarization==3 and antenna_polarization==1:
                    X_to_use = Xodd
                    Y_to_use = Yodd
                    Z_to_use = Zodd
                    T_to_use = Todd
            
                dx = antX - X_to_use
                dy = antY - Y_to_use
                dz = antZ - Z_to_use
                    
                print(dx, dy, dz)
                print(' R:', sqrt(dx*dx + dy*dy + dz*dz))
                dt = sqrt(dx*dx + dy*dy + dz*dz)*c_air_inverse + T_to_use + total_delay
                print(' model:', dt)
                dt -= measurement_slice[ antenna_index ]
                print(' error:', dt)
            else:
                print( " antenna not used in event" )
                
            print()
    
            
    def fit_by_antenna(self, np.ndarray[double , ndim=1] guess):
        ### returns the SSqE for each antenna, and the number of measurments for each antenna
        cdef np.ndarray[double , ndim=1] ant_SSqE = np.zeros(self.num_antennas, dtype=np.double)
        cdef np.ndarray[long , ndim=1] ant_num = np.zeros(self.num_antennas, dtype=np.int)
        
        cdef double[:] station_delays = guess[ : self.num_station_delays]
        cdef double[:] antenna_delays = guess[self.num_station_delays : self.num_station_delays+self.num_antenna_recalibrations]
        cdef double[:] event_XYZTs = guess[self.num_station_delays+self.num_antenna_recalibrations : ]
            
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        
        cdef double Xodd = 0
        cdef double Yodd = 0
        cdef double Zodd = 0
        cdef double Todd = 0
        
        cdef double X_to_use
        cdef double Y_to_use
        cdef double Z_to_use
        cdef double T_to_use
        
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double dt
        
        cdef int event_i
        cdef int antenna_i
        cdef int station_i
        cdef int recalibrate_i
        cdef double total_delay
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        cdef long current_param_i = 0
        cdef long event_polarization
        cdef int antenna_polarization # 0 for even, 1 for odd
        for event_i in range(self.num_events):
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            
            event_polarization = self.event_polarizations[ event_i ]
                        
            X = event_XYZTs[ current_param_i + 0]
            Y = event_XYZTs[ current_param_i + 1]
            Z = event_XYZTs[ current_param_i + 2]
            Z = fabs(Z)
            T = event_XYZTs[ current_param_i + 3]
            
            current_param_i += 4
            if event_polarization == 2:
                Todd = event_XYZTs[ current_param_i ]
                current_param_i += 1
            elif  event_polarization == 3:
                Xodd = event_XYZTs[ current_param_i + 0]
                Yodd = event_XYZTs[ current_param_i + 1]
                Zodd = event_XYZTs[ current_param_i + 2]
                Zodd = fabs(Zodd)
                Todd = event_XYZTs[ current_param_i + 3]
                current_param_i += 4
            
            antenna_polarization = -1
            for antenna_i in range(self.num_antennas):
                antenna_polarization += 1
                if antenna_polarization == 2:
                    antenna_polarization = 0
                
                if filter_slice[antenna_i] and (event_polarization==2 or event_polarization==3 or (event_polarization==0 and antenna_polarization==0) or (event_polarization==1 and antenna_polarization==1)):
                    
                    X_to_use = X
                    Y_to_use = Y
                    Z_to_use = Z
                    T_to_use = T
                    if event_polarization==2 and antenna_polarization==1:
                        T_to_use = Todd
                    elif event_polarization==3 and antenna_polarization==1:
                        X_to_use = Xodd
                        Y_to_use = Yodd
                        Z_to_use = Zodd
                        T_to_use = Todd
                    
                    dx = self.antenna_locations[antenna_i, 0] - X_to_use
                    dy = self.antenna_locations[antenna_i, 1] - Y_to_use
                    dz = self.antenna_locations[antenna_i, 2] - Z_to_use
                    
                    total_delay = 0.0
                    station_i = self.station_indexes[ antenna_i ]
                    if station_i != self.num_station_delays:
                        total_delay = guess[ station_i ]
                    recalibrate_i =  self.antenna_recalibration_indeces[ antenna_i ]
                    if recalibrate_i != -1:
                        total_delay += antenna_delays[ recalibrate_i ]
                        
                    dt = sqrt(dx*dx + dy*dy + dz*dz)*c_air_inverse + T_to_use + total_delay
                    dt -= measurement_slice[ antenna_i ]
                    
                    ant_SSqE[antenna_i] += dt*dt
                    ant_num[antenna_i] += 1
                    
                    
        return ant_SSqE, ant_num
    
    def analytical_covariance_matrix(self, solution, weight=1.0e-9):
        
        cdef double inv_weight_sq = 1.0/(weight*weight)
        
        N = len(solution) #- self.num_station_delays - self.num_antenna_recalibrations
        hess = np.zeros( (N, N), dtype=np.double )
        
        cdef double[:] measurement_slice
        cdef np.uint8_t[:] filter_slice
        cdef double X
        cdef double Y
        cdef double Z
        cdef double T
        cdef double Xodd = 0
        cdef double Yodd = 0
        cdef double Zodd = 0
        cdef double Todd = 0
        cdef double X_to_use = 0
        cdef double Y_to_use = 0
        cdef double Z_to_use = 0
        cdef double T_to_use = 0
        cdef double dx
        cdef double dy
        cdef double dz
        cdef double distance
        cdef double inv_D
        cdef double f
        cdef double dfdx
        cdef double dfdy
        cdef double dfdz
        cdef double ddfddx
        cdef double ddfddy
        cdef double ddfddz
        cdef double ddfdxdy
        cdef double ddfdxdz
        cdef double ddfdydz
        cdef int X_i
        cdef int Y_i
        cdef int Z_i
        cdef int T_i
        cdef int X_i_odd = 0
        cdef int Y_i_odd = 0
        cdef int Z_i_odd = 0
        cdef int T_i_odd = 0
        cdef int X_i_toUse
        cdef int Y_i_toUse
        cdef int Z_i_toUse
        cdef int T_i_toUse
        cdef int stat_cal_i
        cdef int ant_cal_i
        
        cdef int event_i
        cdef long event_polarization
        cdef int antenna_i
        cdef int antenna_polarization
        cdef int solution_offset = self.num_station_delays + self.num_antenna_recalibrations
        for event_i in range(self.num_events):
            measurement_slice = self.measurement_times[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            filter_slice = self.measurement_filter[event_i*self.num_antennas : (event_i+1)*self.num_antennas]
            event_polarization = self.event_polarizations[ event_i ]
                        
            X_i = solution_offset + 0
            Y_i = solution_offset + 1
            Z_i = solution_offset + 2
            T_i = solution_offset + 3
            
            X = solution[ X_i ]
            Y = solution[ Y_i ]
            Z = solution[ Z_i ]
            Z = fabs(Z)
            T = solution[ T_i ]
            solution_offset += 4
            
            if event_polarization == 2:
                T_i_odd = solution_offset
                Todd = solution[ T_i_odd ]
                solution_offset += 1
                
            elif  event_polarization == 3:
                X_i_odd = solution_offset
                Y_i_odd = solution_offset + 1
                Z_i_odd = solution_offset + 2
                T_i_odd = solution_offset + 3
                
                Xodd = solution[ X_i_odd ]
                Yodd = solution[ Y_i_odd ]
                Zodd = solution[ Z_i_odd]
                Zodd = fabs(Zodd)
                Todd = solution[ T_i_odd ]
                solution_offset += 4
#            
            antenna_polarization = -1
            for antenna_i in range(self.num_antennas):
                antenna_polarization += 1
                if antenna_polarization == 2:
                    antenna_polarization = 0
                    
                if filter_slice[antenna_i] and (event_polarization==2 or event_polarization==3 or (event_polarization==0 and antenna_polarization==0) or (event_polarization==1 and antenna_polarization==1)):
                    
                    X_toUse = X
                    Y_toUse = Y
                    Z_toUse = Z
                    T_toUse = T
                    X_i_toUse = X_i
                    Y_i_toUse = Y_i
                    Z_i_toUse = Z_i
                    T_i_toUse = T_i
                    if event_polarization==2 and antenna_polarization==1:
                        T_toUse = Todd
                        T_i_toUse = T_i_odd
                    elif event_polarization==3 and antenna_polarization==1: ## what? should this be '3' and not '4'??
                        X_toUse = Xodd
                        Y_toUse = Yodd
                        Z_toUse = Zodd
                        T_toUse = Todd
                        X_i_toUse = X_i_odd
                        Y_i_toUse = Y_i_odd
                        Z_i_toUse = Z_i_odd
                        T_i_toUse = T_i_odd
                        
                        
                    dx = X_toUse - self.antenna_locations[antenna_i, 0]
                    dy = Y_toUse - self.antenna_locations[antenna_i, 1]
                    dz = Z_toUse - self.antenna_locations[antenna_i, 2]
                    
                    distance = np.sqrt( dx*dx + dy*dy + dz*dz )                    
                    
                    f = distance*c_air_inverse + T_toUse
                    f -= measurement_slice[antenna_i]
                    stat_cal_i = self.station_indexes[ antenna_i ]
                    if stat_cal_i != self.num_station_delays:
                        f += solution[ stat_cal_i ]
                    ant_cal_i =  self.antenna_recalibration_indeces[ antenna_i ]
                    if ant_cal_i != -1:
                        ant_cal_i += self.num_station_delays
                        f += solution[ ant_cal_i ]
                    
                    #### pure spatial component ####
                    inv_D = 1.0/(distance*c_air)
                    dfdx = dx*inv_D
                    dfdy = dy*inv_D
                    dfdz = dz*inv_D
                    
                    inv_D = -1.0/(distance*distance*distance*c_air)
                    ddfddx = dx*dx*inv_D
                    ddfdxdy = dx*dy*inv_D
                    ddfdxdz = dx*dz*inv_D
                    ddfddy = dy*dy*inv_D
                    ddfdydz = dy*dz*inv_D
                    ddfddz = dz*dz*inv_D
                    
                    inv_D = 1.0/(distance*c_air)
                    ddfddx += inv_D
                    ddfddy += inv_D
                    ddfddz += inv_D
                    
                    hess[X_i_toUse, X_i_toUse] += 2*inv_weight_sq*( dfdx*dfdx + f*ddfddx )
                    hess[Y_i_toUse, Y_i_toUse] += 2*inv_weight_sq*( dfdy*dfdy + f*ddfddy )
                    hess[Z_i_toUse, Z_i_toUse] += 2*inv_weight_sq*( dfdz*dfdz + f*ddfddz )
                    
                    hess[X_i_toUse, Y_i_toUse] += 2*inv_weight_sq*( dfdx*dfdy + f*ddfdxdy )
                    hess[Y_i_toUse, X_i_toUse] += 2*inv_weight_sq*( dfdx*dfdy + f*ddfdxdy )
                    
                    hess[X_i_toUse, Z_i_toUse] += 2*inv_weight_sq*( dfdx*dfdz + f*ddfdxdz )
                    hess[Z_i_toUse, X_i_toUse] += 2*inv_weight_sq*( dfdx*dfdz + f*ddfdxdz )
                    
                    hess[Y_i_toUse, Z_i_toUse] += 2*inv_weight_sq*( dfdy*dfdz + f*ddfdydz )
                    hess[Z_i_toUse, Y_i_toUse] += 2*inv_weight_sq*( dfdy*dfdz + f*ddfdydz )
                    
                    #### pure time ####
                    hess[T_i_toUse, T_i_toUse] += 2*inv_weight_sq # so simple!
                    
                    #### mix time and space ####
                    hess[T_i_toUse, X_i_toUse] += 2*inv_weight_sq*( dfdx )
                    hess[X_i_toUse, T_i_toUse] += 2*inv_weight_sq*( dfdx )
                    
                    hess[T_i_toUse, Y_i_toUse] += 2*inv_weight_sq*( dfdy )
                    hess[Y_i_toUse, T_i_toUse] += 2*inv_weight_sq*( dfdy )
                    
                    hess[T_i_toUse, Z_i_toUse] += 2*inv_weight_sq*( dfdz )
                    hess[Z_i_toUse, T_i_toUse] += 2*inv_weight_sq*( dfdz )
                    
                    #### station cal ####
                    if stat_cal_i != self.num_station_delays:
                        #### pure ####
                        hess[stat_cal_i, stat_cal_i] += 2*inv_weight_sq
                        
                        #### mix with space ####
                        hess[stat_cal_i, X_i_toUse] += 2*inv_weight_sq*( dfdx )
                        hess[X_i_toUse, stat_cal_i] += 2*inv_weight_sq*( dfdx )
                        
                        hess[stat_cal_i, Y_i_toUse] += 2*inv_weight_sq*( dfdy )
                        hess[Y_i_toUse, stat_cal_i] += 2*inv_weight_sq*( dfdy )
                        
                        hess[stat_cal_i, Z_i_toUse] += 2*inv_weight_sq*( dfdz )
                        hess[Z_i_toUse, stat_cal_i] += 2*inv_weight_sq*( dfdz )
                    
                        #### mix with time ####
                        hess[stat_cal_i, T_i_toUse] += 2*inv_weight_sq
                        hess[T_i_toUse, stat_cal_i] += 2*inv_weight_sq
                        
                    #### finally, antenna calibration ####
                    if ant_cal_i != -1:
                        #### pure ####
                        hess[ant_cal_i, ant_cal_i] += 2*inv_weight_sq
                        
                        #### mix with space ####
                        hess[ant_cal_i, X_i_toUse] += 2*inv_weight_sq*( dfdx )
                        hess[X_i_toUse, ant_cal_i] += 2*inv_weight_sq*( dfdx )
                        
                        hess[ant_cal_i, Y_i_toUse] += 2*inv_weight_sq*( dfdy )
                        hess[Y_i_toUse, ant_cal_i] += 2*inv_weight_sq*( dfdy )
                        
                        hess[ant_cal_i, Z_i_toUse] += 2*inv_weight_sq*( dfdz )
                        hess[Z_i_toUse, ant_cal_i] += 2*inv_weight_sq*( dfdz )
                    
                        #### mix with time ####
                        hess[ant_cal_i, T_i_toUse] += 2*inv_weight_sq
                        hess[T_i_toUse, ant_cal_i] += 2*inv_weight_sq
                    
                        #### mix with stat. cal ####
                        if stat_cal_i != self.num_station_delays:
                            hess[stat_cal_i, ant_cal_i] += 2*inv_weight_sq
                            hess[ant_cal_i, stat_cal_i] += 2*inv_weight_sq
                    
#        hess = hess[self.num_station_delays:self.num_station_delays+self.num_antenna_recalibrations, self.num_station_delays:self.num_station_delays+self.num_antenna_recalibrations]
                    
#        print(hess)
        cov_mat = np.linalg.inv( hess )      
        cov_mat *= 2
#        print()
#        print()
#        print()
#        print(cov_mat)
        
        return cov_mat
                        
                    
                
        
    
    
    
    
    