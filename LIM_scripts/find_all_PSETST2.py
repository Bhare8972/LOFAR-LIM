#!/usr/bin/env python3

##internal
import time
from os import mkdir
from os.path import isdir

##import external packages
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.linalg import lstsq

##my packages
from LoLIM.utilities import log, processed_data_dir, v_air
from LoLIM.read_pulse_data import read_station_info, refilter_pulses, curtain_plot_CodeLog, AntennaPulse_dict__TO__AntennaTime_dict, getNwriteBin_modData
from LoLIM.porta_code import code_logger
from LoLIM.IO.binary_IO import write_long, write_double_array, write_string, write_double

#PSE_next_unique_index = 0
#class PointSourceEvent:
#    def __init__(self, pulse_dict, guess_location, station_info_dict):
#        global PSE_next_unique_index
#        self.unique_index = PSE_next_unique_index
#        PSE_next_unique_index += 1
#        
#        self.pulse_dict = pulse_dict
#        self.station_info_dict = station_info_dict
#        
#        ant_locations = []
#        PolE_times = []
#        PolO_times = []
#        self.antennas_included = []
#        
#        self.num_even_ant = 0
#        self.num_odd_ant = 0
#        
#        for pulse in self.pulse_dict.values():
#            loc = pulse.antenna_info.location
#            ant_locations.append(loc)
#            self.antennas_included.append( pulse.even_antenna_name )
#            
#            PolE_times.append( pulse.PolE_peak_time ) ###NOTE that peak times are infinity if something is wrong with antenna
#            PolO_times.append( pulse.PolO_peak_time ) ###NOTE that peak times are infinity if something is wrong with antenna
#            
#            if np.isfinite( pulse.PolE_peak_time ):
#                self.num_even_ant += 1
#            if np.isfinite( pulse.PolO_peak_time ):
#                self.num_odd_ant += 1
#            
#        self.antenna_locations = np.array(ant_locations)
#        PolE_times = np.array(PolE_times)
#        PolO_times = np.array(PolO_times)
#        
#        self.residual_workspace = np.zeros(len(PolE_times))
#        self.jacobian_workspace = np.zeros( (len(PolE_times), 4) )
#         
#        ### fit even polarisation ###
#        ## TODO: fit simultaniously!
#        self.ant_times = PolE_times
#        EvenPol_min = least_squares(self.objective_RES, guess_location, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', max_nfev=1000)
#        self.PolE_loc = EvenPol_min.x
#        self.PolE_RMS = np.sqrt( self.SSqE(self.PolE_loc)/float(self.num_even_ant) )
#        
#        ### fit odd polarisation ###
#        self.ant_times = PolO_times
#        OddPol_min = least_squares(self.objective_RES, guess_location, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', max_nfev=1000)
#        self.PolO_loc = OddPol_min.x
#        self.PolO_RMS = np.sqrt( self.SSqE(self.PolO_loc)/float(self.num_odd_ant) )
#        
#        
#        print()
#        print("PSE", self.unique_index)
#        print("  even red chi sq:", self.PolE_RMS, ',', self.num_even_ant, "antennas")
#        print("  even_loc:", self.PolE_loc)
#        print("  ", EvenPol_min.message)
#        print("  odd red chi sq:", self.PolO_RMS, ',', self.num_odd_ant, "antennas")
#        print("  odd loc:", self.PolO_loc)
#        print("  ", OddPol_min.message)
#        print()
#        
#        ## free resources
#        del self.residual_workspace
#        del self.jacobian_workspace
#        del self.ant_times
#        del self.antenna_locations
#        
#    def objective_RES(self, XYZT):
#        
#        self.residual_workspace[:] = self.ant_times 
#        self.residual_workspace[:] -= XYZT[3]
#        self.residual_workspace[:] *= v_air
#        self.residual_workspace[:] *= self.residual_workspace[:]
#        
#        R2 = self.antenna_locations - XYZT[0:3]
#        R2 *= R2
#        self.residual_workspace[:] -= R2[:,0]
#        self.residual_workspace[:] -= R2[:,1]
#        self.residual_workspace[:] -= R2[:,2]
#        
#        self.residual_workspace[ np.logical_not(np.isfinite(self.residual_workspace)) ] = 0.0
#        
#        return self.residual_workspace
#    
#    def objective_JAC(self, XYZT):
#        self.jacobian_workspace[:, 0:3] = XYZT[0:3]
#        self.jacobian_workspace[:, 0:3] -= self.antenna_locations
#        self.jacobian_workspace[:, 0:3] *= -2.0
#        
#        self.jacobian_workspace[:, 3] = self.ant_times
#        self.jacobian_workspace[:, 3] -= XYZT[3]
#        self.jacobian_workspace[:, 3] *= -v_air*v_air*2
#        
#        mask = np.logical_not(np.isfinite(self.jacobian_workspace[:, 3])) 
#        self.jacobian_workspace[mask, :] = 0
#        
#        return self.jacobian_workspace
#    
#    def SSqE(self, XYZT):
#        R2 = self.antenna_locations - XYZT[0:3]
#        R2 *= R2
#        
#        theory = np.sum(R2, axis=1)
#        np.sqrt(theory, out=theory)
#        
#        theory *= 1.0/v_air
#        theory += XYZT[3] - self.ant_times
#        
#        theory *= theory
#        
#        
#        theory[ np.logical_not(np.isfinite(theory) ) ] = 0.0
#        
#        return np.sum(theory)
#    
#    def get_ModelTime(self, station_info, polarity,inclusion=0):
#        ret = []
#        for ant_name in station_info.sorted_antenna_names:
#            
#            if inclusion != 1:
#                included = ant_name in self.antennas_included
#                if inclusion==1 and not included:
#                    ret.append(np.inf)
#                    continue
#                if inclusion==2 and included:
#                    ret.append(np.inf)
#                    continue
#                    
#            ant_loc = station_info.AntennaInfo_dict[ant_name].location
#            
#            if polarity == 0:
#                dx = np.linalg.norm( self.PolE_loc[0:3]-ant_loc)/v_air
#                ret.append( dx+self.PolE_loc[3] )
#            elif polarity == 1:
#                dx = np.linalg.norm( self.PolO_loc[0:3]-ant_loc)/v_air
#                ret.append( dx+self.PolO_loc[3] )
#                
#        return ret
#        
#    
#    def save_as_binary(self, fout):
#        
#        write_long(fout, self.unique_index)
#        write_double(fout, self.PolE_RMS)
#        write_double(fout, 0.0) ## reduced chi-squared
#        write_double(fout, 0.0) ## power
#        write_double_array(fout, self.PolE_loc)
#        write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0])) ## standerd errors
#        
#        write_double(fout, self.PolO_RMS)
#        write_double(fout, 0.0) ## reduced chi-squared
#        write_double(fout, 0.0) ## power
#        write_double_array(fout, self.PolO_loc)
#        write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0]))## standerd errors
#        
#        write_double(fout, 5.0E-9) ##seconds per sample
#        write_long(fout, self.num_even_ant)
#        write_long(fout, self.num_odd_ant)
#        
#        write_long(fout, 1) ## 1 means that we do save trace data
#        
#        write_long(fout, len(self.pulse_dict))
#        for ant_name, pulse in self.pulse_dict.items():
#            write_string(fout, ant_name)
#            write_long(fout, pulse.section_number)
#            write_long(fout, pulse.unique_index)
#            write_long(fout, pulse.starting_index)
#            write_long(fout, pulse.antenna_status)
#            
#            write_double(fout, pulse.PolE_peak_time)
#            write_double(fout, pulse.PolE_estimated_timing_error)
#            write_double(fout, pulse.PolE_time_offset)
#            write_double(fout, np.max(pulse.even_antenna_hilbert_envelope))
#            write_double(fout, pulse.even_data_STD)
#            
#            write_double(fout, pulse.PolO_peak_time)
#            write_double(fout, pulse.PolO_estimated_timing_error)
#            write_double(fout, pulse.PolO_time_offset)
#            write_double(fout, np.max(pulse.odd_antenna_hilbert_envelope))
#            write_double(fout, pulse.odd_data_STD)
#            
#            write_double_array(fout, pulse.even_antenna_hilbert_envelope)
#            write_double_array(fout, pulse.even_antenna_data)
#            write_double_array(fout, pulse.odd_antenna_hilbert_envelope)
#            write_double_array(fout, pulse.odd_antenna_data)
#            
#        
#        
#
###### roaming hyper-bubble #####
#def count_antennas(XYZT, half_R_t, antPos_dicts, antTime_dicts):
#    max_diff = 0.0
#    n_ant = 0
#    for sname, antPulse_dict in antTime_dicts.items():
#        for ant_name, pulse_list in antPulse_dict.items():
#                    
#            ant_loc = antPos_dicts[sname][ant_name]
#            DT = np.linalg.norm( XYZT[0:3]-ant_loc )/v_air
#                    
#            min_box = XYZT[3] + DT - half_R_t
#            max_box = XYZT[3] + DT + half_R_t
#
#            minInd = np.searchsorted(antTime_dicts[sname][ant_name], min_box)
#                    
#            maxInd = np.searchsorted(antTime_dicts[sname][ant_name], max_box)
#                    
#            if (maxInd-minInd)>0:
#                n_ant += 1
#                delta_T = antTime_dicts[sname][ant_name][minInd:maxInd] - (XYZT[3] + DT)
#                max_delta = np.max( np.abs(delta_T) )
#                if max_delta>max_diff:
#                    max_diff = max_delta
#            
#    return n_ant, max_diff
#        
#def AlgorithmOfTheRoamingHypersphere(min_half_R_t, initial_half_R_t, initial_XYZT, antPos_dicts, antTime_dicts):
#    
#    initial_XYZT = np.array(initial_XYZT)
#    initial_XYZT[0:3] *= 1.0/v_air ##put everything in seconds, so that Nelder-Mead tollarances are consistent
#    
#    def obj_fun(XYZT):
#        ### hope that this is safe to modify....=
#        scaled_XYZT = np.array(XYZT)
#        scaled_XYZT[0:3] *= v_air ##put correct things back into meters
#        
#        N, trash = count_antennas(scaled_XYZT, current_half_R_t, antPos_dicts, antTime_dicts)
#        if N == 0:
#            return 2
#        else:
#            return 1.0/N
#    
#    current_half_R_t = initial_half_R_t
#    current_XYZT  = initial_XYZT
#    i = 0
#    while current_half_R_t > min_half_R_t:
#        i += 1
#        
#        min_ret = minimize(obj_fun, current_XYZT, method='Nelder-Mead', options={'xatol':current_half_R_t/100.0})
#        current_XYZT = min_ret.x
#        
#        
#        rescaled_XYZT = np.array(current_XYZT)
#        rescaled_XYZT[0:3] *= v_air ##put correct things back into meters
#        current_N,  max_dif = count_antennas(rescaled_XYZT, current_half_R_t, antPos_dicts, antTime_dicts)
##        print("RHS itter:", i, current_half_R_t, current_N, " "*50, end='\r', flush=True)
#        
##        print()
##        print( current_half_R_t )
##        print( current_N, i)
##        print( rescaled_XYZT, min_ret.message )
##        print( max_dif/current_half_R_t,  current_N)
#
#        new_current_half_R_t = max_dif*0.99
#        
#        if not new_current_half_R_t<current_half_R_t:
#            current_half_R_t *= 0.75
#        else: 
#            current_half_R_t = new_current_half_R_t
#        
#    i += 1
#    current_half_R_t = min_half_R_t
#    min_ret = minimize(obj_fun, current_XYZT, method='Nelder-Mead', options={'xatol':current_half_R_t/100.0})
#    current_XYZT = min_ret.x
#    
#    rescaled_XYZT = np.array(current_XYZT)
#    rescaled_XYZT[0:3] *= v_air ##put correct things back into meters
#    current_N,  max_dif = count_antennas(rescaled_XYZT, current_half_R_t, antPos_dicts, antTime_dicts)
#    print("RHS itter:", i, current_half_R_t, current_N, " "*50)
#        
#    
##    current_XYZT[0:3] *= v_air ##back into meters
#
#    return rescaled_XYZT, current_N
#
##def AlgorithmOfTheRoamingHypersphere_2(min_half_R_t, initial_half_R_t, initial_XYZT, antPos_dicts, antTime_dicts):
##    
##    #### write algorithm that works simular to the minimization loop that runs after the hyperspher algorithm. Except reduce the time diff each time. 
##    
##    ## have two loops, an inner loop that minimizes location each run acn clauclates number of antennas in radius. Stops running once number of antennas is stable. 
##    
##    ## an outer loop that reduces radius each turn. Stops once radius is small enough. 
##    
##    
##    current_half_R_t = initial_half_R_t
##    
##    pulse_indeces, residuals= get_pulse_indecesNres(initial_XYZT, antPos_dicts, PolE_pulseTime_dicts, current_half_R_t)
##    current_num_antennas = len(pulse_indeces)
##    
##    current_location = np.array( initial_XYZT )
###    current_location *= 1.0/v_air ##convert units to seconds
##    
##    outer_i = 0
##    while current_half_R_t > min_half_R_t and current_num_antennas>=num_ant_threshold:
##        outer_i += 1
##        
##        pulse_indeces, residuals= get_pulse_indecesNres(current_location, antPos_dicts, PolE_pulseTime_dicts, current_half_R_t)
##        current_num_antennas = len(pulse_indeces)
##        print("outer loop", outer_i, current_half_R_t, current_num_antennas)
##        
##        inner_i = 0
##        while True:
##            inner_i += 1
##            current_location[2] = np.abs(current_location[2])
##
##            EvenPol_min = least_squares(individual_obj_fun_RES, current_location, jac=individual_obj_fun_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', 
##                                        max_nfev=1000, args=(antPos_dicts, PolE_pulseTime_dicts, pulse_indeces))
##            
##            rescaled_loc = np.array(EvenPol_min.x)
###            rescaled_loc[0:3] *= v_air
##
##            pulse_indeces, residuals= get_pulse_indecesNres(rescaled_loc, antPos_dicts, PolE_pulseTime_dicts, current_half_R_t)
##            new_num_antennas = len(pulse_indeces)
##                    
##            print("RHS loop", outer_i, inner_i)
##            print("  ", EvenPol_min.message)
##            print("   loc:", rescaled_loc)
##            print("   num ant:", new_num_antennas, "RMS:", np.sqrt( np.sum(residuals**2)/new_num_antennas ) )
##                    
##            if new_num_antennas ==0:
##                print("WARINGING! zero ant")
##                break
##            
##            max_residual = np.max(np.abs(residuals))
##            print("   maximum residual:", max_residual)
##                    
##            if new_num_antennas<current_num_antennas:
##                print("WARNING! number of antennas is reduced!")
##                    
##            if new_num_antennas <5:
##                print("not enough ant")
##                break
##            elif new_num_antennas == current_num_antennas:
##                current_num_antennas = new_num_antennas
##                current_location = EvenPol_min.x
##                break
##            else:
##                current_num_antennas = new_num_antennas
##                current_location = EvenPol_min.x
##                
##                
##        new_current_half_R_t = max_residual*0.99
##        if not new_current_half_R_t<current_half_R_t:
##            current_half_R_t *= 0.75
##        else: 
##            current_half_R_t = new_current_half_R_t
#        
#        
#    
#
###### routines for initial fitting #####
#
#
#def get_pulse_indecesNres(XYZT, antPos_dicts, antTime_dicts, T_window):
#    XYZT[2] = np.abs(XYZT[2])
#    
#    pulse_indeces = []
#    pulse_residuals = []
#    for sname, pulse_dict in antTime_dicts.items():
#        for ant_name, pulse_times in pulse_dict.items():
#            
#            ant_loc = antPos_dicts[sname][ant_name]
#            DT = np.linalg.norm( XYZT[0:3]-ant_loc )/v_air
#            
#            residuals = np.abs( antTime_dicts[sname][ant_name] - (XYZT[3] + DT) )
#            idx = np.argmin( residuals )
#            
#            if residuals[idx]<T_window:
#                pulse_residuals.append( residuals[idx] )
#                pulse_indeces.append( [idx, sname, ant_name] )
#            
#    return pulse_indeces, np.array(pulse_residuals)
#
### objective functions
#def individual_obj_fun_RES(XYZT, antPos_dicts, antTime_dicts, pulse_indeces):
#    global residual_workspace
#    XYZT[2] = np.abs(XYZT[2])
#    
#    ant_i = -1
#    for pulse_i, sname, ant_name in pulse_indeces:
#        ant_i += 1
#        
#        if pulse_i<0:
#            residual_workspace[ant_i] = 0.0
#            continue
#            
#        pulse_time = antTime_dicts[sname][ant_name][ pulse_i ]
#         
#        ant_location = antPos_dicts[sname][ant_name]
#        
#        R2 = ant_location - XYZT[0:3]
#        R2 *= R2
#        R2 = np.sum(R2)
#        
#        ant_T_diff = pulse_time - XYZT[3]
#        ant_T_diff *= v_air
#        ant_T_diff *= ant_T_diff
#        
#        ant_T_diff -= R2
#        
#        residual_workspace[ant_i] = ant_T_diff
#                
#                
#    return residual_workspace
#
#def individual_obj_fun_JAC(XYZT, antPos_dicts, antTime_dicts, pulse_indeces):
#    global jacobian_workspace
#    XYZT[2] = np.abs(XYZT[2])
#    
#    ant_i = -1
#    for pulse_i, sname, ant_name in pulse_indeces:
#        ant_i += 1
#            
#        if pulse_i<0:
#            jacobian_workspace[ant_i] = 0.0
#            continue
#        
#        pulse_time = antTime_dicts[sname][ant_name][ pulse_i ]
#        ant_location = antPos_dicts[sname][ant_name]
#            
#        jacobian_workspace[ant_i, 0:3] = 2.0*( ant_location - XYZT[0:3] )
#        jacobian_workspace[ant_i, 3] = 2.0*(v_air*v_air)* ( XYZT[3] - pulse_time )
#                
#    return jacobian_workspace


def station_combinations(stations, num_to_pick=5, exclude_station=None):
    def recursive_combiner(current_solution, viable_stations, num_to_add):
        if num_to_add == 0:
            return [current_solution]
        
        ret = []
        
        N = len(viable_stations) - num_to_add + 1
        for i in range(N):
            if exclude_station is not None and viable_stations[i]==exclude_station:
                continue
            
            new_solution = list(current_solution)
            new_solution.append( viable_stations[i] )
            
            ret.append( recursive_combiner( new_solution, viable_stations[i+1:], num_to_add-1 ) )
            
        return [item for sublist in ret for item in sublist]
    
    return recursive_combiner([], stations, num_to_pick)

def make_pulse_combos( station_list, pulse_dict ):
    solutions = [[]]
    for sname in station_list:
        new_solutions = []
        
        for s in solutions:
            for new_pulse in pulse_dict[sname]:
                new_s = list(s)
                new_s.append( new_pulse )
                new_solutions.append( new_s )
                
        solutions = new_solutions
        
    return solutions



class S1_solver:
    def __init__(self, antenna_locations):
        self.ant_locs = np.array( antenna_locations )
        self.size = len(antenna_locations) - 1
        self.matrix = np.empty( (self.size, 4) )
        
        R2_vector = np.empty( len(antenna_locations) )
        R2_vector[0] = np.sum( antenna_locations[0]*antenna_locations[0] )
        
        for i in range(self.size):
            R2_vector[i+1] = np.sum( antenna_locations[i+1]*antenna_locations[i+1] )
            DR = antenna_locations[0] - antenna_locations[i+1]
            self.matrix[i,:3] = DR
            
        self.pre_K_vector = R2_vector[0] - R2_vector[1:]
        
        self.T_vector = np.empty(self.size + 1)
        self.K_vector = np.array(self.pre_K_vector)
        
    def hyperplane(self, arrival_times):
        
        if np.any( np.logical_not( np.isfinite(arrival_times) ) ):
            return None, None
        
        self.T_vector[:] = arrival_times
        self.matrix[:,3] = arrival_times[0]
        self.matrix[:,3] -= self.T_vector[1:]
        self.matrix[:,3] *= -v_air*v_air

        ### make K_vector avoiding allocating new memory
        self.K_vector[:] = self.T_vector[1:]
        self.K_vector *= self.T_vector[1:]
        self.K_vector -= arrival_times[0]**2
        self.K_vector *= v_air*v_air
        
        self.K_vector += self.pre_K_vector
        self.K_vector *= 0.5
        
        solution, junk,junk,junk = lstsq(self.matrix, self.K_vector)
        
        
        if solution[2]<0 or solution[2]>10000:
            solution[2] = 6000

        return solution, self.RMS( solution )
    
    def residuals(self, XYZT):
        XYZT[2] = np.abs(XYZT[2])
        
        theory = self.ant_locs - XYZT[:3]
        
        theory = np.linalg.norm( theory, axis=1 )
        
        theory *= 1.0/v_air
        theory += XYZT[3]
        theory -= self.T_vector
        return theory
    
    def RMS(self, XYZT):
        res = self.residuals(XYZT)
        res *= res
        ave = np.sum(res)/(self.size+1)
        return np.sqrt(ave)
    
    def residual_jac(self, XYZT):
        XYZT[2] = np.abs(XYZT[2])
        
        re = np.empty( (self.size+1, 4) )
        
        re[:,:3] = XYZT[:3]
        re[:,:3] -= self.ant_locs
        R = np.linalg.norm( re[:,:3], axis=1 )
        re[:,0] /= R
        re[:,1] /= R
        re[:,2] /= R
        re[:,:3] *= 1.0/v_air
        
        re[:,3] = 1.0
        
        return re
        
    
    def LS_fit(self, arrival_times, guess_XYZT):
        
#        fit_res = least_squares(self.residuals, guess_XYZT, jac=self.residual_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
        fit_res = least_squares(self.residuals, guess_XYZT, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
        
                    
        return fit_res.x,  self.RMS(fit_res.x) 
    
class synth_pulse:
    class fake_station:
        def __init__(self, same, loc):
            self.station_name = sname
            self.loc = loc
            
        def get_station_location(self):
            return self.loc
            
        
    class fake_antenna:
        def __init__(self, loc):
            self.location = loc
    
    def __init__(self, ant_loc, sname, source_XYZT):
            self.station_info = self.fake_station(sname, ant_loc)
            self.antenna_info = self.fake_antenna(ant_loc)
            self.PolE_peak_time = np.linalg.norm(ant_loc-source_XYZT[:3])/v_air + source_XYZT[3]
            
    def peak_time(self):
            return self.PolE_peak_time
        
    def best_SNR(self):
        return 100.0

if __name__ == "__main__":
    
    timeID = "D20170929T202255.000Z"
    output_folder = "allPSE_runTST"
    
    plot_station_map = True
    
    stations_to_exclude =  ["CS028","RS106", "RS305", "RS205", "CS201", "RS407", "RS406"]
    
    num_blocks_per_step = 100
    initial_block = 3500
    num_steps = 100
    
    ant_timing_calibrations = "cal_tables/TxtAntDelay"
    polarization_flips = "polarization_flips.txt"
    bad_antennas = "bad_antennas.txt"
    additional_antenna_delays = "ant_delays.txt"
    
    station_delays = "/station_delays.txt"
    
    min_signal_SNR = 0.0
    
    timing_bin_mode = 10.0E-6 ## only keep strongest pulse in this time bin
    max_RMS_fit = 100E-9
    min_num_ant = 100
    
    
    search_location = np.array( [-1.58423240e+04,   9.08114847e+03,   4.0e+03] )
    
#    hyphersphere_initial_size = 20000.0
#    hypersphere_min_size = 10.0E-9 ##1000
     
#    max_ant_time_residual = 100.0E-9 #### if the residual of an antenna time is greater than this, the antenna is thrown out ## 1000E-9
#    num_ant_thr1eshold = 50
    
    
    
    #### setup directory variables ####
    processed_data_dir = processed_data_dir(timeID)
    
    data_dir = processed_data_dir + "/" + output_folder
    if not isdir(data_dir):
        mkdir(data_dir)
        
    logging_folder = data_dir + '/logs_and_plots'
    if not isdir(logging_folder):
        mkdir(logging_folder)

    #Setup logger and open initial data set
    log.set(logging_folder + "/log_out.txt") ## TODo: save all output to a specific output folder
    log.take_stderr()
    log.take_stdout()
    
    
    #### print details
    print("Time ID:", timeID)
    print("output folder name:", output_folder)
    print("date and time run:", time.strftime("%c") )
    print("stations to exclude:", stations_to_exclude)
    print("initial block", initial_block, "num blocks per step:", initial_block, "num steps", num_steps)
    print("min signal SNR", min_signal_SNR)
    print("antenna timing calibrations:", ant_timing_calibrations)
    print("antenna polarization flips file:", polarization_flips)
    print("bad antennas file:", bad_antennas)
    print("station delays file:", station_delays)
    print("ant delays file:", additional_antenna_delays)
    print()
#    print("Hypersphere search location:", search_location)
#    print("Hypersphere initial size (m):", hyphersphere_initial_size)
#    print("Hypersphere min size (s):", hypersphere_min_size)
#    print("maximum residual:", max_ant_time_residual)
#    print("num ant threshold:", num_ant_threshold)
    
    ##station data
    print()
    print("opening station data")
    
    polarization_flips = processed_data_dir + '/' + polarization_flips
    bad_antennas = processed_data_dir + '/' + bad_antennas
    station_delays = processed_data_dir + '/' + station_delays
    ant_timing_calibrations = processed_data_dir + '/' + ant_timing_calibrations
    additional_antenna_delays = processed_data_dir + '/' + additional_antenna_delays
    
    StationInfo_dict = read_station_info(timeID, stations_to_exclude=stations_to_exclude, bad_antennas_fname=bad_antennas, ant_delays_fname=additional_antenna_delays, 
                                         pol_flips_fname=polarization_flips, station_delays_fname=station_delays, txt_cal_table_folder=ant_timing_calibrations)
        
    print("using stations:", StationInfo_dict.keys())

    if plot_station_map:
        CL = code_logger(logging_folder+ "/station_map")
        CL.add_statement("import numpy as np")
        CL.add_statement("import matplotlib.pyplot as plt")
        
        for sname,sdata in StationInfo_dict.items():
            ant_X = np.array([ ant.location[0] for ant in sdata.AntennaInfo_dict.values() ])
            ant_Y = np.array([ ant.location[1] for ant in sdata.AntennaInfo_dict.values() ])
            station_X = np.average(ant_X)
            station_Y = np.average(ant_Y)
            
            CL.add_function("plt.scatter", ant_X, ant_Y)
            CL.add_function("plt.annotate", sname, xy=(station_X, station_Y), size=30)
            
        CL.add_statement( "plt.tick_params(axis='both', which='major', labelsize=30)" )
        CL.add_statement( "plt.show()" )
        CL.save()
        
    ### get limited set of stations to use for stage 1. Throw out all but one core station
    stage1_station_names = [sname for sname in StationInfo_dict.keys() if sname[:2]=="RS"]
    for sname in StationInfo_dict.keys():
        if sname[:2] =="CS":
            stage1_station_names.append( sname ) ### add one core station
            break
        
    print("stage 1 stations:  ", stage1_station_names)

    ### now loop over steps
    for iter_i in range(num_steps):
        print( "Opening Data. Itter:", iter_i )
        
        current_block = initial_block + iter_i*num_blocks_per_step
        
        AntennaPulse_dicts = {sname:sdata.read_pulse_data(approx_min_block=current_block, approx_max_block=current_block+num_blocks_per_step) for sname,sdata in StationInfo_dict.items()}

        ##refilter pulses
        #### NOTE: we are only keeping pulses if they have good even polarization, due to way pulse finding works
        AntennaPulse_dicts_refiltered = {sname:refilter_pulses(AntPulse_dict, min_signal_SNR, pol_requirement=3) for sname,AntPulse_dict in AntennaPulse_dicts.items()}
        
        ## SYNTH data
#        source_XYZT = np.append( search_location, [1.5] )
#        for sname, antPulse_dict in AntennaPulse_dicts_refiltered.items():
#            for ant_name in antPulse_dict.keys():
#                ant_loc = StationInfo_dict[sname].AntennaInfo_dict[ant_name].location
#                new_pulse = synth_pulse(ant_loc, sname, source_XYZT)
#                antPulse_dict[ant_name] = [new_pulse]
            
        
        ## now we sort pulses for each antenna, so that we only have one pulse per bin of width timing_bin_mode
        last_pulse_time = -np.inf
        for sname, AntPulse_dict in AntennaPulse_dicts_refiltered.items():
            for antname, pulse_list in AntPulse_dict.items():
                ##note: pulse_list is already sorted by time
                if len(pulse_list) == 0:
                    continue
                
                new_list = []
                bin_end_time = pulse_list[0].peak_time() + timing_bin_mode
                pulses_to_choose = []
                for pulse in pulse_list:
                    if pulse.peak_time() > bin_end_time:
                        bin_end_time += timing_bin_mode
                        new_found_pulse = max( pulses_to_choose, key=lambda x: x.best_SNR() )
                        new_list.append( new_found_pulse )
                        pulses_to_choose = []
                        
                    pulses_to_choose.append( pulse )
                    
                new_found_pulse = max( pulses_to_choose, key=lambda x: x.best_SNR() )
                new_list.append( new_found_pulse )
                    
#                AntPulse_dict[ antname ] = new_list
                
                LT = new_list[-1].peak_time()
                if LT > last_pulse_time:
                    last_pulse_time = LT
                
        
        #### now we try to find sources
        num_active_ant = min_num_ant
        while num_active_ant>=min_num_ant:
            
            #### STAGE 1: get guess of source location. ONLY USE REMOTE STATIONS  + 1 core station
            
            # first we find the earliest pulse
            num_active_ant = 0
            earliest_pulse = None
            for sname in stage1_station_names:
                
                for antname, pulse_list in  AntennaPulse_dicts_refiltered[ sname ].items(): 
                    if len(pulse_list)==0:
                        continue
                    
                    num_active_ant += 1
                    pulse = pulse_list[0]
                    
                    if earliest_pulse is None or pulse.peak_time() < earliest_pulse.peak_time():
                        earliest_pulse = pulse
                        
            print()
            referance_station = earliest_pulse.station_info.station_name
            print("   next earliest pulse at T:", earliest_pulse.peak_time(), '/', last_pulse_time, "station", referance_station)
            
            
            #### find all pulses in viable time
            ref_station_loc = earliest_pulse.station_info.get_station_location()
            viable_pulses = {}
            for sname, AntPulse_dict in AntennaPulse_dicts_refiltered.items():
                if sname == referance_station:
                    continue
                
                latest_viable_time = earliest_pulse.peak_time() + np.linalg.norm( ref_station_loc - StationInfo_dict[sname].get_station_location() )/v_air
                
                viable_pulses[sname] = {}
                for antname, pulse_list in AntPulse_dict.items():
                    viable_pulse_list = []
                    
                    for pulse in pulse_list:
                        if pulse.peak_time() < latest_viable_time:
                            viable_pulse_list.append( pulse )
                        else:
                            break
                        
                    viable_pulses[sname][antname] = viable_pulse_list
                    
                    
            #### limit pulses for stage 1
            stage_1_pulses = {}
            stage_1_antennas = {}
            for sname in stage1_station_names:
                if sname == referance_station:
                    continue
                
                best_ant = None
                max_num_pulses = 0
            
                for ant_name, viable_pulse_list in viable_pulses[sname].items():
                    
                    if len(viable_pulse_list) > max_num_pulses:
                        max_num_pulses = len(viable_pulse_list)
                        best_ant = ant_name
                        
                if best_ant is not None:
                    stage_1_pulses[sname] = viable_pulses[sname][best_ant]
                    stage_1_antennas[sname] = StationInfo_dict[sname].AntennaInfo_dict[best_ant]
                    
                    
            #### if not enough stations, then quit early
            if len(stage_1_pulses) < 5:
                print("      not enough stations with viable pulses")
                ref_ant = earliest_pulse.even_antenna_name
                AntennaPulse_dicts_refiltered[ referance_station ][ ref_ant ].remove( earliest_pulse )
                continue
                
            ### test out combinations of 6 stations, including referance station
            stat_combos = station_combinations( list(stage_1_pulses.keys()), 5, referance_station )
            best_RMS = np.inf
            best_station_list = None
            best_pulse_list = None
            best_location = None
            best_solver = None
            
            best_LS_RMS = np.inf
            best_LS_RMS_unPol = np.inf
            
            for stat_list in stat_combos:
                ant_locs = [stage_1_antennas[sname].location for sname in stat_list]
                ant_locs.append( earliest_pulse.antenna_info.location )
                
                solver = S1_solver( ant_locs )
                
                pulse_combo_list = make_pulse_combos( stat_list, stage_1_pulses ) 
                for pulse_list in pulse_combo_list:
                    pulse_list.append( earliest_pulse )
                    PolE_peak_times = [ pulse.PolE_peak_time for pulse in pulse_list ]
                    
                    location, RMS = solver.hyperplane( PolE_peak_times )
                
                    if RMS is not None and RMS<best_RMS:
                        best_RMS = RMS
                        best_station_list = stat_list
                        best_pulse_list = pulse_list
                        best_location = location
                        best_solver = solver
                    
                    LS_loc, LSRMS = solver.LS_fit( PolE_peak_times, np.append(search_location, [best_location[3]] ) )
                    if LSRMS is not None and LSRMS<best_LS_RMS:
                        best_LS_RMS = LSRMS
                    if LSRMS is not None and LSRMS<best_LS_RMS_unPol:
                        best_LS_RMS_unPol = LSRMS
                        
                        
                    print("HP:", RMS, "LS fit:", LSRMS)
                        
                    i =0
                    for sname, allpulses in stage_1_pulses.items():
                        all_T = [pulse.PolE_peak_time for pulse in allpulses]
                        
                        
                        
                        
#                    LS_loc, LSRMS = solver.LS_fit( [ pulse.PolO_peak_time for pulse in best_pulse_list ], np.append(search_location, [best_location[3]] ) )
#                    if LSRMS is not None and LSRMS<best_LS_RMS_unPol:
#                        best_LS_RMS_unPol = LSRMS
            
            best_location, best_RMS = solver.LS_fit([ pulse.PolE_peak_time for pulse in best_pulse_list ],  best_location)     
            print("      found RMS:", best_RMS, best_location, 'best LS rms:', best_LS_RMS, best_LS_RMS_unPol)
            
                      
            if best_RMS > max_RMS_fit or not np.isfinite(best_RMS):
                print("      RMS fit too high")
                ref_ant = earliest_pulse.even_antenna_name
                AntennaPulse_dicts_refiltered[ referance_station ][ ref_ant ].remove( earliest_pulse )
                continue
            
            print("YAY!!!!")
            print()
            print()
                
                
                
                

