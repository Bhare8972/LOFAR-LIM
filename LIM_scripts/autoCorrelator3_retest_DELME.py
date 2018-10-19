#!/usr/bin/env python3

#python
import time
from os import mkdir, listdir
from os.path import isdir, isfile
from itertools import chain
from pickle import load

#external
import numpy as np
from scipy.optimize import least_squares, minimize, approx_fprime
from scipy.signal import hilbert
from matplotlib import pyplot as plt


import h5py


#mine
from LoLIM.prettytable import PrettyTable
from LoLIM.utilities import log, processed_data_dir, v_air, SId_to_Sname, Sname_to_SId_dict, RTD
from LoLIM.IO.binary_IO import read_long, write_long, write_double_array, write_string, write_double
from LoLIM.antenna_response import LBA_ant_calibrator
from LoLIM.porta_code import code_logger, pyplot_emulator
from LoLIM.signal_processing import parabolic_fit, remove_saturation, data_cut_at_index
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1
from LoLIM.findRFI import window_and_filter
from LoLIM.autoCorrelator_tools import stationDelay_fitter
#from RunningStat import RunningStat


inv_v_air = 1.0/v_air

#### some random utilities
def none_max(lst):
    """given a list of numbers, return maximum, ignoreing None"""
    
    ret = -np.inf
    for a in lst:
        if (a is not None) and (a>ret):
            ret=a
            
    return ret
                
    
def get_radius_ze_az( XYZ ):
    radius = np.linalg.norm( XYZ )
    ze = np.arccos( XYZ[2]/radius )
    az = np.arctan2( XYZ[1], XYZ[0] )
    return radius, ze, az
    

#### main code


class stochastic_fitter:
    def __init__(self, source_object_list, initial_guess=None, quick_kill=None):
        print("running stochastic fitter")
        self.quick_kill = quick_kill
        self.source_object_list = source_object_list
    
    ## assume globals: 
#    max_itters_per_loop 
#    itters_till_convergence
#    max_jitter_width
#    min_jitter_width 
#    cooldown
    
#    sorted_antenna_names
    
        self.num_antennas = len(sorted_antenna_names)
        self.num_measurments = self.num_antennas*len(source_object_list)
        self.num_delays = len(station_order)
        
        #### make guess ####
        self.num_DOF = -self.num_delays
        self.solution = np.zeros(  self.num_delays+4*len(source_object_list) )
        self.solution[:self.num_delays] = current_delays_guess
        param_i = self.num_delays
        for PSE in source_object_list:
            self.solution[param_i:param_i+4] = PSE.guess_XYZT
            param_i += 4
            self.num_DOF += PSE.num_DOF()
            
        if initial_guess is not None: ## use initial guess instead, if given
            self.solution = initial_guess
            
        self.initial_guess = np.array( self.solution )
            
        
        self.rerun()
        

            
            
        
        
    def objective_fun(self, sol, do_print=False):
        workspace_sol = np.zeros(self.num_measurments, dtype=np.double)
        delays = sol[:self.num_delays]
        ant_i = 0
        param_i = self.num_delays
        for PSE in self.source_object_list:
            
            PSE.try_location_LS(delays, sol[param_i:param_i+4], workspace_sol[ant_i:ant_i+self.num_antennas])
            
            ant_i += self.num_antennas
            param_i += 4
            
        filter = np.logical_not( np.isfinite(workspace_sol) )
        workspace_sol[ filter ]  = 0.0
        
        if do_print:
            print("num func nans:", np.sum(filter))
            
        return workspace_sol
    #        workspace_sol *= workspace_sol
    #        return np.sum(workspace_sol)
        
    
    def objective_jac(self, sol, do_print=False):
        workspace_jac = np.zeros((self.num_measurments, self.num_delays+4*len(self.source_object_list)), dtype=np.double)
    
            
        delays = sol[:self.num_delays]
        ant_i = 0
        param_i = self.num_delays
        for PSE in self.source_object_list:
            
            PSE.try_location_JAC(delays, sol[param_i:param_i+4],  workspace_jac[ant_i:ant_i+self.num_antennas, param_i:param_i+4],  
                                 workspace_jac[ant_i:ant_i+self.num_antennas, 0:self.num_delays])
            
            filter = np.logical_not( np.isfinite(workspace_jac[ant_i:ant_i+self.num_antennas, param_i+3]) )
            workspace_jac[ant_i:ant_i+self.num_antennas, param_i:param_i+4][filter] = 0.0
            workspace_jac[ant_i:ant_i+self.num_antennas, 0:self.num_delays][filter] = 0.0
            
            ant_i += self.num_antennas
            param_i += 4
    

        
        if do_print:
            print("num jac nans:", np.sum(filter))
        
        return workspace_jac
            
    def rerun(self):
        
        current_guess = np.array( self.solution )
        
        #### first itteration ####
        fit_res = least_squares(self.objective_fun, current_guess, jac=self.objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
#        fit_res = least_squares(self.objective_fun, current_guess, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
        
        current_guess = fit_res.x
        current_fit = fit_res.cost
        
        current_temperature = max_jitter_width
        new_guess = np.array( current_guess )
        while current_temperature>min_jitter_width: ## loop over each 'temperature'
            
            print("  stochastic run. Temp:", current_temperature)
            
            itters_since_change = 0
            has_improved = False
            for run_i in range(max_itters_per_loop):
                print("  run:", run_i, ':', itters_since_change, " "*10, end="\r")
#                print("  itter", run_i)
                
                ## jitter the initial guess ##
                new_guess[:self.num_delays] = np.random.normal(scale=current_temperature, size=self.num_delays) + current_guess[:self.num_delays] ## note use of best_solution, allows for translation. Faster convergence?
            
                param_i = self.num_delays
                for PSE in self.source_object_list:
                    new_guess[param_i:param_i+3] = np.random.normal(scale=current_temperature*v_air, size=3) + current_guess[param_i:param_i+3]
#                    new_guess[param_i+2] = np.abs(new_guess[param_i+2])## Z should be positive
                    
                    new_guess[param_i+3] = PSE.estimate_T(new_guess[:self.num_delays], new_guess[param_i:param_i+4])
                    param_i += 4
                
                #### FIT!!! ####
                fit_res = least_squares(self.objective_fun, new_guess, jac=self.objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
#                fit_res = least_squares(self.objective_fun, new_guess, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
#                print("    cost:", fit_res.cost)
        
                if fit_res.cost < current_fit:
                    current_fit = fit_res.cost
                    current_guess = fit_res.x
                    if (fit_res.cost-current_fit)/current_fit > 0.001:
                        itters_since_change = 0
                        has_improved = True
                else:
                    itters_since_change += 1
                if itters_since_change == itters_till_convergence:
                    break
                
            print(" "*30,end="\r")
                
            if has_improved:
                current_temperature /= cooldown
            else:
                current_temperature /= strong_cooldown
                    
            
            total_RMS = 0.0
            new_station_delays = current_guess[:self.num_delays] 
            param_i = self.num_delays
            for PSE in self.source_object_list:
                total_RMS += PSE.SSqE_fit( new_station_delays,  current_guess[param_i:param_i+4] )
                param_i += 4
                
            total_RMS = np.sqrt(total_RMS/self.num_DOF)
            print("    RMS fit", total_RMS, "num runs:", run_i+1, "cost", current_fit)
            
            if self.quick_kill is not None and total_RMS>self.quick_kill:
                print("    quick kill exceeded")
                break
            
        if run_i+1 == max_itters_per_loop:
            print("  not converged!")
            self.converged = False
        else:
            self.converged = True
            
            
            
        #### get individual fits per station and PSE
        self.PSE_fits = []
        self.PSE_RMS_fits = []
        new_station_delays = current_guess[:self.num_delays] 
        param_i = self.num_delays
        for source in self.source_object_list:
            self.PSE_fits.append( source.RMS_fit_byStation(new_station_delays, 
                         current_guess[param_i:param_i+4]) )
            
            SSQE = source.SSqE_fit(new_station_delays, current_guess[param_i:param_i+4])
#            source.SSqE_fitNprint( new_station_delays,  current_guess[param_i:param_i+4] )
            self.PSE_RMS_fits.append( np.sqrt(SSQE/source.num_DOF()) )
            
            param_i += 4
            
            
            
        #### check which stations have fits
        self.stations_with_fits = [False]*( len(station_order)+1 )
            
        for stationfits in self.PSE_fits:
            for snum, (sname, fit) in enumerate(zip( chain(station_order, [referance_station]),  stationfits )):
                if fit is not None:
                    self.stations_with_fits[snum] = True
            
            
            
        #### edit solution for stations that don't have guess
        for snum, has_fit in enumerate(self.stations_with_fits[:-1]): #ignore last station, as is referance station
            if not has_fit:
                current_guess[ snum ] = self.initial_guess[ snum ]
            
            
        #### save results
        self.solution = current_guess
        self.cost = current_fit
        self.RMS = total_RMS
        

        
        
        
        
        
    def employ_result(self, source_object_list):
        """set the result to the guess location of the sources, and return the station timing offsets"""
        
        param_i = self.num_delays
        for PSE in source_object_list:
            PSE.guess_XYZT[:] =  self.solution[param_i:param_i+4]
            param_i += 4
            
        return self.solution[:self.num_delays]
    
    def print_locations(self, source_object_list):
        
        param_i = self.num_delays
        for source, RMSfit in zip(source_object_list, self.PSE_RMS_fits):
            print("source", source.ID)
            print("  RMS:", RMSfit)
            print("  loc:", self.solution[param_i:param_i+4])
            param_i += 4
    
    def print_station_fits(self, source_object_list):
        
        fit_table = PrettyTable()
        fit_table.field_names = ['id'] + station_order + [referance_station] + ['total']
        fit_table.float_format = '.2E'
        
        for source, RMSfit, stationfits in zip(source_object_list, self.PSE_RMS_fits, self.PSE_fits):
            new_row = ['']*len(fit_table.field_names)
            new_row[0] = source.ID
            new_row[-1] = RMSfit
            
            for i,stat_fit in enumerate(stationfits):
                if stat_fit is not None:
                    new_row[i+1] = stat_fit
                    
            fit_table.add_row( new_row )
            
        print( fit_table )
                    
    def print_delays(self, original_delays):
        for sname, delay, original in zip(station_order, self.solution[:self.num_delays], original_delays):
            print("'"+sname+"' : ",delay,', ## diff to guess:', delay-original)
    
    def get_stations_with_fits(self):
        return self.stations_with_fits
    
    

class stochastic_fitter_dt:
    def __init__(self, source_object_list, initial_guess=None, quick_kill=None):
        print("running stochastic fitter")
        self.quick_kill = quick_kill
        self.source_object_list = source_object_list
    
    ## assume globals: 
#    max_itters_per_loop 
#    itters_till_convergence
#    max_jitter_width
#    min_jitter_width 
#    cooldown
    
#    sorted_antenna_names
    
        self.num_antennas = len(sorted_antenna_names)
        self.num_measurments = self.num_antennas*len(source_object_list)
        self.num_delays = len(station_order)
        
        self.station_indeces = np.empty( len(ant_locs), dtype=np.int )
        for station_index, index_range in enumerate(station_to_antenna_index_list):
            first,last = index_range
            self.station_indeces[first:last] = station_index
    
        self.fitter = stationDelay_fitter(ant_locs, self.station_indeces, len(self.source_object_list), self.num_delays)
        for source in self.source_object_list:
            self.fitter.set_event( source.pulse_times )
        
        #### make guess ####
        self.num_DOF = -self.num_delays
        self.solution = np.zeros(  self.num_delays+4*len(source_object_list) )
        self.solution[:self.num_delays] = current_delays_guess
        param_i = self.num_delays
        for PSE in source_object_list:
            self.solution[param_i:param_i+4] = PSE.guess_XYZT
            param_i += 4
            self.num_DOF += PSE.num_DOF()
            
        if initial_guess is not None: ## use initial guess instead, if given
            self.solution = initial_guess
            
        self.initial_guess = np.array( self.solution )
            
        
        self.rerun()
        

            
            
    def rerun(self):
        
        current_guess = np.array( self.solution )
        
        #### first itteration ####
        fit_res = least_squares(self.fitter.objective_fun, current_guess, jac=self.fitter.objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
#        fit_res = least_squares(self.objective_fun, current_guess, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
        
        current_guess = fit_res.x
        current_fit = fit_res.cost
        
        current_temperature = max_jitter_width
        new_guess = np.array( current_guess )
        while current_temperature>min_jitter_width: ## loop over each 'temperature'
            
            print("  stochastic run. Temp:", current_temperature)
            
            itters_since_change = 0
            has_improved = False
            for run_i in range(max_itters_per_loop):
                print("  run:", run_i, ':', itters_since_change, " "*10, end="\r")
#                print("  itter", run_i)
                
                ## jitter the initial guess ##
                new_guess[:self.num_delays] = np.random.normal(scale=current_temperature, size=self.num_delays) + current_guess[:self.num_delays] ## note use of best_solution, allows for translation. Faster convergence?
            
                param_i = self.num_delays
                for PSE in self.source_object_list:
                    new_guess[param_i:param_i+3] = np.random.normal(scale=current_temperature*v_air, size=3) + current_guess[param_i:param_i+3]
#                    new_guess[param_i+2] = np.abs(new_guess[param_i+2])## Z should be positive
                    
                    new_guess[param_i+3] = PSE.estimate_T(new_guess[:self.num_delays], new_guess[param_i:param_i+4])
                    param_i += 4
                
                #### FIT!!! ####
                fit_res = least_squares(self.fitter.objective_fun, current_guess, jac=self.fitter.objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
#                fit_res = least_squares(self.objective_fun, new_guess, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
#                print("    cost:", fit_res.cost)
        
                if fit_res.cost < current_fit:
                    current_fit = fit_res.cost
                    current_guess = fit_res.x
                    if (fit_res.cost-current_fit)/current_fit > 0.001:
                        itters_since_change = 0
                        has_improved = True
                else:
                    itters_since_change += 1
                if itters_since_change == itters_till_convergence:
                    break
                
            print(" "*30,end="\r")
                
            if has_improved:
                current_temperature /= cooldown
            else:
                current_temperature /= strong_cooldown
                    
            
            total_RMS = 0.0
            new_station_delays = current_guess[:self.num_delays] 
            param_i = self.num_delays
            for PSE in self.source_object_list:
                total_RMS += PSE.SSqE_fit( new_station_delays,  current_guess[param_i:param_i+4] )
                param_i += 4
                
            total_RMS = np.sqrt(total_RMS/self.num_DOF)
            print("    RMS fit", total_RMS, "num runs:", run_i+1, "cost", current_fit)
            
            if self.quick_kill is not None and total_RMS>self.quick_kill:
                print("    quick kill exceeded")
                break
            
        if run_i+1 == max_itters_per_loop:
            print("  not converged!")
            self.converged = False
        else:
            self.converged = True
            
            
            
        #### get individual fits per station and PSE
        self.PSE_fits = []
        self.PSE_RMS_fits = []
        new_station_delays = current_guess[:self.num_delays] 
        param_i = self.num_delays
        for source in self.source_object_list:
            self.PSE_fits.append( source.RMS_fit_byStation(new_station_delays, 
                         current_guess[param_i:param_i+4]) )
            
            SSQE = source.SSqE_fit(new_station_delays, current_guess[param_i:param_i+4])
#            source.SSqE_fitNprint( new_station_delays,  current_guess[param_i:param_i+4] )
            self.PSE_RMS_fits.append( np.sqrt(SSQE/source.num_DOF()) )
            
            param_i += 4
            
            
            
        #### check which stations have fits
        self.stations_with_fits = [False]*( len(station_order)+1 )
            
        for stationfits in self.PSE_fits:
            for snum, (sname, fit) in enumerate(zip( chain(station_order, [referance_station]),  stationfits )):
                if fit is not None:
                    self.stations_with_fits[snum] = True
            
            
            
        #### edit solution for stations that don't have guess
        for snum, has_fit in enumerate(self.stations_with_fits[:-1]): #ignore last station, as is referance station
            if not has_fit:
                current_guess[ snum ] = self.initial_guess[ snum ]
            
            
        #### save results
        self.solution = current_guess
        self.cost = current_fit
        self.RMS = total_RMS
        

        
        
        
        
        
    def employ_result(self, source_object_list):
        """set the result to the guess location of the sources, and return the station timing offsets"""
        
        param_i = self.num_delays
        for PSE in source_object_list:
            PSE.guess_XYZT[:] =  self.solution[param_i:param_i+4]
            param_i += 4
            
        return self.solution[:self.num_delays]
    
    def print_locations(self, source_object_list):
        
        param_i = self.num_delays
        for source, RMSfit in zip(source_object_list, self.PSE_RMS_fits):
            print("source", source.ID)
            print("  RMS:", RMSfit)
            print("  loc:", self.solution[param_i:param_i+4])
            param_i += 4
    
    def print_station_fits(self, source_object_list):
        
        fit_table = PrettyTable()
        fit_table.field_names = ['id'] + station_order + [referance_station] + ['total']
        fit_table.float_format = '.2E'
        
        for source, RMSfit, stationfits in zip(source_object_list, self.PSE_RMS_fits, self.PSE_fits):
            new_row = ['']*len(fit_table.field_names)
            new_row[0] = source.ID
            new_row[-1] = RMSfit
            
            for i,stat_fit in enumerate(stationfits):
                if stat_fit is not None:
                    new_row[i+1] = stat_fit
                    
            fit_table.add_row( new_row )
            
        print( fit_table )
                    
    def print_delays(self, original_delays):
        for sname, delay, original in zip(station_order, self.solution[:self.num_delays], original_delays):
            print("'"+sname+"' : ",delay,', ## diff to guess:', delay-original)
    
    def get_stations_with_fits(self):
        return self.stations_with_fits
    
    
class stochastic_fitter_FitLocs:
    def __init__(self, source_object_list, quick_kill=None):
        print("running stochastic fitter")
        self.quick_kill = quick_kill
        self.source_object_list = source_object_list
    
    ## assume globals: 
#    max_itters_per_loop 
#    itters_till_convergence
#    max_jitter_width
#    min_jitter_width 
#    cooldown
    
#    sorted_antenna_names
    
        self.num_antennas = len(sorted_antenna_names)
        self.num_measurments = self.num_antennas*len(source_object_list)
        self.num_delays = len(station_order)
        
        self.used_delays = np.array(current_delays_guess)
        
        #### make guess ####
        self.num_DOF = 0
        self.solution = np.zeros(  4*len(source_object_list) )
        param_i = 0
        for PSE in source_object_list:
            self.solution[param_i:param_i+4] = PSE.guess_XYZT
            param_i += 4
            self.num_DOF += PSE.num_DOF()
            
            
        self.initial_guess = np.array( self.solution )
            
        
        self.rerun()
        

            
            
        
        
    def objective_fun(self, sol, do_print=False):
        workspace_sol = np.zeros(self.num_measurments, dtype=np.double)
        ant_i = 0
        param_i = 0
        for PSE in self.source_object_list:
            
            PSE.try_location_LS(self.used_delays, sol[param_i:param_i+4], workspace_sol[ant_i:ant_i+self.num_antennas])
            
            ant_i += self.num_antennas
            param_i += 4
            
        filter = np.logical_not( np.isfinite(workspace_sol) )
        workspace_sol[ filter ]  = 0.0
        
        if do_print:
            print("num func nans:", np.sum(filter))
            
        return workspace_sol
    #        workspace_sol *= workspace_sol
    #        return np.sum(workspace_sol)
        
    
    def objective_jac(self, sol, do_print=False):
        workspace_jac = np.zeros((self.num_measurments, self.num_delays+4*len(self.source_object_list)), dtype=np.double)
    
        ant_i = 0
        param_i = 0
        for PSE in self.source_object_list:
            
            PSE.try_location_JAC(self.used_delays, sol[param_i:param_i+4],  workspace_jac[ant_i:ant_i+self.num_antennas, param_i:param_i+4],  
                                 workspace_jac[ant_i:ant_i+self.num_antennas, 0:self.num_delays])
            
            filter = np.logical_not( np.isfinite(workspace_jac[ant_i:ant_i+self.num_antennas, param_i+3]) )
            workspace_jac[ant_i:ant_i+self.num_antennas, param_i:param_i+4][filter] = 0.0
            workspace_jac[ant_i:ant_i+self.num_antennas, 0:self.num_delays][filter] = 0.0
            
            ant_i += self.num_antennas
            param_i += 4
    

        
        if do_print:
            print("num jac nans:", np.sum(filter))
        
        return workspace_jac[:, self.num_delays:]
            
    def rerun(self):
        
        current_guess = np.array( self.solution )
        
        #### first itteration ####
        fit_res = least_squares(self.objective_fun, current_guess, jac=self.objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
#        fit_res = least_squares(self.objective_fun, current_guess, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
        
        current_guess = fit_res.x
        current_fit = fit_res.cost
        current_fit_obj = fit_res
        
        current_temperature = max_jitter_width
        new_guess = np.array( current_guess )
        while current_temperature>min_jitter_width: ## loop over each 'temperature'
            
            print("  stochastic run. Temp:", current_temperature)
            
            itters_since_change = 0
            has_improved = False
            for run_i in range(max_itters_per_loop):
                print("  run:", run_i, ':', itters_since_change, " "*10, end="\r")
#                print("  itter", run_i)
                
                ## jitter the initial guess ##
                param_i = 0
                for PSE in self.source_object_list:
                    new_guess[param_i:param_i+3] = np.random.normal(scale=current_temperature*v_air, size=3) + current_guess[param_i:param_i+3]
#                    new_guess[param_i+2] = np.abs(new_guess[param_i+2])## Z should be positive
                    
                    new_guess[param_i+3] = PSE.estimate_T(self.used_delays, new_guess[param_i:param_i+4])
                    param_i += 4
                
                #### FIT!!! ####
                fit_res = least_squares(self.objective_fun, new_guess, jac=self.objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
#                fit_res = least_squares(self.objective_fun, new_guess, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
#                print("    cost:", fit_res.cost)
        
                if fit_res.cost < current_fit:
                    current_fit = fit_res.cost
                    current_fit_obj = fit_res
                    current_guess = fit_res.x
                    if (fit_res.cost-current_fit)/current_fit > 0.001:
                        itters_since_change = 0
                        has_improved = True
                else:
                    itters_since_change += 1
                if itters_since_change == itters_till_convergence:
                    break
                
            print(" "*30,end="\r")
                
            if has_improved:
                current_temperature /= cooldown
            else:
                current_temperature /= strong_cooldown
                    
            
            total_RMS = 0.0
            param_i = 0
            for PSE in self.source_object_list:
                total_RMS += PSE.SSqE_fit( self.used_delays,  current_guess[param_i:param_i+4] )
                param_i += 4
                
            total_RMS = np.sqrt(total_RMS/self.num_DOF)
            print("    RMS fit", total_RMS, "num runs:", run_i+1, "success", current_fit_obj.success)
            
            if self.quick_kill is not None and total_RMS>self.quick_kill:
                print("    quick kill exceeded")
                break
            
        if run_i+1 == max_itters_per_loop:
            print("  not converged!")
            self.converged = False
        else:
            self.converged = True
            
            
            
        #### get individual fits per station and PSE
        self.PSE_fits = []
        self.PSE_RMS_fits = []
        param_i = 0
        for source in self.source_object_list:
            self.PSE_fits.append( source.RMS_fit_byStation(self.used_delays, 
                         current_guess[param_i:param_i+4]) )
            
            SSQE = source.SSqE_fit(self.used_delays, current_guess[param_i:param_i+4])
#            source.SSqE_fitNprint( new_station_delays,  current_guess[param_i:param_i+4] )
            self.PSE_RMS_fits.append( np.sqrt(SSQE/source.num_DOF()) )
            
            param_i += 4
            
            
            
        #### check which stations have fits
        self.stations_with_fits = [False]*( len(station_order)+1 )
            
        for stationfits in self.PSE_fits:
            for snum, (sname, fit) in enumerate(zip( chain(station_order, [referance_station]),  stationfits )):
                if fit is not None:
                    self.stations_with_fits[snum] = True
            
            
            
        #### save results
        self.solution = current_guess
        self.cost = current_fit
        self.RMS = total_RMS
        

        
        
        
        
        
    def employ_result(self, source_object_list):
        """set the result to the guess location of the sources, and return the station timing offsets"""
        
        param_i = 0
        for PSE in source_object_list:
            PSE.guess_XYZT[:] =  self.solution[param_i:param_i+4]
            param_i += 4
            
        return self.used_delays
    
    def print_locations(self, source_object_list):
        
        param_i = 0
        for source, RMSfit in zip(source_object_list, self.PSE_RMS_fits):
            print("source", source.ID)
            print("  RMS:", RMSfit)
            print("  loc:", self.solution[param_i:param_i+4])
            param_i += 4
    
    def print_station_fits(self, source_object_list):
        
        fit_table = PrettyTable()
        fit_table.field_names = ['id'] + station_order + [referance_station] + ['total']
        fit_table.float_format = '.2E'
        
        for source, RMSfit, stationfits in zip(source_object_list, self.PSE_RMS_fits, self.PSE_fits):
            new_row = ['']*len(fit_table.field_names)
            new_row[0] = source.ID
            new_row[-1] = RMSfit
            
            for i,stat_fit in enumerate(stationfits):
                if stat_fit is not None:
                    new_row[i+1] = stat_fit
                    
            fit_table.add_row( new_row )
            
        print( fit_table )
                    
    def print_delays(self, original_delays):
        for sname, delay, original in zip(station_order, self.used_delays, original_delays):
            print("'"+sname+"' : ",delay,', ## diff to guess:', delay-original)
    
    def get_stations_with_fits(self):
        return self.stations_with_fits


#### source object ####
## represents a potential source
## keeps track of a stations on the prefered station, and stations on other stations that could correlate and are considered correlated
## contains utilities for fitting, and for finding RMS in total and for each station
## also contains utilities for plotting and saving info

## need to handle inseartion of random error, and that choosen SSPE can change

class source_object():
## assume: guess_location , ant_locs, station_to_antenna_index_list, station_to_antenna_index_dict, referance_station, station_order,
#    sorted_antenna_names, station_locations,
    # are global

    def __init__(self, ID,  input_fname, location, stations_to_exclude, antennas_to_exclude ):
        self.ID = ID
        self.stations_to_exclude = stations_to_exclude
        self.antennas_to_exclude = antennas_to_exclude
        self.data_file = h5py.File(input_fname, "r")
        self.guess_XYZT = np.array( location )

                 
            
    def prep_for_fitting(self, polarization):
        self.polarization = polarization
        
        self.pulse_times = np.empty( len(sorted_antenna_names) )
        self.pulse_times[:] = np.nan
        self.waveforms = [None]*len(self.pulse_times)
        self.waveform_startTimes = [None]*len(self.pulse_times)
        
        #### first add times from referance_station
        for sname in chain(station_order, [referance_station]):
            if sname not in self.stations_to_exclude:
                self.add_known_station(sname)
                
                
        #### setup some temp storage for fitting
        self.tmp_LS2_data = np.empty( len(sorted_antenna_names) )
                
                
    def remove_station(self, sname):
            
        antenna_index_range = station_to_antenna_index_dict[sname]
        self.pulse_times[ antenna_index_range[0]:antenna_index_range[1] ] = np.nan
        
    def has_station(self, sname):
        antenna_index_range = station_to_antenna_index_dict[sname]
        return np.sum(np.isfinite(self.pulse_times[ antenna_index_range[0]:antenna_index_range[1] ])) > 0
         
    def add_known_station(self, sname):
        self.remove_station( sname )
        
        if sname in self.data_file:
            station_group= self.data_file[sname]
        else:
            return 0
        
        antenna_index_range = station_to_antenna_index_dict[sname]
        for ant_i in range(antenna_index_range[0], antenna_index_range[1]):
            ant_name = sorted_antenna_names[ant_i]
            
            if ant_name in station_group:
                ant_data = station_group[ant_name]
                
                start_time = ant_data.attrs['starting_index']*5.0E-9
                
                pt = ant_data.attrs['PolE_peakTime'] if self.polarization==0 else ant_data.attrs['PolO_peakTime']
                waveform = ant_data[1,:] if self.polarization==0 else ant_data[3,:]
                start_time += ant_data.attrs['PolE_timeOffset'] if self.polarization==0 else ant_data.attrs['PolO_timeOffset']
                amp = np.max(waveform)
                
                if not np.isfinite(pt):
                    pt = np.nan
                if amp<min_antenna_amplitude or (ant_name in self.antennas_to_exclude) or (ant_name in bad_antennas):
                    pt = np.nan
                        
                self.pulse_times[ ant_i ] = pt
                self.waveforms[ ant_i ] = waveform
                self.waveform_startTimes[ ant_i ] = start_time 
                
        return np.sum(np.isfinite( self.pulse_times[antenna_index_range[0]:antenna_index_range[1]] ) )        
            
        
    def try_location_LS(self, delays, XYZT_location, out):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        delta_X_sq = ant_locs[:,0]-X
        delta_Y_sq = ant_locs[:,1]-Y
        delta_Z_sq = ant_locs[:,2]-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
            
        out[:] = T - self.pulse_times
        
        ##now account for delays
        for index_range, delay in zip(station_to_antenna_index_list,  delays):
            first,last = index_range
            
            out[first:last] += delay ##note the wierd sign
                
                
        out *= v_air
        out *= out ##this is now delta_t^2 *C^2
        
        out -= delta_X_sq
        out -= delta_Y_sq
        out -= delta_Z_sq
    
    def try_location_JAC(self, delays, XYZT_location, out_loc, out_delays):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        out_loc[:,0] = X
        out_loc[:,0] -= ant_locs[:,0]
        out_loc[:,0] *= -2
        
        out_loc[:,1] = Y
        out_loc[:,1] -= ant_locs[:,1]
        out_loc[:,1] *= -2
        
        out_loc[:,2] = Z
        out_loc[:,2] -= ant_locs[:,2]
        out_loc[:,2] *= -2
        
        
        out_loc[:,3] = T - self.pulse_times
        out_loc[:,3] *= 2*v_air*v_air
        
        delay_i = 0
        for index_range, delay in zip(station_to_antenna_index_list,  delays):
            first,last = index_range
            
            out_loc[first:last,3] += delay*2*v_air*v_air
            out_delays[first:last,delay_i] = out_loc[first:last,3]
                
            delay_i += 1
            
    def try_location_LS2(self, delays, XYZT_location, out):
        X,Y,Z,T = XYZT_location
#        Z = np.abs(Z)
        
        self.tmp_LS2_data[:] = ant_locs[:,0]
        self.tmp_LS2_data[:] -= X
        self.tmp_LS2_data[:] *= self.tmp_LS2_data[:]
        out[:] = self.tmp_LS2_data
        
        self.tmp_LS2_data[:] = ant_locs[:,1]
        self.tmp_LS2_data[:] -= Y
        self.tmp_LS2_data[:] *= self.tmp_LS2_data[:]
        out[:] += self.tmp_LS2_data
        
        self.tmp_LS2_data[:] = ant_locs[:,2]
        self.tmp_LS2_data[:] -= Z
        self.tmp_LS2_data[:] *= self.tmp_LS2_data[:]
        out[:] += self.tmp_LS2_data
        
        np.sqrt( out, out=out )
        out *= inv_v_air
        
        out += T
        out -= self.pulse_times
        
        ##now account for delays
        for index_range, delay in zip(station_to_antenna_index_list,  delays):
            first,last = index_range
            
            out[first:last] += delay ##note the wierd sign
    
    def try_location_JAC2(self, delays, XYZT_location, out_loc, out_delays):
        X,Y,Z,T = XYZT_location
#        Z = np.abs(Z)
        
        out_loc[:,0] = X
        out_loc[:,0] -= ant_locs[:,0]
        
        out_loc[:,1] = Y
        out_loc[:,1] -= ant_locs[:,1]
        
        out_loc[:,2] = Z
        out_loc[:,2] -= ant_locs[:,2]
        
        
        out_delays[:,0] = out_loc[:,0] ## use as temporary storage
        out_delays[:,0] *= out_delays[:,0]
        out_loc[:,3] = out_delays[:,0] ## also use as temporary storage
         
        out_delays[:,0] = out_loc[:,1] ## use as temporary storage
        out_delays[:,0] *= out_delays[:,0]
        out_loc[:,3] += out_delays[:,0]
        
        out_delays[:,0] = out_loc[:,2] ## use as temporary storage
        out_delays[:,0] *= out_delays[:,0]
        out_loc[:,3] += out_delays[:,0]
        
        np.sqrt( out_loc[:,3], out = out_loc[:,3] )
         
        out_loc[:,0] /= out_loc[:,3]
        out_loc[:,1] /= out_loc[:,3]
        out_loc[:,2] /= out_loc[:,3]
        
        out_loc[:,0] *= inv_v_air
        out_loc[:,1] *= inv_v_air
        out_loc[:,2] *= inv_v_air
         
        out_loc[:,3] = 1
        
        delay_i = 0
        out_delays[:] = 0.0
        for index_range, delay in zip(station_to_antenna_index_list,  delays):
            first,last = index_range
            
            out_delays[first:last,delay_i] = 1
                
            delay_i += 1
            
    def num_DOF(self):
        return np.sum( np.isfinite(self.pulse_times) ) - 3 ## minus three or four?
            
    def estimate_T(self, delays, XYZT_location):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        delta_X_sq = ant_locs[:,0]-X
        delta_Y_sq = ant_locs[:,1]-Y
        delta_Z_sq = ant_locs[:,2]-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
            
        workspace = delta_X_sq+delta_Y_sq
        workspace += delta_Z_sq
        
        np.sqrt(workspace, out=workspace)
        
        workspace[:] -= self.pulse_times*v_air ## this is now source time
        
        ##now account for delays
        for index_range, delay in zip(station_to_antenna_index_list,  delays):
            first,last = index_range
            workspace[first:last] += delay*v_air ##note the wierd sign
                
                
        ave_error = np.nanmean( workspace )
        return -ave_error/v_air
            
    def SSqE_fit(self, delays, XYZT_location):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        delta_X_sq = ant_locs[:,0]-X
        delta_Y_sq = ant_locs[:,1]-Y
        delta_Z_sq = ant_locs[:,2]-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
        distance = delta_X_sq
        distance += delta_Y_sq
        distance += delta_Z_sq
        
        np.sqrt(distance, out=distance)
        distance *= 1.0/v_air
        
        distance += T
        distance -= self.pulse_times
        
        ##now account for delays
        for index_range, delay in zip(station_to_antenna_index_list,  delays):
            first,last = index_range
            if first is not None:
                distance[first:last] += delay ##note the wierd sign
                
        distance *= distance
        return np.nansum(distance)    
                
    
    def RMS_fit_byStation(self, delays, XYZT_location):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        delta_X_sq = ant_locs[:,0]-X
        delta_Y_sq = ant_locs[:,1]-Y
        delta_Z_sq = ant_locs[:,2]-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
        distance = delta_X_sq
        distance += delta_Y_sq
        distance += delta_Z_sq
        
        np.sqrt(distance, out=distance)
        distance *= 1.0/v_air
        
        distance += T
        distance -= self.pulse_times
        
        ##now account for delays
        for index_range, delay in zip(station_to_antenna_index_list,  delays):
            first,last = index_range
            if first is not None:
                distance[first:last] += delay ##note the wierd sign
                
        distance *= distance
        
        ret = []
        for index_range in station_to_antenna_index_list:
            first,last = index_range
            
            data = distance[first:last]
            nDOF = np.sum( np.isfinite(data) )
            if nDOF == 0:
                ret.append( None )
            else:
               ret.append( np.sqrt( np.nansum(data)/nDOF ) )
               
        ## need to do referance station
        first,last = station_to_antenna_index_dict[ referance_station ]
        data = distance[first:last]
        nDOF = np.sum( np.isfinite(data) )
        if nDOF == 0:
            ret.append( None )
        else:
           ret.append( np.sqrt( np.nansum(data)/nDOF ) )
        
        return ret
    
    def plot_waveforms(self, station_timing_offsets, fname=None):
        
        if fname is None:
            plotter = plt
        else:
            CL = code_logger(fname)
            CL.add_statement("import numpy as np")
            plotter = pyplot_emulator(CL)
        
        most_min_t = np.inf
        snames_not_plotted = []
        
        for sname, offset in zip(  chain(station_order,[referance_station]),  chain(station_timing_offsets,[0.0]) ):
            index_range = station_to_antenna_index_dict[sname]
            
            if sname in self.data_file:
                station_data = self.data_file[sname]
            else:
                continue
            
            
            min_T = np.inf
                
            max_T = -np.inf
            for ant_i in range(index_range[0], index_range[1]):
                ant_name = sorted_antenna_names[ant_i]
                if ant_name not in station_data:
                    continue
                
                ant_data = station_data[ant_name]
        
                PolE_peak_time = ant_data.attrs['PolE_peakTime'] - offset
                PolO_peak_time = ant_data.attrs['PolO_peakTime'] - offset

                PolE_hilbert = ant_data[1,:]
                PolO_hilbert = ant_data[3,:]
            
                PolE_trace = ant_data[0,:]
                PolO_trace = ant_data[2,:]

                PolE_T_array = (np.arange(len(PolE_hilbert)) + ant_data.attrs['starting_index'] )*5.0E-9  + ant_data.attrs['PolE_timeOffset']
                PolO_T_array = (np.arange(len(PolO_hilbert)) + ant_data.attrs['starting_index'] )*5.0E-9  + ant_data.attrs['PolO_timeOffset']
                
                PolE_T_array -= offset
                PolO_T_array -= offset
        
                PolE_amp = np.max(PolE_hilbert)
                PolO_amp = np.max(PolO_hilbert)
                amp = max(PolE_amp, PolO_amp)
                PolE_hilbert = PolE_hilbert/(amp*3.0)
                PolO_hilbert = PolO_hilbert/(amp*3.0)
                PolE_trace   = PolE_trace/(amp*3.0)
                PolO_trace   = PolO_trace/(amp*3.0)
        
        
                if PolE_amp < min_antenna_amplitude:
                    PolE_peak_time = np.inf
                if PolO_amp < min_antenna_amplitude:
                    PolO_peak_time = np.inf
        
                plotter.plot( PolE_T_array, ant_i+PolE_hilbert, 'g' )
                plotter.plot( PolE_T_array, ant_i+PolE_trace, 'g' )
                plotter.plot( [PolE_peak_time, PolE_peak_time], [ant_i, ant_i+2.0/3.0], 'g')
                
                plotter.plot( PolO_T_array, ant_i+PolO_hilbert, 'm' )
                plotter.plot( PolO_T_array, ant_i+PolO_trace, 'm' )
                plotter.plot( [PolO_peak_time, PolO_peak_time], [ant_i, ant_i+2.0/3.0], 'm')
                
                plotter.annotate( ant_name, xy=[PolO_T_array[-1], ant_i], size=7)
                
                max_T = max(max_T, PolE_T_array[-1], PolO_T_array[-1])
                min_T = min(min_T, PolE_T_array[0], PolO_T_array[0])
                most_min_t = min(most_min_t, min_T)
                
            if min_T<np.inf:
                plotter.annotate( sname, xy=[min_T, np.average(index_range)], size=15)
            else:
                snames_not_plotted.append( sname )
                
        for sname in snames_not_plotted:
            index_range = station_to_antenna_index_dict[sname]
            plotter.annotate( sname, xy=[most_min_t, np.average(index_range)], size=15)
                
        plotter.show()
        
        if fname is not None:
            CL.save()
    
    def plot_selected_waveforms(self, station_timing_offsets, fname=None):
        
        if fname is None:
            plotter = plt
        else:
            CL = code_logger(fname)
            CL.add_statement("import numpy as np")
            plotter = pyplot_emulator(CL)
        
        most_min_t = np.inf
        snames_not_plotted = []
        
        for sname, offset in zip(  chain(station_order,[referance_station]),  chain(station_timing_offsets,[0.0]) ):
            index_range = station_to_antenna_index_dict[sname]
            
            min_T = np.inf
            max_T = -np.inf
            for ant_i in range(index_range[0], index_range[1]):
                ant_name = sorted_antenna_names[ant_i]
                
                pulse_time = self.pulse_times[ ant_i ]
                waveform = self.waveforms[ ant_i ]
                startTime = self.waveform_startTimes[ ant_i ]
                
                if not np.isfinite( pulse_time ):
                    continue
                
                T_array = np.arange(len(waveform))*5.0E-9  +  (startTime - offset)
                
                
                amp = np.max(waveform)
                waveform = waveform/(amp*3.0)
        
                plotter.plot( T_array, ant_i+waveform, 'g' )
                plotter.plot( [pulse_time-offset, pulse_time-offset], [ant_i, ant_i+2.0/3.0], 'm')
                
                plotter.annotate( ant_name, xy=[T_array[-1], ant_i], size=7)
                
                max_T = max(max_T, T_array[-1])
                min_T = min(min_T, T_array[0])
                most_min_t = min(most_min_t, min_T)
                
            if min_T<np.inf:
                plotter.annotate( sname, xy=[min_T, np.average(index_range)], size=15)
            else:
                snames_not_plotted.append( sname )
                
        for sname in snames_not_plotted:
            index_range = station_to_antenna_index_dict[sname]
            plotter.annotate( sname, xy=[most_min_t, np.average(index_range)], size=15)
                
        plotter.show()
        
        if fname is not None:
            CL.save()
            
class Part1_input_manager:
    def __init__(self, input_files):
        self.max_num_input_files = 10
        if len(input_files) > self.max_num_input_files:
            print("TOO MANY INPUT FOLDERS!!!")
            quit()
        
        self.input_files = input_files
        
        self.input_data = []
        for folder_i, folder in enumerate(input_files):
            input_folder = processed_data_dir + "/" + folder +'/'
            
            file_list = [(int(f.split('_')[1][:-3])*self.max_num_input_files+folder_i ,input_folder+f) for f in listdir(input_folder) if f.endswith('.h5')] ## get all file names, and get the 'ID' for the file name
            file_list.sort( key=lambda x: x[0] ) ## sort according to ID
            self.input_data.append( file_list )
        
    def known_source(self, ID):
        
        file_i = int(ID/self.max_num_input_files)
        folder_i = ID - file_i*self.max_num_input_files
        file_list = self.input_data[ folder_i ]
        
        return [info for info in file_list if info[0]==ID][0]
        

np.set_printoptions(precision=10, threshold=np.inf)
#if __name__ == "__main__": 
#    
#    #### TODO: make code that looks for pulses on all antennas, ala analyze amplitudes
#    ## probably need a seperate code that just fits location of one source, so can be tuned by hand, then fed into main code
#    ## prehaps just add one station at a time? (one closest to known stations) seems to be general stratigey
#    
#    timeID = "D20170929T202255.000Z"
#    output_folder = "autoCorrelator3_fitter_reTST"
#    
#    part1_inputs = ["autoCorrelator3_part1C"]#, "autoCorrelator3_fromIPSE"]
#    
#    
#    #### fitter settings ####
#    max_itters_per_loop = 2000
#    itters_till_convergence = 100
#    max_jitter_width = 100000E-9
#    min_jitter_width = 1E-9
#    cooldown = 10.0 ## 10.0
#    strong_cooldown = 100.0
#    
##    ## quality control
#    min_antenna_amplitude = 10
##    max_station_RMS = 5.0E-9
##    min_stations = 4
##    min_antenna_per_station = 4
#    
#    #### initial guesses ####
#    referance_station = "CS002"
#    
#    guess_timings = { ##OLD STANDARD
#'CS003' :  1.40494600712e-06 , ## diff to guess: -7.66779057095e-10
#'CS004' :  4.31090399655e-07 , ## diff to guess: 6.99747558283e-10
#'CS005' :  -2.19120784288e-07 , ## diff to guess: 9.7074831305e-10
#'CS006' :  4.33554497556e-07 , ## diff to guess: -8.43001702525e-11
#'CS007' :  3.99902678123e-07 , ## diff to guess: -5.9939666121e-10
#'CS011' :  -5.8560385891e-07 , ## diff to guess: 2.89824346402e-10
#'CS013' :  -1.81325669666e-06 , ## diff to guess: -1.34754955008e-09
#'CS017' :  -8.43878144671e-06 , ## diff to guess: -1.37869682555e-09
#'CS021' :  9.24163752585e-07 , ## diff to guess: -1.48380056922e-09
#'CS030' :  -2.74037597057e-06 , ## diff to guess: -2.78214833694e-09
#'CS032' :  -1.57466749415e-06 , ## diff to guess: 2.83849237954e-09
#'CS101' :  -8.16817259789e-06 , ## diff to guess: -4.65243457996e-09
#'CS103' :  -2.8518179599e-05 , ## diff to guess: -3.69610176348e-09
#'RS208' :  6.94747112635e-06 , ## diff to guess: 5.55896927948e-08
#'CS301' :  -7.18344260299e-07 , ## diff to guess: 5.53169999063e-09
#'CS302' :  -5.35619408893e-06 , ## diff to guess: 9.98450123074e-09
#'RS306' :  7.02305935406e-06 , ## diff to guess: 4.40228038518e-08
#'RS307' :  6.92860064373e-06 , ## diff to guess: 8.07099571938e-08
#'RS310' :  7.02115632779e-06 , ## diff to guess: 1.53739165001e-07
#'CS401' :  -9.52017247995e-07 , ## diff to guess: 1.9274824037e-09
#'RS406' :  7.02462236919e-06 , ## diff to guess: 5.07479726915e-05
#'RS409' :  7.03940588106e-06 , ## diff to guess: 3.39804652173e-08
#'CS501' :  -9.6076184928e-06 , ## diff to guess: -5.34064484748e-09
#'RS503' :  6.94774095106e-06 , ## diff to guess: -1.75724349742e-08
#'RS508' :  7.06896423643e-06 , ## diff to guess: -4.27040985914e-08
#'RS509' :  7.11591455679e-06 , ## diff to guess: -4.30170606727e-08
#        }
#   
#        
#    if referance_station in guess_timings:
#        del guess_timings[referance_station]
#    
#    
#    #### these are sources whose location have been fitted, and stations are associated
#    known_sources = [0,10,30,50,80]
#    
#    ### locations of fitted sources
#    known_source_locations = {
#0 :[ -15693.542761 , 10614.7690102 , 4833.01819705 , 1.22363744148 ],
#10 :[ -14746.6987709 , 10164.4935227 , 5153.23554352 , 1.22778566789 ],
#20 :[ -15733.992892 , 10817.6206629 , 4888.45795366 , 1.22938141588 ], ## OLAF SAYS NOT SUITABLE
#30 :[ -14837.260161 , 10240.4210371 , 5200.02019411 , 1.2307873997 ],
#40 :[ -15045.1734315 , 10401.3870684 , 5085.78988291 , 1.23751673723 ],
#50 :[ -15728.0633707 , 10828.3878524 , 4930.20022273 , 1.24428590365 ],
#60 :[ -15076.0691784 , 10428.6773094 , 5120.20382588 , 1.24440896162 ],
#70 :[ -18648.664056 , 9482.97793092 , 3094.45610521 , 1.27982455954 ],
#80 :[ -15176.5061353 , 8481.5636753 , 4742.15114273 , 1.3357053429 ],
#90 :[ -15176.4922093 , 8166.78593544 , 4297.68674851 , 1.36449659607 ],
#    }
#    
#    ### polarization of fitted sources
#    known_polarizations = {
#0 : 1 ,
#10 : 1 ,
#20 : 1 ,
#30 : 1 ,
#40 : 1 ,
#50 : 1 ,
#60 : 1 ,
#70 : 1 ,
#80 : 1 ,
#90 : 0 ,
#    }
#    
#    
#    ## these are stations to exclude
#    stations_to_exclude = {
#            0:['CS021', 'RS406'],
#            10:['RS208'],
#            20:['CS003', 'RS306', 'RS406', 'RS409'],
#            30:['CS002', 'RS306', 'RS307', 'RS310', 'RS406', 'RS409', 'RS508', 'RS509'],
#            40:['RS406'],
#            50:['CS017', "RS406"],
#            60:[],
#            70:['CS007', 'CS013', 'CS401', 'RS306', 'RS310', 'RS406', 'RS409', 'RS503', 'RS508', 'RS509'],
#            80:['RS307', 'RS310', 'RS503', 'RS508', 'RS509'],
#            90:['CS002', 'RS208', 'RS306', 'RS307', 'RS406']
#            }
#    
#    antennas_to_exclude = {
#            0:['146007060'],
#            10:[], 
#            20:[],
#            30:[],
#            40:['161010080', '183002016'],
#            50:[],
#            60:['128011094', '128011092', '128005044', '128003028', '128010080', '128000000',
#               '150005044', '150009076'],
#            70:['003004032', '003010080', '004008064', '011011092', '011009078', '017006050',
#               '021008064', '021001014', '030006048', '030001014', '032006048', '032002016', 
#               '032005046', '101002016', '141006048', '142000000', '142008064', '412007060',
#               '142007062', '181005044'],
#            80:[],
#            90:['032006048'],
#            }
#    
#    bad_antennas = [
#            ##CS002
#            '003008064', ##CS003
#            ##CS004
#            ##CS005
#            ##CS006
#            ##CS007
#            ##CS011
#            ##CS013
#            ##CS017
#            '021011092', ##CS021
#            '030003028', ##CS030
#            '032010080',##CS032
#            ##CS101
#            ##CS103
#            ##CS301
#            ##CS302
#            ##CS401
#            '181006048', ##CS501
#            ##RS208
#            ##RS306
#            ##RS307
#            ##RS310
#            ##RS406
#            '169005046', '169009076',##RS409
#            ##RS503
#            '188011094',##RS508
#            ##RS509
#                    ]


if __name__ == "__main__": 
    
    #### TODO: make code that looks for pulses on all antennas, ala analyze amplitudes
    ## probably need a seperate code that just fits location of one source, so can be tuned by hand, then fed into main code
    ## prehaps just add one station at a time? (one closest to known stations) seems to be general stratigey
    
    timeID = "D20170929T202255.000Z"
    output_folder = "autoCorrelator3_fitter_reTST"
    
    part1_inputs = ["autoCorrelator3_fromLOCA"]
    
    
    #### fitter settings ####
#    max_itters_per_loop = 2000
#    itters_till_convergence = 100
#    max_jitter_width = 100000E-9
#    min_jitter_width = 1E-9
#    cooldown = 10.0 ## 10.0
#    strong_cooldown = 100.0
    
    max_itters_per_loop = 5000
    itters_till_convergence = 1000
    max_jitter_width = 100000E-9
    min_jitter_width = 1E-13
    cooldown = 10.0 ## 10.0
    strong_cooldown = 10.0
    
#    ## quality control
    min_antenna_amplitude = 10
#    max_station_RMS = 5.0E-9
#    min_stations = 4
#    min_antenna_per_station = 4
    
    #### initial guesses ####
    referance_station = "CS002"
    
    guess_timings = { ##OLD STANDARD
'CS003' :  1.40494600712e-06 , ## diff to guess: -7.66779057095e-10
'CS004' :  4.31090399655e-07 , ## diff to guess: 6.99747558283e-10
'CS005' :  -2.19120784288e-07 , ## diff to guess: 9.7074831305e-10
'CS006' :  4.33554497556e-07 , ## diff to guess: -8.43001702525e-11
'CS007' :  3.99902678123e-07 , ## diff to guess: -5.9939666121e-10
'CS011' :  -5.8560385891e-07 , ## diff to guess: 2.89824346402e-10
'CS013' :  -1.81325669666e-06 , ## diff to guess: -1.34754955008e-09
'CS017' :  -8.43878144671e-06 , ## diff to guess: -1.37869682555e-09
'CS021' :  9.24163752585e-07 , ## diff to guess: -1.48380056922e-09
'CS030' :  -2.74037597057e-06 , ## diff to guess: -2.78214833694e-09
'CS032' :  -1.57466749415e-06 , ## diff to guess: 2.83849237954e-09
'CS101' :  -8.16817259789e-06 , ## diff to guess: -4.65243457996e-09
'CS103' :  -2.8518179599e-05 , ## diff to guess: -3.69610176348e-09
'RS208' :  6.94747112635e-06 , ## diff to guess: 5.55896927948e-08
'CS301' :  -7.18344260299e-07 , ## diff to guess: 5.53169999063e-09
'CS302' :  -5.35619408893e-06 , ## diff to guess: 9.98450123074e-09
'RS306' :  7.02305935406e-06 , ## diff to guess: 4.40228038518e-08
'RS307' :  6.92860064373e-06 , ## diff to guess: 8.07099571938e-08
'RS310' :  7.02115632779e-06 , ## diff to guess: 1.53739165001e-07
'CS401' :  -9.52017247995e-07 , ## diff to guess: 1.9274824037e-09
'RS406' :  7.02462236919e-06 , ## diff to guess: 5.07479726915e-05
'RS409' :  7.03940588106e-06 , ## diff to guess: 3.39804652173e-08
'CS501' :  -9.6076184928e-06 , ## diff to guess: -5.34064484748e-09
'RS503' :  6.94774095106e-06 , ## diff to guess: -1.75724349742e-08
'RS508' :  7.06896423643e-06 , ## diff to guess: -4.27040985914e-08
'RS509' :  7.11591455679e-06 , ## diff to guess: -4.30170606727e-08
        }
    
#    guess_timings = { #stat delays 2
#'CS002':  0.0,
#'CS003':  1.40436380151e-06 ,
#'CS004':  4.31343360778e-07 ,
#'CS005':  -2.18883924536e-07 ,
#'CS006':  4.33532992523e-07 ,
#'CS007':  3.99644095007e-07 ,
#'CS011':  -5.85451477265e-07 ,
#'CS013':  -1.81434735154e-06 ,
#'CS017':  -8.4398374875e-06 ,
#'CS021':  9.23663075135e-07 ,
#'CS030':  -2.74255354078e-06,
#'CS032':  -1.57305580305e-06,
#'CS101':  -8.17154277682e-06,
#'CS103':  -2.85194082718e-05,
#'RS208':  6.97951240511e-06 ,
#'CS301':  -7.15482701536e-07 ,
#'CS302':  -5.35024064624e-06 ,
#'RS306':  7.04283154727e-06,
#'RS307':  6.96315727897e-06 ,
#'RS310':  7.04140267551e-06,
#'CS401':  -9.5064990747e-07 ,
#'RS406':  6.96866309712e-06,
#'RS409':  7.02251772331e-06,    
#'CS501':  -9.61256584076e-06, 
#'RS503':  6.93934919654e-06 ,
#'RS508':  6.98208245779e-06 ,
#'RS509':  7.01900854365e-06
#        }
    
#    guess_timings = { ##incl. 70
#    'CS003' :  1.40406621911e-06 , ## diff to guess: -2.97582399388e-10
#'CS004' :  4.31385690892e-07 , ## diff to guess: 4.23301135518e-11
#'CS005' :  -2.1859126247e-07 , ## diff to guess: 2.92662065886e-10
#'CS006' :  4.33737447458e-07 , ## diff to guess: 2.04454935474e-10
#'CS007' :  3.99653069882e-07 , ## diff to guess: 8.97487473944e-12
#'CS011' :  -5.8495505038e-07 , ## diff to guess: 4.96426885262e-10
#'CS013' :  -1.81520175476e-06 , ## diff to guess: -8.54403221457e-10
#'CS017' :  -8.43959639123e-06 , ## diff to guess: 2.41096273183e-10
#'CS021' :  9.22589892036e-07 , ## diff to guess: -1.07318309883e-09
#'CS030' :  -2.74480264644e-06 , ## diff to guess: -2.24910565969e-09
#'CS032' :  -1.57307500032e-06 , ## diff to guess: -1.91972721945e-11
#'CS101' :  -8.17262937308e-06 , ## diff to guess: -1.08659626483e-09
#'CS103' :  -2.85184469337e-05 , ## diff to guess: 9.61338126864e-10
#'RS208' :  7.00240434182e-06 , ## diff to guess: 2.28919367117e-08
#'CS301' :  -7.13604373643e-07 , ## diff to guess: 1.87832789252e-09
#'CS302' :  -5.34835991794e-06 , ## diff to guess: 1.88072830146e-09
#'RS306' :  7.03842827887e-06 , ## diff to guess: -4.40326840323e-09
#'RS307' :  6.96945123387e-06 , ## diff to guess: 6.29395489957e-09
#'RS310' :  7.01466477193e-06 , ## diff to guess: -2.67379035788e-08
#'CS401' :  -9.51330977655e-07 , ## diff to guess: -6.81070184752e-10
#'RS406' :  6.90112556039e-06 , ## diff to guess: -6.75375367306e-08
#'RS409' :  6.95010394068e-06 , ## diff to guess: -7.24137826317e-08
#'CS501' :  -9.61540017339e-06 , ## diff to guess: -2.83433262773e-09
#'RS503' :  6.93000222477e-06 , ## diff to guess: -9.34697176934e-09
#'RS508' :  6.93106911104e-06 , ## diff to guess: -5.10133467456e-08
#'RS509' :  6.951187713e-06 , ## diff to guess: -6.78208306452e-08
#}
    
#    guess_timings = { ##incl. 80
#    'CS003' :  1.40421212443e-06 , ## diff to guess: 1.45905323785e-10
#'CS004' :  4.31476626482e-07 , ## diff to guess: 9.09355901378e-11
#'CS005' :  -2.18644873592e-07 , ## diff to guess: -5.36111219721e-11
#'CS006' :  4.33642073207e-07 , ## diff to guess: -9.53742505377e-11
#'CS007' :  3.99567203241e-07 , ## diff to guess: -8.58666412326e-11
#'CS011' :  -5.85176003945e-07 , ## diff to guess: -2.20953564802e-10
#'CS013' :  -1.81476263252e-06 , ## diff to guess: 4.39122237939e-10
#'CS017' :  -8.43968557631e-06 , ## diff to guess: -8.91850772514e-11
#'CS021' :  9.23141209733e-07 , ## diff to guess: 5.51317696771e-10
#'CS030' :  -2.74368600917e-06 , ## diff to guess: 1.11663726943e-09
#'CS032' :  -1.57282881181e-06 , ## diff to guess: 2.46188505452e-10
#'CS101' :  -8.17238302702e-06 , ## diff to guess: 2.4634606036e-10
#'CS103' :  -2.85191140031e-05 , ## diff to guess: -6.67069383516e-10
#'RS208' :  6.99245574092e-06 , ## diff to guess: -9.94860090402e-09
#'CS301' :  -7.14120995008e-07 , ## diff to guess: -5.16621364673e-10
#'CS302' :  -5.34875595775e-06 , ## diff to guess: -3.96039814521e-10
#'RS306' :  7.04094171439e-06 , ## diff to guess: 2.51343552499e-09
#'RS307' :  6.96606917708e-06 , ## diff to guess: -3.38205679167e-09
#'RS310' :  7.01572098334e-06 , ## diff to guess: 1.05621141033e-09
#'CS401' :  -9.50847114833e-07 , ## diff to guess: 4.8386282209e-10
#'RS406' :  6.92543926746e-06 , ## diff to guess: 2.43137070715e-08
#'RS409' :  6.96613970787e-06 , ## diff to guess: 1.60357671861e-08
#'CS501' :  -9.61421061808e-06 , ## diff to guess: 1.18955531296e-09
#'RS503' :  6.93372265995e-06 , ## diff to guess: 3.72043518085e-09
#'RS508' :  6.94119719961e-06 , ## diff to guess: 1.01280885735e-08
#'RS509' :  6.96395578682e-06 , ## diff to guess: 1.27680738193e-08
#}
    
#    guess_timings = { ##incl. 90
#    'CS003' :  1.40433044347e-06 , ## diff to guess: 1.18319036699e-10
#'CS004' :  4.31943207728e-07 , ## diff to guess: 4.66581246497e-10
#'CS005' :  -2.18443505428e-07 , ## diff to guess: 2.01368163929e-10
#'CS006' :  4.33757338521e-07 , ## diff to guess: 1.15265313632e-10
#'CS007' :  3.99692113558e-07 , ## diff to guess: 1.24910316831e-10
#'CS011' :  -5.84956359991e-07 , ## diff to guess: 2.1964395439e-10
#'CS013' :  -1.81429034028e-06 , ## diff to guess: 4.72292240749e-10
#'CS017' :  -8.43955579972e-06 , ## diff to guess: 1.29776591566e-10
#'CS021' :  9.23414452423e-07 , ## diff to guess: 2.73242690094e-10
#'CS030' :  -2.74333318147e-06 , ## diff to guess: 3.52827699726e-10
#'CS032' :  -1.57273568113e-06 , ## diff to guess: 9.31306827023e-11
#'CS101' :  -8.17190662339e-06 , ## diff to guess: 4.76403631906e-10
#'CS103' :  -2.85188458982e-05 , ## diff to guess: 2.68104873575e-10
#'RS208' :  6.98776026878e-06 , ## diff to guess: -4.69547213655e-09
#'CS301' :  -7.14432948322e-07 , ## diff to guess: -3.1195331374e-10
#'CS302' :  -5.34887182438e-06 , ## diff to guess: -1.15866626379e-10
#'RS306' :  7.03856487269e-06 , ## diff to guess: -2.37684169666e-09
#'RS307' :  6.96056645903e-06 , ## diff to guess: -5.50271804922e-09
#'RS310' :  7.00673018169e-06 , ## diff to guess: -8.99080165234e-09
#'CS401' :  -9.5077618853e-07 , ## diff to guess: 7.09263026434e-11
#'RS406' :  6.93192024438e-06 , ## diff to guess: 6.48097691801e-09
#'RS409' :  6.96030515155e-06 , ## diff to guess: -5.83455632346e-09
#'CS501' :  -9.61352286266e-06 , ## diff to guess: 6.87755416147e-10
#'RS503' :  6.93514290261e-06 , ## diff to guess: 1.42024266275e-09
#'RS508' :  6.94635371243e-06 , ## diff to guess: 5.15651281721e-09
#'RS509' :  6.96778730878e-06 , ## diff to guess: 3.8315219551e-09
#}
    
#    guess_timings = { ## Olaf's farorites
#    'CS003' :  1.4042163013e-06 , ## diff to guess: -1.14142167168e-10
#'CS004' :  4.31491732703e-07 , ## diff to guess: -4.51475025084e-10
#'CS005' :  -2.18688092672e-07 , ## diff to guess: -2.44587243874e-10
#'CS006' :  4.33633641144e-07 , ## diff to guess: -1.23697376933e-10
#'CS007' :  3.99583732343e-07 , ## diff to guess: -1.08381214875e-10
#'CS011' :  -5.85199559216e-07 , ## diff to guess: -2.43199224813e-10
#'CS013' :  -1.8147630918e-06 , ## diff to guess: -4.72751522454e-10
#'CS017' :  -8.43989595123e-06 , ## diff to guess: -3.40151513264e-10
#'CS021' :  9.23188743313e-07 , ## diff to guess: -2.25709109864e-10
#'CS030' :  -2.74366262143e-06 , ## diff to guess: -3.29439957314e-10
#'CS032' :  -1.57259636164e-06 , ## diff to guess: 1.39319494223e-10
#'CS101' :  -8.17257920552e-06 , ## diff to guess: -6.7258213271e-10
#'CS103' :  -2.85190817028e-05 , ## diff to guess: -2.35804637477e-10
#'RS208' :  6.99818676139e-06 , ## diff to guess: 1.04264926079e-08
#'CS301' :  -7.14031248782e-07 , ## diff to guess: 4.01699540062e-10
#'CS302' :  -5.34786808017e-06 , ## diff to guess: 1.00374420585e-09
#'RS306' :  7.04712650703e-06 , ## diff to guess: 8.56163434299e-09
#'RS307' :  6.9780725263e-06 , ## diff to guess: 1.75060672712e-08
#'RS310' :  7.05219954302e-06 , ## diff to guess: 4.54693613315e-08
#'CS401' :  -9.50484502263e-07 , ## diff to guess: 2.91686267432e-10
#'RS406' :  6.93089620534e-06 , ## diff to guess: -1.0240390396e-09
#'RS409' :  7.01056175752e-06 , ## diff to guess: 5.02566059676e-08
#'CS501' :  -9.6145851877e-06 , ## diff to guess: -1.06232503527e-09
#'RS503' :  6.93299835701e-06 , ## diff to guess: -2.14454560191e-09
#'RS508' :  6.95755989836e-06 , ## diff to guess: 1.12061859289e-08
#'RS509' :  6.98948454237e-06 , ## diff to guess: 2.16972335925e-08
#}


        
    if referance_station in guess_timings:
        del guess_timings[referance_station]
    
    
#    #### these are sources whose location have been fitted, and stations are associated
#    known_sources = [0,30,40,50,60,70]#,80,90, 100]
##    known_sources = [0,30]
##    known_sources = [40,50,60]
##    known_sources = [70]
#    
#    ### locations of fitted sources
#    known_source_locations = {
#0 :[ -15693.542761 , 10614.7690102 , 4833.01819705 , 1.22363744148 ],
#10 :[ -14746.6987709 , 10164.4935227 , 5153.23554352 , 1.22778566789 ],
#20 :[ -14837.260161 , 10240.4210371 , 5200.02019411 , 1.2307873997 ],
#30 :[ -15728.0633707 , 10828.3878524 , 4930.20022273 , 1.24428590365 ],
#40 :[ -15176.5061353 , 8481.5636753 , 4742.15114273 , 1.3357053429 ],
#50 :[ -1.60622566e+04 ,  1.00105856e+04 ,  3.82465424e+03 ,  1.17760563e+00],
#60 :[ -1.58093802e+04,   8.19398319e+03,   1.18307602e+03 ,  1.21527513e+00],
#70 :[ -1.57274095e+04,   1.07813916e+04 ,  4.84121261e+03 ,  1.23034882e+00],
#80 :[ -17713.36726129,   9953.00658086,   3476.37485781, 1.2690714758],
#90 :[-15684.61938993 ,  8999.12262615,   2934.83549653, 1.15655002631],
#100 :[-15254.54018532 ,  8836.1191801,    3126.99536402, 1.17993760262],
#    }
#    
#    ### polarization of fitted sources
#    known_polarizations = {
#0 : 1 ,
#10 : 1 ,
#20 : 1 ,
#30 : 1 ,
#40 : 1 ,
#50 : 1 ,
#60:  1 ,
#70:  1 ,
#80:  1,
#90:  1,
#100: 1,
#    }
#    
#    
#    ## these are stations to exclude
#    stations_to_exclude = {
#            0:['RS406'], ## why is RS306 such a bad fit? RS406 is saturated
#            10:["RS406"], ##RS 406 is TERRIBLE fit, probably wrong pulse. Maybe there was saturation
#            20:['RS409', 'RS509'], ## RS409 seems to be wrong pulse. Don't understand why RS509 is such a bad fit
#            30:[],
#            40:['RS310', 'RS509'], ## RS310 has pulses moving through it. RS509 does not have pulse
#            50:['CS002', 'CS003', 'CS004', 'CS005', 'CS006', 'CS007', 'CS011'], ##event is weak on the core
#            60:['RS406'], ##probably got wrong pulse. May saturation issues?
#            70:['RS406', "RS508"],##RS406: wrong pulse, RS508 has complex structure. RERUN FIND PULSES
#            80:['RS406'], ## RS406 has wrong pulse.RERUN FIND PULSES
#            90:['RS406', 'RS306', 'RS409'],#wrong pulse.RERUN FIND PULSES RS409 has very complicated pulse
#            100:['RS307', 'RS406'],
#            }
#    
#    antennas_to_exclude = {
#            0:['146007060', '146007062'],
#            10:['188008064', '188005044'],
#            20:['150007062', '150007060', '150005044'],
#            30:[],
#            40:['188007062', '188005044'],
#            50:['017007060', '101004032', '141006048', 
#                '166002016', '166001012', '166006050', '166003028', '166003030', '166005044', 
#                '166005046', '166007060', '166007062', '166009076', '166011094',
#                '188006050'],
#            60:['021009078'],
#            70:['188008064','188001014','188011092'],
#            80:['146000000', '146010080', '146007060', '146009078'],
#            90:[],
#            100:['146010080', '146005046', '146007062', '146009076', '146009078', '146011094', '183002016',
#                 '183004032', '183006048', '183008064', '183001012', '183001014', '183003028', '183009076',
#                 '183011094', '002001014', '003008064', '003010080', '007008064', '007007062', '013008064',
#                 '013001014', '013003028', '013007062', '013007060', '021008064', '021001014', '021007062',
#                 '021009076', '021009078', '030006048', '030008064', '030007062', '030001012', '032002016', '032004032',
#                 '141006048', '161010080', '161009076', '161011092', '161011094', '181000000', '181003028'], ## a lot of weak pulses....
#            }
#    
#    bad_antennas = [
#            ##CS002
#            '003008064', ##CS003
#            ##CS004
#            ##CS005
#            ##CS006
#            ##CS007
#            ##CS011
#            ##CS013
#            ##CS017
#            '021009078', '021011092', ##CS021
#            '030003028', ##CS030
#            '032010080',##CS032
#            ##CS101
#            ##CS103
#            ##CS301
#            ##CS302
#            ##CS401
#            '181006048', ##CS501
#            ##RS208
#            ##RS306
#            ##RS307
#            ##RS310
#            ##RS406
#            '169005046', '169009076',##RS409
#            ##RS503  check 183006048, 183003030
#            '188011094',##RS508
#            ##RS509
#                    ]
        
        
        
            #### these are sources whose location have been fitted, and stations are associated
#    known_sources = [0,30,40,50,60,70,80,90, 100]
    known_sources = [100]
    
#    known_sources = [0,30]
#    known_sources = [40,50,60]
#    known_sources = [70]
    
    ### locations of fitted sources
    known_source_locations = {
0 :[ -15693.542761 , 10614.7690102 , 4833.01819705 , 1.22363744148 ],
10 :[ -14746.6987709 , 10164.4935227 , 5153.23554352 , 1.22778566789 ],
20 :[ -14837.260161 , 10240.4210371 , 5200.02019411 , 1.2307873997 ],
30 :[ -15728.0633707 , 10828.3878524 , 4930.20022273 , 1.24428590365 ],
40 :[ -15176.5061353 , 8481.5636753 , 4742.15114273 , 1.3357053429 ],
50 :[ -1.60622566e+04 ,  1.00105856e+04 ,  3.82465424e+03 ,  1.17760563e+00],
60 :[ -1.58093802e+04,   8.19398319e+03,   1.18307602e+03 ,  1.21527513e+00],
70 :[ -1.57274095e+04,   1.07813916e+04 ,  4.84121261e+03 ,  1.23034882e+00],
80 :[ -17713.36726129,   9953.00658086,   3476.37485781, 1.2690714758],
90 :[-15684.61938993 ,  8999.12262615,   2934.83549653, 1.15655002631],
100 :[-15254.54018532 ,  8836.1191801,    3126.99536402, 1.17993760262],
    }
    
    ### polarization of fitted sources
    known_polarizations = {
0 : 1 ,
10 : 1 ,
20 : 1 ,
30 : 1 ,
40 : 1 ,
50 : 1 ,
60:  1 ,
70:  1 ,
80:  1,
90:  1,
100: 1,
    }
    
    
    ## these are stations to exclude
    stations_to_exclude = {
            0:['RS406'], ## why is RS306 such a bad fit? RS406 is saturated
            10:["RS406"], ##RS 406 is TERRIBLE fit, probably wrong pulse. Maybe there was saturation
            20:['RS409', 'RS509'], ## RS409 seems to be wrong pulse. Don't understand why RS509 is such a bad fit
            30:[],
            40:['RS310', 'RS509','RS508'], ## RS310 has pulses moving through it. RS509 does not have pulse
            50:['CS002', 'CS003', 'CS004', 'CS005', 'CS006', 'CS007', 'CS011','RS503'], ##event is weak on the core
            60:['RS406'], ##probably got wrong pulse. May saturation issues?
            70:['RS406', "RS508"],##RS406: wrong pulse, RS508 has complex structure. RERUN FIND PULSES
            80:['RS406'], ## RS406 has wrong pulse.RERUN FIND PULSES
            90:['RS406', 'RS306', 'RS409'],#wrong pulse.RERUN FIND PULSES RS409 has very complicated pulse
            100:['RS307', 'RS406'],
            }
    
    antennas_to_exclude = {
            0:['146007060', '146007062'],
            10:['188008064', '188005044'],
            20:['150007062', '150007060', '150005044'],
            30:[],
            40:['188007062', '188005044'],
            50:['017007060', '101004032', '141006048', 
                '166002016', '166001012', '166006050', '166003028', '166003030', '166005044', 
                '166005046', '166007060', '166007062', '166009076', '166011094',
                '188006050'],
            60:['021009078'],
            70:['188008064','188001014','188011092'],
            80:['146000000', '146010080', '146007060', '146009078'],
            90:[],
            100:['146010080', '146005046', '146007062', '146009076', '146009078', '146011094', '183002016',
                 '183004032', '183006048', '183008064', '183001012', '183001014', '183003028', '183009076',
                 '183011094', '002001014', '003008064', '003010080', '007008064', '007007062', '013008064',
                 '013001014', '013003028', '013007062', '013007060', '021008064', '021001014', '021007062',
                 '021009076', '021009078', '030006048', '030008064', '030007062', '030001012', '032002016', '032004032',
                 '141006048', '161010080', '161009076', '161011092', '161011094', '181000000', '181003028'], ## a lot of weak pulses....
            }
    
    bad_antennas = [
            ##CS002
            '003008064', ##CS003
            ##CS004
            ##CS005
            ##CS006
            ##CS007
            ##CS011
            ##CS013
            ##CS017
            '021009078', '021011092', ##CS021
            '030003028', ##CS030
            '032010080',##CS032
            ##CS101
            ##CS103
            ##CS301
            ##CS302
            ##CS401
            '181006048', ##CS501
            ##RS208
            ##RS306
            ##RS307
            ##RS310
            ##RS406
            '169005046', '169009076',##RS409
            '183006048', '183003030', ##RS503  check 183006048, 183003030
            '188011094',##RS508
            ##RS509
                    ]

    antenna_calibrator = LBA_ant_calibrator(timeID)
    
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
    
        #### open data and data processing stuff ####
    print("loading data")
    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(fpaths, force_metadata_ant_pos=True) for sname,fpaths in raw_fpaths.items() if sname in chain(guess_timings.keys(), [referance_station]) }
    
    data_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in  chain(guess_timings.keys(), [referance_station]) }
    
    antenna_calibrator = LBA_ant_calibrator(timeID)
    
    
    
    #### sort antennas and stations ####
    station_order = list(guess_timings.keys())## note this doesn't include reference station
    sorted_antenna_names = []
    station_to_antenna_index_dict = {}
    ant_loc_dict = {}
    
    for sname in station_order + [referance_station]:
        first_index = len(sorted_antenna_names)
        
        stat_data = raw_data_files[sname]
        even_ant_names = stat_data.get_antenna_names()[::2]
        even_ant_locs = stat_data.get_LOFAR_centered_positions()[::2]
        
        sorted_antenna_names += even_ant_names
        
        for ant_name, ant_loc in zip(even_ant_names,even_ant_locs):
            ant_loc_dict[ant_name] = ant_loc
                
        station_to_antenna_index_dict[sname] = (first_index, len(sorted_antenna_names))
    
    ant_locs = np.zeros( (len(sorted_antenna_names), 3))
    for i, ant_name in enumerate(sorted_antenna_names):
        ant_locs[i] = ant_loc_dict[ant_name]
    
    station_locations = {sname:ant_locs[station_to_antenna_index_dict[sname][0]] for sname in station_order + [referance_station]}
    station_to_antenna_index_list = [station_to_antenna_index_dict[sname] for sname in station_order]
    
    
    #### sort the delays guess, and account for station locations ####
    current_delays_guess = np.array([guess_timings[sname] for sname in station_order])
    original_delays = np.array( current_delays_guess )        
    
    
    
    
    #### open info from part 1 ####

    input_manager = Part1_input_manager( part1_inputs )



    #### first we fit the known sources ####
    current_sources = []
#    next_source = 0
    for knownID in known_sources:
        
        source_ID, input_name = input_manager.known_source( knownID )
        
        print("prep fitting:", source_ID)
            
            
        location = known_source_locations[source_ID]
        
        ## make source
        source_to_add = source_object(source_ID, input_name, location, stations_to_exclude[source_ID], antennas_to_exclude[source_ID] )
        current_sources.append( source_to_add )


        polarity = known_polarizations[source_ID]

            
        source_to_add.prep_for_fitting(polarity)
        
    print()
    print("fitting known sources")
#    fitter = stochastic_fitter(current_sources)
#    fitter = stochastic_fitter_dt(current_sources)
    fitter = stochastic_fitter_FitLocs(current_sources)
    
    fitter.employ_result( current_sources )
    stations_with_fits = fitter.get_stations_with_fits()
    
    print()
    fitter.print_locations( current_sources )
    print()
    fitter.print_station_fits( current_sources )
    print()
    fitter.print_delays( original_delays )
    print()
    print()
    
    print("locations")
    for source in current_sources:
        print(source.ID,':[', source.guess_XYZT[0], ',', source.guess_XYZT[1], ',', source.guess_XYZT[2], ',', source.guess_XYZT[3], '],')
    
    print()
    print()
    print("REL LOCS")
    rel_loc = current_sources[0].guess_XYZT
    for source in current_sources:
        print(source.ID, rel_loc-source.guess_XYZT)
    
        
        
    
    
    
    
    