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
from LoLIM.utilities import logger, processed_data_dir, v_air, SId_to_Sname, Sname_to_SId_dict, RTD
from LoLIM.IO.binary_IO import read_long, write_long, write_double_array, write_string, write_double
from LoLIM.antenna_response import LBA_ant_calibrator
from LoLIM.porta_code import code_logger, pyplot_emulator
from LoLIM.signal_processing import parabolic_fit, remove_saturation, data_cut_at_index
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1
from LoLIM.findRFI import window_and_filter
from LoLIM.stationTimings.autoCorrelator_tools import stationDelay_fitter
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
#
#
#class stochastic_fitter:
#    def __init__(self, source_object_list, initial_guess=None, quick_kill=None):
#        print("running stochastic fitter")
#        self.quick_kill = quick_kill
#        self.source_object_list = source_object_list
#    
#    ## assume globals: 
##    max_itters_per_loop 
##    itters_till_convergence
##    max_jitter_width
##    min_jitter_width 
##    cooldown
#    
##    sorted_antenna_names
#    
#        self.num_antennas = len(sorted_antenna_names)
#        self.num_measurments = self.num_antennas*len(source_object_list)
#        self.num_delays = len(station_order)
#        
#        #### make guess ####
#        self.num_DOF = -self.num_delays
#        self.solution = np.zeros(  self.num_delays+4*len(source_object_list) )
#        self.solution[:self.num_delays] = current_delays_guess
#        param_i = self.num_delays
#        for PSE in source_object_list:
#            self.solution[param_i:param_i+4] = PSE.guess_XYZT
#            param_i += 4
#            self.num_DOF += PSE.num_DOF()
#            
#        if initial_guess is not None: ## use initial guess instead, if given
#            self.solution = initial_guess
#            
#        self.initial_guess = np.array( self.solution )
#            
#        
#        self.rerun()
#        
#
#            
#            
#        
#        
#    def objective_fun(self, sol, do_print=False):
#        workspace_sol = np.zeros(self.num_measurments, dtype=np.double)
#        delays = sol[:self.num_delays]
#        ant_i = 0
#        param_i = self.num_delays
#        for PSE in self.source_object_list:
#            
#            PSE.try_location_LS(delays, sol[param_i:param_i+4], workspace_sol[ant_i:ant_i+self.num_antennas])
#            
#            ant_i += self.num_antennas
#            param_i += 4
#            
#        filter = np.logical_not( np.isfinite(workspace_sol) )
#        workspace_sol[ filter ]  = 0.0
#        
#        if do_print:
#            print("num func nans:", np.sum(filter))
#            
#        return workspace_sol
#    #        workspace_sol *= workspace_sol
#    #        return np.sum(workspace_sol)
#        
#    
#    def objective_jac(self, sol, do_print=False):
#        workspace_jac = np.zeros((self.num_measurments, self.num_delays+4*len(self.source_object_list)), dtype=np.double)
#    
#            
#        delays = sol[:self.num_delays]
#        ant_i = 0
#        param_i = self.num_delays
#        for PSE in self.source_object_list:
#            
#            PSE.try_location_JAC(delays, sol[param_i:param_i+4],  workspace_jac[ant_i:ant_i+self.num_antennas, param_i:param_i+4],  
#                                 workspace_jac[ant_i:ant_i+self.num_antennas, 0:self.num_delays])
#            
#            filter = np.logical_not( np.isfinite(workspace_jac[ant_i:ant_i+self.num_antennas, param_i+3]) )
#            workspace_jac[ant_i:ant_i+self.num_antennas, param_i:param_i+4][filter] = 0.0
#            workspace_jac[ant_i:ant_i+self.num_antennas, 0:self.num_delays][filter] = 0.0
#            
#            ant_i += self.num_antennas
#            param_i += 4
#    
#
#        
#        if do_print:
#            print("num jac nans:", np.sum(filter))
#        
#        return workspace_jac
#    
#    def get_RMS(self, solution):
#        
#        total_RMS = 0.0
#        new_station_delays = solution[:self.num_delays] 
#        param_i = self.num_delays
#        for PSE in self.source_object_list:
#            total_RMS += PSE.SSqE_fit( new_station_delays,  solution[param_i:param_i+4] )
#            param_i += 4
#            
#        return np.sqrt(total_RMS/self.num_DOF)
#        
#            
#    def rerun(self):
#        
#        current_guess = np.array( self.solution )
#        
#        #### first itteration ####
#        fit_res = least_squares(self.objective_fun, current_guess, jac=self.objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
##        fit_res = least_squares(self.objective_fun, current_guess, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
#        
#        
#        
#        print("guess RMS:", self.get_RMS(current_guess))
#        current_guess = fit_res.x
#        current_fit = fit_res.cost
#        print("new RMS:", self.get_RMS(current_guess), current_fit, fit_res.status)
#        
#        current_temperature = max_jitter_width
#        new_guess = np.array( current_guess )
#        while current_temperature>min_jitter_width: ## loop over each 'temperature'
#            
#            print("  stochastic run. Temp:", current_temperature)
#            
#            itters_since_change = 0
#            has_improved = False
#            for run_i in range(max_itters_per_loop):
#                print("  run:", run_i, ':', itters_since_change, " "*10, end="\r")
##                print("  itter", run_i)
#                
#                ## jitter the initial guess ##
#                new_guess[:self.num_delays] = np.random.normal(scale=current_temperature, size=self.num_delays) + current_guess[:self.num_delays] ## note use of best_solution, allows for translation. Faster convergence?
#            
#                param_i = self.num_delays
#                for PSE in self.source_object_list:
#                    new_guess[param_i:param_i+3] = np.random.normal(scale=current_temperature*v_air, size=3) + current_guess[param_i:param_i+3]
##                    new_guess[param_i+2] = np.abs(new_guess[param_i+2])## Z should be positive
#                    
#                    new_guess[param_i+3] = PSE.estimate_T(new_guess[:self.num_delays], new_guess[param_i:param_i+4])
#                    param_i += 4
#                
#                #### FIT!!! ####
#                fit_res = least_squares(self.objective_fun, new_guess, jac=self.objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
##                fit_res = least_squares(self.objective_fun, new_guess, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
##                print("    cost:", fit_res.cost)
#        
#                if fit_res.cost < current_fit:
#                    current_fit = fit_res.cost
#                    current_guess = fit_res.x
#                    if (fit_res.cost-current_fit)/current_fit > 0.001:
#                        itters_since_change = 0
#                        has_improved = True
#                else:
#                    itters_since_change += 1
#                if itters_since_change == itters_till_convergence:
#                    break
#                
#            print(" "*30,end="\r")
#                
#            if has_improved:
#                current_temperature /= cooldown
#            else:
#                current_temperature /= strong_cooldown
#                    
#            
#            total_RMS = 0.0
#            new_station_delays = current_guess[:self.num_delays] 
#            param_i = self.num_delays
#            for PSE in self.source_object_list:
#                total_RMS += PSE.SSqE_fit( new_station_delays,  current_guess[param_i:param_i+4] )
#                param_i += 4
#                
#            total_RMS = np.sqrt(total_RMS/self.num_DOF)
#            print("    RMS fit", total_RMS, "num runs:", run_i+1, "cost", current_fit)
#            
#            if self.quick_kill is not None and total_RMS>self.quick_kill:
#                print("    quick kill exceeded")
#                break
#            
#        if run_i+1 == max_itters_per_loop:
#            print("  not converged!")
#            self.converged = False
#        else:
#            self.converged = True
#            
#            
#            
#        #### get individual fits per station and PSE
#        self.PSE_fits = []
#        self.PSE_RMS_fits = []
#        new_station_delays = current_guess[:self.num_delays] 
#        param_i = self.num_delays
#        for source in self.source_object_list:
#            self.PSE_fits.append( source.RMS_fit_byStation(new_station_delays, 
#                         current_guess[param_i:param_i+4]) )
#            
#            SSQE = source.SSqE_fit(new_station_delays, current_guess[param_i:param_i+4])
##            source.SSqE_fitNprint( new_station_delays,  current_guess[param_i:param_i+4] )
#            self.PSE_RMS_fits.append( np.sqrt(SSQE/source.num_DOF()) )
#            
#            param_i += 4
#            
#            
#            
#        #### check which stations have fits
#        self.stations_with_fits = [False]*( len(station_order)+1 )
#            
#        for stationfits in self.PSE_fits:
#            for snum, (sname, fit) in enumerate(zip( chain(station_order, [referance_station]),  stationfits )):
#                if fit is not None:
#                    self.stations_with_fits[snum] = True
#            
#            
#            
#        #### edit solution for stations that don't have guess
#        for snum, has_fit in enumerate(self.stations_with_fits[:-1]): #ignore last station, as is referance station
#            if not has_fit:
#                current_guess[ snum ] = self.initial_guess[ snum ]
#            
#            
#        #### save results
#        self.solution = current_guess
#        self.cost = current_fit
#        self.RMS = total_RMS
#        
#
#        
#        
#        
#        
#        
#    def employ_result(self, source_object_list):
#        """set the result to the guess location of the sources, and return the station timing offsets"""
#        
#        param_i = self.num_delays
#        for PSE in source_object_list:
#            PSE.guess_XYZT[:] =  self.solution[param_i:param_i+4]
#            param_i += 4
#            
#        return self.solution[:self.num_delays]
#    
#    def print_locations(self, source_object_list):
#        
#        param_i = self.num_delays
#        for source, RMSfit in zip(source_object_list, self.PSE_RMS_fits):
#            print("source", source.ID)
#            print("  RMS:", RMSfit)
#            print("  loc:", self.solution[param_i:param_i+4])
#            param_i += 4
#    
##    def print_station_fits(self, source_object_list):
##        
##        fit_table = PrettyTable()
##        fit_table.field_names = ['id'] + station_order + [referance_station] + ['total']
##        fit_table.float_format = '.2E'
##        
##        for source, RMSfit, stationfits in zip(source_object_list, self.PSE_RMS_fits, self.PSE_fits):
##            new_row = ['']*len(fit_table.field_names)
##            new_row[0] = source.ID
##            new_row[-1] = RMSfit
##            
##            for i,stat_fit in enumerate(stationfits):
##                if stat_fit is not None:
##                    new_row[i+1] = stat_fit
##                    
##            fit_table.add_row( new_row )
##            
##        print( fit_table )
#        
#    def print_station_fits(self, source_object_list, num_stat_per_table):
#        
#        stations_to_print = station_order + [referance_station]
#        current_station_i = 0
#        while len(stations_to_print) > 0:
#            stations_this_run = stations_to_print[:num_stat_per_table]
#            stations_to_print = stations_to_print[len(stations_this_run):]
#            
#            fit_table = PrettyTable()
#            fit_table.field_names = ['id'] + stations_this_run + ['total']
#            fit_table.float_format = '.2E'
#            
#            for source, RMSfit, stationfits in zip(source_object_list, self.PSE_RMS_fits, self.PSE_fits):
#                new_row = ['']*len(fit_table.field_names)
#                new_row[0] = source.ID
#                new_row[-1] = RMSfit
#                
#                for i,stat_fit in enumerate(stationfits[current_station_i:current_station_i+len(stations_this_run)]):
#                    if stat_fit is not None:
#                        new_row[i+1] = stat_fit
#                        
#                fit_table.add_row( new_row )
#                
#            print( fit_table )
#            print()
#            current_station_i += len(stations_this_run)
#                    
#    def print_delays(self, original_delays):
#        for sname, delay, original in zip(station_order, self.solution[:self.num_delays], original_delays):
#            print("'"+sname+"' : ",delay,', ## diff to guess:', delay-original)
#    
#    def get_stations_with_fits(self):
#        return self.stations_with_fits
#    
#    

class stochastic_fitter_dt:
    def __init__(self, source_object_list, initial_guess=None, quick_kill=None):
#        print("running stochastic fitter")
        self.quick_kill = quick_kill
        self.first_source = source_object_list[0]
        self.source_object_list = source_object_list[1:]
        source_object_list = self.source_object_list
    
    ## assume globals: 
#    max_itters_per_loop 
#    itters_till_convergence
#    max_jitter_width
#    min_jitter_width 
#    cooldown
    
#    sorted_antenna_names
    
        self.num_antennas = len(sorted_antenna_names)
        self.num_measurments = self.num_antennas*(len(source_object_list)+1)
        self.num_delays = len(station_order)
        
        self.station_indeces = np.empty( len(ant_locs), dtype=np.int )
        for station_index, index_range in enumerate(station_to_antenna_index_list):
            first,last = index_range
            self.station_indeces[first:last] = station_index
    
        self.fitter = stationDelay_fitter(ant_locs, self.station_indeces, len(self.source_object_list)+1, self.num_delays)
        
        self.fitter.set_event( self.first_source.pulse_times )
        for source in self.source_object_list:
            self.fitter.set_event( source.pulse_times )
            
#        self.one_fitter = stationDelay_fitter(ant_locs, self.station_indeces, 1, self.num_delays)
#        self.one_fitter.set_event( self.source_object_list[0] )
        
        #### make guess ####
        self.num_DOF = -self.num_delays
        self.solution = np.zeros(  self.num_delays+4*len(source_object_list)+4 )
        self.solution[:self.num_delays] = current_delays_guess
        param_i = self.num_delays
        
        self.solution[param_i:param_i+4] = self.first_source.guess_XYZT
        param_i += 4
        self.num_DOF += self.first_source.num_DOF()
        
        for PSE in source_object_list:
            self.solution[param_i:param_i+4] = PSE.guess_XYZT
            param_i += 4
            self.num_DOF += PSE.num_DOF()
            
            
        if initial_guess is not None: ## use initial guess instead, if given
            self.solution = initial_guess
            
        self.initial_guess = np.array( self.solution )
            
        
        self.rerun()
        
        
    def modify_solution(self, solution):
        solution[self.num_delays:self.num_delays+3] = self.first_source.guess_XYZT[:3]
        
    def objective_fun(self, guess):
#        self.guess_temp[:self.num_delays] = guess[:self.num_delays]
#        self.guess_temp[self.guess_temp:self.guess_temp+4] = self.first_source.guess_XYZT
#        self.guess_temp[self.guess_temp+4:] = guess[self.num_delays:]
#        
#        return self.fitter.objective_fun( self.guess_temp )
        self.modify_solution(guess)
    
        return self.fitter.objective_fun( guess)
        

    
    def get_RMS(self, solution):
        self.modify_solution(solution)
        
        solution[self.num_delays:self.num_delays+3] = self.first_source.guess_XYZT[:3]
        
        total_RMS = 0.0
        new_station_delays = solution[:self.num_delays] 
        param_i = self.num_delays
        
        #### first source ####
#        total_RMS += self.first_source.SSqE_fit( new_station_delays,  self.first_source.guess_XYZT )
#        print(self.first_source.guess_XYZT, solution[param_i:param_i+4])
        total_RMS += self.first_source.SSqE_fit( new_station_delays,  solution[param_i:param_i+4] )
        param_i += 4
        
        
        
        #### now all the other ones ###
        for PSE in self.source_object_list:
            total_RMS += PSE.SSqE_fit( new_station_delays,  solution[param_i:param_i+4] )
            param_i += 4
            
        return np.sqrt(total_RMS/self.num_DOF)
            
            
    def rerun(self):
        
        current_guess = np.array( self.solution )
#        print('D', current_guess)
        
        #### first itteration ####
        fit_res = least_squares(self.objective_fun, current_guess, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
        
#        print( self.fitter.objective_jac(current_guess) )
        
        current_guess = fit_res.x
        self.modify_solution( current_guess )
#        print('E', fit_res.status, current_guess)
#        quit()
        current_fit = fit_res.cost
        
        total_RMS = self.get_RMS( current_guess )
        
#        current_temperature = max_jitter_width
#        new_guess = np.array( current_guess )
#        while current_temperature>min_jitter_width: ## loop over each 'temperature'
#            
#            print("  stochastic run. Temp:", current_temperature)
#            
#            itters_since_change = 0
#            has_improved = False
#            for run_i in range(max_itters_per_loop):
#                print("  run:", run_i, ':', itters_since_change, " "*10, end="\r")
##                print("  itter", run_i)
#                
#                ## jitter the initial guess ##
#                new_guess[:self.num_delays] = np.random.normal(scale=current_temperature, size=self.num_delays) + current_guess[:self.num_delays] ## note use of best_solution, allows for translation. Faster convergence?
#            
#                param_i = self.num_delays
#                
#                
#                new_guess[param_i:param_i+3] = np.random.normal(scale=current_temperature*v_air, size=3) + current_guess[param_i:param_i+3]
#                new_guess[param_i+3] = self.first_source.estimate_T(new_guess[:self.num_delays], new_guess[param_i:param_i+4])
#                param_i += 4
#                
#                for PSE in self.source_object_list:
#                    new_guess[param_i:param_i+3] = np.random.normal(scale=current_temperature*v_air, size=3) + current_guess[param_i:param_i+3]
##                    new_guess[param_i+2] = np.abs(new_guess[param_i+2])## Z should be positive
#                    
#                    new_guess[param_i+3] = PSE.estimate_T(new_guess[:self.num_delays], new_guess[param_i:param_i+4])
#                    param_i += 4
#                
#                #### FIT!!! ####
#                fit_res = least_squares(self.objective_fun, current_guess, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
##                print("    cost:", fit_res.cost)
#        
#                if fit_res.cost < current_fit:
#                    current_fit = fit_res.cost
#                    current_guess = fit_res.x
#                    if (fit_res.cost-current_fit)/current_fit > 0.001:
#                        itters_since_change = 0
#                        has_improved = True
#                else:
#                    itters_since_change += 1
#                if itters_since_change == itters_till_convergence:
#                    break
#                
#            print(" "*30,end="\r")
#                
#            if has_improved:
#                current_temperature /= cooldown
#            else:
#                current_temperature /= strong_cooldown
#                    
#            total_RMS = self.get_RMS( current_guess )
#            print("    RMS fit", total_RMS, "num runs:", run_i+1, "cost", current_fit)
#            
#            if self.quick_kill is not None and total_RMS>self.quick_kill:
#                print("    quick kill exceeded")
#                break
#            
#        if run_i+1 == max_itters_per_loop:
#            print("  not converged!")
#            self.converged = False
#        else:
        self.converged = True
            
            
            
#        #### get individual fits per station and PSE
#        self.PSE_fits = []
#        self.PSE_RMS_fits = []
#        new_station_delays = current_guess[:self.num_delays] 
#        param_i = self.num_delays
#        
#        
#        self.PSE_fits.append( self.first_source.RMS_fit_byStation(new_station_delays, 
#                     current_guess[param_i:param_i+4]) )
#        SSQE = self.first_source.SSqE_fit(new_station_delays, current_guess[param_i:param_i+4])
#        self.PSE_RMS_fits.append( np.sqrt(SSQE/self.first_source.num_DOF()) )
#        
#        param_i += 4
#        
#        for source in self.source_object_list:
#            self.PSE_fits.append( source.RMS_fit_byStation(new_station_delays, 
#                         current_guess[param_i:param_i+4]) )
#            
#            SSQE = source.SSqE_fit(new_station_delays, current_guess[param_i:param_i+4])
##            source.SSqE_fitNprint( new_station_delays,  current_guess[param_i:param_i+4] )
#            self.PSE_RMS_fits.append( np.sqrt(SSQE/source.num_DOF()) )
#            
#            param_i += 4
#            
#            
#            
#        #### check which stations have fits
#        self.stations_with_fits = [False]*( len(station_order)+1 )
#            
#        for stationfits in self.PSE_fits:
#            for snum, (sname, fit) in enumerate(zip( chain(station_order, [referance_station]),  stationfits )):
#                if fit is not None:
#                    self.stations_with_fits[snum] = True
#            
#            
#            
#        #### edit solution for stations that don't have guess
#        for snum, has_fit in enumerate(self.stations_with_fits[:-1]): #ignore last station, as is referance station
#            if not has_fit:
#                current_guess[ snum ] = self.initial_guess[ snum ]
            
            
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
    
    def print_station_fits(self, source_object_list, num_stat_per_table):
        
        stations_to_print = station_order + [referance_station]
        current_station_i = 0
        while len(stations_to_print) > 0:
            stations_this_run = stations_to_print[:num_stat_per_table]
            stations_to_print = stations_to_print[len(stations_this_run):]
            
            fit_table = PrettyTable()
            fit_table.field_names = ['id'] + stations_this_run + ['total']
            fit_table.float_format = '.2E'
            
            for source, RMSfit, stationfits in zip(source_object_list, self.PSE_RMS_fits, self.PSE_fits):
                new_row = ['']*len(fit_table.field_names)
                new_row[0] = source.ID
                new_row[-1] = RMSfit
                
                for i,stat_fit in enumerate(stationfits[current_station_i:current_station_i+len(stations_this_run)]):
                    if stat_fit is not None:
                        new_row[i+1] = stat_fit
                        
                fit_table.add_row( new_row )
                
            print( fit_table )
            print()
            current_station_i += len(stations_this_run)
                    
    def print_delays(self, original_delays):
        for sname, delay, original in zip(station_order, self.solution[:self.num_delays], original_delays):
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
        
#        print(delta_X_sq)
        np.sqrt(workspace, out=workspace)
        
#        print(self.pulse_times)
#        print(workspace)
        workspace[:] -= self.pulse_times*v_air ## this is now source time
        
        ##now account for delays
        for index_range, delay in zip(station_to_antenna_index_list,  delays):
            first,last = index_range
            workspace[first:last] += delay*v_air ##note the wierd sign
                
#        print(workspace)
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
            
class Part1_input_manager:
    def __init__(self, input_files):
        self.max_num_input_files = 10
        if len(input_files) > self.max_num_input_files:
            print("TOO MANY INPUT FOLDERS!!!")
            quit()
        
        self.input_files = input_files
        
        self.input_data = []
        for folder_i, folder in enumerate(input_files):
            input_folder = processed_data_folder + "/" + folder +'/'
            
            file_list = [(int(f.split('_')[1][:-3])*self.max_num_input_files+folder_i ,input_folder+f) for f in listdir(input_folder) if f.endswith('.h5')] ## get all file names, and get the 'ID' for the file name
            file_list.sort( key=lambda x: x[0] ) ## sort according to ID
            self.input_data.append( file_list )
        
    def known_source(self, ID):
        
        file_i = int(ID/self.max_num_input_files)
        folder_i = ID - file_i*self.max_num_input_files
        file_list = self.input_data[ folder_i ]
        
        return [info for info in file_list if info[0]==ID][0]
        

np.set_printoptions(precision=10, threshold=np.inf)

## some global settings
num_stat_per_table = 10
#### these globals are holdovers
#station_locations = None ## to be set
#station_to_antenna_index_list = None## to be set
#stations_with_fits = None## to be set
#station_to_antenna_index_dict = None
def run_fitter(timeID, output_folder, pulse_input_folders, guess_timings, souces_to_fit, guess_source_locations,
               source_polarizations, source_stations_to_exclude, source_antennas_to_exclude, bad_ants,
               X_deviations, Y_deviations, Z_deviations,
               ref_station="CS002", min_ant_amplitude=10, max_stoch_loop_itters = 2000, min_itters_till_convergence = 100,
               initial_jitter_width = 100000E-9, final_jitter_width = 1E-9, cooldown_fraction = 10.0, strong_cooldown_fraction = 100.0,
               fitter = "dt"):
    
    ##### holdovers. These globals need to be fixed, so not global....
    global station_locations, station_to_antenna_index_list, stations_with_fits, station_to_antenna_index_dict
    global referance_station, station_order, sorted_antenna_names, min_antenna_amplitude, ant_locs, bad_antennas
    global max_itters_per_loop, itters_till_convergence, min_jitter_width, max_jitter_width, cooldown, strong_cooldown
    global current_delays_guess, processed_data_folder
    
    referance_station = ref_station
    min_antenna_amplitude = min_ant_amplitude
    bad_antennas = bad_ants
    max_itters_per_loop = max_stoch_loop_itters
    itters_till_convergence = min_itters_till_convergence
    max_jitter_width = initial_jitter_width
    min_jitter_width = final_jitter_width
    cooldown = cooldown_fraction
    strong_cooldown = strong_cooldown_fraction
    
    if referance_station in guess_timings:
        ref_T = guess_timings[referance_station]
        guess_timings = {station:T-ref_T for station,T in guess_timings.items() if station != referance_station}
        
    processed_data_folder = processed_data_dir(timeID)
    
    data_dir = processed_data_folder + "/" + output_folder
    if not isdir(data_dir):
        mkdir(data_dir)
        
    logging_folder = data_dir + '/logs_and_plots'
    if not isdir(logging_folder):
        mkdir(logging_folder)



    #Setup logger and open initial data set
    log = logger()
    log.set(logging_folder + "/log_out.txt") ## TODo: save all output to a specific output folder
    log.take_stderr()
    log.take_stdout()
    
    
    print("timeID:", timeID)
    print("date and time run:", time.strftime("%c") )
    print("input folders:", pulse_input_folders)
    print("source IDs to fit:", souces_to_fit)
    print("guess locations:", guess_source_locations)
    print("polarization to use:", source_polarizations)
    print("source stations to exclude:", source_stations_to_exclude)
    print("source antennas to exclude:", source_antennas_to_exclude)
    print("bad antennas:", bad_ants)
    print("referance station:", ref_station)
    print("fitter type:", fitter)
    print("guess delays:", guess_timings)
    print()
    print()
    
        #### open data and data processing stuff ####
    print("loading data")
    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(fpaths, force_metadata_ant_pos=True) for sname,fpaths in raw_fpaths.items() if sname in chain(guess_timings.keys(), [referance_station]) }
    
    
    
    
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
    station_to_antenna_index_list = [station_to_antenna_index_dict[sname] for sname in station_order + [referance_station]]
    
    
    #### sort the delays guess, and account for station locations ####
    current_delays_guess = np.array([guess_timings[sname] for sname in station_order])
    original_delays = np.array( current_delays_guess )        
    
    
    
    
    #### open info from part 1 ####

    input_manager = Part1_input_manager( pulse_input_folders )



    #### first we fit the known sources ####
    current_sources = []
#    next_source = 0
    for knownID in souces_to_fit:
        
        source_ID, input_name = input_manager.known_source( knownID )
        
        print("prep fitting:", source_ID)
            
            
        location = guess_source_locations[source_ID]
        
        ## make source
        source_to_add = source_object(source_ID, input_name, location, source_stations_to_exclude[source_ID], source_antennas_to_exclude[source_ID] )
        current_sources.append( source_to_add )


        polarity = source_polarizations[source_ID]

            
        source_to_add.prep_for_fitting(polarity)
        
    print()
    print("fitting known sources")
    if fitter!='dt':
        print("only dt")
        quit()
        
    first_source = current_sources[0]
    original_XYZT = np.array( first_source.guess_XYZT )
    
    if X_deviations is not None:
        RMS_values = np.empty( len(X_deviations) )
        
        first_source.guess_XYZT[:] = original_XYZT
        for i in range(len(X_deviations)):
        
            first_source.guess_XYZT[0] = original_XYZT[0] + X_deviations[i]
            fitter = stochastic_fitter_dt(current_sources)
            RMS_values[i] = fitter.RMS
            
            print('X', X_deviations[i], fitter.RMS)
            
        plt.plot(X_deviations, RMS_values)
        plt.show()
        
    
    if Y_deviations is not None:
        RMS_values = np.empty( len(Y_deviations) )
        
        first_source.guess_XYZT[:] = original_XYZT
        for i in range(len(Y_deviations)):
        
            first_source.guess_XYZT[1] = original_XYZT[1] + Y_deviations[i]
            fitter = stochastic_fitter_dt(current_sources)
            RMS_values[i] = fitter.RMS
            
            print('Y', Y_deviations[i], fitter.RMS)
            
        plt.plot(Y_deviations, RMS_values)
        plt.show()
    
        
    if Z_deviations is not None:
        RMS_values = np.empty( len(Z_deviations) )
        
        first_source.guess_XYZT[:] = original_XYZT
        for i in range(len(Z_deviations)):
        
            first_source.guess_XYZT[2] = original_XYZT[2] + Z_deviations[i]
            fitter = stochastic_fitter_dt(current_sources)
            RMS_values[i] = fitter.RMS
            
            print('Z', Z_deviations[i], fitter.RMS)
            
        plt.plot(Z_deviations, RMS_values)
        plt.show()
    
    