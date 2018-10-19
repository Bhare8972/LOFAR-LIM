#!/usr/bin/env python3

#python
import time
from os import mkdir
from os.path import isdir, isfile
from itertools import chain
from pickle import load

#external
import numpy as np
from scipy.optimize import least_squares, minimize, approx_fprime
from scipy.signal import hilbert
from matplotlib import pyplot as plt

#mine
from LoLIM.prettytable import PrettyTable
from LoLIM.utilities import log, processed_data_dir, v_air, SId_to_Sname, Sname_to_SId_dict, RTD
from LoLIM.IO.binary_IO import read_long, write_long, write_double_array, write_string, write_double
from LoLIM.antenna_response import LBA_ant_calibrator
from LoLIM.porta_code import code_logger, pyplot_emulator
from LoLIM.signal_processing import parabolic_fit
#from RunningStat import RunningStat

from LoLIM.read_pulse_data import writeTXT_station_delays,read_station_info, curtain_plot_CodeLog

from LoLIM.planewave_functions import read_SSPW_timeID

#### some random utilities
def none_max(lst):
    """given a list of numbers, return maximum, ignoreing None"""
    
    ret = -np.inf
    for a in lst:
        if (a is not None) and (a>ret):
            ret=a
            
    return ret

def combine_SSPW(SSPW_A, SSPW_B):
    """for two SSPW, first loads all data from file. Then combines antenna data from SSPW_B into SSPW_A"""
    
    SSPW_A.reload_data()
    SSPW_B.reload_data()

    for ant_name, ant_data in SSPW_B.ant_data.items():
        SSPW_A.ant_data[ant_name] = ant_data
                
    
def get_radius_ze_az( XYZ ):
    radius = np.linalg.norm( XYZ )
    ze = np.arccos( XYZ[2]/radius )
    az = np.arctan2( XYZ[1], XYZ[0] )
    return radius, ze, az
    

#### main code


class stochastic_fitter:
    def __init__(self, source_object_list, initial_guess=None):
        print("running stochastic fitter")
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
                print("  run:", run_i, ':', itters_since_change, end="\r")
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
                    itters_since_change = 0
#                    print("    improved")
                    has_improved = True
                else:
                    itters_since_change += 1
                if itters_since_change == itters_till_convergence:
                    break
            
            print(end="\r")
                
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
            print("source", source.SSPW.unique_index)
            print("  RMS:", RMSfit)
            print("  loc:", self.solution[param_i:param_i+4])
            param_i += 4
    
    def print_station_fits(self, source_object_list):
        
        fit_table = PrettyTable()
        fit_table.field_names = ['id'] + station_order + [referance_station] + ['total']
        fit_table.float_format = '.2E'
        
        for source, RMSfit, stationfits in zip(source_object_list, self.PSE_RMS_fits, self.PSE_fits):
            new_row = ['']*len(fit_table.field_names)
            new_row[0] = source.SSPW.unique_index
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


#### source object ####
## represents a potential source
## keeps track of a SSPW on the prefered station, and SSPW on other stations that could correlate and are considered correlated
## contains utilities for fitting, and for finding RMS in total and for each station
## also contains utilities for plotting and saving info

## need to handle inseartion of random error, and that choosen SSPE can change

class source_object():
## assume: guess_location , ant_locs, station_to_antenna_index_list, station_to_antenna_index_dict, referance_station, station_order, SSPW_dict,
#    sorted_antenna_names, station_locations, SSPW_dict
    # are global
    
    def do_combine_SSPW(self, SSPW, SSPW_list):
        combine_list = []
        for combination in self.SSPW_to_combine:
            if SSPW.unique_index in combination:
                combine_list = combination
                break
            
        for SSPW_ID in combine_list:
            if SSPW_ID == SSPW.unique_index:
                continue
            
            for new_SSPW in SSPW_list:
                if new_SSPW.unique_index == SSPW_ID:
                    break
                
            combine_SSPW( SSPW, new_SSPW )
            self.source_exclusions.append( SSPW_ID ) ## so we don't use this SSPW in the future
    
    def __init__(self, ref_SSPW_index,  viable_SSPW_indeces, source_exclusions, location=None, SSPW_to_combine=[], suspicious_SSPW_list=[] ):
        self.suspicious_SSPW_list = suspicious_SSPW_list
        self.SSPW_to_combine = SSPW_to_combine
        self.source_exclusions = source_exclusions
        self.SSPW_in_use = {}
        
        
        #### first we need to find the SSPW on the referance station
        self.SSPW = None
        for SSPW in SSPW_dict[ referance_station ]:
            if SSPW.unique_index == ref_SSPW_index:
                self.SSPW = SSPW
                
                if not self.SSPW.timeseries_loaded:
                    self.SSPW.reload_data()
                    
                self.do_combine_SSPW( self.SSPW, SSPW_dict[ referance_station ]  )
                break
                
        if self.SSPW is None:
            print("cannot find SSPW")
            quit()
            
        
        #### now we find SSPW on other stations
        self.viable_SSPW = {referance_station:[self.SSPW]}
        self.num_stations_with_unique_viable_SSPW = 0
        for sname, SSPW_ids in viable_SSPW_indeces.items():
            SSPW_list = []
            
            for ID in SSPW_ids:
                if ID in source_exclusions:
                    continue
                
                for SSPW in SSPW_dict[ sname ]:
                    if SSPW.unique_index == ID:
                        if not SSPW.timeseries_loaded:
                            SSPW.reload_data()
                        SSPW_list.append( SSPW )
                        self.do_combine_SSPW( SSPW, SSPW_dict[ sname ]  )
                        break
                    
            self.viable_SSPW[sname] = SSPW_list
            if len(SSPW_list) == 1 and not SSPW_list[0].unique_index in self.suspicious_SSPW_list:
                self.num_stations_with_unique_viable_SSPW += 1
        
        if location is None:
            guess_time = SSPW.ZAT[2] - np.linalg.norm( station_locations[referance_station]-guess_location )/v_air
            self.guess_XYZT = np.append( guess_location, [guess_time] )
        else:
            self.guess_XYZT = np.array( location )

                 
            
    def prep_for_fitting(self, polarization):
        self.polarization = polarization
        
        self.pulse_times = np.empty( len(sorted_antenna_names) )
        self.pulse_times[:] = np.nan
        self.waveforms = [None]*len(self.pulse_times)
        self.waveform_startTimes = [None]*len(self.pulse_times)
        
        #### first add times from referance_station
        self.add_SSPW( self.SSPW )
                
        #### next we add in stations that only have 1 unique SSPW
        for sname in station_order:
            SSPW_list = self.viable_SSPW[sname]
            if len(SSPW_list) == 1  and not SSPW_list[0].unique_index in self.suspicious_SSPW_list:
                SSPW_to_add = SSPW_list[0]
                
                self.add_SSPW( SSPW_to_add )
        
                        
    def prep_for_fitting_knownFit(self, polarity, SSPW_associations ):
        self.polarization = polarity
        
        self.pulse_times = np.empty( len(sorted_antenna_names) )
        self.pulse_times[:] = np.nan
        self.waveforms = [None]*len(self.pulse_times)
        self.waveform_startTimes = [None]*len(self.pulse_times)
        
        for sname, SSPW_ID in SSPW_associations.items():
            SSPW_list = self.viable_SSPW[sname]
            for SSPW in SSPW_list:
                if SSPW.unique_index == SSPW_ID:
                    break## found correct SSPW
                
            self.add_SSPW( SSPW )
                
                
    def remove_station(self, sname):
        if sname in self.SSPW_in_use:
            del self.SSPW_in_use[ sname ]
            
        antenna_index_range = station_to_antenna_index_dict[sname]
        self.pulse_times[ antenna_index_range[0]:antenna_index_range[1] ] = np.nan
        
    def has_station(self, sname):
        return (sname in self.SSPW_in_use)
         
    def add_SSPW(self, SSPW):
        
        self.remove_station( SSPW.sname )
        
        self.SSPW_in_use[ SSPW.sname ] = SSPW.unique_index
        
        if not SSPW.timeseries_loaded:
            SSPW.reload_data()
        
        antenna_index_range = station_to_antenna_index_dict[SSPW.sname]
        
        for ant_i in range(antenna_index_range[0], antenna_index_range[1]):
            ant_name = sorted_antenna_names[ant_i]
            if ant_name in SSPW.ant_data:
                ant_info = SSPW.ant_data[ant_name]
                start_time = ant_info.pulse_starting_index*5.0E-9
                
                if self.polarization != 3:
                    pt = ant_info.PolE_peak_time if self.polarization==0 else ant_info.PolO_peak_time
                    waveform = ant_info.PolE_hilbert_envelope if self.polarization==0 else ant_info.PolO_hilbert_envelope
                    start_time += ant_info.PolE_time_offset if self.polarization==0 else ant_info.PolO_time_offset
                    amp = np.max(waveform)
                    
                    if pt == np.inf:
                        pt = np.nan
                    if amp<min_antenna_amplitude:
                        pt = np.nan
                else:
                    PolE_data = ant_info.PolE_antenna_data
                    PolO_data = ant_info.PolO_antenna_data
                
                    if np.max( PolE_data ) < min_antenna_amplitude or np.max( PolO_data ) < min_antenna_amplitude:
                        pt = np.nan
                        waveform = None
                    else:
                        ant_loc = ant_locs[ ant_i ]
                        
                        radius, zenith, azimuth = get_radius_ze_az( self.guess_XYZT[:3]-ant_loc )
                
                        antenna_calibrator.FFT_prep(ant_name, PolE_data, PolO_data)
                        antenna_calibrator.apply_time_shift(0.0, ant_info.PolO_time_offset-ant_info.PolE_time_offset)
                        polE_cal, polO_cal = antenna_calibrator.apply_GalaxyCal()
                        
                        if not np.isfinite(polE_cal) or not np.isfinite(polO_cal):
                            pt = np.nan
                            waveform = None
                            
                        else:
                            antenna_calibrator.unravelAntennaResponce(zenith*RTD, azimuth*RTD)
                            zenith_data, azimuth_data = antenna_calibrator.getResult()
                
                            ZeD_R = np.real( zenith_data )
                            ZeD_I = np.imag( zenith_data )
                            Az_R = np.real( azimuth_data )
                            Az_I = np.imag( azimuth_data )
                            
                            total_amplitude_waveform = np.sqrt( ZeD_R*ZeD_R + ZeD_I*ZeD_I + Az_R*Az_R + Az_I*Az_I )
                            waveform = total_amplitude_waveform
                            
                            peak_finder = parabolic_fit( total_amplitude_waveform )
                            pt = (peak_finder.peak_index + ant_info.pulse_starting_index)*5.0E-9 + ant_info.PolE_time_offset
                            start_time += ant_info.PolE_time_offset
                
            
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
    
    def SSqE_fitNprint(self, delays, XYZT_location):
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
#        print(distance)
#        print()
#        print(self.pulse_times)
#        distance -= self.pulse_times
        
        ##now account for delays
        for sname, index_range, delay in zip(station_order, station_to_antenna_index_list,  delays):
            first,last = index_range
            
            print(sname, self.SSPW.unique_index, delay)
            print(sorted_antenna_names[first:last])
            print( distance[first:last] )
            print( self.pulse_times[first:last] )
            print( )
            
#        quit()
                
    
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
            
            min_T = np.inf
            for SSPW in self.viable_SSPW[sname]:
                
                max_T = -np.inf
                for ant_i in range(index_range[0], index_range[1]):
                    ant_name = sorted_antenna_names[ant_i]
                    if ant_name not in SSPW.ant_data:
                        continue
                    
                    ant_info = SSPW.ant_data[ant_name]
            
                    PolE_peak_time = ant_info.PolE_peak_time - offset
                    PolO_peak_time = ant_info.PolO_peak_time - offset
    
                    PolE_hilbert = ant_info.PolE_hilbert_envelope
                    PolO_hilbert = ant_info.PolO_hilbert_envelope
                
                    PolE_trace = ant_info.PolE_antenna_data
                    PolO_trace = ant_info.PolO_antenna_data
    
                    PolE_T_array = (np.arange(len(PolE_hilbert)) + ant_info.pulse_starting_index )*5.0E-9  + ant_info.PolE_time_offset
                    PolO_T_array = (np.arange(len(PolO_hilbert)) + ant_info.pulse_starting_index )*5.0E-9  + ant_info.PolO_time_offset
                    
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
                    
                plotter.annotate( SSPW.unique_index, xy=[max_T, index_range[1]-1.0/3.0], size=15)
                
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
                    
            if sname in self.SSPW_in_use:
                plotter.annotate( self.SSPW_in_use[ sname ], xy=[max_T, index_range[1]-1.0/3.0], size=15)
                
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
    def __init__(self, input_files, inject_sources):
        self.input_files = input_files
        
        self.input_data = []
        for fname in input_files:
            input_fname = processed_data_dir + "/" + fname + '/out'
            with open(input_fname, 'rb') as fin:
                input_SSPW_sort = load(fin)
                self.input_data.append( input_SSPW_sort )
                
        if len(inject_sources)>0:
            self.input_data.append(  [ (data[referance_station][0], data) for data in inject_sources ] )
                
        self.indeces = np.zeros( len(self.input_data), dtype=int )
#        self.last_file_index = None
        
    def known_source(self, ID):
        
        for current_i, index in enumerate(self.indeces):
            if index == len(self.input_data[ current_i ]):
                continue
            
            ref_SSPW_index, viable_SSPW_indeces = self.input_data[ current_i ][ index ]
            
            if ref_SSPW_index in bad_sources:
                self.indeces[ current_i ] += 1
                return self.known_source( ID )
            
            if ref_SSPW_index==ID:
                self.indeces[ current_i ] += 1
                return ref_SSPW_index, viable_SSPW_indeces
            
        return None
        
    def next(self):
#        #### first check for sources that are well known
#        for current_i, index in enumerate(self.indeces):
#            ref_SSPW_index, viable_SSPW_indeces = self.input_data[ current_i ][ index ]
#            
#            if ref_SSPW_index in bad_sources:
#                self.indeces[ current_i ] += 1
#                return self.next()
#            
#            elif ref_SSPW_index in known_sources:
#                self.indeces[ current_i ] += 1
##                self.last_file_index = current_i
#                return ref_SSPW_index, viable_SSPW_indeces
            
        #### if we are here, no sources are well known. check for sources that are currently being fitted
        best_file_i = 0
        for current_i, index in enumerate(self.indeces):
            if index == len(self.input_data[ current_i ]):
                continue
            
            ref_SSPW_index, viable_SSPW_indeces = self.input_data[ current_i ][ index ]
            
            if ref_SSPW_index in planewave_exclusions:
                self.indeces[ current_i ] += 1
#                self.last_file_index = current_i
                return ref_SSPW_index, viable_SSPW_indeces
            
            if index < self.indeces[best_file_i]:
                best_file_i = current_i
                
        #### if we are here, then no fitted sources left, only fully uknown sources
        ## return the "highest" unkown source
        
#        self.last_file_index = best_file_i
        return self.input_data[best_file_i][ self.indeces[best_file_i] ]
            


np.set_printoptions(precision=10, threshold=np.inf)
if __name__ == "__main__": 
    
    #### TODO: make code that looks for pulses on all antennas, ala analyze amplitudes
    ## probably need a seperate code that just fits location of one source, so can be tuned by hand, then fed into main code
    ## prehaps just add one station at a time? (one closest to known stations) seems to be general stratigey
    
    timeID = "D20170929T202255.000Z"
    output_folder = "autoCorrelator_unCal"
    
#    part1_input = "autoCorrelator_Part1"
    part1_inputs = ["autoCorrelator_Part1", "autoCorrelator_Part1_2"]
    
    
    SSPW_folder = 'SSPW'
    first_block = 3450
    num_blocks = 600
    
    
    #### fitter settings ####
    max_itters_per_loop = 2000
    itters_till_convergence = 100
    max_jitter_width = 100000E-9
    min_jitter_width = 1E-9
    cooldown = 10.0 ## 10.0
    strong_cooldown = 100.0
    
    #### source quality requirments ####
    min_stations = 4
    max_station_RMS = 5.0E-9
    min_antenna_amplitude = 10
    
    #### initial guesses ####
    referance_station = "CS002"
    guess_location = np.array( [1.72389621e+04,   9.50496918e+03, 2.37800915e+03] )
    
    guess_timings = {
'CS003' :  1.40571278618e-06 , ## diff to guess: 8.15937912233e-11
'CS004' :  4.30390652097e-07 , ## diff to guess: -1.0595566894e-09
'CS005' :  -2.20091532601e-07 , ## diff to guess: -5.23323229879e-10
'CS006' :  4.33638797726e-07 , ## diff to guess: 4.86616846895e-10
'CS007' :  4.00502074784e-07 , ## diff to guess: 7.34541540001e-10
'CS011' :  -5.85893683256e-07 , ## diff to guess: 5.92241536001e-10
'CS013' :  -1.81190914711e-06 , ## diff to guess: 8.11356409305e-10
'CS017' :  -8.43740274988e-06 , ## diff to guess: 3.23570949456e-09
'CS021' :  9.25647553154e-07 , ## diff to guess: -9.34026291847e-10
'CS030' :  -2.73759382223e-06 , ## diff to guess: 1.62030082312e-09
'CS032' :  -1.57750598653e-06 , ## diff to guess: -6.03828285099e-09
'CS101' :  -8.16352016331e-06 , ## diff to guess: 8.78074551323e-09
'CS103' :  -2.85144834972e-05 , ## diff to guess: 9.74383792332e-09
'RS208' :  6.89188143356e-06 , ## diff to guess: -1.23633423675e-07
'CS301' :  -7.2387596029e-07 , ## diff to guess: -7.60291555168e-09
'CS302' :  -5.36617859016e-06 , ## diff to guess: -1.97962223555e-08
'RS306' :  6.97903655021e-06 , ## diff to guess: -1.27342313225e-07
'RS307' :  6.84789068654e-06 , ## diff to guess: -2.62305606129e-07
'RS310' :  6.86741716279e-06 , ## diff to guess: -6.44600536056e-07
'CS401' :  -9.53944730399e-07 , ## diff to guess: -6.72613292808e-09
'RS406' :  -4.37233503223e-05 , ## diff to guess: 0.0
'RS409' :  7.00542541584e-06 , ## diff to guess: -7.54111679614e-07
'CS501' :  -9.60227784795e-06 , ## diff to guess: 5.90821201896e-09
'RS503' :  6.96531338603e-06 , ## diff to guess: 1.17710254312e-08
'RS508' :  7.11166833502e-06 , ## diff to guess: -1.33879926581e-07
'RS509' :  7.15893161746e-06 , ## diff to guess: -2.99785568996e-07
        }
        
    guess_timing_error = 5E-6
    guess_is_from_curtain_plot = False ## if true, must take station locations into account in order to get true offsets
    
    if referance_station in guess_timings:
        del guess_timings[referance_station]
    
    
    #### these are sources whose location have been fitted, and SSPW are associated
    known_sources = [275988, 278749, 280199, 274969, 275426, 276467, 274359, 274360]
    bad_sources = [  275989, 278750,         274968, 275427, 276468] ## these are sources that should not be fit for one reason or anouther
    
    ### locations of fitted sources
    known_source_locations = {
        275988 :[ -17209.1237384 , 9108.42794861 , 2773.14255942 , 1.17337196519 ],
        278749 :[ -15588.9336476 , 8212.16720085 , 1870.97456308 , 1.20861208031 ],
        280199 :[ -15700.0394701 , 10843.2074395 , 4885.98794791 , 1.22938987942 ],
        274969 :[ -16106.1557939 , 9882.41368417 , 3173.38595461 , 1.16010470252 ],
        275426 :[ -15653.5926819 , 9805.92570475 , 3558.68260696 , 1.16635803567 ],
        276467 :[ -15989.0445204 , 10252.7493955 , 3860.49112931 , 1.17974406565 ],
        274359 :[ -15847.7399188 , 9151.7837243 , 3657.10059838 , 1.1531769444 ],
        274360 :[ -15826.2306908 , 9138.95497711 , 3677.7045732 , 1.1532138795 ],
        285199 :[ -15162.8047845 , 8518.51940209 , 4742.84792164 , 1.33581775529 ],
    }
    
    ### polarization of fitted sources
    known_polarizations = {
        275988 : 3 ,
        278749 : 3 ,
        280199 : 3 ,
        274969 : 3 ,
        275426 : 3 ,
        276467 : 3 ,
        274359 : 0 ,
        274360 : 3 ,
        285199 : 3 ,
    }

    #### SSPW associated with known sources
    known_SSPW_associations = {
        275988 :{
          'CS002': 275988 ,
          'CS003': 24748 ,
          'CS004': 14200 ,
          'CS005': 88756 ,
          'CS006': 57214 ,
          'CS007': 146309 ,
          'CS011': 320165 ,
          'CS013': 242590 ,
          'CS017': 109146 ,
          'CS021': 157424 ,
          'CS030': 254161 ,
          'CS032': 212655 ,
          'CS101': 36666 ,
          'CS103': 177242 ,
          'RS208': 100043 ,
          'CS301': 78487 ,
          'CS302': 46855 ,
          'RS306': 223578 ,
          'RS307': 331331 ,
          'CS401': 234914 ,
          'CS501': 120972 ,
          'RS503': 264891 ,
          'RS509': 131377 ,
        },
        278749 :{
          'CS002': 278749 ,
          'CS003': 27636 ,
          'CS004': 16653 ,
          'CS005': 91465 ,
          'CS006': 59926 ,
          'CS007': 148951 ,
          'CS011': 322779 ,
          'CS013': 245389 ,
          'CS017': 111986 ,
          'CS021': 159903 ,
          'CS030': 256660 ,
          'CS032': 215217 ,
          'CS101': 39074 ,
          'CS103': 179617 ,
          'CS301': 80884 ,
          'CS302': 49269 ,
          'RS307': 333870 ,
          'CS401': 236365 ,
          'CS501': 123427 ,
          'RS503': 267510 ,
          'RS409': 203996 ,
        },
        280199 :{
          'CS002': 280199 ,
          'CS003': 29170 ,
          'CS004': 17941 ,
          'CS005': 92973 ,
          'CS006': 61367 ,
          'CS007': 150387 ,
          'CS011': 324197 ,
          'CS013': 246868 ,
          'CS017': 113547 ,
          'CS021': 161298 ,
          'CS030': 257980 ,
          'CS032': 216549 ,
          'CS101': 40362 ,
          'CS103': 180885 ,
          'CS302': 50561 ,
          'CS401': 237257 ,
          'CS501': 124716 ,
          'RS503': 268846 ,
          'RS307': 335211 ,
          'RS310': 141874 ,
        },
        274969 :{
          'CS002': 274969 ,
          'CS003': 23683 ,
          'CS004': 13255 ,
          'CS005': 87738 ,
          'CS006': 56136 ,
          'CS007': 145284 ,
          'CS011': 319157 ,
          'CS013': 241518 ,
          'CS017': 108091 ,
          'CS021': 156470 ,
          'CS030': 253194 ,
          'CS032': 211661 ,
          'CS101': 35724 ,
          'CS103': 176359 ,
          'RS208': 99056 ,
          'CS302': 45962 ,
          'RS306': 222520 ,
          'RS307': 330365 ,
          'CS401': 234365 ,
          'RS503': 263923 ,
        },
        275426 :{
          'CS002': 275426 ,
          'CS003': 24178 ,
          'CS004': 13711 ,
          'CS005': 88213 ,
          'CS006': 56657 ,
          'CS011': 319648 ,
          'CS013': 242031 ,
          'CS017': 108580 ,
          'CS021': 156912 ,
          'CS030': 253646 ,
          'CS032': 212137 ,
          'CS101': 36177 ,
          'CS103': 176782 ,
          'CS301': 78006 ,
          'CS302': 46387 ,
          'RS307': 330823 ,
          'CS401': 234639 ,
          'CS501': 120471 ,
          'RS503': 264392 ,
          'CS007': 145749 ,
          'RS208': 99549 ,
        },
        276467 :{
          'CS002': 276467 ,
          'CS003': 25287 ,
          'CS004': 14637 ,
          'CS005': 89274 ,
          'CS006': 57673 ,
          'CS007': 146788 ,
          'CS011': 320634 ,
          'CS013': 243118 ,
          'CS017': 109701 ,
          'CS021': 157874 ,
          'CS030': 254609 ,
          'CS032': 213131 ,
          'CS101': 37091 ,
          'CS103': 177678 ,
          'CS301': 78918 ,
          'CS302': 47306 ,
          'CS401': 235180 ,
          'CS501': 121428 ,
#          'RS503': 265345 ,
          'RS208': 100559 ,
          'RS310': 139439 ,
        },
        274359 :{
          'CS002': 274359 ,
          'CS003': 23074 ,
          'CS004': 12701 ,
          'CS005': 87161 ,
          'CS006': 55515 ,
          'CS007': 144697 ,
          'CS011': 318581 ,
          'CS013': 240917 ,
          'CS017': 107475 ,
          'CS021': 155974 ,
          'CS030': 252601 ,
          'CS103': 175823 ,
          'CS301': 77035 ,
          'CS302': 45417 ,
          'RS307': 329828 ,
          'CS401': 234180 ,
          'RS409': 199737 ,
          'RS503': 263303 ,
          'RS508': 309188 ,
          'RS509': 129999 ,
          'CS032': 211073 ,
          'CS101': 35148 ,
        },
        274360 :{
          'CS002': 274360 ,
          'CS004': 12702 ,
          'CS005': 87163 ,
          'CS006': 55516 ,
          'CS017': 107477 ,
          'CS103': 175824 ,
          'CS301': 77037 ,
          'CS302': 45418 ,
          'RS307': 329830 ,
          'RS310': 137879 ,
          'RS508': 309189 ,
        },
    }

    inject_sources = [
        {
        "CS002":[274359],
        "CS003":[23074,23075],
        "CS004":[12701],
        "CS005":[87161,87162],
        "CS006":[55515],
        "CS007":[144697],
        "CS011":[318581,318582],
        "CS013":[240917,240918],
        "CS017":[107475,107476],
        "CS021":[155974],
        "CS030":[252601, 252602],
        "CS032":[211073],
        "CS101":[35148],
        "CS103":[175823],
        "RS208":[98543],
        "CS301":[77035,77036],
        "CS302":[45417],
        "RS306":[],
        "RS307":[329828,329829],
        "RS310":[],
        "CS401":[234180],
        "RS406":[],
        "RS409":[199737],
        "CS501":[119469],
        "RS503":[263303,263304],
        "RS508":[309188],
        "RS509":[129999],
        },
        {
        "CS002":[274360],
        "CS003":[],
        "CS004":[12702],
        "CS005":[87163,87164],
        "CS006":[55516],
        "CS007":[],
        "CS011":[],
        "CS013":[],
        "CS017":[107477],
        "CS021":[],
        "CS030":[],
        "CS032":[],
        "CS101":[35149],
        "CS103":[175824,175825],
        "RS208":[],
        "CS301":[77037],
        "CS302":[45418],
        "RS306":[],
        "RS307":[329830,329831,329832],
        "RS310":[137879],
        "CS401":[],
        "RS406":[185833],
        "RS409":[],
        "CS501":[],
        "RS503":[],
        "RS508":[309189],
        "RS509":[]
        },
#        {
#        "CS002":[274361],
#        "CS003":[],
#        "CS004":[12702],
#        "CS005":[87163,87164],
#        "CS006":[],
#        "CS007":[],
#        "CS011":[],
#        "CS013":[240919],
#        "CS017":[],
#        "CS021":[155975],
#        "CS030":[252603],
#        "CS032":[],
#        "CS101":[35150],
#        "CS103":[175826],
#        "RS208":[98544,98545],
#        "CS301":[],
#        "CS302":[],
#        "RS306":[],
#        "RS307":[329833],
#        "RS310":[137880],
#        "CS401":[137880],
#        "RS406":[],
#        "RS409":[199738,199739],
#        "CS501":[119470],
#        "RS503":[],
#        "RS508":[309191,309192],
#        "RS509":[130000,130001]     
#        }
    ]

    
    ### this is a dictionary that helps control which planewaves are correlated together
    ### Key is unique index of planewaves on referance station
    ### if the value is a list, then the items in the list should be unique_indeces of planewaves on other stations to NOT associate with this one
    planewave_exclusions = {
            275988:[223579, 100044, 131378],
            278749:[133731, 141099, 226374, 226375],#[133731, 141099, 226374, 226375, 333870, 333869],
            280199:[180886, 205404],#[180886,192177,192175,192174,192173,141874,14875, 205404, 124716, 314360, 314361, 134607],
            274969:[],
            275426:[],
            276467:[],
            
            274359:[],
            274360:[],
            
            285199:[185126],
            285276:[66315, 66316]
            }
    
    ## these are SSPW that don't look right, but could be okay. They are treated as if there are mulitple SSPW are viable on that station
    suspicious_SSPW = {
            280199:[141874,141875,  335211, 134607, 314360,314361],
            
            275426:[99548,99549,18919],
            
            276467:[331787,331788,139439],
            
            274359:[98543, 211073, 35148],
            274360:[185833],
            
            285199:[197951, 273340],
            285276:[210417, 273437]
            
            } 
    
    ### some SSPW are actually the same event, due to an error in previous processesing. This says which SSPW should be combined
    SSPW_to_combine = {
            275988: [ [275988,275989], [139072,139073], [131376,131377], [131379,131380] ],
            
            278749: [ [278749,278750], [148950,148951], [333869,333870], [203996,203997] ],
            
            280199: [ [141874,141875], [192173,192174,192175], [314360,314361]],
            
            274969: [ [274969,274968], [87738,87739],   [330364,330365], ],
            
            275426:[ [275426,275427], [24177,24178],  [14578,14579], [99548,99549], [330824,330823]],
            
            276467:[ [276467,276468], [89274,89275], [331787,331788] ],
            
            274359:[ [23074,23075],  [87161,87162], [318581,318582], [240917,240918], [107475,107476], [77035,77036], [263303,263304], 
                    [252601, 252602], [329828,329829]],
                    
            274360:[ [87163,87164], [175824,175825], [329830,329831,329832] ],
            }

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
    
    
    #### read SSPW ####
    print("reading SSPW")
    SSPW_data = read_SSPW_timeID(timeID, SSPW_folder, data_loc="/home/brian/processed_files", min_block=first_block, max_block=first_block+num_blocks, load_timeseries=False)
    SSPW_dict = SSPW_data["SSPW_dict"]
    ant_loc_dict = SSPW_data["ant_locations"]
    
    
    
    #### sort antennas and stations ####
    station_order = list(guess_timings.keys())## note this doesn't include reference station
    sorted_antenna_names = []
    station_to_antenna_index_dict = {}
    
    for sname in station_order + [referance_station]:
        s_id = Sname_to_SId_dict[ sname ]
        first_index = len(sorted_antenna_names)
        
        for ant_name in ant_loc_dict.keys():
            if int(ant_name[:3]) == s_id:
                sorted_antenna_names.append( ant_name )
                
        station_to_antenna_index_dict[sname] = (first_index, len(sorted_antenna_names))
    
    ant_locs = np.zeros( (len(sorted_antenna_names), 3))
    for i, ant_name in enumerate(sorted_antenna_names):
        ant_locs[i] = ant_loc_dict[ant_name]
    
    station_locations = {sname:ant_locs[station_to_antenna_index_dict[sname][0]] for sname in station_order + [referance_station]}
    station_to_antenna_index_list = [station_to_antenna_index_dict[sname] for sname in station_order]
    
    
    #### sort the delays guess, and account for station locations ####
    current_delays_guess = np.array([guess_timings[sname] for sname in station_order])
    if guess_is_from_curtain_plot:
        reference_propagation_delay = np.linalg.norm( guess_location - station_locations[referance_station] )/v_air
        for stat_i, sname in enumerate(station_order):
            propagation_delay =  np.linalg.norm( guess_location - station_locations[sname] )/v_air
            current_delays_guess[ stat_i ] -= propagation_delay - reference_propagation_delay
    original_delays = np.array( current_delays_guess )        
    
    
    
    
    #### open info from part 1 ####
#    input_fname = processed_data_dir + "/" + part1_input + '/out'
#    with open(input_fname, 'rb') as fin:
#        input_SSPW_sort = load(fin)

    input_manager = Part1_input_manager( part1_inputs, inject_sources )



    #### first we fit the known sources ####
    current_sources = []
#    next_source = 0
    for knownID in known_sources:
        
        ref_SSPW_index, viable_SSPW_indeces = input_manager.known_source( knownID )
        
        print("prep fitting:", ref_SSPW_index)
            
        
        ## we have a source to add. Get info.
        source_exclusions = []
        if ref_SSPW_index in planewave_exclusions:
            source_exclusions = planewave_exclusions[ref_SSPW_index]
            
        SSPW_combine_list = []
        if ref_SSPW_index in SSPW_to_combine:
            SSPW_combine_list = SSPW_to_combine[ref_SSPW_index]
            
        suspicious_SSPW_list = []
        if ref_SSPW_index in suspicious_SSPW:
            suspicious_SSPW_list = suspicious_SSPW[ref_SSPW_index]
            
        location = known_source_locations[ref_SSPW_index]
            
            
        ## make source
        source_to_add = source_object(ref_SSPW_index,  viable_SSPW_indeces, source_exclusions, location, SSPW_combine_list, suspicious_SSPW_list )
        current_sources.append( source_to_add )


        polarity = known_polarizations[ref_SSPW_index]
        SSPW_associations = known_SSPW_associations[ref_SSPW_index]

            
        source_to_add.prep_for_fitting_knownFit(polarity, SSPW_associations )
            
#        source_to_add.plot_waveforms( current_delays_guess, logging_folder+'/source_'+str(source_to_add.SSPW.unique_index) )
        source_to_add.plot_selected_waveforms( current_delays_guess, logging_folder+'/sourceSELECT_'+str(source_to_add.SSPW.unique_index) )
        
    if len(current_sources) > 0:
        
        print()
        print("fitting known sources")
        fitter = stochastic_fitter(current_sources)
        stations_with_fits = fitter.get_stations_with_fits()
        
        print()
        fitter.print_locations( current_sources )
        print()
        fitter.print_station_fits( current_sources )
        print()
        fitter.print_delays( original_delays )
        print()
        print()
    else:
        stations_with_fits = [False]*(len(station_order)+1) 
        


    #### loop over adding SSPW ####
    print("attempt to fit sources")
#    stations_with_fits = [False]*(len(station_order)+1) 
    while True:
        
#        ref_SSPW_index, viable_SSPW_indeces = input_SSPW_sort[next_source]
#        next_source += 1
        ref_SSPW_index, viable_SSPW_indeces = input_manager.next()
        
        
        if ref_SSPW_index in bad_sources: 
            ## this is bad source, ignore it
            continue
        
        
        source_exclusions = []
        if ref_SSPW_index in planewave_exclusions:
            source_exclusions = planewave_exclusions[ ref_SSPW_index ]
            
        SSPW_combine_list = []
        if ref_SSPW_index in SSPW_to_combine:
            SSPW_combine_list = SSPW_to_combine[ref_SSPW_index]
            
        suspicious_SSPW_list = []
        if ref_SSPW_index in suspicious_SSPW:
            suspicious_SSPW_list = suspicious_SSPW[ref_SSPW_index]
            
        location = None
        if ref_SSPW_index in known_source_locations:
            location = known_source_locations[ ref_SSPW_index ]
        
        source_to_add = source_object(ref_SSPW_index,  viable_SSPW_indeces, source_exclusions, location, SSPW_combine_list, suspicious_SSPW_list )
        
        
        print("using SSPW", source_to_add.SSPW.unique_index, ":", source_to_add.num_stations_with_unique_viable_SSPW, "with unique SSPW")
            
        ## get initial fit
        current_sources.append( source_to_add )
        
        if source_to_add.SSPW.unique_index in known_polarizations:
            print()
            print("forced using polarity", known_polarizations[source_to_add.SSPW.unique_index])
            
            source_to_add.prep_for_fitting( known_polarizations[source_to_add.SSPW.unique_index] )
            
            source_to_add.plot_waveforms( current_delays_guess, logging_folder+'/source_'+str(source_to_add.SSPW.unique_index) )
#            source_to_add.plot_waveforms_intensity( current_delays_guess, logging_folder+'/sourceSUM_'+str(source_to_add.SSPW.unique_index) )
            fitter = stochastic_fitter(current_sources)
            
        else:
            print()
            print("testing even polarization")
            source_to_add.prep_for_fitting( 0 )
            source_to_add.plot_waveforms( current_delays_guess, logging_folder+'/source_'+str(source_to_add.SSPW.unique_index) )
#            source_to_add.plot_waveforms_intensity( current_delays_guess, logging_folder+'/sourceSUM_'+str(source_to_add.SSPW.unique_index) )
            even_fitter = stochastic_fitter(current_sources)
            while not even_fitter.converged:
                print("not converged, rerun")
                even_fitter.rerun()
            
            print()
            print("testing odd polarization")
            source_to_add.prep_for_fitting( 1 )
            odd_fitter = stochastic_fitter(current_sources)
            while not odd_fitter.converged:
                print("not converged, rerun")
                odd_fitter.rerun()
            
            if even_fitter.RMS < odd_fitter.RMS:
                print("choosing even antennas")
                source_to_add.prep_for_fitting( 0 )
                fitter = even_fitter
            else:
                print("choosing odd antennas")
                source_to_add.prep_for_fitting( 1 )
                fitter = odd_fitter
            
        print()
        fitter.print_locations( current_sources )
        print()
        fitter.print_station_fits( current_sources )
        print()
        fitter.print_delays( original_delays )
        print()
        print()
            
        if ref_SSPW_index not in planewave_exclusions:
            break
        
        ## filter out bad fits
        looping = True
        previous_solution = fitter.solution
        have_changed = False
        while looping:
            have_changed = False
            looping = False
            print()
            
            ## see if need to remove station from source, or discard source altogether
            worst_station = None
            worst_fit = 0.0
            num_stations = 0
            for sname, fit in zip(station_order + [referance_station], fitter.PSE_fits[-1]):
                
                if (fit is not None) and (fit > worst_fit):
                    worst_fit = fit
                    worst_station = sname
                    
                if fit is not None:
                    num_stations += 1
                    
            
            if worst_fit > max_station_RMS:
                looping = True
                have_changed = True
                ## need to remove a station
                print("remove station", worst_station, "from source", source_to_add.SSPW.unique_index)
                source_to_add.remove_station( worst_station )
                    
                
            ## remove source if too few stations
            if num_stations < min_stations:
                print("removing source", source_to_add.SSPW.unique_index)
                have_changed = True
                looping = False
                current_sources.pop( )
                previous_solution = None
                
            if have_changed:
                
                fitter = stochastic_fitter(current_sources, previous_solution)        
                while not fitter.converged:
                    print("not converged, rerun")
                    fitter.rerun()
                
                fitter.print_station_fits( current_sources )
            
        if have_changed:
            print()
            fitter.print_locations( current_sources )
            print()
            fitter.print_delays( original_delays )
            print()
            print()
        else:
            print("no change: continueing")
        
        current_delays_guess = fitter.employ_result( current_sources )
        
        
        previous_stations_with_fits = stations_with_fits
        stations_with_fits = fitter.get_stations_with_fits()
        print("testing added SSPW for source", source_to_add.SSPW.unique_index)
        for stat_i, (sname,fit) in enumerate( zip( chain(station_order, [referance_station]), stations_with_fits) ):
            if not fit:
                continue
            
            if not source_to_add.has_station(sname):
                print()
                print("  testing", sname)
                
                best_SSPW = None
                best_fitter = None
                best_fit = np.inf
                for SSPW in source_to_add.viable_SSPW[sname]:
                    num_ant_added = source_to_add.add_SSPW( SSPW )
                    if num_ant_added<3:
                        print("   ", SSPW.unique_index, '- not enough antennas')
                        continue
                    
                    tmp_fitter = stochastic_fitter(current_sources) 
                    new_fit = tmp_fitter.PSE_fits[-1][stat_i]
                    print("   ", SSPW.unique_index, ':', new_fit)
                    
                    if new_fit<best_fit:
                        best_fit = new_fit
                        best_fitter = tmp_fitter
                        best_SSPW = SSPW
                        
                if best_fit < max_station_RMS:
                    print("  choosing", SSPW.unique_index)
                    source_to_add.add_SSPW( best_SSPW )
                    fitter = best_fitter
                    print()
                    fitter.print_station_fits( current_sources )
                    print()
                    print()
                    current_delays_guess = fitter.employ_result( current_sources )
                else:
                    print("  no SSPW choosen")
                    source_to_add.remove_station( sname )
                    
        print()
        print("ploting selected event waveforms")
        source_to_add.plot_selected_waveforms( current_delays_guess, logging_folder+'/sourceSELECT_'+str(source_to_add.SSPW.unique_index) )
                    
        
        print()
        print()
        print("testing adding SSPW to old events for new stations")
        for stat_i, (sname,fit,previous_fit) in enumerate( zip( chain(station_order, [referance_station]), stations_with_fits, previous_stations_with_fits) ):
            if not fit: ## at least one station needs to have this sname
                continue
            if previous_fit: ## the previous sources can have this SSPW
                continue
            
            print()
            print("  testing", sname)
            for source_i, source in enumerate(current_sources[:-1]): ## loop over all older sources
                print("    source", source.SSPW.unique_index)
                           
                best_SSPW = None
                best_fitter = None
                best_fit = np.inf
                
                for SSPW in source.viable_SSPW[sname]:
                    num_ant_added = source.add_SSPW( SSPW )
                    if num_ant_added<3:
                        print("     ", SSPW.unique_index, '- not enough antennas')
                        continue
                    
                    tmp_fitter = stochastic_fitter(current_sources) 
                    new_fit = tmp_fitter.PSE_fits[source_i][stat_i]
                    print("     ", SSPW.unique_index, ':', new_fit)
                    
                    if new_fit<best_fit:
                        best_fit = new_fit
                        best_fitter = tmp_fitter
                        best_SSPW = SSPW
                        
                        
                if best_fit < max_station_RMS:
                    print("  choosing", best_SSPW.unique_index)
                    source.add_SSPW( best_SSPW )
                    fitter = best_fitter
                    print()
                    fitter.print_station_fits( current_sources )
                    print()
                    print()
                    current_delays_guess = fitter.employ_result( current_sources )
                else:
                    print("  no SSPW choosen")
                    source.remove_station( sname )
        print()
        print()
        
        print("RESULTS:")
        fitter.print_delays( original_delays )       
                    
        print()
        print()
        print("locations")
        for source in current_sources:
            print(source.SSPW.unique_index,':[', source.guess_XYZT[0], ',', source.guess_XYZT[1], ',', source.guess_XYZT[2], ',', source.guess_XYZT[3], '],')
            
        print()
        print("polarizations")
        for source in current_sources:
            print(source.SSPW.unique_index,':', source.polarization, ',')
            
        print()
        print("used SSPW")
        for source in current_sources:
            print(source.SSPW.unique_index,':{')
            for sname,SSPW_ID in source.SSPW_in_use.items():
                print("  '"+sname+"':",SSPW_ID,',')
            print('},')
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    