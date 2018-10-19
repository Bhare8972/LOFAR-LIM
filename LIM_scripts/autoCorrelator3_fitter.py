#!/usr/bin/env python3


##### THIS NEEDS TO BE FIXED. THROW OUT STATIONS ONLY WITH REALLY BAD RMS (100 ns)
#### THROW OUT EVENTS WITH BAD RMS (5ns or 2ns depending)


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
#from RunningStat import RunningStat


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

    def __init__(self, ID,  input_fname, exclusions, location=None, suspicious_station_list=[] ):
        self.ID = ID
        self.suspicious_station_list = suspicious_station_list
        self.exclusions = exclusions
        
        
        self.data_file = h5py.File(input_fname, "r")
        
        if location is None:
            guess_time = np.inf
            ref_stat = None
            for sname, stat_data in self.data_file.items():
                for ant_data in stat_data.values():
                    T = ant_data.attrs['PolE_peakTime']
                    if T<guess_time:
                        guess_time = T
                        ref_stat = sname
            
            guess_time = guess_time - np.linalg.norm( station_locations[ref_stat]-guess_location )/v_air
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
        for sname in chain(station_order, [referance_station]):
            if sname not in chain(self.exclusions, self.suspicious_station_list):
                num_ant = self.add_known_station(sname)
                if num_ant < min_antenna_per_station:
                    self.remove_station( sname )
                
                
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
                
                if self.polarization != 3:
                    pt = ant_data.attrs['PolE_peakTime'] if self.polarization==0 else ant_data.attrs['PolO_peakTime']
                    waveform = ant_data[1,:] if self.polarization==0 else ant_data[3,:]
                    start_time += ant_data.attrs['PolE_timeOffset'] if self.polarization==0 else ant_data.attrs['PolO_timeOffset']
                    amp = np.max(waveform)
                    
                    if not np.isfinite(pt):
                        pt = np.nan
                    if amp<min_antenna_amplitude:
                        pt = np.nan
                else:
                    PolE_data = ant_data[0,:]
                    PolO_data = ant_data[3,:]
                
                    if np.max( PolE_data ) < min_antenna_amplitude or np.max( PolO_data ) < min_antenna_amplitude:
                        pt = np.nan
                        waveform = None
                    else:
                        ant_loc = ant_locs[ ant_i ]
                        
                        radius, zenith, azimuth = get_radius_ze_az( self.guess_XYZT[:3]-ant_loc )
                
                        antenna_calibrator.FFT_prep(ant_name, PolE_data, PolO_data)
                        antenna_calibrator.apply_time_shift(0.0, ant_data.attrs['PolO_timeOffset']-ant_data.attrs['PolE_timeOffset'])
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
                            pt = (peak_finder.peak_index + ant_data.attrs['starting_index'])*5.0E-9 + ant_data.attrs['PolE_timeOffset']
                            start_time += ant_data.attrs['PolE_timeOffset']
                
            
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
        Z = np.abs(Z)
        
        out[:] = ant_locs[:,0]
        out[:] -= X
        out *= out
        
        delta_sq = ant_locs[:,1]-Y
        delta_sq *= delta_sq
        out += delta_sq
        
        delta_sq[:] = ant_locs[:,2]-Z
        delta_sq *= delta_sq
        out += delta_sq
        
        np.sqrt( out, out=out )
        out *= 1.0/v_air
        
        out += T
        out -=self.pulse_times
        
        ##now account for delays
        for index_range, delay in zip(station_to_antenna_index_list,  delays):
            first,last = index_range
            
            out[first:last] += delay ##note the wierd sign
    
    def try_location_JAC2(self, delays, XYZT_location, out_loc, out_delays):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
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
        
        out_loc[:,0] *= 1.0/v_air
        out_loc[:,1] *= 1.0/v_air
        out_loc[:,2] *= 1.0/v_air
         
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
        self.input_files = input_files
        
        self.input_data = []
        current_max = 0
        for folder in input_files:
            input_folder = processed_data_dir + "/" + folder +'/'
            
            file_list = [(int(f.split('_')[1][:-3])+current_max ,input_folder+f) for f in listdir(input_folder) if f.endswith('.h5')] ## get all file names, and get the 'ID' for the file name
            file_list.sort( key=lambda x: x[0] ) ## sort according to ID
            self.input_data.append( file_list ) 
            
            current_max = max(file_list, key=lambda x: x[0])[0]
                
        self.indeces = np.zeros( len(self.input_data), dtype=int )
        
    def known_source(self, ID):
        
        for current_i, index in enumerate(self.indeces):
            if index == len(self.input_data[ current_i ]):
                continue
            
            source_ID, file_name = self.input_data[ current_i ][ index ]
            
            if source_ID in bad_sources:
                self.indeces[ current_i ] += 1
                return self.known_source( ID )
            
            if source_ID==ID:
                self.indeces[ current_i ] += 1
                return source_ID, file_name
            
        return None
        
    def next(self):
        #### if we are here, no sources are well known. check for sources that are currently being fitted
        best_file_i = 0
        for current_i, index in enumerate(self.indeces):
            if index == len(self.input_data[ current_i ]):
                continue
            
            source_ID, file_name = self.input_data[ current_i ][ index ]
            
            if source_ID in station_exclusions:
                self.indeces[ current_i ] += 1
                return source_ID, file_name
            
            if index < self.indeces[best_file_i]:
                best_file_i = current_i
                
                
        ret = self.input_data[best_file_i][ self.indeces[best_file_i] ]
        self.indeces[best_file_i] += 1
        return ret
            


np.set_printoptions(precision=10, threshold=np.inf)
if __name__ == "__main__": 
    
    #### TODO: make code that looks for pulses on all antennas, ala analyze amplitudes
    ## probably need a seperate code that just fits location of one source, so can be tuned by hand, then fed into main code
    ## prehaps just add one station at a time? (one closest to known stations) seems to be general stratigey
    
    timeID = "D20170929T202255.000Z"
    output_folder = "autoCorrelator3_fitter_fromIPSE"
    
#    part1_input = "autoCorrelator_Part1"
    part1_inputs = ["autoCorrelator3_fromIPSE"]
    
    
    #### fitter settings ####
    max_itters_per_loop = 2000
    itters_till_convergence = 100
    max_jitter_width = 100000E-9
    min_jitter_width = 1E-9
    cooldown = 10.0 ## 10.0
    strong_cooldown = 100.0
    
    ## quality control
    min_antenna_amplitude = 10
    max_station_RMS = 5.0E-9
    min_stations = 4
    min_antenna_per_station = 4
    
    #### initial guesses ####
    referance_station = "CS002"
    guess_location = np.array( [1.72389621e+04,   9.50496918e+03, 2.37800915e+03] )
    
    guess_timings = {
'CS002':   0.0,
'CS003':   1.40494600712e-06,
'CS004':   4.31090399655e-07,
'CS005':   -2.19120784288e-07,
'CS006':   4.33554497556e-07,
'CS007':   3.99902678123e-07,
'CS011':   -5.8560385891e-07,
'CS013':   -1.81325669666e-06, 
'CS017':   -8.43878144671e-06,
'CS021':   9.24163752585e-07,
'CS030':   -2.74037597057e-06,
'CS032':   -1.57466749415e-06,
'CS101':   -8.16817259789e-06,
'CS103':   -2.8518179599e-05,
'RS208':   6.94747112635e-06,
'CS301':   -7.18344260299e-07,
'CS302':   -5.35619408893e-06,
'RS306':   7.02305935406e-06,
'RS307':   6.92860064373e-06,
'RS310':   7.02115632779e-06,
'CS401':   -9.52017247995e-07,
'RS406':   7.02462236919e-06,
'RS409':   7.03940588106e-06,
'CS501':   -9.6076184928e-06,
'RS503':   6.94774095106e-06,
'RS508':   7.06896423643e-06,
'RS509':   7.11591455679e-06
        }
        
    guess_is_from_curtain_plot = False ## if true, must take station locations into account in order to get true offsets
    
    if referance_station in guess_timings:
        del guess_timings[referance_station]
    
    
    #### these are sources whose location have been fitted, and stations are associated
#    known_sources = [0,1,2,4,5,6]
#    bad_sources = [3,7,8 ] ## these are sources that should not be fit for one reason or anouther
    known_sources = [0,1,2,3,4,5,6,7,8,9,10,11]
    bad_sources = [ ] ## these are sources that should not be fit for one reason or anouther
    
    ### locations of fitted sources
    known_source_locations = {
0 :[ -17886.9863488 , 9934.26940962 , 3029.84742808 , 1.26907124207 ],
1 :[ -17902.307183 , 9933.52685635 , 3025.87334528 , 1.26920653245 ],
2 :[ -17768.264943 , 10031.4353928 , 2740.7739126 , 1.26909386199 ],
3 :[ -17562.9207211 , 9109.65962727 , 2187.46943922 , 1.1753118302 ],
4 :[ -16712.3179488 , 10030.4914577 , 3005.19197644 , 1.25997870982 ],
5 :[ -15431.2200327 , 8839.41661906 , 2712.1452132 , 1.18001625539 ],
6 :[ -15839.9011294 , 9763.49982414 , 3127.08221584 , 1.16684101626 ],
7 :[ -16347.6759968 , 9363.48942884 , 3420.47148857 , 1.17120459855 ],
8 :[ -16477.0942466 , 9519.54616374 , 3347.33132217 , 1.17262310872 ],
9 :[ -16426.6048681 , 9984.05807972 , 2182.64632747 , 1.26032082112 ],
10 :[ -16413.8163908 , 9843.94008507 , 2638.5240372 , 1.25761899141 ],
11 :[ -16016.9601222 , 10659.0291436 , 3685.77797706 , 1.19685566371 ],

    }
    
    ### polarization of fitted sources
    known_polarizations = {
#        0 : 1 ,
#        1 : 0 ,
#        2 : 1 ,
#        4 : 1 ,
#        5 : 1 ,
#        6 : 1 ,
    }
    
    known_polarizations = {i:0 for i in range(40)}## to force one pol

    ### this is a dictionary that helps control which stations are correlated together
    ### Key is unique index of planewaves on referance station
    ### if the value is a list, then the items in the list should be unique_indeces of planewaves on other stations to NOT associate with this one
    station_exclusions = {
            }
    
    ## these are stations that don't look right, but could be okay. They are treated as if there are mulitple stations are viable on that station
    suspicious_stations = {
            0:['RS306', 'RS307', 'RS310', 'RS406'],
            1:['RS306', 'RS307', 'RS406'],
            2:['RS306', 'RS406', 'RS409', "RS508"],
            3:['CS007', 'CS032', 'CS101', 'CS103', 'CS302', 'RS310', 'CS401', 'RS406', 'RS508', 'RS509'],
            4:['RS509'],
            5:['CS030', 'RS306', 'RS406', 'RS503', 'RS508'],
            6:['CS032', 'RS306', 'RS406', 'RS409', 'RS509'],
            7:['RS406', 'RS409'],
            8:['RS307', 'RS406', 'RS409', 'RS509'],
            9:['CS032', 'RS306', 'RS307', 'RS310', 'RS508'],
            10:['RS307', 'RS406','RS409','RS508'],
            11:['CS007', 'RS406', 'RS409', 'RS508']
            }
    
    stations_to_retest = ['RS406']
    
    ### ADD RECOVER STATIONS, stations to use location to recover data from file after fitting


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
    if guess_is_from_curtain_plot:
        reference_propagation_delay = np.linalg.norm( guess_location - station_locations[referance_station] )/v_air
        for stat_i, sname in enumerate(station_order):
            propagation_delay =  np.linalg.norm( guess_location - station_locations[sname] )/v_air
            current_delays_guess[ stat_i ] -= propagation_delay - reference_propagation_delay
    original_delays = np.array( current_delays_guess )        
    
    
    
    
    #### open info from part 1 ####

    input_manager = Part1_input_manager( part1_inputs )



    #### first we fit the known sources ####
    current_sources = []
#    next_source = 0
    for knownID in known_sources:
        
        source_ID, input_name = input_manager.known_source( knownID )
        
        print("prep fitting:", source_ID)
            
        
        ## we have a source to add. Get info.
        source_exclusions = []
        if source_ID in station_exclusions:
            source_exclusions = station_exclusions[source_ID]
            
        suspicious_stat_list = []
        if source_ID in suspicious_stations:
            suspicious_stat_list = suspicious_stations[source_ID]
            
        location = known_source_locations[source_ID]
            
            
        ## make source
        source_to_add = source_object(source_ID,  input_name, source_exclusions, location, suspicious_stat_list )
        current_sources.append( source_to_add )


        polarity = known_polarizations[source_ID]

            
        source_to_add.prep_for_fitting(polarity)
            
        source_to_add.plot_selected_waveforms( current_delays_guess, logging_folder+'/sourceSELECT_'+str(source_to_add.ID) )
        
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
        
        
        
    ### try to re-test stations, that is, stations that we think the wrong SSPW are fitted
    print("NOW RE-TESTING STATIONS")
    for sname in stations_to_retest:
        print()
        print("  RETESTING", sname)
        stat_i = len(station_order) if sname==referance_station else station_order.index(sname)
        
        ## first, catalog current situation
        best_sources_with_fit = []
        best_fitter = fitter
        for source_i, source in enumerate(current_sources):
            fit = fitter.PSE_fits[source_i][stat_i]
            if fit is not None and fit < max_station_RMS:
                best_sources_with_fit.append( source_i )
                
        for source_i, sourceA in enumerate(current_sources): ## choose a source as our primary fit
            print()
            print("    testing source", sourceA.ID)
            ### first, remove this station from all sources
            for sourceB in current_sources:
                sourceB.remove_station(sname)
                
            ## now only add station back to main test source
            num_ant = sourceA.add_known_station(sname)
            if num_ant < min_antenna_per_station:
                print("    not enough antennas")
                sourceA.remove_station(sname)
                continue
            
            tmp_fitter = stochastic_fitter(current_sources, quick_kill=5.0*fitter.RMS)

            if tmp_fitter.PSE_fits[source_i][stat_i] > max_station_RMS:
                print("      no good! RMS:", tmp_fitter.PSE_fits[source_i][stat_i])
                sourceA.remove_station(sname)
                continue
                 
            print("      good! (for now):", tmp_fitter.PSE_fits[source_i][stat_i])
            
            # now we attempt to add in more sources
            current_sources_with_fit = [source_i]
            for source_iB, sourceB in enumerate(current_sources[source_i+1:], start=source_i+1): ## ignore previous sources to avoid dbl counting
                print("      attempting to add:", sourceB.ID)
                
                num_ant = sourceB.add_known_station(sname)
                if num_ant < min_antenna_per_station:
                    print("     not enough antennas")
                    sourceB.remove_station(sname)
                    continue
            
                tmp_fitter2 = stochastic_fitter(current_sources, initial_guess=tmp_fitter.solution, quick_kill=5.0*tmp_fitter.RMS)

                if tmp_fitter2.PSE_fits[source_iB][stat_i] < max_station_RMS:
                    print("       good! adding source. RMS:", tmp_fitter2.PSE_fits[source_iB][stat_i])
                    current_sources_with_fit.append( source_iB )
                    tmp_fitter = tmp_fitter2
                else:
                    print("       no good! RMS:", tmp_fitter2.PSE_fits[source_iB][stat_i])
                    sourceB.remove_station(sname)
                    
            ## now we test how good we are!
            if len(current_sources_with_fit) > len(best_sources_with_fit):
                print("      this is best configuration so far")
                best_fitter = tmp_fitter
                best_sources_with_fit = current_sources_with_fit
                best_fitter.print_station_fits( current_sources )
                
        ### now implement best fit
        for source in current_sources:
            source.remove_station(sname)
            
        for source_i in best_sources_with_fit:
            current_sources[source_i].add_known_station(sname)
            
        fitter = best_fitter
        stations_with_fits = fitter.get_stations_with_fits()
        

    #### loop over adding stations ####
    print("attempt to fit sources")
    while True:
        
        source_ID, input_name = input_manager.next()
        
        
        if source_ID in bad_sources: 
            ## this is bad source, ignore it
            continue
        
        
        source_exclusions = []
        if source_ID in station_exclusions:
            source_exclusions = station_exclusions[ source_ID ]
            
        suspicious_stat_list = []
        if source_ID in suspicious_stations:
            suspicious_stat_list = suspicious_stations[source_ID]
            
        location = None
        if source_ID in known_source_locations:
            location = known_source_locations[ source_ID ]
        
        source_to_add = source_object(source_ID,  input_name, source_exclusions, location, suspicious_stat_list )
        
        print()
        print("using source", source_to_add.ID)
            
        ## get initial fit
        current_sources.append( source_to_add )
        
        if source_to_add.ID in known_polarizations:
            print()
            print("forced using polarity", known_polarizations[source_to_add.ID])
            
            source_to_add.prep_for_fitting( known_polarizations[source_to_add.ID] )
            
            source_to_add.plot_waveforms( current_delays_guess, logging_folder+'/source_'+str(source_to_add.ID) )
            fitter = stochastic_fitter(current_sources)
            
        else:
            print()
            print("testing even polarization")
            source_to_add.prep_for_fitting( 0 )
            source_to_add.plot_waveforms( current_delays_guess, logging_folder+'/source_'+str(source_to_add.ID) )
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
            
#        if source_ID not in station_exclusions:
#            break
        
        ## filter out bad fits
        looping = True
        previous_solution = fitter.solution
        have_changed = False
        removed_source = False
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
                print("remove station", worst_station, "from source", source_to_add.ID)
                source_to_add.remove_station( worst_station )
                    
                
            ## remove source if too few stations
            if num_stations < min_stations:
                print("removing source", source_to_add.ID)
                have_changed = True
                looping = False
                current_sources.pop( )
                previous_solution = None
                removed_source = True
                
            if have_changed:
                
                fitter = stochastic_fitter(current_sources, previous_solution, quick_kill=1.0E-7)        
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
        
        if removed_source:
            continue
        
        
        previous_stations_with_fits = stations_with_fits
        stations_with_fits = fitter.get_stations_with_fits()
        print("testing added stations for source", source_to_add.ID)
        for stat_i, (sname,fit) in enumerate( zip( chain(station_order, [referance_station]), stations_with_fits) ):
            if not fit:
                continue
            
            if not source_to_add.has_station(sname):
                print()
                print("  testing", sname)
                
                num_ant_added = source_to_add.add_known_station( sname )
                if num_ant_added < min_antenna_per_station:
                    source_to_add.remove_station( sname )
                    print("   - not enough antennas")
                    continue
                
                tmp_fitter = stochastic_fitter(current_sources, quick_kill = fitter.RMS*5) 
                new_fit = tmp_fitter.PSE_fits[-1][stat_i]
                print("    fit:", new_fit)
                        
                if new_fit < max_station_RMS:
                    print("  adding station")
                    fitter = tmp_fitter
                    print()
                    fitter.print_station_fits( current_sources )
                    print()
                    print()
                    current_delays_guess = fitter.employ_result( current_sources )
                else:
                    print("  no stations choosen")
                    source_to_add.remove_station( sname )
                    
        print()
        print("ploting selected event waveforms")
        source_to_add.plot_selected_waveforms( current_delays_guess, logging_folder+'/sourceSELECT_'+str(source_to_add.ID) )
                    
        
        print()
        print()
        print("testing adding stations to old events for new stations")
        for stat_i, (sname,fit,previous_fit) in enumerate( zip( chain(station_order, [referance_station]), stations_with_fits, previous_stations_with_fits) ):
            if not fit: ## at least one station needs to have this sname
                continue
            if previous_fit: ## the previous sources can have this stations
                continue
            
            print()
            print("  testing", sname)
            for source_i, source in enumerate(current_sources[:-1]): ## loop over all older sources
                print("    source", source.ID)
                
                num_ant_added = source.add_known_station( sname )
                if num_ant_added < min_antenna_per_station:
                    print("      not enough antennas")
                    source.remove_station( sname )
                    continue
                
                tmp_fitter = stochastic_fitter(current_sources, quick_kill = fitter.RMS*5) 
                new_fit = tmp_fitter.PSE_fits[source_i][stat_i]
                print("      fit:", new_fit)
                        
                if new_fit < max_station_RMS:
                    print("      adding station")
                    fitter = tmp_fitter
                    print()
                    fitter.print_station_fits( current_sources )
                    print()
                    print()
                    current_delays_guess = fitter.employ_result( current_sources )
                else:
                    print("      no stations choosen")
                    source.remove_station( sname )
        print()
        print()
        
        print("RESULTS:")
        fitter.print_station_fits( current_sources )
        
        print()
        print()
        fitter.print_delays( original_delays )       
                    
        print()
        print()
        print("locations")
        for source in current_sources:
            print(source.ID,':[', source.guess_XYZT[0], ',', source.guess_XYZT[1], ',', source.guess_XYZT[2], ',', source.guess_XYZT[3], '],')
            
        print()
        print("polarizations")
        for source in current_sources:
            print(source.ID,':', source.polarization, ',')
            
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    