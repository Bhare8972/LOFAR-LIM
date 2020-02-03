#!/usr/bin/env python3

#python
import time
from os import mkdir, listdir
from os.path import isdir, isfile
from itertools import chain
from pickle import load
from random import choice

#external
import numpy as np
from scipy.optimize import least_squares, minimize, approx_fprime
from scipy.signal import hilbert
from matplotlib import pyplot as plt


import h5py


#mine
from LoLIM.prettytable import PrettyTable
from LoLIM.utilities import logger, processed_data_dir, v_air, SId_to_Sname, Sname_to_SId_dict, RTD
#from LoLIM.IO.binary_IO import read_long, write_long, write_double_array, write_string, write_double
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
            
#        self.one_fitter = stationDelay_fitter(ant_locs, self.station_indeces, 1, self.num_delays)
#        self.one_fitter.set_event( self.source_object_list[0] )
        
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
            
        self.fitter.prep_for_random_pert()
        

            
            
    def rerun(self, deviation, antenna_error_deviation):
        
        self.fitter.random_perturbation( deviation, antenna_error_deviation )
        
        new_guess = np.array(self.solution)
        new_guess[:self.num_delays] += np.random.normal(scale=100E-9, size=self.num_delays)
        new_guess[self.num_delays+3::4] += np.random.normal(scale=100E-9, size=len(self.source_object_list))
        
        new_guess[self.num_delays::4] = 0.0
        new_guess[self.num_delays+1::4] = 0.0
        new_guess[self.num_delays+2::4] = 0.0
        
        fit_res = least_squares(self.fitter.objective_fun, self.solution, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')

        self.last_fit = fit_res.x

        new_station_delays = fit_res.x[:self.num_delays]

        total_RMS = self.fitter.RMS(fit_res.x, self.num_DOF)
        
        return new_station_delays, total_RMS
    
    def employ_result(self, source_object_list):
        
        param_i = self.num_delays
        for PSE in source_object_list:
            PSE.append_solution( self.last_fit[param_i:param_i+4] )
            param_i += 4
            
class stochastic_fitter_dt_loc:
    def __init__(self, source_object_list):
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
            
#        self.one_fitter = stationDelay_fitter(ant_locs, self.station_indeces, 1, self.num_delays)
#        self.one_fitter.set_event( self.source_object_list[0] )
        
        #### make guess ####
        self.num_DOF = -self.num_delays
        self.solution = np.zeros(  4*len(source_object_list) )
        param_i = 0
        for PSE in source_object_list:
            self.solution[param_i:param_i+4] = PSE.guess_XYZT
            param_i += 4
            self.num_DOF += PSE.num_DOF()
            
        self.fitter.prep_for_random_pert()
        
        self.tmp_array = np.zeros(  self.num_delays+4*len(source_object_list) )
        self.current_delays = current_delays_guess   
        
    def obj_func(self, vals):
        self.tmp_array[:self.num_delays] = self.current_delays
        self.tmp_array[self.num_delays:] = vals
        return self.fitter.objective_fun( self.tmp_array )
        
    def rerun(self, station_delays, deviation, antenna_error_deviation):
        
        self.fitter.random_perturbation( deviation, antenna_error_deviation )
        self.current_delays = station_delays
        
        new_guess = np.array(self.solution)
        new_guess[3::4] += np.random.normal(scale=100E-9, size=len(self.source_object_list))
        
        new_guess[::4] = 0.0
        new_guess[1::4] = 0.0
        new_guess[2::4] = 0.0
        
        fit_res = least_squares(self.obj_func, self.solution, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')

        self.last_fit = fit_res.x

        self.tmp_array[:self.num_delays] = self.current_delays
        self.tmp_array[self.num_delays:] = fit_res.x
        total_RMS = self.fitter.RMS(self.tmp_array, self.num_DOF)
        
        return total_RMS
    
    def employ_result(self, source_object_list):
        
        param_i = 0
        for PSE in source_object_list:
            PSE.append_solution( self.last_fit[param_i:param_i+4] )
            param_i += 4

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

    def __init__(self, ID,  input_fname, location, stations_to_exclude, antennas_to_exclude, num_runs ):
        self.ID = ID
        self.stations_to_exclude = stations_to_exclude
        self.antennas_to_exclude = antennas_to_exclude
        self.data_file = h5py.File(input_fname, "r")
        self.guess_XYZT = np.array( location )

        self.solutions = np.empty( (num_runs, 4), dtype=np.double )
        self.sol_i = 0
                 
            
    def prep_for_fitting(self, polarization, station_delay_guess_dict):
        self.polarization = polarization
        
        self.pulse_times = np.empty( len(sorted_antenna_names) )
        self.pulse_times[:] = np.nan
        
        #### first add times from referance_station
        for sname in chain(station_order):
            if sname not in self.stations_to_exclude:
                self.add_known_station(sname, station_delay_guess_dict[sname])
        if referance_station not in self.stations_to_exclude:
            self.add_known_station(referance_station, 0.0)
                
                
        #### setup some temp storage for fitting
        self.tmp_LS2_data = np.empty( len(sorted_antenna_names) )
                
                
    def remove_station(self, sname):
            
        antenna_index_range = station_to_antenna_index_dict[sname]
        self.pulse_times[ antenna_index_range[0]:antenna_index_range[1] ] = np.nan
#        
#    def has_station(self, sname):
#        antenna_index_range = station_to_antenna_index_dict[sname]
#        return np.sum(np.isfinite(self.pulse_times[ antenna_index_range[0]:antenna_index_range[1] ])) > 0
         
    def add_known_station(self, sname, delay):
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
                
#                start_time = ant_data.attrs['starting_index']*5.0E-9
                
                pt = ant_data.attrs['PolE_peakTime'] if self.polarization==0 else ant_data.attrs['PolO_peakTime']
                waveform = ant_data[1,:] if self.polarization==0 else ant_data[3,:]
#                start_time += ant_data.attrs['PolE_timeOffset'] if self.polarization==0 else ant_data.attrs['PolO_timeOffset']
                amp = np.max(waveform)
                
                if not np.isfinite(pt):
                    pt = np.nan
                if amp<min_antenna_amplitude or (ant_name in self.antennas_to_exclude) or (ant_name in bad_antennas):
                    pt = np.nan
                        
                self.pulse_times[ ant_i ] = np.linalg.norm( ant_locs[ant_i] - self.guess_XYZT[0:3] )*inv_v_air + self.guess_XYZT[3] + delay
                
        return np.sum(np.isfinite( self.pulse_times[antenna_index_range[0]:antenna_index_range[1]] ) )        
    
    
    def num_DOF(self):
        return np.sum( np.isfinite(self.pulse_times) ) - 3 ## minus three or four?
    
    def append_solution(self, new_solution):
        self.solutions[self.sol_i, : ] = new_solution
        self.sol_i += 1
        
#    def try_location_LS(self, delays, XYZT_location, out):
#        X,Y,Z,T = XYZT_location
#        Z = np.abs(Z)
#        
#        delta_X_sq = ant_locs[:,0]-X
#        delta_Y_sq = ant_locs[:,1]-Y
#        delta_Z_sq = ant_locs[:,2]-Z
#        
#        delta_X_sq *= delta_X_sq
#        delta_Y_sq *= delta_Y_sq
#        delta_Z_sq *= delta_Z_sq
#        
#            
#        out[:] = T - self.pulse_times
#        
#        ##now account for delays
#        for index_range, delay in zip(station_to_antenna_index_list,  delays):
#            first,last = index_range
#            
#            out[first:last] += delay ##note the wierd sign
#                
#                
#        out *= v_air
#        out *= out ##this is now delta_t^2 *C^2
#        
#        out -= delta_X_sq
#        out -= delta_Y_sq
#        out -= delta_Z_sq
    
#    def try_location_JAC(self, delays, XYZT_location, out_loc, out_delays):
#        X,Y,Z,T = XYZT_location
#        Z = np.abs(Z)
#        
#        out_loc[:,0] = X
#        out_loc[:,0] -= ant_locs[:,0]
#        out_loc[:,0] *= -2
#        
#        out_loc[:,1] = Y
#        out_loc[:,1] -= ant_locs[:,1]
#        out_loc[:,1] *= -2
#        
#        out_loc[:,2] = Z
#        out_loc[:,2] -= ant_locs[:,2]
#        out_loc[:,2] *= -2
#        
#        
#        out_loc[:,3] = T - self.pulse_times
#        out_loc[:,3] *= 2*v_air*v_air
#        
#        delay_i = 0
#        for index_range, delay in zip(station_to_antenna_index_list,  delays):
#            first,last = index_range
#            
#            out_loc[first:last,3] += delay*2*v_air*v_air
#            out_delays[first:last,delay_i] = out_loc[first:last,3]
#                
#            delay_i += 1
#            
#    def try_location_LS2(self, delays, XYZT_location, out):
#        X,Y,Z,T = XYZT_location
##        Z = np.abs(Z)
#        
#        self.tmp_LS2_data[:] = ant_locs[:,0]
#        self.tmp_LS2_data[:] -= X
#        self.tmp_LS2_data[:] *= self.tmp_LS2_data[:]
#        out[:] = self.tmp_LS2_data
#        
#        self.tmp_LS2_data[:] = ant_locs[:,1]
#        self.tmp_LS2_data[:] -= Y
#        self.tmp_LS2_data[:] *= self.tmp_LS2_data[:]
#        out[:] += self.tmp_LS2_data
#        
#        self.tmp_LS2_data[:] = ant_locs[:,2]
#        self.tmp_LS2_data[:] -= Z
#        self.tmp_LS2_data[:] *= self.tmp_LS2_data[:]
#        out[:] += self.tmp_LS2_data
#        
#        np.sqrt( out, out=out )
#        out *= inv_v_air
#        
#        out += T
#        out -= self.pulse_times
#        
#        ##now account for delays
#        for index_range, delay in zip(station_to_antenna_index_list,  delays):
#            first,last = index_range
#            
#            out[first:last] += delay ##note the wierd sign
#    
#    def try_location_JAC2(self, delays, XYZT_location, out_loc, out_delays):
#        X,Y,Z,T = XYZT_location
##        Z = np.abs(Z)
#        
#        out_loc[:,0] = X
#        out_loc[:,0] -= ant_locs[:,0]
#        
#        out_loc[:,1] = Y
#        out_loc[:,1] -= ant_locs[:,1]
#        
#        out_loc[:,2] = Z
#        out_loc[:,2] -= ant_locs[:,2]
#        
#        
#        out_delays[:,0] = out_loc[:,0] ## use as temporary storage
#        out_delays[:,0] *= out_delays[:,0]
#        out_loc[:,3] = out_delays[:,0] ## also use as temporary storage
#         
#        out_delays[:,0] = out_loc[:,1] ## use as temporary storage
#        out_delays[:,0] *= out_delays[:,0]
#        out_loc[:,3] += out_delays[:,0]
#        
#        out_delays[:,0] = out_loc[:,2] ## use as temporary storage
#        out_delays[:,0] *= out_delays[:,0]
#        out_loc[:,3] += out_delays[:,0]
#        
#        np.sqrt( out_loc[:,3], out = out_loc[:,3] )
#         
#        out_loc[:,0] /= out_loc[:,3]
#        out_loc[:,1] /= out_loc[:,3]
#        out_loc[:,2] /= out_loc[:,3]
#        
#        out_loc[:,0] *= inv_v_air
#        out_loc[:,1] *= inv_v_air
#        out_loc[:,2] *= inv_v_air
#         
#        out_loc[:,3] = 1
#        
#        delay_i = 0
#        out_delays[:] = 0.0
#        for index_range, delay in zip(station_to_antenna_index_list,  delays):
#            first,last = index_range
#            
#            out_delays[first:last,delay_i] = 1
#                
#            delay_i += 1
#            
#            
#    def estimate_T(self, delays, XYZT_location):
#        X,Y,Z,T = XYZT_location
#        Z = np.abs(Z)
#        
#        delta_X_sq = ant_locs[:,0]-X
#        delta_Y_sq = ant_locs[:,1]-Y
#        delta_Z_sq = ant_locs[:,2]-Z
#        
#        delta_X_sq *= delta_X_sq
#        delta_Y_sq *= delta_Y_sq
#        delta_Z_sq *= delta_Z_sq
#        
#            
#        workspace = delta_X_sq+delta_Y_sq
#        workspace += delta_Z_sq
#        
##        print(delta_X_sq)
#        np.sqrt(workspace, out=workspace)
#        
##        print(self.pulse_times)
##        print(workspace)
#        workspace[:] -= self.pulse_times*v_air ## this is now source time
#        
#        ##now account for delays
#        for index_range, delay in zip(station_to_antenna_index_list,  delays):
#            first,last = index_range
#            workspace[first:last] += delay*v_air ##note the wierd sign
#                
##        print(workspace)
#        ave_error = np.nanmean( workspace )
#        return -ave_error/v_air
            
#    def SSqE_fit(self, delays, XYZT_location):
#        X,Y,Z,T = XYZT_location
#        Z = np.abs(Z)
#        
#        delta_X_sq = ant_locs[:,0]-X
#        delta_Y_sq = ant_locs[:,1]-Y
#        delta_Z_sq = ant_locs[:,2]-Z
#        
#        delta_X_sq *= delta_X_sq
#        delta_Y_sq *= delta_Y_sq
#        delta_Z_sq *= delta_Z_sq
#        
#        distance = delta_X_sq
#        distance += delta_Y_sq
#        distance += delta_Z_sq
#        
#        np.sqrt(distance, out=distance)
#        distance *= 1.0/v_air
#        
#        distance += T
#        distance -= self.pulse_times
#        
#        ##now account for delays
#        for index_range, delay in zip(station_to_antenna_index_list,  delays):
#            first,last = index_range
#            if first is not None:
#                distance[first:last] += delay ##note the wierd sign
#                
#        distance *= distance
#        return np.nansum(distance)    
                
    
#    def RMS_fit_byStation(self, delays, XYZT_location):
#        X,Y,Z,T = XYZT_location
#        Z = np.abs(Z)
#        
#        delta_X_sq = ant_locs[:,0]-X
#        delta_Y_sq = ant_locs[:,1]-Y
#        delta_Z_sq = ant_locs[:,2]-Z
#        
#        delta_X_sq *= delta_X_sq
#        delta_Y_sq *= delta_Y_sq
#        delta_Z_sq *= delta_Z_sq
#        
#        distance = delta_X_sq
#        distance += delta_Y_sq
#        distance += delta_Z_sq
#        
#        np.sqrt(distance, out=distance)
#        distance *= 1.0/v_air
#        
#        distance += T
#        distance -= self.pulse_times
#        
#        ##now account for delays
#        for index_range, delay in zip(station_to_antenna_index_list,  delays):
#            first,last = index_range
#            if first is not None:
#                distance[first:last] += delay ##note the wierd sign
#                
#        distance *= distance
#        
#        ret = []
#        for index_range in station_to_antenna_index_list:
#            first,last = index_range
#            
#            data = distance[first:last]
#            nDOF = np.sum( np.isfinite(data) )
#            if nDOF == 0:
#                ret.append( None )
#            else:
#               ret.append( np.sqrt( np.nansum(data)/nDOF ) )
#               
#        ## need to do referance station
#        first,last = station_to_antenna_index_dict[ referance_station ]
#        data = distance[first:last]
#        nDOF = np.sum( np.isfinite(data) )
#        if nDOF == 0:
#            ret.append( None )
#        else:
#           ret.append( np.sqrt( np.nansum(data)/nDOF ) )
#        
#        return ret
#    
#    def plot_waveforms(self, station_timing_offsets, fname=None):
#        
#        if fname is None:
#            plotter = plt
#        else:
#            CL = code_logger(fname)
#            CL.add_statement("import numpy as np")
#            plotter = pyplot_emulator(CL)
#        
#        most_min_t = np.inf
#        snames_not_plotted = []
#        
#        for sname, offset in zip(  chain(station_order,[referance_station]),  chain(station_timing_offsets,[0.0]) ):
#            index_range = station_to_antenna_index_dict[sname]
#            
#            if sname in self.data_file:
#                station_data = self.data_file[sname]
#            else:
#                continue
#            
#            
#            min_T = np.inf
#                
#            max_T = -np.inf
#            for ant_i in range(index_range[0], index_range[1]):
#                ant_name = sorted_antenna_names[ant_i]
#                if ant_name not in station_data:
#                    continue
#                
#                ant_data = station_data[ant_name]
#        
#                PolE_peak_time = ant_data.attrs['PolE_peakTime'] - offset
#                PolO_peak_time = ant_data.attrs['PolO_peakTime'] - offset
#
#                PolE_hilbert = ant_data[1,:]
#                PolO_hilbert = ant_data[3,:]
#            
#                PolE_trace = ant_data[0,:]
#                PolO_trace = ant_data[2,:]
#
#                PolE_T_array = (np.arange(len(PolE_hilbert)) + ant_data.attrs['starting_index'] )*5.0E-9  + ant_data.attrs['PolE_timeOffset']
#                PolO_T_array = (np.arange(len(PolO_hilbert)) + ant_data.attrs['starting_index'] )*5.0E-9  + ant_data.attrs['PolO_timeOffset']
#                
#                PolE_T_array -= offset
#                PolO_T_array -= offset
#        
#                PolE_amp = np.max(PolE_hilbert)
#                PolO_amp = np.max(PolO_hilbert)
#                amp = max(PolE_amp, PolO_amp)
#                PolE_hilbert = PolE_hilbert/(amp*3.0)
#                PolO_hilbert = PolO_hilbert/(amp*3.0)
#                PolE_trace   = PolE_trace/(amp*3.0)
#                PolO_trace   = PolO_trace/(amp*3.0)
#        
#        
#                if PolE_amp < min_antenna_amplitude:
#                    PolE_peak_time = np.inf
#                if PolO_amp < min_antenna_amplitude:
#                    PolO_peak_time = np.inf
#        
#                plotter.plot( PolE_T_array, ant_i+PolE_hilbert, 'g' )
#                plotter.plot( PolE_T_array, ant_i+PolE_trace, 'g' )
#                plotter.plot( [PolE_peak_time, PolE_peak_time], [ant_i, ant_i+2.0/3.0], 'g')
#                
#                plotter.plot( PolO_T_array, ant_i+PolO_hilbert, 'm' )
#                plotter.plot( PolO_T_array, ant_i+PolO_trace, 'm' )
#                plotter.plot( [PolO_peak_time, PolO_peak_time], [ant_i, ant_i+2.0/3.0], 'm')
#                
#                plotter.annotate( ant_name, xy=[PolO_T_array[-1], ant_i], size=7)
#                
#                max_T = max(max_T, PolE_T_array[-1], PolO_T_array[-1])
#                min_T = min(min_T, PolE_T_array[0], PolO_T_array[0])
#                most_min_t = min(most_min_t, min_T)
#                
#            if min_T<np.inf:
#                plotter.annotate( sname, xy=[min_T, np.average(index_range)], size=15)
#            else:
#                snames_not_plotted.append( sname )
#                
#        for sname in snames_not_plotted:
#            index_range = station_to_antenna_index_dict[sname]
#            plotter.annotate( sname, xy=[most_min_t, np.average(index_range)], size=15)
#                
#        plotter.show()
#        
#        if fname is not None:
#            CL.save()
#    
#    def plot_selected_waveforms(self, station_timing_offsets, fname=None):
#        
#        if fname is None:
#            plotter = plt
#        else:
#            CL = code_logger(fname)
#            CL.add_statement("import numpy as np")
#            plotter = pyplot_emulator(CL)
#        
#        most_min_t = np.inf
#        snames_not_plotted = []
#        
#        for sname, offset in zip(  chain(station_order,[referance_station]),  chain(station_timing_offsets,[0.0]) ):
#            index_range = station_to_antenna_index_dict[sname]
#            
#            min_T = np.inf
#            max_T = -np.inf
#            for ant_i in range(index_range[0], index_range[1]):
#                ant_name = sorted_antenna_names[ant_i]
#                
#                pulse_time = self.pulse_times[ ant_i ]
#                waveform = self.waveforms[ ant_i ]
#                startTime = self.waveform_startTimes[ ant_i ]
#                
#                if not np.isfinite( pulse_time ):
#                    continue
#                
#                T_array = np.arange(len(waveform))*5.0E-9  +  (startTime - offset)
#                
#                
#                amp = np.max(waveform)
#                waveform = waveform/(amp*3.0)
#        
#                plotter.plot( T_array, ant_i+waveform, 'g' )
#                plotter.plot( [pulse_time-offset, pulse_time-offset], [ant_i, ant_i+2.0/3.0], 'm')
#                
#                plotter.annotate( ant_name, xy=[T_array[-1], ant_i], size=7)
#                
#                max_T = max(max_T, T_array[-1])
#                min_T = min(min_T, T_array[0])
#                most_min_t = min(most_min_t, min_T)
#                
#            if min_T<np.inf:
#                plotter.annotate( sname, xy=[min_T, np.average(index_range)], size=15)
#            else:
#                snames_not_plotted.append( sname )
#                
#        for sname in snames_not_plotted:
#            index_range = station_to_antenna_index_dict[sname]
#            plotter.annotate( sname, xy=[most_min_t, np.average(index_range)], size=15)
#                
#        plotter.show()
#        
#        if fname is not None:
#            CL.save()
            
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
               ref_station="CS002", min_ant_amplitude=10, num_itters=1000, error_deviation=0.5E-9, antenna_error=0.5E-9,
               source_XYZ_to_test=[]):
    
    ##### holdovers. These globals need to be fixed, so not global....
    global station_locations, station_to_antenna_index_list, stations_with_fits, station_to_antenna_index_dict
    global referance_station, station_order, sorted_antenna_names, min_antenna_amplitude, ant_locs, bad_antennas
    global current_delays_guess, processed_data_folder
    
    referance_station = ref_station
    min_antenna_amplitude = min_ant_amplitude
    bad_antennas = bad_ants
    
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
    print("guess delays:", guess_timings)
    print('pulse error:', error_deviation)
    print('antenna error:', antenna_error)
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
#    original_delays = np.array( current_delays_guess )        
    
    
    
    
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
        source_to_add = source_object(source_ID, input_name, location, source_stations_to_exclude[source_ID], source_antennas_to_exclude[source_ID], num_itters )
        current_sources.append( source_to_add )


        polarity = source_polarizations[source_ID]

            
        source_to_add.prep_for_fitting(polarity, guess_timings)
        
    print("prepping test sources")
    test_sources = []
    for XYZ in source_XYZ_to_test:
        XYZT = np.append(XYZ, [0.0])
        base_ID = choice(souces_to_fit)
        
        
        ## make source
        source_ID, input_name = input_manager.known_source( base_ID )
        source_to_add = source_object(source_ID, input_name, XYZT, source_stations_to_exclude[source_ID], source_antennas_to_exclude[source_ID], num_itters )
        
        polarity = source_polarizations[source_ID]
        source_to_add.prep_for_fitting(polarity, guess_timings)
        
        
        test_sources.append( source_to_add )






    fitter = stochastic_fitter_dt(current_sources)
    location_fitter = stochastic_fitter_dt_loc( test_sources )
    
    all_delays = np.empty( (num_itters, fitter.num_delays), dtype=np.double )
    all_RMSs = np.empty( num_itters, dtype=np.double )
    loc_RMSs = np.empty( num_itters, dtype=np.double )
    for i in range(num_itters):
        all_delays[ i, :], all_RMSs[i] = fitter.rerun(error_deviation, antenna_error)
    
        fitter.employ_result( current_sources )
        print('run', i, 'RMS:', all_RMSs[i])
        
        if len(test_sources) != 0:
            station_delays = all_delays[ i ]
            loc_RMSs[i] = location_fitter.rerun(station_delays, error_deviation, antenna_error)
            location_fitter.employ_result( test_sources )
            print("    loc. RMS", loc_RMSs[i])
            
        
    print()
    print()
    print("station timing errors:")
    for i, sname in zip( range(fitter.num_delays), station_order):
        print(sname, ":", np.std(all_delays[:,i]) )
        
    print()
    print()
    
    ### get average X, Y, Z for each itteraion
    ave_X = np.zeros( num_itters )
    ave_Y = np.zeros( num_itters )
    ave_Z = np.zeros( num_itters )
    
    for source in current_sources:
        ave_X += source.solutions[: , 0]
        ave_Y += source.solutions[: , 1]
        ave_Z += source.solutions[: , 2]
        
    ave_X /= len(current_sources)
    ave_Y /= len(current_sources)
    ave_Z /= len(current_sources)
    
    print("absolute location errors:")
    print("X", np.std(ave_X), "Y", np.std(ave_Y), "Z", np.std(ave_Z))
    
    print()
    print()
    print("relative location errors")
    
    for source in current_sources:
        source.solutions[: , 0] -= ave_X
        source.solutions[: , 1] -= ave_Y
        source.solutions[: , 2] -= ave_Z
    
        print("source", source.ID)
        print("  ", np.std(source.solutions[:,0]), np.std(source.solutions[:,1]), np.std(source.solutions[:,2]))
        
    print()
    print()
    
    print("average RMS", np.average(all_RMSs), "std of RMS", np.std(all_RMSs))
    
    
    ### same for location fits
    if len(test_sources) != 0:
        print()
        print()
        print("location source tests")
        ave_X = np.zeros( num_itters )
        ave_Y = np.zeros( num_itters )
        ave_Z = np.zeros( num_itters )
        
        for source in test_sources:
            ave_X += source.solutions[: , 0]
            ave_Y += source.solutions[: , 1]
            ave_Z += source.solutions[: , 2]
            
        ave_X /= len(test_sources)
        ave_Y /= len(test_sources)
        ave_Z /= len(test_sources)
        
        print("absolute location errors:")
        print("X", np.std(ave_X), "Y", np.std(ave_Y), "Z", np.std(ave_Z))
        
        print()
        print()
        print("relative location errors")
        
        for i, source in enumerate(test_sources):
            source.solutions[: , 0] -= ave_X
            source.solutions[: , 1] -= ave_Y
            source.solutions[: , 2] -= ave_Z
        
            print("loc. source", i)
            print("  ", np.std(source.solutions[:,0]), np.std(source.solutions[:,1]), np.std(source.solutions[:,2]))
            
        print()
        print()
        
        print("average RMS", np.average(loc_RMSs), "std of RMS", np.std(loc_RMSs))
    
    
    
    
    