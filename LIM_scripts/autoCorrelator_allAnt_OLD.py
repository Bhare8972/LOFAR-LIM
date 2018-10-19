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
from LoLIM.signal_processing import parabolic_fit, remove_saturation
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1
from LoLIM.findRFI import window_and_filter
#from RunningStat import RunningStat

from LoLIM.read_pulse_data import writeTXT_station_delays,read_station_info, curtain_plot_CodeLog

from LoLIM.planewave_functions import read_SSPW_timeID

#### some random utilities

                
    
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
            self.solution[param_i:param_i+4] = PSE.XYZT
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
                print("  run:", run_i, end="\r")
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
            
            
            
        #### save results
        self.solution = current_guess
        self.cost = current_fit
        self.RMS = total_RMS
        

        
        
        
        
        
    def employ_result(self, source_object_list):
        """set the result to the guess location of the sources, and return the station timing offsets"""
        
        param_i = self.num_delays
        for PSE in source_object_list:
            PSE.XYZT[:] =  self.solution[param_i:param_i+4]
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
    


#### source object ####
class source:
    def __init__(self, ID, XYZT):
        self.ID = ID
        self.XYZT = XYZT
        
        self.PolE_data = [None]*len(sorted_antenna_names)
        self.PolO_data = [None]*len(sorted_antenna_names)
        
        self.PolE_ant_time_offset = np.zeros( len(sorted_antenna_names) )
        self.PolO_ant_time_offset = np.zeros( len(sorted_antenna_names) )
        
        self.data_index_starts = [None]*len(sorted_antenna_names)
        
        self.calibrated_waveform = [None]*len(sorted_antenna_names)
        self.pulse_times = np.empty( len(sorted_antenna_names) )
        self.pulse_times[:] = np.nan
        
    def find_pulses(self, XYZT, stations_to_exclude=[]):
        for sname in station_order + [referance_station]:
            data_file = raw_data_files[sname]
            data_filter = data_filters[sname]
            
            start_index,end_index = station_to_antenna_index_dict[sname]
            
            timing_callibration_delays = data_file.get_timing_callibration_delays()
            known_station_delay = -data_file.get_nominal_sample_number()*5.0E-9
            trace_width = int(num_data_ponts/2)
            
            for ant_i in range(start_index,end_index):
                    
                ant_data_index = dataFile_ant_index[ ant_i ]
                ant_loc = ant_locs[ ant_i ]
                
                PolE_time_offset = timing_callibration_delays[ant_data_index] + known_station_delay
                PolO_time_offset = timing_callibration_delays[ant_data_index+1] + known_station_delay
                
                dt = np.linalg.norm( XYZT[:3]-ant_loc )/v_air
                predicted_arrival_time = dt + XYZT[3]
                
                if sname == referance_station:
                    station_timing_calibration = 0.0
                else:
                    station_timing_calibration = guess_timings[sname]
                    
                
                data_arrival_index = int( predicted_arrival_time/5.0E-9 + PolE_time_offset/5.0E-9 + station_timing_calibration/5.0E-9 )
                local_data_index = int( data_filter.blocksize*0.5 )
                data_start_sample = data_arrival_index - local_data_index
            
                PolE_data = data_file.get_data(data_start_sample, data_filter.blocksize, antenna_index=ant_data_index  )
                PolO_data = data_file.get_data(data_start_sample, data_filter.blocksize, antenna_index=ant_data_index+1 )
                
                PolE_data = np.array( PolE_data, dtype=np.float )
                PolO_data = np.array( PolO_data, dtype=np.float )
                
                remove_saturation(PolE_data, positive_saturation_amplitude, negative_saturation_amplitude)
                remove_saturation(PolO_data, positive_saturation_amplitude, negative_saturation_amplitude)
                
                PolE_data = data_filter.filter( PolE_data )[ local_data_index-trace_width : local_data_index+trace_width ]
                PolO_data = data_filter.filter( PolO_data )[ local_data_index-trace_width : local_data_index+trace_width ]
                
                self.PolE_data[ ant_i ] = PolE_data
                self.PolO_data[ ant_i ] = PolO_data
                
                self.PolE_ant_time_offset[ ant_i ] = PolE_time_offset
                self.PolO_ant_time_offset[ ant_i ] = PolO_time_offset
                
                self.data_index_starts[ ant_i ] = data_arrival_index-trace_width
                
                self.calibrated_waveform[ ant_i ] = None
                self.pulse_times[ ant_i ] = np.nan
                
                if np.max( PolE_data) < min_antenna_amplitude or np.max( PolO_data) < min_antenna_amplitude:
                    continue
                    
                radius, zenith, azimuth = get_radius_ze_az( XYZT[:3]-ant_loc )
                
                antenna_calibrator.FFT_prep(sorted_antenna_names[ant_i], PolE_data, PolO_data)
                antenna_calibrator.apply_time_shift(0.0, PolE_time_offset-PolO_time_offset)
                polE_cal, polO_cal = antenna_calibrator.apply_GalaxyCal()
                
                if not np.isfinite(polE_cal) or not np.isfinite(polO_cal):
                    print("callibration issues with", sname, sorted_antenna_names[ant_i])
                    continue
                
                antenna_calibrator.unravelAntennaResponce(zenith*RTD, azimuth*RTD)
                zenith_data, azimuth_data = antenna_calibrator.getResult()        
                
                Ze_R = np.real( zenith_data )
                Ze_I = np.imag( zenith_data )
                Az_R = np.real( azimuth_data )
                Az_I = np.imag( azimuth_data )
                
                Ze_int = Ze_R*Ze_R + Ze_I*Ze_I
                Az_int = Az_R*Az_R + Az_I*Az_I
                
                total_amplitude_waveform = np.sqrt( Ze_int + Az_int )
                
                peak_finder = parabolic_fit( total_amplitude_waveform )
                pt = (peak_finder.peak_index + data_arrival_index-trace_width)*5.0E-9 - PolE_time_offset
                
                self.calibrated_waveform[ ant_i ] = total_amplitude_waveform
                
                if sname not in stations_to_exclude:
                    self.pulse_times[ ant_i ] = pt
                
    def relocation_algorithm(self, located_stations):
        
        stations_to_exclude = [ sname for sname in station_order if sname not in located_stations]
        
        print("relocating:", self.ID)
        while True:
            print()
            print("excluding", len(stations_to_exclude))
            while True:
                ## find pulses
                self.find_pulses( self.XYZT, stations_to_exclude )
                
                ## locate
                fit_res = least_squares(self.relocation_fit, self.XYZT, jac='2-point', method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
                
                distance = np.linalg.norm( fit_res.x[:3] - self.XYZT[:3] )
                print("  loc:", fit_res.x)
                print("  D:", distance )
                
                self.XYZT = fit_res.x
                if distance<1:
                    break
                
            if len(stations_to_exclude) == 0:
                break
                
            ## find station to include, one closest to an included station
            closest_distance = np.inf
            best_station = None
            for excluded_station in stations_to_exclude:
                exc_stat_loc = station_locations[excluded_station]
                
                for sname,sloc in station_locations.items():
                    if sname in stations_to_exclude:
                        continue
                    
                    D = np.linalg.norm(sloc-exc_stat_loc)
                    if D<closest_distance:
                        closest_distance = D
                        best_station = excluded_station
                        
            stations_to_exclude.remove( best_station )
            
            
            
    def relocation_fit(self, XYZT):
            
        ret = np.zeros( len(self.pulse_times) )
        
        dx = ant_locs[:,0]-XYZT[0]
        dy = ant_locs[:,1]-XYZT[1]
        dz = ant_locs[:,2]-XYZT[2]
        
        ret[:] = dx*dx
        ret += dy*dy
        ret += dz*dz
        np.sqrt(ret, out=ret)
        
        ret /= v_air
        
        ret += XYZT[3] - self.pulse_times
        
        for index_range, station_timing in zip(station_to_antenna_index_list, current_delays_guess):
            ret[index_range[0]:index_range[1]] += station_timing
            
        is_fin = np.isfinite(ret) 
        filter = np.logical_not( is_fin )
        self.numDOF = np.sum( is_fin )
        ret[filter] = 0.0
        
        return ret
    
    def predictive_curtain_plot(self, sname, plotter=plt):
        data_file = raw_data_files[sname]
        data_filter = data_filters[sname]
        
        timing_callibration_delays = data_file.get_timing_callibration_delays()
        data_file_ant_names = data_file.get_antenna_names()
        
        known_station_delay = -data_file.get_nominal_sample_number()*5.0E-9
        
        for ant_i in range(station_to_antenna_index_dict[sname][0], station_to_antenna_index_dict[sname][1]):
            ant_name = sorted_antenna_names[ ant_i ]
            ##### plot data from data file
           
    #            else:
            ant_data_index = data_file_ant_names.index( ant_name )
    #            print(local_ant_i, ant_name)
    #            print("  ", SSPW.PolE_time_offsets[local_ant_i], timing_callibration_delays[ant_data_index] + known_station_delay, station_timing_calibration[sname])
                
            ant_loc = ant_locs[ ant_i ]
            dt = np.linalg.norm( self.XYZT[:3]-ant_loc )/v_air # - np.linalg.norm( self.source_XYZT[:3]-arrival_loc )/v_air
    #            predicted_arrival_time = dt + ave_arrival_time
            predicted_arrival_time = dt + self.XYZT[3]
    #            print("  ", SSPW.pulse_times[local_ant_i], predicted_arrival_time)
    #            print()
    
            
            if sname == referance_station:
                station_timing_calibration = 0.0
            else:
                station_timing_calibration = guess_timings[sname]
    
            total_timing_offset = timing_callibration_delays[ant_data_index] + known_station_delay + station_timing_calibration
            
            data_start_sample = int( predicted_arrival_time/5.0E-9 - data_filter.blocksize*0.5 + total_timing_offset/5.0E-9 )
            
            PolE_data = data_file.get_data(data_start_sample, data_filter.blocksize, antenna_index=ant_data_index  )
            PolO_data = data_file.get_data(data_start_sample, data_filter.blocksize, antenna_index=ant_data_index+1 )
            
            PolE_data = np.array( PolE_data, dtype=np.float )
            PolO_data = np.array( PolO_data, dtype=np.float )
            
            PolE_data = np.real( data_filter.filter( PolE_data ) )
            PolO_data = np.real( data_filter.filter( PolO_data ) )
            
            PolE_T_array = ( np.arange(data_filter.blocksize) + data_start_sample )*5.0E-9 - total_timing_offset
            PolO_T_array = ( np.arange(data_filter.blocksize) + data_start_sample )*5.0E-9 - total_timing_offset
            
            amp = max( np.max(PolE_data), np.max(PolO_data) )*3
            PolE_data = PolE_data/(2*amp)
            PolO_data = PolO_data/(2*amp)
            
            plotter.plot(PolE_T_array, PolE_data+ant_i, 'g')
            plotter.plot(PolO_T_array, PolO_data+ant_i, 'm')
            
            plotter.plot( [predicted_arrival_time, predicted_arrival_time], [ant_i, ant_i+2.0/3.0], 'k')
            
            
            
            #### try to plot known data
            PolE_data = np.real( self.PolE_data[ant_i] )
            PolO_data = np.real( self.PolO_data[ant_i] )
            PolE_HE = np.abs( self.PolE_data[ant_i] )
            PolO_HE = np.abs( self.PolO_data[ant_i] )
            
            PolE_data  /= (2*amp)
            PolO_data /= (2*amp)
            PolE_HE  /= (2*amp)
            PolO_HE /= (2*amp)
    
            PolE_timing_offset = -self.PolE_ant_time_offset[ant_i] - station_timing_calibration  #timing_callibration_delays[ant_data_index] + known_station_delay + station_timing_calibration[sname]
            PolO_timing_offset = -self.PolO_ant_time_offset[ant_i] - station_timing_calibration# timing_callibration_delays[ant_data_index+1] + known_station_delay + station_timing_calibration[sname]
        
            PolE_tarray = (np.arange(len(PolE_data)) + self.data_index_starts[ant_i])*5.0E-9 + PolE_timing_offset
            PolO_tarray = (np.arange(len(PolO_data)) + self.data_index_starts[ant_i])*5.0E-9 + PolO_timing_offset
            
            plotter.plot( PolE_tarray, PolE_data+ant_i+0.5, 'g' )
            plotter.plot( PolO_tarray, PolO_data+ant_i+0.5, 'm' )
            plotter.plot( PolE_tarray, PolE_HE+ant_i+0.5, 'g' )
            plotter.plot( PolO_tarray, PolO_HE+ant_i+0.5, 'm' )
            
            if np.isfinite( self.pulse_times[ant_i] ):
                pt = self.pulse_times[ant_i] - station_timing_calibration
                plotter.plot( [pt,pt], [ant_i, ant_i+1], 'r' )
                
            if self.calibrated_waveform[ ant_i ] is not None:
                waveform = self.calibrated_waveform[ ant_i ]
                waveform *= max(np.max(PolE_HE), np.max(PolO_HE))/np.max(waveform)
                plotter.plot( PolE_tarray, waveform+ant_i+0.5, 'r' )
                
                
        plotter.show()
        
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

np.set_printoptions(precision=10, threshold=np.inf)
if __name__ == "__main__": 
    
    timeID = "D20170929T202255.000Z"
    output_folder = "autoCorrelator_allAnt"
    
    
    #### fitter settings ####
    max_itters_per_loop = 2000
    itters_till_convergence = 100
    max_jitter_width = 100000E-9
    min_jitter_width = 1E-9
    cooldown = 10.0 ## 10.0
    strong_cooldown = 100.0
    
    #### source quality requirments ####
    min_antenna_amplitude = 10
    num_data_ponts = 50
    
    positive_saturation_amplitude = 2046
    negative_saturation_amplitude = -2047
    
    #### initial guesses ####
    referance_station = "CS002"
    
    guess_timings = {
    'CS003' :  1.40563119239e-06 , ## diff to guess: 4.55994078331e-07
    'CS004' :  4.31450208786e-07 , ## diff to guess: 7.28970916176e-07
    'CS005' :  -2.19568209371e-07 , ## diff to guess: -1.49553002018e-07
    'CS006' :  4.33152180879e-07 , ## diff to guess: 4.5146170079e-08
    'CS007' :  3.99767533244e-07 , ## diff to guess: 7.41660299165e-08
    'CS011' :  -5.86485924792e-07 , ## diff to guess: -1.17442456584e-06
    'CS013' :  -1.81272050352e-06 , ## diff to guess: 1.25835038937e-06
    'CS017' :  -8.44063845937e-06 , ## diff to guess: -3.0099081864e-06
    'CS021' :  9.26581579446e-07 , ## diff to guess: 2.58530724568e-06
    'CS030' :  -2.73921412305e-06 , ## diff to guess: 3.18376778734e-06
    'CS032' :  -1.57146770368e-06 , ## diff to guess: 4.20935139017e-06
    'CS101' :  -8.17230090882e-06 , ## diff to guess: -4.70418011408e-06
    'CS103' :  -2.85242273351e-05 , ## diff to guess: -1.12064955665e-05
    'RS208' :  7.01551485723e-06 , ## diff to guess: -1.04428593978e-05
    'CS301' :  -7.16273044738e-07 , ## diff to guess: 6.18694157532e-07
    'CS302' :  -5.3463823678e-06 , ## diff to guess: 7.54484020861e-06
    'RS306' :  7.10637886343e-06 , ## diff to guess: 4.348439054e-05
    'RS307' :  7.11019629267e-06 , ## diff to guess: 4.67900967941e-05
    'RS310' :  7.51201769885e-06 , ## diff to guess: 9.40071404642e-05
    'CS401' :  -9.47218597471e-07 , ## diff to guess: 4.83164553717e-06
    'RS409' :  7.75953709545e-06 , ## diff to guess: 0.000106087887834
    'CS501' :  -9.60818605997e-06 , ## diff to guess: 1.29372282703e-06
    'RS503' :  6.9535423606e-06 , ## diff to guess: 7.70400202009e-06
    'RS508' :  7.2455482616e-06 , ## diff to guess: -2.19581336478e-05
    'RS509' :  7.45871718646e-06 , ## diff to guess: 9.76876679784e-06

        }
    
    
    if referance_station in guess_timings:
        del guess_timings[referance_station]
    
    

    ### locations of fitted sources
    known_source_locations = { ### KEY is just an ID, value is XYZT
        275988 :[ -1.7334770736e+04,   9.1209964351e+03,   2.6220062583e+03, 1.1733716468e+00],
        278749 :[ -1.5704816070e+04,   8.2185753809e+03,   1.6427109863e+03, 1.2086118059e+00],
#        280199 :[ -1.5836301764e+04,   1.0867984453e+04,   4.8170092025e+03, 1.2293895281e+00],
#        274969 :[ -1.6234762843e+04,   9.8971580929e+03,   3.0231639229e+03, 1.1601043965e+00],
#        275426 :[ -1.5888132157e+04,   9.8483264619e+03,   3.4646984245e+03, 1.1663573674e+00],
        276467 :[ -1.6126183340e+04,   1.0307452810e+04,   3.8319314319e+03, 1.1797436117e+00],
#        274359 :[ -1.6070388856e+04,   9.1908620715e+03,   3.5981290415e+03, 1.1531762594e+00],
        
#        274360 :[ -15945.8337978 , 9150.01178357 , 3664.26294234 , 1.15321353185 ],
    }
    
    original_antennas = {
            275988: ['CS002','CS003','CS004','CS005','CS006','CS007','CS011','CS013',
                     'CS017','CS021','CS030','CS032','CS101','CS103','RS208','CS301',
                     'CS302','RS306','RS307','CS401','CS501','RS503','RS509'],
                     
            278749: ['CS002','CS003','CS004','CS005','CS006','CS007','CS011','CS013',
                     'CS017','CS021','CS030','CS032','CS101','CS103','CS301','CS302',
                     'RS307','CS401','CS501','RS503','RS409'],
                     
            280199: ['CS002','CS003','CS004','CS005','CS006','CS007','CS011','CS013',
                     'CS017','CS021','CS030','CS032','CS101','CS103','CS302','CS401',
                     'CS501','RS503','RS307','RS310'],
                     
            274969: ['CS002','CS003','CS004','CS005','CS006','CS007','CS011','CS013',
                     'CS017','CS021','CS030','CS032','CS101','CS103','RS208','CS302',
                     'RS306','RS307','CS401','RS503'],
                     
            275426: ['CS002','CS003','CS004','CS005','CS006','CS011','CS013','CS017',
                     'CS021','CS030','CS032','CS101','CS103','CS301','CS302','RS307',
                     'CS401','CS501','RS503','CS007','RS208'],
                     
            276467: ['CS002','CS003','CS004','CS005','CS006','CS007','CS011','CS013',
                     'CS017','CS021','CS030','CS032','CS101','CS103','CS301','CS302',
                     'CS401','CS501','RS503','RS208','RS310'],
                     
            274359: ['CS002','CS003','CS004','CS005','CS006','CS007','CS011','CS013',
                     'CS017','CS021','CS030','CS103','CS301','CS302','RS307','CS401',
                     'RS409','RS503','RS508','RS509','CS032','CS101'],
                     
#            274360: ['CS002','CS004','CS005','CS006','CS017','CS103','CS301','CS302',
#                     'RS307','RS310','RS508'],
            }
    
    sources_to_relocate = []
    
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
    print("sorting info")
    station_order = list(guess_timings.keys())## note this doesn't include reference station
    sorted_antenna_names = []
    ant_locs = []
    dataFile_ant_index = []
    station_to_antenna_index_dict = {}
    
    for sname in station_order + [referance_station]:
        first_index = len(sorted_antenna_names)
        
        stat_input = raw_data_files[sname]
        antenna_names = stat_input.get_antenna_names()
        antenna_locations = stat_input.get_LOFAR_centered_positions()
        
        for ant_i in range(0, len(antenna_names), 2): ## loop over even antennas
            sorted_antenna_names.append( antenna_names[ant_i] )
            ant_locs.append( antenna_locations[ant_i] )
            dataFile_ant_index.append( ant_i )
                
        station_to_antenna_index_dict[sname] = (first_index, len(sorted_antenna_names))
    
    ant_locs = np.array( ant_locs ) 
    dataFile_ant_index = np.array( dataFile_ant_index ) 
    
    station_locations = {sname:ant_locs[station_to_antenna_index_dict[sname][0]] for sname in station_order + [referance_station]}
    station_to_antenna_index_list = [station_to_antenna_index_dict[sname] for sname in station_order]
    
    #### sort the delays guess, and account for station locations ####
    current_delays_guess = np.array([guess_timings[sname] for sname in station_order])

    original_delays = np.array( current_delays_guess )        
    
    
    print("locate sources on all antennas")
    
    current_sources = []
    for ID, location in known_source_locations.items():
        new_source = source(ID, location  )
        print()
        print("making source", ID)
        if ID in sources_to_relocate:
            new_source.relocation_algorithm( original_antennas[ID] )
        else:
            new_source.find_pulses( new_source.XYZT )
        
        for sname in station_order:
            CL = code_logger( logging_folder+'/source_'+str(ID)+"_"+sname  )
            CL.add_statement("import numpy as np")
            plotter = pyplot_emulator(CL)
            
            new_source.predictive_curtain_plot( sname, plotter )
        
            CL.save()
        
        current_sources.append( new_source )
            
        
    while True:
        print()
        print()
        
        MC_fitter = stochastic_fitter( current_sources )
        print()
        print()
            
        MC_fitter.print_locations( current_sources )
        print()
        print()
        MC_fitter.print_station_fits( current_sources )
        print()
        print()
        MC_fitter.print_delays( original_delays )
        
        current_delays_guess = MC_fitter.employ_result( current_sources )
        
        for source in current_sources:
            source.find_pulses( source.XYZT )
            
        
        
        
        
        
    
    
    
    
    