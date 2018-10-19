#!/usr/bin/env python3

#python
import time
from os import mkdir
from os.path import isdir, isfile
from copy import deepcopy

#external
import numpy as np
from scipy.optimize import least_squares, minimize

#mine
from utilities import log, processed_data_dir, v_air, SNum_to_SName_dict
from read_PSE import read_PSE_timeID, writeBin_modData_T2
from read_pulse_data import writeTXT_ant_delays
from binary_IO import write_long, write_double, write_double_array, write_string

class PSE_re_fitter:
    def __init__(self, PSE, new_ant_delays, ant_locs_dict):
        self.PSE = PSE
        self.PSE.load_antenna_data( self.PSE.have_trace_data )
        
        self.new_ant_delays = new_ant_delays
        
        ant_locations = []
        PolE_times = []
        PolO_times = []
        
        self.num_even_ant = 0
        self.num_odd_ant = 0
        
        for ant_name, ant_info in self.PSE.antenna_data.items():
            
            ant_locations.append( ant_locs_dict[ant_name] )
            PolE_delay = 0.0
            PolO_delay = 0.0
            
            if ant_name in new_ant_delays:
                PolE_delay, PolO_delay = new_ant_delays[ant_name]
                
            PolE_times.append( ant_info.PolE_peak_time - PolE_delay)
            PolO_times.append( ant_info.PolO_peak_time - PolO_delay)
            
            
            if np.isfinite( ant_info.PolE_peak_time ):
                self.num_even_ant += 1
            if np.isfinite( ant_info.PolO_peak_time ):
                self.num_odd_ant += 1
                
                
        self.antenna_locations = np.array(ant_locations)
        PolE_times = np.array(PolE_times)
        PolO_times = np.array(PolO_times)
        
        self.residual_workspace = np.zeros(len(PolE_times))
        self.jacobian_workspace = np.zeros( (len(PolE_times), 4) )
        
        ### fit even polarisation ###
        self.ant_times = PolE_times
        self.PolE_RMS_prefit = np.sqrt( self.SSqE(self.PSE.PolE_loc)/float(self.num_even_ant) ) 
        EvenPol_min = least_squares(self.objective_RES, self.PSE.PolE_loc, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', max_nfev=1000)
        self.PolE_loc = EvenPol_min.x
        self.PolE_RMS = np.sqrt( self.SSqE(self.PolE_loc)/float(self.num_even_ant) )
        
        self.PolE_dS = np.linalg.norm( self.PSE.PolE_loc[0:3]-self.PolE_loc[0:3] )
        self.PolE_dt = np.abs( self.PSE.PolE_loc[3]-self.PolE_loc[3] )
        
        ### fit odd polarisation ###
        self.ant_times = PolO_times
        self.PolO_RMS_prefit = np.sqrt( self.SSqE(self.PSE.PolO_loc)/float(self.num_odd_ant) ) 
        OddPol_min = least_squares(self.objective_RES, self.PSE.PolO_loc, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', max_nfev=1000)
        self.PolO_loc = OddPol_min.x
        self.PolO_RMS = np.sqrt( self.SSqE(self.PolO_loc)/float(self.num_odd_ant) )
        
        self.PolO_dS = np.linalg.norm( self.PSE.PolO_loc[0:3]-self.PolO_loc[0:3] )
        self.PolO_dt = np.abs( self.PSE.PolO_loc[3]-self.PolO_loc[3] )
        
        
        print()
        print("PSE", self.PSE.unique_index)
        print("  old even RMS:", self.PSE.PolE_RMS, "prefit:", self.PolE_RMS_prefit, "new RMS", self.PolE_RMS)
        print("    dS:", self.PolE_dS, 'dt:', self.PolE_dt)
        print("  ", EvenPol_min.message)
        print("  old Odd RMS:", self.PSE.PolO_RMS, "prefit:", self.PolO_RMS_prefit, "new RMS", self.PolO_RMS)
        print("    dS:", self.PolO_dS, 'dt:', self.PolO_dt)
        print("  ", OddPol_min.message)
        print()
        
        ## free resources
        del self.residual_workspace
        del self.jacobian_workspace
        del self.ant_times
        del self.antenna_locations
        
    def objective_RES(self, XYZT):
        
        self.residual_workspace[:] = self.ant_times 
        self.residual_workspace[:] -= XYZT[3]
        self.residual_workspace[:] *= v_air
        self.residual_workspace[:] *= self.residual_workspace[:]
        
        R2 = self.antenna_locations - XYZT[0:3]
        R2 *= R2
        self.residual_workspace[:] -= R2[:,0]
        self.residual_workspace[:] -= R2[:,1]
        self.residual_workspace[:] -= R2[:,2]
        
        self.residual_workspace[ np.logical_not(np.isfinite(self.residual_workspace)) ] = 0.0
        
        return self.residual_workspace
    
    def objective_JAC(self, XYZT):
        self.jacobian_workspace[:, 0:3] = XYZT[0:3]
        self.jacobian_workspace[:, 0:3] -= self.antenna_locations
        self.jacobian_workspace[:, 0:3] *= -2.0
        
        self.jacobian_workspace[:, 3] = self.ant_times
        self.jacobian_workspace[:, 3] -= XYZT[3]
        self.jacobian_workspace[:, 3] *= -v_air*v_air*2
        
        mask = np.logical_not(np.isfinite(self.jacobian_workspace[:, 3])) 
        self.jacobian_workspace[mask, :] = 0
        
        return self.jacobian_workspace
    
    def SSqE(self, XYZT):
        R2 = self.antenna_locations - XYZT[0:3]
        R2 *= R2
        
        theory = np.sum(R2, axis=1)
        np.sqrt(theory, out=theory)
        
        theory *= 1.0/v_air
        theory += XYZT[3] - self.ant_times
        
        theory *= theory
        
        
        theory[ np.logical_not(np.isfinite(theory) ) ] = 0.0
        
        return np.sum(theory)    
    
    
    def save_as_binary(self, fout):
        
        write_long(fout, self.PSE.unique_index)
        write_double(fout, self.PolE_RMS)
        write_double(fout, 0.0) ## reduced chi-squared
        write_double(fout, 0.0) ## power
        write_double_array(fout, self.PolE_loc)
        write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0])) ## standerd errors
        
        write_double(fout, self.PolO_RMS)
        write_double(fout, 0.0) ## reduced chi-squared
        write_double(fout, 0.0) ## power
        write_double_array(fout, self.PolO_loc)
        write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0]))## standerd errors
        
        write_double(fout, 5.0E-9) ##seconds per sample
        write_long(fout, self.num_even_ant)
        write_long(fout, self.num_odd_ant)
        
        write_long(fout, 1) ## 1 means that we do save trace data
        
        write_long(fout, len(self.PSE.antenna_data))
        for ant_name, ant_info in self.PSE.antenna_data.items():
            PolE_delay = 0.0
            PolO_delay = 0.0
            if ant_name in self.new_ant_delays:
                PolE_delay, PolO_delay = new_ant_delays[ant_name]
            
            write_string(fout, ant_name)
            write_long(fout, ant_info.section_number)
            write_long(fout, ant_info.unique_index)
            write_long(fout, ant_info.starting_index)
            write_long(fout, ant_info.antenna_status)
            
            write_double(fout, ant_info.PolE_peak_time)
            write_double(fout, ant_info.PolE_estimated_timing_error)
            write_double(fout, ant_info.PolE_time_offset)
            write_double(fout, ant_info.PolE_HE_peak)
            write_double(fout, ant_info.PolE_data_std)
            
            write_double(fout, ant_info.PolO_peak_time)
            write_double(fout, ant_info.PolO_estimated_timing_error)
            write_double(fout, ant_info.PolO_time_offset)
            write_double(fout, ant_info.PolO_HE_peak)
            write_double(fout, ant_info.PolO_data_std)
            
            if self.PSE.have_trace_data:
                write_double_array(fout, ant_info.even_antenna_hilbert_envelope)
                write_double_array(fout, ant_info.even_antenna_data)
                write_double_array(fout, ant_info.odd_antenna_hilbert_envelope)
                write_double_array(fout, ant_info.odd_antenna_data)
            

if __name__=="__main__":
    ##opening data
    timeID = "D20160712T173455.100Z"
    output_folder = "PSE_ant_delays"
    
    PSE_folder = "allPSE_new3"
    max_RMS = 2.0E-9
    min_STD = 3.0
    min_num_events = 50
    
    
    
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
    print("PSE folder:", PSE_folder)
    print("max RMS:", max_RMS)
    print("min std:", min_STD)
    print("min num events:", min_num_events)
    
    
    
    PSE_info = read_PSE_timeID(timeID, PSE_folder)
    PSE_list = PSE_info["PSE_list"]
    ant_locs = PSE_info["ant_locations"]
    old_ant_delays = PSE_info["ant_delays"]
    
    
    
    ant_delay_dict = {}
    
    for PSE in PSE_list:
        PSE.load_antenna_data(True)
        
        for ant_name, ant_info in PSE.antenna_data.items():
            loc = ant_locs[ant_name]
            
            if ant_name not in ant_delay_dict:
                ant_delay_dict[ant_name] = [[],  []]
            
            if (ant_info.antenna_status==0 or ant_info.antenna_status==2) and PSE.PolE_RMS<max_RMS:
                model = np.linalg.norm(PSE.PolE_loc[0:3] - loc)/v_air + PSE.PolE_loc[3]
                data = ant_info.PolE_peak_time
                ant_delay_dict[ant_name][0].append( data - model )
            
            if (ant_info.antenna_status==0 or ant_info.antenna_status==1) and PSE.PolO_RMS<max_RMS:
                model = np.linalg.norm(PSE.PolO_loc[0:3] - loc)/v_air + PSE.PolO_loc[3]
                data = ant_info.PolO_peak_time
                ant_delay_dict[ant_name][1].append( data - model )
        
    total_ant_delays = deepcopy( old_ant_delays )
    new_ant_delays = {}
    for ant_name, (even_delays, odd_delays) in ant_delay_dict.items():
        PolE_ave = np.average(even_delays)
        PolE_std = np.std(even_delays)
        PolE_N = len(even_delays)
        PolE_err = PolE_std/np.sqrt(PolE_N)
        PolE_significant = PolE_ave>min_STD*PolE_err and PolE_N>=min_num_events and PolE_ave>PolE_std
        
        PolO_ave = np.average(odd_delays)
        PolO_std = np.std(odd_delays)
        PolO_N = len(odd_delays)
        PolO_err = PolO_std/np.sqrt(PolO_N)
        PolO_significant = PolO_ave>min_STD*PolO_err and PolO_N>=min_num_events and PolO_ave>PolO_std

        print(ant_name)
        print("   even:", PolE_ave, "+/-", PolE_err, '(', PolE_std, PolE_N, ')', PolE_significant)
        print("    odd:", PolO_ave, "+/-", PolO_err, '(', PolO_std, PolO_N, ')', PolO_significant)
        
        if PolE_significant or PolO_significant:
            if ant_name not in total_ant_delays:
                total_ant_delays[ant_name] = [0.0, 0.0]
                
            if ant_name not in new_ant_delays:
                new_ant_delays[ant_name] = [0.0, 0.0]
                
            if PolE_significant:
                total_ant_delays[ant_name][0] += PolE_ave
                new_ant_delays[ant_name][0] += PolE_ave
                
            if PolO_significant:
                total_ant_delays[ant_name][1] += PolO_ave
                new_ant_delays[ant_name][1] += PolO_ave
        
    print()
    print()
                
     
    writeTXT_ant_delays(data_dir + '/ant_delays.txt', new_ant_delays)
        
    
    new_PSE_list = [PSE_re_fitter(PSE, new_ant_delays, ant_locs) for PSE in PSE_list]
    
    
    with open(data_dir + '/point_sources_all', 'wb') as fout:
        write_long(fout, 5)
        writeBin_modData_T2(fout, station_delay_dict=PSE_info['stat_delays'], ant_delay_dict=total_ant_delays, bad_ant_list=PSE_info['bad_ants'], flipped_pol_list=PSE_info['pol_flips'])
    
        write_long(fout, 2) ## means antenna location data is next
        write_long(fout, len(ant_locs))
        for ant_name, ant_loc in ant_locs.items():
            write_string(fout, ant_name)
            write_double_array(fout, ant_loc)
            
        write_long(fout,3) ## means PSE data is next
        write_long(fout, len(new_PSE_list))
        for PSE in new_PSE_list:
            PSE.save_as_binary(fout)
    
    
    
    
    
    
    
    