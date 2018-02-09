#!/usr/bin/env python3

##internal
import time
from os import mkdir, listdir
from os.path import isdir
from shutil import copyfile

##import external packages
import numpy as np
from scipy.optimize import least_squares
#import matplotlib.pyplot as plt

##my packages
from utilities import log, processed_data_dir, v_air, upsample_N_envelope, parabolic_fit
from binary_IO import write_long, write_double, write_string, write_double_array
from read_PSE import read_PSE_file
#from porta_code import code_logger

def peak_finder(waveform, threshold):
    
    above_previous =  waveform[1:-1]>waveform[:-2]
    above_next = waveform[1:-1]>waveform[2:]
    
    peak_indeces = np.where(np.logical_and( above_previous, above_next ))[0] + 1
    
    above_threshold = waveform[ peak_indeces ] > threshold
    
    return peak_indeces[above_threshold]

class PSE_analyzer:
    
    class subEvent_data:
        def __init__(self):
            self.RMS_fit = 0
            self.polarization = 0
            self.location = 0
            self.times = 0
            self.parabolic_information_list = 0
            self.num_ant = 0
            
    
    def __init__(self, PSE, ant_locations):
        self.PSE = PSE
        
        self.PSE.load_antenna_data(True)
        
        antenna_locations = []
        even_T = []
        odd_T = []
        
        even_parabolic_data = []
        odd_parabolic_data = []
        
        self.num_even_ant = 0
        self.num_odd_ant = 0
        
        self.antenna_data = self.PSE.antenna_data.values()
        for ant_data in self.antenna_data:
            ant_name = ant_data.ant_name
            loc = ant_locations[ant_name]
            
            if ant_data.antenna_status == 0 or ant_data.antenna_status == 2:
                
                peaks = peak_finder(ant_data.even_antenna_hilbert_envelope, ant_data.PolE_data_std*1.2533)
                tmp_peak_t = []
                tmp_parabolic_info = []
                for index in peaks:
                    para_fiter = parabolic_fit(ant_data.even_antenna_hilbert_envelope, index, 5)
                    tmp_parabolic_info.append( para_fiter )
                    tmp_peak_t.append(   (para_fiter.peak_index+ant_data.starting_index)*5.0E-9 + ant_data.PolE_time_offset   )
                    
                even_T.append( np.array(tmp_peak_t) )
                even_parabolic_data.append( tmp_parabolic_info )
                
                
                self.num_even_ant += 1
            else:
                even_T.append( [np.inf] )
                
            if ant_data.antenna_status == 0 or ant_data.antenna_status == 1:
                
                peaks = peak_finder(ant_data.odd_antenna_hilbert_envelope, ant_data.PolO_data_std*1.2533)
                tmp_peak_t = []
                tmp_parabolic_info = []
                for index in peaks:
                    para_fiter = parabolic_fit(ant_data.odd_antenna_hilbert_envelope, index, 5)
                    tmp_parabolic_info.append( para_fiter )
                    tmp_peak_t.append(   (para_fiter.peak_index+ant_data.starting_index)*5.0E-9 + ant_data.PolO_time_offset   )
                    
                odd_T.append( np.array(tmp_peak_t) )
                odd_parabolic_data.append( tmp_parabolic_info )
                
                self.num_odd_ant += 1
            else:
                odd_T.append( [np.inf] )
            
            antenna_locations.append(loc)
            
        self.num_ant = len(even_T)
        
        self.antenna_locations = np.array( antenna_locations )
        
        self.residual_workspace = np.zeros(len(even_T))
        self.jacobian_workspace = np.zeros( (len(even_T), 4) )
        self.index_workspace = np.zeros(len(even_T), dtype=int)
        
        
        print()
        print("PSE", self.PSE.unique_index)
        print(" original:")
        print("  even red chi sq:", self.PSE.PolE_RMS)
        print("  even_loc:", self.PSE.PolE_loc)
        print("  odd red chi sq:", self.PSE.PolO_RMS)
        print("  odd loc:", self.PSE.PolO_loc)
         
        
        ### even pol ###
        self.even_event_data_list = []
            
        while True:
            all_ant_have_pulses = True
            for pulse_times in even_T:
                if not np.any( np.isfinite(pulse_times) ):
                    all_ant_have_pulses = False
                    break
                    
            if not all_ant_have_pulses:
                break

            
            self.ant_times = even_T
            
            min_fit = least_squares(self.objective_RES, self.PSE.PolE_loc, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')

    
            indeces = self.get_time_indeces(min_fit.x)
            event_times = []
            parabolic_info = []
            
            for ant_i in range(self.num_ant):
                if indeces[ant_i] >= 0:
                    event_times.append( self.ant_times[ant_i][ indeces[ant_i] ] )
                    parabolic_info.append( even_parabolic_data[ant_i][ indeces[ant_i] ]  )
                    
                    self.ant_times[ant_i][ indeces[ant_i] ]  = np.inf
                else:
                    event_times.append( np.inf )
                    parabolic_info.append( None )
            event_times = np.array(event_times)
    
            residuals = self.residuals( min_fit.x, event_times )

            RMS =  np.std(residuals)
    
            print(" new even-pol sub-event")
            print("   ", min_fit.x)
            print("   ", RMS )
            print("   ", np.sum( np.abs(residuals)>3*RMS), len(residuals)  )
            
            
            if RMS <= max_fit:
                
                data_obj = PSE_analyzer.subEvent_data()
                data_obj.RMS_fit = RMS
                data_obj.polarization = 0
                data_obj.location = min_fit.x
                data_obj.times = event_times
                data_obj.parabolic_information_list = parabolic_info
                data_obj.num_ant = np.sum( np.isfinite(event_times) )
                
                self.even_event_data_list.append( data_obj )
            
        
        ### odd pol ###
        self.odd_event_data_list = []
            
        while True:
            all_ant_have_pulses = True
            for pulse_times in odd_T:
                if not np.any( np.isfinite(pulse_times) ):
                    all_ant_have_pulses = False
                    break
                    
            if not all_ant_have_pulses:
                break

            
            self.ant_times = odd_T
            
            min_fit = least_squares(self.objective_RES, self.PSE.PolO_loc, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')

    
            indeces = self.get_time_indeces(min_fit.x)
            event_times = []
            parabolic_info = []
            
            for ant_i in range(self.num_ant):
                if indeces[ant_i] >= 0:
                    event_times.append( self.ant_times[ant_i][ indeces[ant_i] ] )
                    parabolic_info.append( odd_parabolic_data[ant_i][ indeces[ant_i] ]  )
                    
                    self.ant_times[ant_i][ indeces[ant_i] ]  = np.inf
                    
                else:
                    event_times.append( np.inf )
                    parabolic_info.append( None )
            event_times = np.array(event_times)
    
            residuals = self.residuals( min_fit.x, event_times )

            RMS =  np.std(residuals)
    
            print(" new odd-pol sub-event")
            print("   ", min_fit.x)
            print("   ", RMS )
            print("   ", np.sum( np.abs(residuals)>3*RMS), len(residuals)  )
            
            if RMS <= max_fit:
                
                data_obj = PSE_analyzer.subEvent_data()
                data_obj.RMS_fit = RMS
                data_obj.polarization = 1
                data_obj.location = min_fit.x
                data_obj.times = event_times
                data_obj.parabolic_information_list = parabolic_info
                data_obj.num_ant = np.sum( np.isfinite(event_times) )
                
                self.odd_event_data_list.append( data_obj )
                    
        

        
        ## free resources
        del self.residual_workspace
        del self.jacobian_workspace
        del self.antenna_locations
        del self.ant_times
        
    def write(self, fout):
        sub_event_index = 0
        
        for event in self.even_event_data_list:
            
            write_long(fout, 1)
            
            write_long(fout, sub_event_index )
            sub_event_index += 1
            
            write_long(fout, self.PSE.unique_index)
            
            write_double(fout, event.RMS_fit)
            write_double(fout, 0.0) ##reduced chi-squared
            write_double(fout, 0.0) ##estimated power
            write_double_array(fout, event.location )
            write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0]) )
            
            ## odd pol stuff
            write_double(fout, 0.0)
            write_double(fout, 0.0)
            write_double(fout, 0.0)
            write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0]))
            write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0]))
            
            write_double(fout, 5.0E-9) ##seconds per sample
            write_long(fout, event.num_ant)
            write_long(fout, 0)
            write_long(fout, 0) ##does not have trace data
            
            write_long(fout, event.num_ant)
            for t, para_info, ant_info in zip(event.times, event.parabolic_information_list, self.antenna_data):
                if np.isfinite(t):
                    write_string(fout, ant_info.ant_name)
                    write_long(fout, ant_info.section_number)
                    write_long(fout, ant_info.unique_index)
                    write_long(fout, ant_info.starting_index)
                    write_long(fout, ant_info.antenna_status)
                    
                    write_double(fout, t)
                    write_double(fout, para_info.peak_index_error_ratio*ant_info.PolE_data_std*0.655136 )
                    write_double(fout, ant_info.PolE_time_offset )
                    write_double(fout, para_info.amplitude)
                    write_double(fout, ant_info.PolE_data_std)
                    
                    #### odd pol data ####
                    write_double(fout, 0.0)
                    write_double(fout, 0.0)
                    write_double(fout, 0.0)
                    write_double(fout, 0.0)
                    write_double(fout, 0.0)
                    
        
        
        for event in self.odd_event_data_list:
            
            write_long(fout, 1)
            
            write_long(fout, sub_event_index )
            sub_event_index += 1
            
            write_long(fout, self.PSE.unique_index)
            
            ## even pol stuff
            write_double(fout, 0.0)
            write_double(fout, 0.0)
            write_double(fout, 0.0)
            write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0]))
            write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0]))
            
            ## odd pol
            write_double(fout, event.RMS_fit)
            write_double(fout, 0.0) ##reduced chi-squared
            write_double(fout, 0.0) ##estimated power
            write_double_array(fout, event.location )
            write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0]) )
            
            write_double(fout, 5.0E-9) ##seconds per sample
            write_long(fout, 0)
            write_long(fout, event.num_ant)
            write_long(fout, 0) ##does not have trace data
            
            write_long(fout, event.num_ant)
            for t, para_info, ant_info in zip(event.times, event.parabolic_information_list, self.antenna_data):
                if np.isfinite(t):
                    write_string(fout, ant_info.ant_name)
                    write_long(fout, ant_info.section_number)
                    write_long(fout, ant_info.unique_index)
                    write_long(fout, ant_info.starting_index)
                    write_long(fout, ant_info.antenna_status)
                    
                    #### even pol data ####
                    write_double(fout, 0.0)
                    write_double(fout, 0.0)
                    write_double(fout, 0.0)
                    write_double(fout, 0.0)
                    write_double(fout, 0.0)
                    
                    ### odd pol data
                    write_double(fout, t)
                    write_double(fout, para_info.peak_index_error_ratio*ant_info.PolO_data_std*0.655136 )
                    write_double(fout, ant_info.PolO_time_offset )
                    write_double(fout, para_info.amplitude)
                    write_double(fout, ant_info.PolO_data_std)
                    
                
        
        
    def objective_RES(self, XYZT):
        
        R2 = self.antenna_locations - XYZT[0:3]
        R2 *= R2
        R2 = np.sum(R2, axis=1)
        
        for ant_i in range(self.num_ant):
            
            T_array = self.ant_times[ant_i]-XYZT[3]
            T_array *= v_air
            T_array *= T_array
            
            T_array -= R2[ant_i]
            
            peak_i = np.argmin(np.abs(T_array))
            
            self.residual_workspace[ant_i] = T_array[peak_i]
        
        self.residual_workspace[ np.logical_not(np.isfinite(self.residual_workspace)) ] = 0.0
        
        return self.residual_workspace
    
    def get_time_indeces(self, XYZT):
        R2 = self.antenna_locations - XYZT[0:3]
        R2 *= R2
        R2 = np.sum(R2, axis=1)
        
        for ant_i in range(self.num_ant):
            
            T_array = self.ant_times[ant_i]-XYZT[3]
            T_array *= v_air
            T_array *= T_array
            
            T_array -= R2[ant_i]
            
            
            self.index_workspace[ant_i] = np.argmin(np.abs(T_array))
        
        self.index_workspace[ np.logical_not(np.isfinite(self.residual_workspace)) ] = -1
        
        return self.index_workspace
        
    
    def objective_JAC(self, XYZT):
        self.jacobian_workspace[:, 0:3] = XYZT[0:3]
        self.jacobian_workspace[:, 0:3] -= self.antenna_locations
        self.jacobian_workspace[:, 0:3] *= -2.0
        
        pulse_indeces = self.get_time_indeces(XYZT)
        
        for ant_i in range(self.num_ant):
            if pulse_indeces[ant_i]>=0:
                self.jacobian_workspace[ant_i, 3] = self.ant_times[ant_i][ pulse_indeces[ant_i] ]
            else:
                self.jacobian_workspace[ant_i, 3] = np.inf
                
        self.jacobian_workspace[:, 3] -= XYZT[3]
        self.jacobian_workspace[:, 3] *= -v_air*v_air*2
        
        mask = np.logical_not(np.isfinite(self.jacobian_workspace[:, 3])) 
        self.jacobian_workspace[mask, :] = 0
        
        return self.jacobian_workspace
    
    def residuals(self, XYZT, ant_times):
        R2 = self.antenna_locations - XYZT[0:3]
        R2 *= R2
        
        theory = np.sum(R2, axis=1)
        np.sqrt(theory, out=theory)
        
        theory *= 1.0/v_air
        theory += XYZT[3] - ant_times
        
        theory[ np.logical_not(np.isfinite(theory) ) ] = 0.0
        
        return theory
        

if __name__ == "__main__":
    
    timeID = "D20160712T173455.100Z"
    output_folder = "subEvents"
    
    input_folder = "allPSE"
    
    initial_block = 90800
    final_block = 95200
    
    max_fit = 20.0E-9
    
    
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
    print("input folder:", input_folder)
    print("initial block:", initial_block)
    print("final_block:", final_block)
    print("max_fit:", max_fit)
    
    files = [fname for fname in listdir(processed_data_dir+'/'+input_folder) if fname.startswith("point_sources") and ( initial_block<=int(fname.split('_')[-1])<=final_block )]
    files.sort()
    
    for fname in files:
        
        #### step one: copy the file to new location
        copyfile(processed_data_dir+'/'+input_folder+'/'+fname, processed_data_dir+"/"+output_folder+'/'+fname)
        
        #### step two, process each point source and append to new file
            
        with open(processed_data_dir+"/"+output_folder+'/'+fname, 'ab') as fout:
            with open(processed_data_dir+'/'+input_folder+'/'+fname, 'rb') as fin:
                write_long(fout, 4)
                
                PSE_data = read_PSE_file(fin)
                ant_locs = PSE_data['ant_locations']
                PSE_list = PSE_data["PSE_list"]
                
                for PSE in PSE_list:
                    new_analysis = PSE_analyzer(PSE, ant_locs)
                    new_analysis.write( fout )
                    
                write_long(fout, 0)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    