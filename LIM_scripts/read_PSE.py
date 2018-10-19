#!/usr/bin/env python

##__future__
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

##internal
from os import listdir

##import external packages
import numpy as np
import matplotlib.pyplot as plt

##mine 
from LoLIM.read_pulse_data import readBin_modData
from LoLIM.IO.binary_IO import write_long, write_double_array, write_string, write_double, read_long, read_double_array, read_string, read_double, skip_double_array, at_eof
from LoLIM.utilities import processed_data_dir, v_air

def writeBin_modData_T2(fout, station_delay_dict, ant_delay_dict, bad_ant_list, flipped_pol_list):
    """write mod data to a binary file using format 2. Format 1 is defined in read_pulse_data"""
    write_long(fout, len(station_delay_dict))
    for sname, sdelay in station_delay_dict.items():
        write_string(fout, sname)
        write_double(fout, sdelay)

    
    write_long(fout, len(ant_delay_dict))
    for ant_name, (PolE_delay, PolO_delay) in ant_delay_dict.items():
        write_string(fout, ant_name)
        write_double(fout, PolE_delay)
        write_double(fout, PolO_delay)
            
            
    write_long(fout, len(bad_ant_list))
    for ant_name, pol in bad_ant_list:
        write_string(fout, ant_name)
        write_long(fout, pol)
            
    write_long(fout, len(flipped_pol_list))
    for ant_name in flipped_pol_list:
        write_string(fout, ant_name)
            
def readBin_modData_T2(fin):
    """read mod data from a binary file using format 2. Format 1 is defined in read_pulse_data"""
    
    station_delay_dict = {}
    num_stations = read_long(fin)
    for si in range(num_stations):
        sname = read_string(fin)
        delay = read_double(fin)
        station_delay_dict[sname] = delay
        
        
    ant_delay_dict = {}
    num_ant_delays = read_long(fin)
    for adi in range(num_ant_delays):
        ant_name = read_string(fin)
        PolE_delay = read_double(fin)
        PolO_delay = read_double(fin)
        ant_delay_dict[ant_name] = [PolE_delay, PolO_delay]
            
        
    bad_ant_list = []    
    num_bad_ant = read_long(fin)
    for bai in range(num_bad_ant):
        ant_name = read_string(fin)
        pol = read_long(fin)
        bad_ant_list.append([ant_name, pol])
            
        
    flipped_pol_list = []
    num_flipped_pol = read_long(fin)
    for fpi in range(num_flipped_pol):
        ant_name = read_string(fin)
        flipped_pol_list.append(ant_name)
            
    return flipped_pol_list, bad_ant_list, ant_delay_dict, station_delay_dict



def read_PSE_file(fin):
    out = {}
    
    while not at_eof(fin):
        code = read_long(fin)
        
        if code == 1: ## mod data
        
            pol_flips, bad_ants, ant_delays, stat_delays = readBin_modData(fin)
            out['pol_flips'] = pol_flips
            out['bad_ants'] = bad_ants
            out["ant_delays"] = ant_delays
            out["stat_delays"] = stat_delays
            
        elif code == 5: ## mod data, stored useing second format.
        
            pol_flips, bad_ants, ant_delays, stat_delays = readBin_modData_T2(fin)
            out['pol_flips'] = pol_flips
            out['bad_ants'] = bad_ants
            out["ant_delays"] = ant_delays
            out["stat_delays"] = stat_delays
            
        elif code == 2:
            ant_loc_dict = {}
            N_ant = read_long(fin)
            for i in range(N_ant):
                ant_name = read_string(fin)
                ant_loc = read_double_array(fin)
                ant_loc_dict[ant_name] = ant_loc
            out['ant_locations'] = ant_loc_dict
            
        elif code == 3:
            num_PSE = read_long(fin)
            PSE_list = []
            for i in range(num_PSE):
                new_PSE = PointSourceEvent(fin)
                PSE_list.append( new_PSE )
                
            out["PSE_list"] = PSE_list
            
        elif code == 4:
            ### this is a list of sub-events
            PSE_list = []
            while True:
                continue_reading = bool( read_long(fin) )
                if not continue_reading:
                    break
                
                sub_event_index = read_long(fin)
                new_PSE = PointSourceEvent(fin, sub_event_index)
                PSE_list.append( new_PSE )
                
            out["sub_PSE_list"] = PSE_list
            
        #elif code == 6:
            # planewaves, not implemented
            
        ####TODO: add a section that can record arbitrary data
                
    if "sub_PSE_list" not in out:
        out["sub_PSE_list"] = []
            
    return out

def read_PSE_timeID(TimeID, analysis_folder, fname_prefix=None, data_loc=None):
    ## assumes the data is in a particular directory structure:
        ## that the top folder is data_loc (default to "/home/brian/processed_files/")
        ## next folder has the name of TimeID
        ## next folder has the name of the year (found from TimeID)
        ## next folder is analysis_folder
        ## inside is the fname, default to "point_sources_"
        
        
    
    if fname_prefix is None:
        fname_prefix = "point_sources_"
        
    data_loc = processed_data_dir(TimeID, data_loc) + "/" + analysis_folder + "/"
    
    out = None
    for fname in listdir(data_loc):
        if not fname.startswith(fname_prefix):
            continue
        
        with open(data_loc+fname, 'rb') as fin:
            print("reading", data_loc+fname)
            new_info = read_PSE_file(fin)
            if out is None:
                out = new_info
            else:
                out['PSE_list'] += new_info['PSE_list']
                out['sub_PSE_list'] += new_info['sub_PSE_list']
    
    return out



class PointSourceEvent:
    def __init__(self, fin, sub_event_index=None):
        self.file_location = fin.name
        self.read_status = 0 ## zero means just the basics. 1 means we have antenna details. 2 means we have trace data
        
        self.is_sub_event_index = sub_event_index is not None
        self.sub_event_index = sub_event_index
        
        self.unique_index = read_long(fin)
        self.PolE_RMS = read_double(fin)
        read_double(fin)## reduced chi-squared
        read_double(fin)## power
        self.PolE_loc = read_double_array(fin)
        read_double_array(fin) ## standard errors
        
        self.PolO_RMS = read_double(fin)
        read_double(fin)
        read_double(fin)
        self.PolO_loc = read_double_array(fin)
        read_double_array(fin)
        
        
        self.polarization_status = 0  ##0 means both pols are good
        if self.PolE_RMS < 1.0E-15:
            self.polarization_status = 1 ## 1 means even is bad
        elif self.PolO_RMS < 1.0E-15:
            self.polarization_status = 2 ## 2 means odd is bad
        
        
        self.seconds_per_sample = read_double(fin)
        
        self.num_even_antennas = read_long(fin)
        self.num_odd_antennas = read_long(fin)
        
        self.have_trace_data = bool( read_long(fin) )
        
        self.antenna_data_loc = fin.tell()
        
#        print( self.unique_index )
#        print( " even fitval:", self.PolE_RMS )
#        print( "   loc:", self.PolE_loc, "n ant:", self.num_even_antennas )
#        print( " odd fitval:", self.PolO_RMS, "n ant:", self.num_odd_antennas )
#        print( "   loc:", self.PolO_loc )
#        print()
       
        
        num_particpating_ant = read_long(fin)
        for ant_i in range(num_particpating_ant):
            read_string(fin) ##ant_name
            read_long(fin) ##pulse section number
            read_long(fin) ## pulse unique index
            read_long(fin) ##pulse starting index
            read_long(fin) ##pulse antenna status
            
            read_double(fin) ##polE peak time
            read_double(fin) ##PolE est timing error
            read_double(fin) ## PolE timing offset
            read_double(fin) ## hilbert envelope peak
            read_double(fin) ## polE data std
            
            read_double(fin) ##polO peak time
            read_double(fin) ##PolO est timing error
            read_double(fin) ## PolO timing offset
            read_double(fin) ## hilbert envelope peak
            read_double(fin) ## polO data std

            if self.have_trace_data:
                skip_double_array(fin) ## even hilbert
                skip_double_array(fin) ## even data
                skip_double_array(fin) ## odd hilbert
                skip_double_array(fin) ## odd data
            
    def best_location(self):
        if self.PolE_RMS<self.PolO_RMS:
            return self.PolE_loc
        else:
            return self.PolO_loc
        
#    def best_fitval(self):
#        if self.PolE_RMS<self.PolO_RMS:
#            return self.PolE_RMS
#        else:
#            return self.PolO_RMS
#    
#    def best_num_antennas(self):
#        if self.PolE_RMS<self.PolO_RMS:
#            return self.num_even_antennas
#        else:
#            return self.num_odd_antennas
            
    class ant_info_obj:
        def __init__(self, fin, load_trace_data, has_trace_data):
            self.ant_name = read_string(fin) ##ant_name
            self.section_number = read_long(fin) ##pulse section number
            self.unique_index = read_long(fin) ## pulse unique index
            self.starting_index = read_long(fin) ##pulse starting index
            self.antenna_status = read_long(fin) ##pulse antenna status
            # 0 means both are good. 1 means even is bad. 2 means odd is bad, 3 means both are bad
            
            self.PolE_peak_time = read_double(fin) ##polE peak time
            self.PolE_estimated_timing_error = read_double(fin) ##PolE est timing error
            self.PolE_time_offset = read_double(fin) ## PolE timing offset
            self.PolE_HE_peak = read_double(fin) ## hilbert envelope peak
            self.PolE_data_std = read_double(fin) ## polE data std
            
            self.PolO_peak_time = read_double(fin) ##polO peak time
            self.PolO_estimated_timing_error = read_double(fin) ##PolO est timing error
            self.PolO_time_offset = read_double(fin) ## PolO timing offset
            self.PolO_HE_peak = read_double(fin) ## hilbert envelope peak
            self.PolO_data_std = read_double(fin) ## polO data std

            if load_trace_data:
                
                ####.... your fault if you load trace data that it doesn't have ....
                
                self.even_antenna_hilbert_envelope = read_double_array(fin)
                self.even_antenna_data = read_double_array(fin)
                self.odd_antenna_hilbert_envelope = read_double_array(fin)
                self.odd_antenna_data = read_double_array(fin)
            else:
                if has_trace_data:
                    skip_double_array(fin) ## even hilbert
                    skip_double_array(fin) ## even data
                    skip_double_array(fin) ## odd hilbert
                    skip_double_array(fin) ## odd data
                self.even_antenna_data = None
                self.even_antenna_hilbert_envelope = None
                self.odd_antenna_data = None
                self.odd_antenna_hilbert_envelope = None
            
    def load_antenna_data(self, load_trace_data=False):
        if not load_trace_data and self.read_status != 0:
            return ## antenna data is read
        elif load_trace_data and self.read_status == 2:
            return ## trace data is already read
        
        self.antenna_data = {}
        with open(self.file_location, 'rb') as fin:
            fin.seek(self.antenna_data_loc)
            
            num_particpating_ant = read_long(fin)
            for ant_i in range(num_particpating_ant):
                new_ant_data = self.ant_info_obj(fin, load_trace_data, self.have_trace_data)
                self.antenna_data[ new_ant_data.ant_name ] = new_ant_data
                
        if load_trace_data:
            self.read_status = 2
        else:
            self.read_status = 1
                
            
    def plot_trace_data(self, ant_locations, plotter=plt):
        
        if not self.have_trace_data:
            print("cannot plot PSE", self.unique_index)
            return
        
        self.load_antenna_data(True)
        
        ant_names = list(self.antenna_data.keys())
        ant_names.sort()
        
        offset_index = 0
        for ant_name in ant_names:
            ant_info = self.antenna_data[ant_name]
            ant_loc = ant_locations[ant_name]
            
            PolE_model_time = np.linalg.norm( ant_loc-self.PolE_loc[0:3] )/v_air + self.PolE_loc[3]
            PolO_model_time = np.linalg.norm( ant_loc-self.PolO_loc[0:3] )/v_air + self.PolO_loc[3]
            
            PolE_peak_time = ant_info.PolE_peak_time
            PolO_peak_time = ant_info.PolO_peak_time

            PolE_hilbert = ant_info.even_antenna_hilbert_envelope
            PolO_hilbert = ant_info.odd_antenna_hilbert_envelope
            
            PolE_trace = ant_info.even_antenna_data
            PolO_trace = ant_info.odd_antenna_data

            PolE_T_array = (np.arange(len(PolE_hilbert)) + ant_info.starting_index )*5.0E-9  + ant_info.PolE_time_offset
            PolO_T_array = (np.arange(len(PolO_hilbert)) + ant_info.starting_index )*5.0E-9  + ant_info.PolO_time_offset
            
            amp = max(np.max(PolE_hilbert), np.max(PolO_hilbert))
            PolE_hilbert = PolE_hilbert/(amp*3.0)
            PolO_hilbert = PolO_hilbert/(amp*3.0)
            PolE_trace   = PolE_trace/(amp*3.0)
            PolO_trace   = PolO_trace/(amp*3.0)
            
            
            plotter.plot( PolE_T_array, offset_index+PolE_hilbert, 'g' )
            plotter.plot( PolE_T_array, offset_index+PolE_trace, 'g' )
            plotter.plot( [PolE_peak_time, PolE_peak_time], [offset_index, offset_index+2.0/3.0], 'g')
            plotter.plot( [PolE_model_time,PolE_model_time], [offset_index, offset_index+2.0/3.0], 'r')
            plotter.plot( [PolE_model_time,PolE_model_time], [offset_index, offset_index+2.0/3.0], 'go')
            
            plotter.plot( PolO_T_array, offset_index+PolO_hilbert, 'm' )
            plotter.plot( PolO_T_array, offset_index+PolO_trace, 'm' )
            plotter.plot( [PolO_peak_time, PolO_peak_time], [offset_index, offset_index+2.0/3.0], 'm')
            plotter.plot( [PolO_model_time,PolO_model_time], [offset_index, offset_index+2.0/3.0], 'r')
            plotter.plot( [PolO_model_time,PolO_model_time], [offset_index, offset_index+2.0/3.0], 'mo')
                    
            
            max_T = max( np.max(PolE_T_array), np.max(PolO_T_array) )
            
            plotter.annotate( ant_name, xy=[max_T, offset_index+1.0/3.0], size=15)
            
            offset_index += 1
        plotter.show()
                
                
                
                
                
                
                
                
                
                
                
                
                
                