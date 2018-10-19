#!/usr/bin/env python3

#python
import time
from os import mkdir
from os.path import isdir, isfile

#external
import numpy as np
from scipy.optimize import least_squares, minimize
from matplotlib import pyplot as plt

#mine
from LoLIM.prettytable import PrettyTable
from LoLIM.utilities import log, processed_data_dir, v_air, SId_to_Sname
from LoLIM.IO.binary_IO import read_long, write_long, write_double_array, write_string, write_double
from porta_code import code_logger, pyplot_emulator
#from RunningStat import RunningStat

from LoLIM.read_pulse_data import writeTXT_station_delays,read_station_info, curtain_plot_CodeLog, curtain_plot

from LoLIM.read_PSE import read_PSE_timeID
from LoLIM.planewave_functions import read_SSPW_timeID_multiDir


class fitting_PSE:
    def __init__(self, PSE, PSE_polarity, current_delays, delays_to_use):
        self.PSE = PSE
        self.current_delays = current_delays
        self.delays_to_use = delays_to_use
        PSE.load_antenna_data( True )
        
        if PSE.PolE_RMS<PSE.PolO_RMS:
            self.polarity = 0
        else:
            self.polarity = 1
            
        if PSE_polarity is not None:
            self.polarity = PSE_polarity
        
        self.initial_loc = PSE.PolE_loc if self.polarity==0 else PSE.PolO_loc
        
        self.SSPW_list = []
        
    def get_block_indeces(self, sname):
        blocks = []
        for PSE_ant_info in self.PSE.antenna_data.values():
            block  = int(PSE_ant_info.section_number/2)
            if block not in blocks:
                blocks.append( block )
                
        return blocks

    def add_SSPW(self, new_SSPW):
        self.SSPW_list.append(new_SSPW)

    def fitting_prep(self, station_delay_input_order, stations_to_keep, stations_use_PSE, PSE_ant_locs, SSPW_ant_locs):
        self.stations_to_keep = stations_to_keep
        self.stations_use_PSE = stations_use_PSE
        self.ant_locs = PSE_ant_locs
        
        print("PSE", self.PSE.unique_index)
        self.station_delay_input_order = station_delay_input_order
        
        ant_X = []
        ant_Y = []
        ant_Z = []
        pulse_times = []
#        self.station_index_range = {}
        antenna_names = []
        
        for ant_name, ant_info in self.PSE.antenna_data.items():
            station_number = ant_name[0:3]
            sname = SId_to_Sname[ int(station_number) ]
            if not ( (sname in stations_to_keep) or (sname in stations_use_PSE) ) :
                continue
            
            pt = ant_info.PolE_peak_time if self.polarity==0 else ant_info.PolO_peak_time 
            if not np.isfinite(pt):
                continue
            
            if sname in self.current_delays:
                pt += self.current_delays[sname]
            if sname in self.delays_to_use:
                pt -= self.delays_to_use[sname]
            
            ant_loc = PSE_ant_locs[ ant_name ]
            ant_X.append( ant_loc[0] )
            ant_Y.append( ant_loc[1] )
            ant_Z.append( ant_loc[2] )
            
            pulse_times.append( pt )
            antenna_names.append( ant_name )
            
            
        ### add SSPW
        for SSPW in self.SSPW_list:
            sname = SSPW.sname
            print("  SSPW", SSPW.unique_index, SSPW.sname)
            
            n_ant = 0
            for ant_name, ant_info in SSPW.ant_data.items():
                pt = ant_info.PolE_peak_time if self.polarity==0 else ant_info.PolO_peak_time
                
                if not np.isfinite(pt):
                    continue
                
                if sname in self.delays_to_use:
                    pt -= self.delays_to_use[sname]
                
                ant_loc = SSPW_ant_locs[ ant_name ]
                ant_X.append( ant_loc[0] )
                ant_Y.append( ant_loc[1] )
                ant_Z.append( ant_loc[2] )
            
                pulse_times.append( pt )
                antenna_names.append( ant_name )
                n_ant += 1
            print("    num ant:", n_ant)
            
        self.antenna_X = np.array(ant_X)
        self.antenna_Y = np.array(ant_Y)
        self.antenna_Z = np.array(ant_Z)
        self.pulse_times = np.array(pulse_times)
        self.antenna_names = np.array(antenna_names)
        
        sorted_indeces = np.argsort( self.antenna_names )
        self.antenna_X = self.antenna_X[sorted_indeces]
        self.antenna_Y = self.antenna_Y[sorted_indeces]
        self.antenna_Z = self.antenna_Z[sorted_indeces]
        self.pulse_times = self.pulse_times[sorted_indeces]
        self.antenna_names = self.antenna_names[sorted_indeces]
        
        
        self.station_index_range = {}
        current_station = None
        start_index = None
        for idx, ant_name in enumerate(self.antenna_names):
            sname = SId_to_Sname[ int(ant_name[0:3]) ]
            
            if start_index is None:
                start_index = idx
                current_station = sname
            elif sname != current_station:
                self.station_index_range[ current_station ] = [start_index, idx]
                start_index = idx
                current_station = sname
        self.station_index_range[ current_station ] = [start_index, idx+1]
        
        
        self.ordered_station_index_range = [ (self.station_index_range[sname_] if (sname_ in self.station_index_range) else [None,None]) for sname_ in  self.station_delay_input_order]
        
    def add_new_ant_offsets(self, ant_offset_dict):
        for ant_name, offset in ant_offset_dict.items():
            if ant_name in self.antenna_names:
                i = self.antenna_names.index(ant_name)
                self.pulse_times[i] -= offset
            
    def try_location_LS(self, delays, XYZT_location, out):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        delta_X_sq = self.antenna_X-X
        delta_Y_sq = self.antenna_Y-Y
        delta_Z_sq = self.antenna_Z-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
            
        out[:] = T - self.pulse_times
        
        ##now account for delays
        for index_range, delay in zip(self.ordered_station_index_range,  delays):
            first,last = index_range
            if first is not None:
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
        out_loc[:,0] -= self.antenna_X
        out_loc[:,0] *= -2
        
        out_loc[:,1] = Y
        out_loc[:,1] -= self.antenna_Y
        out_loc[:,1] *= -2
        
        out_loc[:,2] = Z
        out_loc[:,2] -= self.antenna_Z
        out_loc[:,2] *= -2
        
        
        out_loc[:,3] = T - self.pulse_times
        out_loc[:,3] *= 2*v_air*v_air
        
        delay_i = 0
        for index_range, delay in zip(self.ordered_station_index_range,  delays):
            first,last = index_range
            if first is not None:
                out_loc[first:last,3] += delay*2*v_air*v_air
                out_delays[first:last,delay_i] = out_loc[first:last,3]
            delay_i += 1
            
    def SSqE_fit(self, delays, XYZT_location):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        delta_X_sq = self.antenna_X-X
        delta_Y_sq = self.antenna_Y-Y
        delta_Z_sq = self.antenna_Z-Z
        
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
        for index_range, delay in zip(self.ordered_station_index_range,  delays):
            first,last = index_range
            if first is not None:
                distance[first:last] += delay ##note the wierd sign
                
        distance *= distance
        return np.sum(distance)
    
    
            
    def RMS_fit_byStation(self, delays, XYZT_location):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        delta_X_sq = self.antenna_X-X
        delta_Y_sq = self.antenna_Y-Y
        delta_Z_sq = self.antenna_Z-Z
        
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
        for index_range, delay in zip(self.ordered_station_index_range,  delays):
            first,last = index_range
            if first is not None:
                distance[first:last] += delay ##note the wierd sign
                
        distance *= distance
        
        ret = {}
        for sname, index_range in self.station_index_range.items():
            first,last = index_range
            
            ret[sname] = np.sqrt( np.sum(distance[first:last])/float(last-first) )
        
        return ret
    
    def get_DataTime(self, sname, sorted_antenna_names):
        out = []
        
        index_range = self.station_index_range[sname]
        if index_range[0] is None:
            return [np.inf]*len(sorted_antenna_names)
        first,last = index_range
        
        ant_names = self.antenna_names[first:last]
        data_times = self.pulse_times[first:last]
            
        for ant_name in sorted_antenna_names:
            if ant_name in ant_names:
                ant_index = np.where(ant_names==ant_name)[0][0]
                out.append( data_times[ ant_index ] )
            else:
                out.append( np.inf )
        
        return out
        
    def get_ModelTime(self, sname, sorted_antenna_names, sdelay):
        out = []
        for ant_name in sorted_antenna_names:
            
            if ant_name in self.antenna_names:
                ant_idx = np.where( self.antenna_names==ant_name )[0][0]
                ant_loc = np.array([ self.antenna_X[ant_idx], self.antenna_Y[ant_idx], self.antenna_Z[ant_idx]])
                
                ant_loc -= self.source_location[0:3]
                model_time = np.linalg.norm( ant_loc )/v_air + self.source_location[3] + sdelay
                
                out.append(model_time)
            else:
                out.append( np.inf )
        return out
    
    
    def estimate_T(self, delays, XYZT_location, workspace):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        delta_X_sq = self.antenna_X-X
        delta_Y_sq = self.antenna_Y-Y
        delta_Z_sq = self.antenna_Z-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
            
        workspace[:] = delta_X_sq
        workspace[:] += delta_Y_sq
        workspace[:] += delta_Z_sq
        
        np.sqrt(workspace, out=workspace)
        
        workspace[:] -= self.pulse_times*v_air ## this is now source time
        
        ##now account for delays
        for index_range, delay in zip(self.ordered_station_index_range,  delays):
            first,last = index_range
            if first is not None:
                workspace[first:last] += delay*v_air ##note the wierd sign
                
                
        ave_error = np.average(workspace)
        return -ave_error/v_air

    def plot_trace_data(self, station_timing_offsets, plotter=plt):
        
        sorted_ant_names = list(self.antenna_names)
        sorted_ant_names.sort()
        
        snames_annotated = []
        
        offset_index = 0
        for ant_name in sorted_ant_names:
            sname = SId_to_Sname[ int(ant_name[0:3]) ]
            ant_loc = self.ant_locs[ant_name]
            
            if (sname in self.stations_to_keep) or (sname in self.stations_use_PSE): ##in PSE
            
                ant_info = self.PSE.antenna_data[ant_name]
            
                PolE_peak_time = ant_info.PolE_peak_time
                PolO_peak_time = ant_info.PolO_peak_time

                PolE_hilbert = ant_info.even_antenna_hilbert_envelope
                PolO_hilbert = ant_info.odd_antenna_hilbert_envelope
            
                PolE_trace = ant_info.even_antenna_data
                PolO_trace = ant_info.odd_antenna_data

                PolE_T_array = (np.arange(len(PolE_hilbert)) + ant_info.starting_index )*5.0E-9  + ant_info.PolE_time_offset
                PolO_T_array = (np.arange(len(PolO_hilbert)) + ant_info.starting_index )*5.0E-9  + ant_info.PolO_time_offset
            else: ## in SSPW
                
                for SSPW in self.SSPW_list:
                    if SSPW.sname == sname:
                        found_SSPW = SSPW
                        break
                    
                ant_info = found_SSPW.ant_data[ant_name]
            
                PolE_peak_time = ant_info.PolE_peak_time
                PolO_peak_time = ant_info.PolO_peak_time

                PolE_hilbert = ant_info.PolE_hilbert_envelope
                PolO_hilbert = ant_info.PolO_hilbert_envelope
            
                PolE_trace = ant_info.PolE_antenna_data
                PolO_trace = ant_info.PolO_antenna_data

                PolE_T_array = (np.arange(len(PolE_hilbert)) + ant_info.pulse_starting_index )*5.0E-9  + ant_info.PolE_time_offset
                PolO_T_array = (np.arange(len(PolO_hilbert)) + ant_info.pulse_starting_index )*5.0E-9  + ant_info.PolO_time_offset
                
        
            
            PolE_T_array -= station_timing_offsets[sname]
            PolO_T_array -= station_timing_offsets[sname]
            
            amp = max(np.max(PolE_hilbert), np.max(PolO_hilbert))
            PolE_hilbert = PolE_hilbert/(amp*3.0)
            PolO_hilbert = PolO_hilbert/(amp*3.0)
            PolE_trace   = PolE_trace/(amp*3.0)
            PolO_trace   = PolO_trace/(amp*3.0)
            
            
            plotter.plot( PolE_T_array, offset_index+PolE_hilbert, 'g' )
            plotter.plot( PolE_T_array, offset_index+PolE_trace, 'g' )
            plotter.plot( [PolE_peak_time, PolE_peak_time], [offset_index, offset_index+2.0/3.0], 'g')
            
            plotter.plot( PolO_T_array, offset_index+PolO_hilbert, 'm' )
            plotter.plot( PolO_T_array, offset_index+PolO_trace, 'm' )
            plotter.plot( [PolO_peak_time, PolO_peak_time], [offset_index, offset_index+2.0/3.0], 'm')
            
            model_time = np.linalg.norm( ant_loc-self.source_location[0:3] )/v_air + self.source_location[3]
            plotter.plot( [model_time,model_time], [offset_index, offset_index+2.0/3.0], 'r')
            if self.polarity == 0:
                plotter.plot( [model_time,model_time], [offset_index, offset_index+2.0/3.0], 'mo')
            elif self.polarity == 1:
                plotter.plot( [model_time,model_time], [offset_index, offset_index+2.0/3.0], 'go')
                    
            max_T = max( np.max(PolE_T_array), np.max(PolO_T_array) )
            min_T = max( np.min(PolE_T_array), np.min(PolO_T_array) )
            
            plotter.annotate( ant_name, xy=[max_T, offset_index+1.0/3.0], size=15)
            
            if sname not in snames_annotated:
                plotter.annotate( sname, xy=[min_T, offset_index+1.0/3.0], size=15)
                snames_annotated.append( sname )
            
            offset_index += 1
        plotter.show()
    

if __name__=="__main__":
    
    ##opening data
#    timeID = "D20170929T202255.000Z"
#    output_folder = "stocastic_fitter_runA"
#    
#    SSPW_folders = ['SSPW2_tmp']
#    PSE_folder = "handcorrelate_SSPW"
#    
#    max_num_itter = 2000
#    itters_till_convergence =  100
#    delay_width = 10000E-9  ##100.0E-9 
#    position_width = delay_width/3.0E-9
#    
#    
#    referance_station = "CS002"
#    stations_to_keep = []## to keep from the PSE
#    stations_to_correlate = ["CS002", "CS003", "CS004", "CS005", "CS006", "CS007", "CS011", "CS013", "CS017", "CS021", "CS030", "CS032", "CS101", "CS103",
#                             "RS208", "CS301", "CS302", "RS306", "RS307", "RS310", "CS401", "RS406", "RS409", "CS501", "RS503", "RS508", "RS509"]  
#                                ## to correlate from the SSPW
#    
#    ## to find delays
#    stations_to_find_offsets = stations_to_correlate + stations_to_keep
#
### -1 means use data from PSE None means no data
#    correlation_table =  {
##  "CS002", "CS003", "CS004", "CS005", "CS006", "CS007", "CS011", "CS013", "CS017", "CS021", "CS030", "CS032", "CS101", "CS103", 
##  "RS208", "CS301", "CS302", "RS306", "RS307", "RS310", "CS401", "RS406", "RS409", "CS501", "RS503", "RS508", "RS509"
#   
#0:[  -1   ,    1808,      -1,      -1,      -1,     -1 ,      -1,      -1,      -1,      -1,      -1,     None,    -1 ,    -1  ,
#        -1,   -1   ,    -1  ,    -1  ,      -1,     -1 ,    -1  ,      -1,      -1,      -1,      -1,     None,  None],
#
#1:[  -1   ,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,       -1,   None,    -1  ,
#      -1  ,   -1   ,    -1  ,    -1  ,    None,     -1 ,    -1  ,    None,      -1,      -1,      -1,       -1,    -1 ],
#  
#2:[  -1   ,      -1,      -1,      -1,      -1,  19428 ,      -1,      -1,      -1,      -1,      -1,       -1,     -1,    -1  ,
#      -1  ,   -1   ,    -1  ,    None,      -1,     -1 ,    -1  ,      -1,      -1,      -1,      -1,       -1,    None],
#   
#3:[  -1   ,    1825,      -1,    None,    6191,      -1,      -1,      -1,      -1,   None ,      -1,       -1,     -1,   24648,
#      -1  ,      -1,    -1  ,    -1  ,      -1,     -1 ,    -1  ,      -1,      -1,      -1,      -1,       -1,    -1 ],
#   
#4:[  -1   ,      -1,      -1,      -1,    6200,      -1,  40265 ,      -1,      -1,      -1,      -1,       -1,     -1,    -1  ,
#      -1  ,   -1   ,    -1  ,    -1  ,    None,  18376 ,    -1  ,      -1,      -1,    None,      -1,     None,    -1 ],
#    
##5:[  -1   ,      -1,    836 ,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,       -1,     -1,    -1  ,
##      -1  ,   -1   ,    -1  ,    -1  ,      -1,     -1 ,    -1  ,      -1,      -1,      -1,      -1,       -1,    -1 ],
#   
#            }
#    
##2:[  -1   ,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,       -1,     -1,    -1  ,
##      -1  ,   -1   ,    -1  ,    -1  ,      -1,     -1 ,    -1  ,      -1,      -1,      -1,      -1,       -1,    -1 ],
#    
#    ### since we are loading multiple SSPW files, the SSPW unique IDs are not actually unique. Either have a list of indeces, where each index
#    ## refers to the file in "SSPW_folders", or a None, which implies all indeces are from the first "SSPW_folders"
#    correlation_table_SSPW_group = { 
#            0:None,
#            1:None,
#            2:None,
#            3:None,
#            4:None,
##            5:None,
#            }
#    
#    PSE_to_plot = []#list( correlation_table.keys() )
#    
#    ##set polarization to fit. Do not need to have PSE. if 0, use even annennas, if 1 use odd antennas. If None, or not-extent in this list,
#    ## then use default polarization
#    PSE_polarization = {
#            0:1,
#            1:1,
#            2:1,
#            3:0,
#            4:0,
##            5:1,
#            }
#
#    
#    initial_guess = np.array(
#       [  1.40326475e-06,   4.30479399e-07,  -2.20650962e-07,
#         4.32830875e-07,   3.98541326e-07,  -5.86309866e-07,
#        -1.81558008e-06,  -8.44069643e-06,   9.23537095e-07,
#        -2.74297829e-06,  -1.57373607e-06,  -8.17222435e-06,
#        -2.85236088e-05,   6.74971391e-06,  -7.19546289e-07,
#        -5.36000378e-06,   6.86710450e-06,   6.49488183e-06,
#         6.21288530e-06,   1.35065442e-06,   2.41064270e-05,
#         4.85765330e-06,  -9.61342940e-06,   6.92730986e-06,
#         6.57015166e-06,   7.93761182e-06,  -1.55398871e+04,
#         8.91884438e+03,   3.54843082e+03,   1.15317825e+00,
#        -1.57070879e+04,   9.04128609e+03,   3.56309168e+03,
#         1.15321446e+00,  -1.55078410e+04,   8.90700728e+03,
#         3.59197303e+03,   1.15324217e+00,  -1.59926465e+04,
#         9.23681333e+03,   3.68261300e+03,   1.15344759e+00,
#        -1.58423240e+04,   9.08114847e+03,   3.34469757e+03,
#         1.15356267e+00]
#            )
    
    
    
    
    
    
    
    
    
    timeID = "D20170929T202255.000Z"
    output_folder = "stocastic_fitter_runA"
    
    SSPW_folders = ['SSPW', 'SSPW2_tmp']
    PSE_folders = ["allPSE_runA", "handcorrelate_SSPW"]
#    PSE_folder = "allPSE_runA"
    
    max_num_itter = 2000
    itters_till_convergence =  100
    delay_width = 1000E-9  ##100.0E-9 
    position_width = delay_width/3.0E-9
    
    
    referance_station = "CS002"
    stations_to_keep = []## to keep from the PSE
    stations_to_correlate = ["CS002", "CS003", "CS004", "CS005", "CS006", "CS007", "CS011", "CS013", "CS017", "CS021", "CS030", "CS032", "CS101", "CS103",
                             "RS208", "CS301", "CS302", "RS306", "RS307", "RS310", "CS401", "RS406", "RS409", "CS501", "RS503", "RS508", "RS509"]  
                                ## to correlate from the SSPW
    
    ## to find delays
    stations_to_find_offsets = stations_to_correlate + stations_to_keep
    
    PSE_ids = { ##for each PSE, give unique ID and file index
            2:(2,0), 
            47:(47,0),
            191:(191,0),
            266:(266,0),
            313:(313,0),
            
            0:(0,1),
            1:(1,1),
            3:(2,1),
            4:(3,1),
            5:(4,1),
            6:(5,1),
            }
    

## -1 means use data from PSE None means no data
    correlation_table =  {
#    "CS002", "CS003", "CS004", "CS005", "CS006", "CS007", "CS011", "CS013", "CS017", "CS021", "CS030", "CS032", "CS101", "CS103", 
#    "RS208", "CS301", "CS302", "RS306", "RS307", "RS310", "CS401", "RS406", "RS409", "CS501", "RS503", "RS508", "RS509"

2:[    -1   ,      -1,      -1,      -1,    None,      -1,    None,      -1,      -1,      -1,    None,       -1,     -1,    -1  ,
        -1  ,   -1   ,    -1  ,    -1  ,    None,     -1 ,    -1  ,      -1,      -1,      -1,      -1,       -1,   -1 ],
   
47:[   -1   ,    None,      -1,      -1,      -1,      -1,      -1,      -1,      -1,    None,    None,    None ,   None,   None ,
        -1  ,    None,    -1  ,    -1  ,      -1,     -1 ,    -1  ,   None ,      -1,      -1,    None,   310086,   None],
    
191:[  -1   ,      -1,    None,    None,    None,   None ,      -1,   None ,   None ,   None ,      -1,     None,     -1,    -1  ,
        None,   -1   ,    -1  ,    -1  ,    None,     -1 ,    -1  ,      -1,      -1,      -1,      -1,       -1,    -1 ],
    
266:[None   ,    None,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,       -1,     -1,    -1  ,
        None,   -1   ,    None,    -1  ,      -1,     -1 ,    -1  ,      -1,      -1,      -1,      -1,       -1,   -1 ],
     
313:[  -1   ,    None,    None,      -1,      -1,    None,    None,      -1,      -1,      -1,    None,     None,   None,    -1  ,
        -1  ,   -1   ,    -1  ,    -1  ,    None,     -1 ,    -1  ,      -1,      -1,      -1,      -1,       -1,   -1 ],
     
     
     
0:[  -1     ,    1808,      -1,      -1,      -1,     -1 ,      -1,      -1,      -1,      -1,      -1,     None,    -1 ,    -1  ,
          -1,   -1   ,    -1  ,    -1  ,    None,     -1 ,    -1  ,      -1,      -1,      -1,      -1,     None,  None],
   
1:[ None    ,      -1,      -1,  None  ,      -1,      -1,      -1,      -1,      -1,      -1,      -1,       -1,   None,    -1  ,
        -1  ,   -1   ,    -1  ,    -1  ,  329831,     -1 ,    -1  ,      -1,      -1,      -1,      -1,     None,    -1 ],
  
3:[  -1     ,      -1,      -1,      -1,      -1,  19428 ,      -1,      -1,      -1,      -1,      -1,       -1,     -1,    -1  ,
      -1    ,   -1   ,    -1  ,    None,  329837,     -1 ,    -1  ,      -1,      -1,      -1,      -1,   309190,    None],
   
4:[  -1     ,    None,      -1,    None,    6191,      -1,      -1,    None,      -1,   None ,    None,       -1,     -1,   24648,
      -1    ,      -1,    None,    -1  ,  329859,     -1 ,    -1  ,      -1,      -1,      -1,    None,       -1,    -1 ],
     
5:[  -1     ,      -1,      -1,      -1,    6200,      -1,  40265 ,      -1,      -1,      -1,      -1,       -1,   None,    -1  ,
      -1    ,   -1   ,    None,    -1  ,    None,  18376 ,    -1  ,      -1,      -1,    None,      -1,     None,    -1 ],
    
     
            }
    
    PSE_order = [0, 2, 191, 266, 3, 47]
## skip: 243, 292, 302, 304
    
#2:[  -1   ,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,       -1,     -1,    -1  ,
#        -1  ,   -1   ,    -1  ,    -1  ,      -1,     -1 ,    -1  ,      -1,      -1,      -1,      -1,       -1,   -1 ],
    
    ### since we are loading multiple SSPW files, the SSPW unique IDs are not actually unique. Either have a list of indeces, where each index
    ## refers to the file in "SSPW_folders", or a None, which implies all indeces are from the first "SSPW_folders"
    correlation_table_SSPW_group = { 
            
#    "CS002", "CS003", "CS004", "CS005", "CS006", "CS007", "CS011", "CS013", "CS017", "CS021", "CS030", "CS032", "CS101", "CS103", 
#    "RS208", "CS301", "CS302", "RS306", "RS307", "RS310", "CS401", "RS406", "RS409", "CS501", "RS503", "RS508", "RS509"
2:0,
47:0,
191:0,
266:0,
313:0,


#    "CS002", "CS003", "CS004", "CS005", "CS006", "CS007", "CS011", "CS013", "CS017", "CS021", "CS030", "CS032", "CS101", "CS103", 
#    "RS208", "CS301", "CS302", "RS306", "RS307", "RS310", "CS401", "RS406", "RS409", "CS501", "RS503", "RS508", "RS509"

0:1,
1:[   1     ,       1,       1,       1,       1,       1,       1,       1,       1,       1,       1,        1,      1,     1  ,
         1  ,    1   ,     1  ,     1  ,       0,      1 ,     1  ,       1,       1,       1,       1,        1,    1 ],
   
3:[   1     ,       1,       1,       1,       1,       1,       1,       1,       1,       1,       1,        1,      1,     1  ,
         1  ,    1   ,     1  ,     1  ,       0,      1 ,     1  ,       1,       0,       1,       1,        0,    1 ],
   
4:[   0     ,       1,       1,       1,       1,       1,       1,       1,       1,       1,       1,        1,      1,     1  ,
         1  ,    1   ,     1  ,     1  ,       0,      1 ,     1  ,       1,       1,       1,       1,        1,    1 ],
   
5:1,
            }
    
    PSE_to_plot = []#list( correlation_table.keys() )
    
    ##set polarization to fit. Do not need to have PSE. if 0, use even annennas, if 1 use odd antennas. If None, or not-extent in this list,
    ## then use default polarization
    PSE_polarization = {
            2:1,
            47:1,
            191:1,
            266:1,
            313:1, 
            
            0:1,
            1:1,
            3:1,
            4:0,
            5:0,
            }


    
    
    initial_guess = np.array(
      [  1.40642048e-06,   4.30353714e-07,  -2.20446034e-07,
         4.33656872e-07,   4.00510047e-07,  -5.87026178e-07,
        -1.81011176e-06,  -8.43797092e-06,   9.28644928e-07,
        -2.73151939e-06,  -1.57714416e-06,  -8.16496814e-06,
        -2.85242441e-05,   6.35983109e-06,  -7.31636551e-07,
        -5.38775023e-06,  -4.30249855e-04,   3.52246913e-06,
        -3.60162399e-06,  -4.32160239e-04,   1.04464472e-05,
         2.28681068e-06,  -9.59837327e-06,   6.95992766e-06,
        -3.93057016e-06,  -2.45852240e-06,  -1.51559350e+04,
         8.76217477e+03,   3.28095055e+03,   1.15317977e+00,
        -1.52430707e+04,   9.07687363e+03,   3.25066712e+03,
         1.16347069e+00,  -1.53942084e+04,   9.05313943e+03,
         4.16399579e+03,   1.17534780e+00,  -1.54462820e+04,
         9.03551143e+03,   3.40389600e+03,   1.17195747e+00,
        -1.51154810e+04,   8.74590160e+03,   3.34264256e+03,
         1.15324371e+00,  -1.68358544e+04,   9.32895432e+03,
         2.74899248e+03,   1.16552508e+00]
      )
    
    
 
    print_fit_table = True
    stations_to_plot = []#'RS106':0, 'RS205':0, 'RS208':0, 'RS305':0, 'RS306':0, 'RS307':0, 'RS406':0, 'RS407':0, 'RS503':0, 'RS508':0, 'RS509':0} ## index is SSPW folder to use
#    stations_to_plot = {sname:0 for sname in stations_to_correlate}
    
#    for sname in stations_to_correlate:
#        for sspw_i in range(len(SSPW_folders)):
#            stations_to_plot.append( [sname,sspw_i] )
    
    SSPW_folders_to_plot = [0]
            
        
    
    
    if referance_station in stations_to_find_offsets:
        stations_to_find_offsets.remove( referance_station )
    
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
    
    
    
    
    
    log("Time ID:", timeID)
    log("output folder name:", output_folder)
    log("date and time run:", time.strftime("%c") )
    log("SSPW folder:", SSPW_folders)
    log("PSE folders:", PSE_folders)
    log("referance station:", referance_station)
    log("stations to keep:", stations_to_keep)
    log("stations to correlate:", stations_to_correlate)
    log("stations to find delays:", stations_to_find_offsets)
    log("max num. iters:", max_num_itter)
    log("num convergence itters:", itters_till_convergence)
    log("position width:", position_width)
    log("delay width:", delay_width)
    log("correlation matrix:", correlation_table)
    log("SSPW group", correlation_table_SSPW_group)
    log("iniital guess:", initial_guess)
    
    

            
    #### open known PSE ####
    print("reading PSE")
#    PSE_data = read_PSE_timeID(timeID, PSE_folder, data_loc="/home/brian/processed_files")
#    PSE_ant_locs = PSE_data["ant_locations"]
#    PSE_list = PSE_data["PSE_list"]
#    old_delays = PSE_data["stat_delays"]
#    PSE_list.sort(key=lambda PSE: PSE.unique_index)
    
    PSE_lists = []
    PSE_delays = []
    for fname in PSE_folders:
        PSE_data = read_PSE_timeID(timeID, fname, data_loc="/home/brian/processed_files")
        PSE_ant_locs = PSE_data["ant_locations"]
        PSE_list = PSE_data["PSE_list"]
        old_delays = PSE_data["stat_delays"]
        PSE_list.sort(key=lambda PSE: PSE.unique_index)
        
        PSE_lists.append( PSE_list )
        PSE_delays.append( old_delays )
    
    
    #### get list of blocks to open in SSPW ####
    PSE_to_correlate = []
    SSPW_blocks_to_open = [ [] for SSPW_file in  SSPW_folders]
    for PSE_num in PSE_order:
        PSE_index, PSE_list_num = PSE_ids[ PSE_num ]
        PSE_list = PSE_lists[ PSE_list_num ]
        PSE_past_delays = PSE_delays[ PSE_list_num ]
        
        
        ## find PSE
        found_PSE = None
        for PSE in PSE_list:
            if PSE.unique_index == PSE_index:
                found_PSE = PSE
                break
        
        if found_PSE is None:
            print("error! cannot find PSE")
            quit()
            
        PSE_polarity = None
        if PSE_num in PSE_polarization:
            PSE_polarity = PSE_polarization[PSE_num]
        
        new_PSE_to_fit = fitting_PSE( found_PSE, PSE_polarity, PSE_past_delays, old_delays)
        
        PSE_to_correlate.append(new_PSE_to_fit)
        
        if len(SSPW_folders) == 0:
            continue
        
        group_indeces = correlation_table_SSPW_group[PSE_num]
        if isinstance(group_indeces, int):
            group_indeces = [group_indeces]*len(stations_to_correlate)
        
        stations_to_PSE_use = []
        for sname, SSPW_group_index in zip(stations_to_correlate, group_indeces):
            block_indeces = new_PSE_to_fit.get_block_indeces( sname )
            for BI in block_indeces:
                if BI not in SSPW_blocks_to_open[ SSPW_group_index ]:
                    SSPW_blocks_to_open[ SSPW_group_index ].append( BI )
        
            
        
            
    
    ##open known SSPW##
    print("reading SSPW")
    SSPW_data_dict = read_SSPW_timeID_multiDir(timeID, SSPW_folders, data_loc="/home/brian/processed_files", stations=stations_to_correlate, 
                                               block_indeces=SSPW_blocks_to_open, blocks_per_file=100) 
    SSPW_multiple_dict = SSPW_data_dict["SSPW_multi_dicts"]
    SSPW_locs_dict = SSPW_data_dict["ant_locations"]
    


    ##### correlate SSPW to PSE according to matrix
    for PSE_num, PSE_to_fit in zip(PSE_order, PSE_to_correlate):
        
        SSPW_indeces = correlation_table[PSE_num]
            
        ## correlate SSPW
        group_indeces = correlation_table_SSPW_group[ PSE_num ]
        if isinstance(group_indeces, int):
            group_indeces = [group_indeces]*len(stations_to_correlate)
        
        stations_to_PSE_use = []
        for sname, SSPW_index, SSPW_group_index in zip(stations_to_correlate, SSPW_indeces, group_indeces):
            
            if SSPW_index is None:
                continue
            if SSPW_index == -1:
                stations_to_PSE_use.append(sname)
                continue
            
            SSPW_dict = SSPW_multiple_dict[SSPW_group_index]
            
            for SSPW in SSPW_dict[sname]:
                if SSPW.unique_index==SSPW_index:
                    PSE_to_fit.add_SSPW(SSPW)
                    break
                    
        ## prep the PSE
        PSE_to_fit.fitting_prep(stations_to_find_offsets, stations_to_keep, stations_to_PSE_use, PSE_ant_locs, SSPW_locs_dict)
        print()
        
            
            
    #### prep for fitting! ####
    N_delays = len(stations_to_find_offsets)
    
    _initial_guess_ = initial_guess ## becouse I'm lazy
    
    initial_guess = np.zeros(N_delays + 4*len(PSE_to_correlate), dtype=np.double)

    i = N_delays
    N_ant = 0
    for PSE in PSE_to_correlate:
        initial_guess[i:i+4] = PSE.initial_loc
        initial_guess[i+2] = np.abs(initial_guess[i+2]) ##make sure Z guess is positive
        i += 4
        N_ant += len(PSE.antenna_X)
        
    if _initial_guess_ is not None:
        initial_guess[:len(_initial_guess_)] = _initial_guess_
        
        
        
        
    workspace_sol = np.zeros(N_ant, dtype=np.double)
#    workspace_jac = np.zeros((N_ant, N_delays+4*len(PSE_to_correlate)), dtype=np.double)
    
    
    
    def objective_fun(sol):
#        global workspace_sol
        workspace_sol = np.zeros(N_ant, dtype=np.double)
        delays = sol[:N_delays]
        ant_i = 0
        param_i = N_delays
        for PSE in PSE_to_correlate:
            N_stat_ant = len(PSE.antenna_X)
            
            PSE.try_location_LS(delays, sol[param_i:param_i+4], workspace_sol[ant_i:ant_i+N_stat_ant])
            ant_i += N_stat_ant
            param_i += 4
            
        
            
        return workspace_sol
        
#        workspace_sol *= workspace_sol
#        return np.sum(workspace_sol)
        
    
    def objective_jac(sol):
#        global workspace_jac
        workspace_jac = np.zeros((N_ant, N_delays+4*len(PSE_to_correlate)), dtype=np.double)
        
        if np.isnan(workspace_jac).any():
            print("JAC NAN A")
            
        delays = sol[:N_delays]
        ant_i = 0
        param_i = N_delays
        for PSE in PSE_to_correlate:
            N_stat_ant = len(PSE.antenna_X)
            
            PSE.try_location_JAC(delays, sol[param_i:param_i+4],  workspace_jac[ant_i:ant_i+N_stat_ant, param_i:param_i+4],  workspace_jac[ant_i:ant_i+N_stat_ant, 0:N_delays])
            ant_i += N_stat_ant
            param_i += 4
        
        if np.isnan(workspace_jac).any():
            print("JAC NAN B")
            
        return workspace_jac
            
    
    print()
    print()
    
    
    initial_RMS = 0.0
    new_station_delays = initial_guess[:N_delays]
    param_i = N_delays
    for PSE in PSE_to_correlate:
        initial_RMS += PSE.SSqE_fit( new_station_delays,  initial_guess[param_i:param_i+4] )
        param_i += 4
    initial_RMS = np.sqrt(initial_RMS/N_ant)
    
    print("initial RMS:", initial_RMS)
    print()
    
    #### run fit N times ####
    best_solution = initial_guess
    best_RMS = initial_RMS
    new_guess = np.array( initial_guess )
    itters_since_change = 0
    for run_i in range(max_num_itter):
        itters_since_change += 1
        
        new_guess[:N_delays] = np.random.normal(scale=delay_width, size=N_delays) + best_solution[:N_delays] ## note use of best_solution, allows for translation. Faster convergence?
        
        param_i = N_delays
        ant_i = 0
        for PSE in PSE_to_correlate:
            new_guess[param_i:param_i+3] = np.random.normal(scale=position_width, size=3) + best_solution[param_i:param_i+3]
            
            N_stat_ant = len(PSE.antenna_X)
            new_guess[param_i+3] = PSE.estimate_T(new_guess[:N_delays], new_guess[param_i:param_i+4], workspace_sol[ant_i:ant_i+N_stat_ant])
            ant_i += N_stat_ant
            param_i += 4
        
        fit_res = least_squares(objective_fun, new_guess, jac=objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
#        fit_res = least_squares(objective_fun, new_guess, jac="3-point", method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
        
        total_RMS = 0.0
        new_station_delays = fit_res.x[:N_delays] 
        param_i = N_delays
        for PSE in PSE_to_correlate:
            total_RMS += PSE.SSqE_fit( new_station_delays,  fit_res.x[param_i:param_i+4] )
            param_i += 4
            
        total_RMS = np.sqrt(total_RMS/N_ant)
            
        print("run", run_i, "fit:", total_RMS)
        print("  ", fit_res.message)
        
        if total_RMS < best_RMS:
            print("  best fit so far")
            best_RMS = total_RMS
            best_solution = fit_res.x
            itters_since_change = 0
            
            print( repr(best_solution) )
            
        print()
        
        if itters_since_change == itters_till_convergence:
            break
        
    if itters_since_change != itters_till_convergence:
        print("solution not converged")
    else:
        print("solution converged")
        
        
    
    ####  print out results ####
            
    if print_fit_table:
        fit_table = PrettyTable()
        fit_table.field_names = ['id'] + stations_to_correlate + ['total']
        fit_table.float_format = '.2E'
        
        fit_table_OLD = PrettyTable()
        fit_table_OLD .field_names = ['id'] + stations_to_keep + ['total']
        fit_table_OLD .float_format = '.2E'
    
    i = N_delays
    total_RMS = 0.0
    new_station_delays = best_solution[:N_delays] 
    for PSE_num, PSE in zip(PSE_order, PSE_to_correlate):
        
        loc = best_solution[i:i+4]
        loc[2] = np.abs(loc[2])
        SSqE = PSE.SSqE_fit( new_station_delays,  loc )
        PSE.source_location = loc
        PSE.RMS_fit = np.sqrt(SSqE/len(PSE.antenna_X))
        
        print("PSE", PSE_num, PSE.PSE.unique_index)
        print("  RMS:", PSE.RMS_fit)
        print("  loc:", loc)
        
        if print_fit_table:
            fit_table_row = [PSE_num]
            station_fits = PSE.RMS_fit_byStation( new_station_delays,  loc )
            
            for sname in stations_to_correlate:
                if sname in station_fits:
                    fit_table_row.append( station_fits[sname] )
                else:
                    fit_table_row.append( '' )
            fit_table_row.append( np.sqrt(SSqE/len(PSE.antenna_X)) )
            fit_table.add_row( fit_table_row )
            
            fit_table_row_OLD = [ PSE_num ]
            for sname in stations_to_keep:
                if sname in station_fits:
                    fit_table_row_OLD.append( station_fits[sname] )
                else:
                    fit_table_row_OLD.append( '' )
            fit_table_row_OLD.append( np.sqrt(SSqE/len(PSE.antenna_X)) )
            fit_table_OLD.add_row( fit_table_row_OLD )
        
        
        i += 4
        total_RMS += SSqE
        
        
    if print_fit_table:
        print()
        print()
        print( fit_table_OLD )
        print()
        print()
        print( fit_table )
        
    print()
    print("total RMS:", np.sqrt(total_RMS/N_ant))
    
    #### open station info ####
    StationInfo_dict = read_station_info(timeID)
    
    ##### make plots of fits ####
    for sname, SSPW_group_index in stations_to_plot:
        sdata = StationInfo_dict[sname]
        SSPW_dict = SSPW_multiple_dict[ SSPW_group_index ]
        
        if sname in SSPW_dict:
            planewave_events = SSPW_dict[sname]
        else:
            planewave_events = []
        
        sdelay = 0.0
        if sname in stations_to_find_offsets:
            sdelay += new_station_delays[ stations_to_find_offsets.index(sname) ]
#        if sname in station_delay_dict:
#            sdelay += station_delay_dict[sname]
        
        
#        SSPW_amp_X = []
#        SSPW_even_amp_Y = []
#        SSPW_odd_amp_Y = []
#        for SSPW in planewave_events:
#            T = SSPW.ZAT[2]
#            even_amp = SSPW.get_ave_even_amp()
#            odd_amp = SSPW.get_ave_odd_amp()
            
#            SSPW_amp_X.append(T)
#            SSPW_even_amp_Y.append(even_amp)
#            SSPW_odd_amp_Y.append(odd_amp)
            
        data_SSPW  = [e.get_DataTime(sdata.sorted_antenna_names) for e in planewave_events]
        annotations  = [e.unique_index for e in planewave_events]
        
        CP = curtain_plot_CodeLog(StationInfo_dict, logging_folder + "/SSPWvsPSE_"+sname+"_"+str(SSPW_group_index))
        CP.addEventList(sname, data_SSPW, 'b', marker='o', size=50, annotation_list=annotations, annotation_size=20)
        

        data_PSE = [PSE.get_DataTime(sname, sdata.sorted_antenna_names) for PSE in PSE_to_correlate if sname in PSE.station_index_range]                
        model_PSE = [PSE.get_ModelTime(sname, sdata.sorted_antenna_names, sdelay) for PSE in PSE_to_correlate if sname in PSE.station_index_range]
        CP.addEventList(sname, data_PSE, 'r', marker='+', size=100)
        CP.addEventList(sname, model_PSE, 'r', marker='o', size=50)
        
        
#        PSE_amp_X = []
#        PSE_even_amp_Y = []
#        PSE_odd_amp_Y = []
        for PSE_num, PSE in zip(PSE_order, PSE_to_correlate):
            distance = np.linalg.norm( PSE.source_location[0:3] - sdata.get_station_location() )
            arrival_time = PSE.source_location[3] + distance/v_air + sdelay
            
            CP.CL.add_function( "plt.axvline", x=arrival_time, c='r' )
            CP.CL.add_function( "plt.annotate", str(PSE_num), xy=(arrival_time, np.max(CP.station_offsets[sname]) ), size=20)
            
#            PSE_amp_X.append(arrival_time)
#            PSE_even_amp_Y.append(PSE.orig_even_amp)
#            PSE_odd_amp_Y.append(PSE.orig_odd_amp)
#            
            
        
#        low_point = np.min(CP.station_offsets[sname])
#        top_point = np.max(CP.station_offsets[sname]) - low_point
#        MAX_AMP = max( np.max(PSE_even_amp_Y),  np.max(PSE_odd_amp_Y),  np.max(SSPW_even_amp_Y),  np.max(SSPW_odd_amp_Y))
#        
#        SSPW_even_amp_Y = np.array(SSPW_even_amp_Y)*top_point/MAX_AMP + low_point
#        SSPW_odd_amp_Y = np.array(SSPW_odd_amp_Y)*top_point/MAX_AMP + low_point
#        PSE_even_amp_Y = np.array(PSE_even_amp_Y)*top_point/MAX_AMP + low_point
#        PSE_odd_amp_Y = np.array(PSE_odd_amp_Y)*top_point/MAX_AMP + low_point
#        
#        CP.CL.add_function("plt.plot", PSE_amp_X, PSE_even_amp_Y, 'g+')
#        CP.CL.add_function("plt.plot", PSE_amp_X, PSE_odd_amp_Y, 'm+')
#        CP.CL.add_function("plt.plot", SSPW_amp_X, SSPW_even_amp_Y, 'go')
#        CP.CL.add_function("plt.plot", SSPW_amp_X, SSPW_odd_amp_Y, 'mo')
            
        CP.save()
            

            
        
        
    #### save point sources to binary ####
#    with open(data_dir + "/PSE_data", 'wb') as fout:
#        write_long(fout, len(PSE_to_correlate))
#        for PSE in PSE_to_correlate:
#            PSE.save_binary(fout)
            
    #### save station delays ####
    print()
    print("new delays:")
    for sname, new_delay in zip(stations_to_find_offsets, new_station_delays):
        if sname not in old_delays:
            old_delays[sname] = 0.0
        old_delays[sname] += new_delay
        print(sname, ":", new_delay)
    writeTXT_station_delays(data_dir + "/station_delays.txt", old_delays)
    
    print("new solution:")
    print( repr(best_solution) )
        
    
    for PSE in PSE_to_correlate:
        if PSE.PSE.unique_index in PSE_to_plot:
            
            CL = code_logger(logging_folder + "/PSE_"+str(PSE.PSE.unique_index))
            plotter = pyplot_emulator( CL )
            
            PSE.plot_trace_data( old_delays, plotter=plotter )
            
            plotter.show()
            CL.save()
            
            
            
            
    flash_location = np.zeros(3)
    param_i = N_delays
    for PSE in PSE_to_correlate:
        flash_location += best_solution[param_i:param_i+3]
        param_i += 4
    flash_location /= len(flash_location)
            
    print()
    print("plotting")
    for SSPW_group_index in SSPW_folders_to_plot:
        
#        CP = curtain_plot_CodeLog(StationInfo_dict, logging_folder + "/planewaves_"+str(SSPW_group_index))
        CP = curtain_plot(StationInfo_dict)
        
        SSPW_dict = SSPW_multiple_dict[ SSPW_group_index ]
        
        for i, sname in enumerate(stations_to_find_offsets+[referance_station]):
            print(sname, '(',i,'/',len(stations_to_find_offsets)+1, ')')
            sdata = StationInfo_dict[sname]
            station_offset = old_delays[sname]
            station_location = sdata.get_station_location()
            propagation_delay = np.linalg.norm( station_location-flash_location )/v_air
            
            data_SSPW  = [e.get_DataTime(sdata.sorted_antenna_names)-station_offset-propagation_delay for e in SSPW_dict[sname]]
            annotations  = [e.unique_index for e in SSPW_dict[sname]]
            CP.addEventList(sname, data_SSPW, 'b', marker='o', size=50, annotation_list=annotations, annotation_size=20)
            
            
            param_i = N_delays
            Y_offsets = CP.station_offsets[sname]
            for PSE_num, PSE in zip(PSE_order, PSE_to_correlate):
                PSE_time = best_solution[param_i+3] + np.linalg.norm( best_solution[param_i:param_i+3]-station_location )/v_air - propagation_delay
                param_i += 4
            
#                CP.CL.add_function( "plt.plot", [PSE_time,PSE_time], [Y_offsets[0], Y_offsets[-1]], c='r' )
#                CP.CL.add_function( "plt.annotate", str(PSE_num), xy=(PSE_time, Y_offsets[-1]) , size=20 )
                plt.plot( [PSE_time,PSE_time], [Y_offsets[0], Y_offsets[-1]], c='r' )
                plt.annotate( str(PSE_num), xy=(PSE_time, Y_offsets[-1]) , size=20 )
            
            
        CP.annotate_station_names(t_offset=0, xt=1)
#        CP.save()
        plt.show()










    
    