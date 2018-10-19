#!/usr/bin/env python3

from os import mkdir, rename
from os.path import isdir, isfile
import time

import numpy as np
from matplotlib import pyplot as plt

import h5py

from scipy.optimize import minimize

from LoLIM.utilities import v_air, processed_data_dir, logger
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.findRFI import FindRFI, window_and_filter
#from LoLIM.IO.metadata import getClockCorrections
from LoLIM.signal_processing import half_hann_window, remove_saturation, num_double_zeros

from LoLIM.interferometry import impulsive_imager_tools as II_tools
from LoLIM.interferometry import read_interferometric_PSE as R_IPSE
from LoLIM.interferometry.interferometry_absBefore import stochastic_minimizer, do_nothing, find_max_duration

            
def multipoint_minimizer(function, test_locs, initial_result=None, options={}, test_spot=None):
    if initial_result is None:
        best_itter = None
    else:
        best_itter = initial_result
        
    for tst_pnt in test_locs:
        
        res = minimize(function, tst_pnt, method="Nelder-Mead", options=options)
        
        if best_itter is None:
            best_itter = res
        elif res.fun < best_itter.fun:
            best_itter = res
            
    if test_spot is not None:
        res = minimize(function, test_spot, method="Nelder-Mead", options=options)
        
        if best_itter is None:
            best_itter = res
        elif res.fun < best_itter.fun:
            best_itter = res
        
    return best_itter

class relocator:
    def __init__(self, timeID, interferometry_header, pulse_length,
                 initial_RFI_block=None, RFI_num_blocks=None, RFI_max_blocks=None):
        self.timeID = timeID
        self.interferometry_header = interferometry_header
        self.pulse_length = pulse_length
        
        self.bounding_box = self.interferometry_header.bounding_box
        self.num_antennas_per_station = self.interferometry_header.num_antennas_per_station
               
        self.bad_antennas = self.interferometry_header.bad_antennas
        self.pol_flips = self.interferometry_header.pol_flips 
        
        print( self.bad_antennas )
        print( self.pol_flips )
        
        self.stations_to_exclude = self.interferometry_header.stations_to_exclude
        self.block_size = self.interferometry_header.block_size
        
        self.upsample_factor = self.interferometry_header.upsample_factor
        self.max_events_perBlock = self.interferometry_header.max_events_perBlock
        
        self.hann_window_fraction = self.interferometry_header.hann_window_fraction
        
        self.prefered_station = self.interferometry_header.prefered_station_name
        
        
        self.do_RFI_filtering = True
        self.use_saved_RFI_info= True
        self.initial_RFI_block = initial_RFI_block
        self.RFI_num_blocks = RFI_num_blocks
        self.RFI_max_blocks = RFI_max_blocks
        
        self.min_pref_ant_amplitude = 10
        self.min_pulse_amplitude = 2
        
        self.positive_saturation = 2046
        self.negative_saturation = -2047
        self.saturation_removal_length = 50
        self.saturation_hann_window_length = 50
        
        self.stage_2_max_itters = 10
        self.stage_2_max_stoch_itters = 50
        self.stage_2_convergence_length = 1
        self.stage_2_break_length = 200
        
        self.use_core_stations_S1 = True
        self.use_core_stations_S2 = False
        
        self.erase_pulses = True
        
    def open_files(self, log_func=do_nothing, force_metadata_ant_pos=True):
        
        #### open data files, and find RFI ####
        raw_fpaths = filePaths_by_stationName(self.timeID)
        self.station_names = []
        self.input_files = []
        self.RFI_filters = []
        self.station_to_antenna_indeces_dict = {}
        for station, fpaths  in raw_fpaths.items():
            if (station not in self.stations_to_exclude) and ( self.use_core_stations_S1 or self.use_core_stations_S2 or (not station[:2]=='CS') or station=="CS002" ):
                log_func("opening", station)
                self.station_names.append( station )
                self.station_to_antenna_indeces_dict[station] = []
                
                new_file = MultiFile_Dal1( fpaths, force_metadata_ant_pos=force_metadata_ant_pos )
                new_file.set_polarization_flips( self.pol_flips )
                self.input_files.append( new_file )
                
                RFI_result = None
                if self.do_RFI_filtering and self.use_saved_RFI_info:
                    self.RFI_filters.append( window_and_filter(timeID=self.timeID, sname=station) )
                
                elif self.do_RFI_filtering:
                    RFI_result = FindRFI(new_file, self.block_size, self.initial_RFI_block, self.RFI_num_blocks, self.RFI_max_blocks, verbose=False, figure_location=None)
                    self.RFI_filters.append( window_and_filter(find_RFI=RFI_result) )
                    
                else: ## only basic filtering
                    self.RFI_filters.append( window_and_filter(blocksize=self.block_size) )
                
                
                
        #### get antenna locations and delays ####
        
        self.num_antennas = len( self.interferometry_header.antenna_data )
        
        self.antenna_locations = np.zeros((self.num_antennas, 3), dtype=np.double)
        self.antenna_delays = np.zeros(self.num_antennas, dtype=np.double)
        self.antenna_data_offsets = np.zeros(self.num_antennas, dtype=np.long)
        self.half_antenna_data_length = np.zeros(self.num_antennas, dtype=np.long)
        self.is_not_core = np.zeros( self.num_antennas, dtype=np.bool )
        self.antennas_to_use = []
        
        for ant_i, antenna_info in enumerate( self.interferometry_header.antenna_data ):
            self.station_to_antenna_indeces_dict[ antenna_info.station ].append( ant_i )
            
            self.antenna_locations[ant_i] = antenna_info.location
            self.antenna_delays[ant_i] = antenna_info.timing_delay
            self.antenna_data_offsets[ant_i] = antenna_info.data_offset
            self.half_antenna_data_length[ant_i] = antenna_info.half_window_length
            
            self.is_not_core[ant_i] =  (not antenna_info.station[:2]=='CS') or antenna_info.station=="CS002"
            
            self.antennas_to_use.append( (self.station_names.index(antenna_info.station), antenna_info.station_antenna_i) )
            
        self.prefered_station_timing_offset = -0.000433356842721#self.interferometry_header.h5_group.attrs["prefered_station_timing_offset"]
        pref_stat_i = self.station_names.index( self.prefered_station )
        self.prefered_station_antenna_timing_offsets = self.input_files[pref_stat_i].get_timing_callibration_delays()
        
        #### find prefered station info ####
        boundingBox_center = np.average(self.bounding_box, axis=-1)
        ant_i = self.station_to_antenna_indeces_dict[ self.prefered_station ][0]
        
        prefered_stat_shortestWindowTime = 0
        for ant_j in range(self.num_antennas):
            window_time, throw, throw = find_max_duration(self.antenna_locations[ant_i], self.antenna_locations[ant_j], self.bounding_box, center=boundingBox_center )
            if window_time>prefered_stat_shortestWindowTime:
                prefered_stat_shortestWindowTime = window_time
                
    
        log_func("prefered station:", self.prefered_station )

        
        
        
        self.startBlock_exclusion = int(0.1*self.block_size) ### TODO: save these to header. 
        self.endBlock_exclusion = int(prefered_stat_shortestWindowTime/5.0E-9) + 1 + int(0.1*self.block_size) ## last bit accounts for hann-window 
        self.active_block_length = self.block_size - self.startBlock_exclusion - self.endBlock_exclusion##the length of block used for looking at data
        
        
        if (not self.use_core_stations_S1) or (not self.use_core_stations_S2):
            core_filtered_ant_locs = np.array( self.antenna_locations[self.is_not_core] )
            core_filtered_ant_delays = np.array( self.antenna_delays[self.is_not_core] )
        
        #### allocate some memory ####
        self.data_block = np.empty((self.num_antennas,self.block_size), dtype=np.complex)
        self.hilbert_envelope_tmp = np.empty(self.block_size, dtype=np.double)
        
        #### initialize stage 1 ####
        if self.use_core_stations_S1:
            self.trace_length_stage1 = 2*np.max(self.half_antenna_data_length )
            S1_ant_locs = self.antenna_locations
            S1_ant_delays = self.antenna_delays
        else:
            self.trace_length_stage1 = 2*np.max(self.half_antenna_data_length[self.is_not_core] )
            S1_ant_locs = core_filtered_ant_locs
            S1_ant_delays = core_filtered_ant_delays
        self.trace_length_stage1 = 2**( int(np.log2( self.trace_length_stage1 )) + 1 )
        self.stage_1_imager = II_tools.image_data_stage1(S1_ant_locs, S1_ant_delays, self.trace_length_stage1, self.upsample_factor)

        #### initialize stage 2 ####
        if self.use_core_stations_S2:
            S2_ant_locs = self.antenna_locations
            S2_ant_delays = self.antenna_delays
        else:
            S2_ant_locs = core_filtered_ant_locs
            S2_ant_delays = core_filtered_ant_delays
        self.trace_length_stage2 = 2**( int(np.log2( self.pulse_length )) + 1 )
        self.stage_2_window = half_hann_window(self.pulse_length, self.hann_window_fraction)
        self.stage_2_imager = II_tools.image_data_stage2_absBefore(S2_ant_locs, S2_ant_delays, self.trace_length_stage2, self.upsample_factor)

        self.erasure_window = 1.0-self.stage_2_window
                
    def save_header(self, h5_header_file):
        header_group = h5_header_file.create_group("header")
        header_group.attrs["timeID"] = self.timeID
        header_group.attrs["bounding_box"] = self.bounding_box
        header_group.attrs["pulse_length"] = self.pulse_length
        header_group.attrs["num_antennas_per_station"] = self.num_antennas_per_station
        header_group.attrs["stations_to_exclude"] =  np.array(self.stations_to_exclude, dtype='S')
        header_group.attrs["do_RFI_filtering"] = self.do_RFI_filtering
        header_group.attrs["initial_RFI_block"] = self.initial_RFI_block
        header_group.attrs["RFI_num_blocks"] = self.RFI_num_blocks
        header_group.attrs["RFI_max_blocks"] = self.RFI_max_blocks
        header_group.attrs["block_size"] = self.block_size
        header_group.attrs["upsample_factor"] = self.upsample_factor
        header_group.attrs["max_events_perBlock"] = self.max_events_perBlock
        header_group.attrs["min_pulse_amplitude"] = self.min_pulse_amplitude
        header_group.attrs["min_pref_ant_amplitude"] = self.min_pref_ant_amplitude
        header_group.attrs["positive_saturation"] = self.positive_saturation
        header_group.attrs["negative_saturation"] = self.negative_saturation
        header_group.attrs["saturation_removal_length"] = self.saturation_removal_length
        header_group.attrs["hann_window_fraction"] = self.hann_window_fraction
        header_group.attrs["stage_1_converg_num"] = self.stage_1_converg_num
        header_group.attrs["stage_1_converg_radius"] = self.stage_1_converg_radius
        header_group.attrs["stage_1_max_itters"] = self.stage_1_max_itters
        header_group.attrs["use_core_stations_S1"] = self.use_core_stations_S1
        header_group.attrs["use_core_stations_S2"] = self.use_core_stations_S2
        header_group.attrs["stage_2_max_itter"] = self.stage_2_max_itters
        header_group.attrs["stage_2_convergence_length"] = self.stage_2_convergence_length
        header_group.attrs["stage_2_break_length"] = self.stage_2_break_length
        header_group.attrs["prefered_station_name"] = self.prefered_station
        header_group.attrs["trace_length_stage1"] = self.trace_length_stage1
        header_group.attrs["trace_length_stage2"] = self.trace_length_stage2
        header_group.attrs["erase_pulses"] = self.erase_pulses
        header_group.attrs["prefered_station_timing_offset"] = self.prefered_station_timing_offset
        
        
#        header_group.attrs["station_timing_offsets"] = self.station_timing_offsets
#        header_group.attrs["additional_antenna_delays"] = self.additional_ant_delays
        header_group.attrs["bad_antennas"] = np.array(self.bad_antennas, dtype='S')
        header_group.attrs["polarization_flips"] = np.array(self.pol_flips, dtype='S')
        
        no_core_i = 0
        for ant_i, (station_i, station_ant_i) in enumerate(self.antennas_to_use):
            data_file = self.input_files[station_i]
            station = self.station_names[station_i]
            
            ant_name = data_file.get_antenna_names()[ station_ant_i ]
            
            
            antenna_group = header_group.create_group( str(ant_i) )
            antenna_group.attrs["antenna_name"] = ant_name
            antenna_group.attrs["location"] = self.antenna_locations[ant_i]
            antenna_group.attrs["timing_delay"] = self.antenna_delays[ant_i] 
            antenna_group.attrs["half_window_length"] = self.half_antenna_data_length[ant_i]
            antenna_group.attrs["data_offset"] = self.antenna_data_offsets[ant_i]
            antenna_group.attrs["station"] = station
            antenna_group.attrs["station_antenna_i"] = station_ant_i
            antenna_group.attrs["with_core_ant_i"] = ant_i
            
            if self.is_not_core[ant_i]:
                antenna_group.attrs["no_core_ant_i"] = no_core_i
                no_core_i += 1
            else:
                antenna_group.attrs["no_core_ant_i"] = np.nan             
    
    
    def process_block(self, start_index, block_index, S1_search_locations, h5_groupobject, log_func=do_nothing):
        
        #### open and filter the data ####
        prefered_ant_i = None
        prefered_ant_dataLoss = np.inf
        for ant_i, (station_i, station_ant_i) in enumerate(self.antennas_to_use):
            data_file = self.input_files[station_i]
            RFI_filter = self.RFI_filters[station_i]
            
            offset = self.antenna_data_offsets[ant_i]
            self.data_block[ant_i, :] = data_file.get_data(start_index+offset, self.block_size, antenna_index=station_ant_i) ## get the data. accounting for the offsets calculated earlier
            
            remove_saturation(self.data_block[ant_i, :], self.positive_saturation, self.negative_saturation, post_removal_length=self.saturation_removal_length, 
                              half_hann_length=self.saturation_hann_window_length)
            
            
            if ant_i in self.station_to_antenna_indeces_dict[ self.prefered_station ]: ## this must be done before filtering
                num_D_zeros = num_double_zeros( self.data_block[ant_i, :] )
                if num_D_zeros < prefered_ant_dataLoss: ## this antenna could be teh antenna, in the prefered station, with least data loss
                    prefered_ant_dataLoss = num_D_zeros
                    prefered_ant_i = ant_i
            
            self.data_block[ant_i, :] = RFI_filter.filter( self.data_block[ant_i, :] )
            
            
        ### make output object ###
        h5_groupobject = h5_groupobject.create_group(str(block_index))
        h5_groupobject.attrs['block_index'] = block_index
        h5_groupobject.attrs['start_index'] = start_index
        h5_groupobject.attrs['prefered_ant_i'] = prefered_ant_i
            
            
        #### find sources ####
        np.abs( self.data_block[ prefered_ant_i ], out=self.hilbert_envelope_tmp )
        for event_i in range(self.max_events_perBlock):
            
            ## find peak ##
            peak_loc = np.argmax(self.hilbert_envelope_tmp[self.startBlock_exclusion : -self.endBlock_exclusion ]) + self.startBlock_exclusion 
            trace_start_loc = peak_loc - int( self.pulse_length/2 ) # pobably account for fact that trace is now longer than pulse_length on prefered antenna
            
            if self.hilbert_envelope_tmp[peak_loc] < self.min_pref_ant_amplitude:
                log_func("peaks are too small. Done searching")
                break
            
            ## select data for stage one ##
            s1_ant_i = 0
            for ant_i, (station_i, station_ant_i) in enumerate(self.antennas_to_use):
                if self.use_core_stations_S1 or self.is_not_core[ant_i]:
                    half_window_length = self.half_antenna_data_length[ant_i]
                    self.stage_1_imager.set_data( self.data_block[ant_i][trace_start_loc : trace_start_loc+2*half_window_length], s1_ant_i )
                    s1_ant_i += 1
                
    
            ## fft and xcorrelation ##
            self.stage_1_imager.prepare_image( prefered_ant_i, self.min_pulse_amplitude )
            log_func("source:", event_i)
                   
            stage_1_result = multipoint_minimizer(self.stage_1_imager.intensity, S1_search_locations)
            log_func("   s1 done")
            
            
            ## select data for stage 2 ##
            previous_solution = stage_1_result
            converged = False
            for stage2loop_i in range(self.stage_2_max_itters):            
            
                problem = False
                s2_ant_i = -1
                for ant_i in range( self.num_antennas ):
                    if self.use_core_stations_S2 or self.is_not_core[ant_i]:
                        s2_ant_i += 1 ## do this here cause of break below
                    
                        modeled_dt = -(  np.linalg.norm( self.antenna_locations[ prefered_ant_i ]-previous_solution.x ) - 
                                       np.linalg.norm( self.antenna_locations[ant_i]-previous_solution.x )   )/v_air
                        modeled_dt -= self.antenna_delays[ prefered_ant_i ] -  self.antenna_delays[ant_i]
                        modeled_dt /= 5.0E-9
                        
                        modeled_dt += peak_loc
                        modeled_dt = int(modeled_dt)
                        
                        if modeled_dt+int(self.pulse_length/2) >= len(self.data_block[ant_i]):
                            problem = True
                            break
                         
                        self.stage_2_imager.set_data( self.data_block[ant_i, modeled_dt-int(self.pulse_length/2):modeled_dt+int(self.pulse_length/2)]*self.stage_2_window, s2_ant_i, -modeled_dt*5.0E-9 )
                        
                        
                if problem:
                    log_func("unknown problem. LOC at", previous_solution.x)
                    self.hilbert_envelope_tmp[peak_loc-int(self.pulse_length/2):peak_loc+int(self.pulse_length/2)] = 0.0
                    continue
                    
                ## fft and and xcorrelation ##
                self.stage_2_imager.prepare_image( self.min_pulse_amplitude )
#                
                BB = np.array( [ [previous_solution.x[0]-50, previous_solution.x[0]+50], [previous_solution.x[1]-50, previous_solution.x[1]+50], [previous_solution.x[2]-50, previous_solution.x[2]+50] ] )
                stage_2_result, s2_itternum, num_stage2_itters = stochastic_minimizer(self.stage_2_imager.intensity_ABSbefore , BB, converg_num=5, test_spot=previous_solution.x,
                                                                           converg_rad=self.stage_2_convergence_length, max_itters=self.stage_2_max_stoch_itters, options={'maxiter':1000})
                
                
                D = np.linalg.norm( stage_2_result.x - previous_solution.x )
                log_func("   s2 itter: {:2d} {:4.2f} {:d}".format(stage2loop_i, -stage_2_result.fun, int(D)) )
                if D < self.stage_2_convergence_length:
                    converged = True
                    break
                elif D > self.stage_2_break_length:
                    converged = False
                    break
                
                previous_solution = stage_2_result
            
            new_stage_1_result = minimize(self.stage_1_imager.intensity, stage_2_result.x, method="Nelder-Mead", options={'maxiter':1000})
            log_func("   old S1: {:4.2f} new SH1: {:4.2f}".format(-stage_1_result.fun, -new_stage_1_result.fun) )
            if stage_1_result.fun < new_stage_1_result.fun:
                S1_S2_distance = np.linalg.norm(stage_1_result.x-stage_2_result.x)
            else:
                S1_S2_distance = np.linalg.norm(new_stage_1_result.x-stage_2_result.x)
                
                
            log_func("   loc: {:d} {:d} {:d}".format(int(stage_2_result.x[0]), int(stage_2_result.x[1]), int(stage_2_result.x[2])) )
            log_func("   S1-S2 distance: {:d} converged: {} ".format( int(S1_S2_distance), converged) )
            log_func("   intensity: {:4.2f} amplitude: {:d} ".format( -stage_2_result.fun, int(self.hilbert_envelope_tmp[peak_loc])) )
            log_func()
            log_func()
            
            
            
            ## save to file ##
            source_dataset = h5_groupobject.create_dataset(str(event_i), (self.num_antennas,self.pulse_length), dtype=np.complex)
            source_dataset.attrs["loc"] = stage_2_result.x
            source_dataset.attrs["unique_index"] = block_index*self.max_events_perBlock + event_i
            
            source_time_s2 =  (peak_loc+start_index+self.antenna_data_offsets[prefered_ant_i])*5.0E-9 - np.linalg.norm( stage_2_result.x - self.antenna_locations[ prefered_ant_i ] )/v_air
            source_time_s2 -= self.prefered_station_timing_offset + self.prefered_station_antenna_timing_offsets[ self.antennas_to_use[prefered_ant_i][1] ]
            source_dataset.attrs["T"] = source_time_s2
            
            source_dataset.attrs["peak_index"] = peak_loc
            source_dataset.attrs["intensity"] = -stage_2_result.fun
            source_dataset.attrs["stage_1_success"] = True
            source_dataset.attrs["stage_1_num_itters"] = len(S1_search_locations)
            source_dataset.attrs["amplitude"] = self.hilbert_envelope_tmp[peak_loc]
            source_dataset.attrs["S1_S2_distance"] = S1_S2_distance
            source_dataset.attrs["converged"] = converged
            
            source_time_s1 =  (peak_loc+start_index+self.antenna_data_offsets[prefered_ant_i])*5.0E-9 - np.linalg.norm( stage_1_result.x - self.antenna_locations[ prefered_ant_i ] )/v_air
            source_time_s1 -= self.prefered_station_timing_offset + self.prefered_station_antenna_timing_offsets[ self.antennas_to_use[prefered_ant_i][1] ]
            source_dataset.attrs["XYZT_s1"] = np.append(stage_1_result.x, [source_time_s1])
            
                
            #### erase the peaks !! ####
#            self.hilbert_envelope_tmp[peak_loc-int(self.pulse_length/2):peak_loc+int(self.pulse_length/2)]  *= self.erasure_window
            self.hilbert_envelope_tmp[peak_loc-int(self.pulse_length/2):peak_loc+int(self.pulse_length/2)] = 0.0
            if converged and self.erase_pulses:
                for ant_i in range( self.num_antennas ):
                    
                    modeled_dt = -(  np.linalg.norm( self.antenna_locations[ prefered_ant_i ]-stage_2_result.x ) - 
                                   np.linalg.norm( self.antenna_locations[ant_i]-stage_2_result.x )   )/v_air
                    modeled_dt -= self.antenna_delays[ prefered_ant_i ] -  self.antenna_delays[ant_i]
                    modeled_dt /= 5.0E-9
                    
                    modeled_dt += peak_loc
                    modeled_dt = int(modeled_dt)
                    
                    source_dataset[ant_i] = self.data_block[ant_i, modeled_dt-int(self.pulse_length/2):modeled_dt+int(self.pulse_length/2)]
                    self.data_block[ant_i, modeled_dt-int(self.pulse_length/2):modeled_dt+int(self.pulse_length/2)] *= self.erasure_window
            
        


#### TODO: if antenna data is too low amplitude, then do not include in correlation.
if __name__ == "__main__":
    #### TODO: how to handle different polarization, probably treat then as different antenna I think
    timeID = "D20170929T202255.000Z"
    input_folder = "interferometry_out4"
    output_folder = "interferometry_out4_reSearch"
    
    skip_blocks_done = True
   
    initial_datapoint = 229376000 ## somehow need to get this from header
    first_block = 0
    blocks_per_run = 174
    run_number =1#/5
    
    block_override = None#282
    
    half_blocks_to_search = 2 ## actual blocks searched is 2*half_blocks_to_search + 1
    min_intensity = 0.8
    max_S1S2_distance = 50
    
    pulse_length = 50
    
    ### load previous data
    processed_data_folder = processed_data_dir(timeID)
    data_dir = processed_data_folder + "/" + input_folder
    interferometry_header, IPSE_list = R_IPSE.load_interferometric_PSE( data_dir )
    
    IPSE_by_block = {}
    for IPSE in IPSE_list:
        block = IPSE.block_index
        if block not in IPSE_by_block:
            IPSE_by_block[block] = []
        
        if IPSE.converged and IPSE.intensity > min_intensity and IPSE.S1_S2_distance < max_S1S2_distance:
            IPSE_by_block[block].append( IPSE )
    
    
    ### create the imager ###
    
    imager_utility = relocator(timeID, interferometry_header, pulse_length)
    

    imager_utility.do_RFI_filtering = True
    imager_utility.use_saved_RFI_info = True
#    imager_utility.initial_RFI_block= 5
#    imager_utility.RFI_num_blocks = 20
#    imager_utility.RFI_max_blocks = 100
    
    imager_utility.max_events_perBlock = 100
    imager_utility.min_pulse_amplitude = 10
    
    imager_utility.half_blocks_to_search = 2
    imager_utility.min_intensity = 40
    
    imager_utility.erase_pulses = True
    
    
    #### set logging ####
    processed_data_folder = processed_data_dir(timeID)
    data_dir = processed_data_folder + "/" + output_folder
    if not isdir(data_dir):
        mkdir(data_dir)
        
    logging_folder = data_dir + '/logs_and_plots'
    if not isdir(logging_folder):
        mkdir(logging_folder)
    
    file_number = 0
    while True:
        fname = logging_folder + "/log_run_"+str(file_number)+".txt"
        if isfile(fname) :
            file_number += 1
        else:
            break
            
    
    logger_function = logger()
    logger_function.set(logging_folder + "/log_run_"+str(file_number)+".txt")
    logger_function.take_stdout()
    logger_function.take_stderr()
    
    
    #### TODO!#### improve log all options
    logger_function("timeID:", timeID)
    logger_function("input folder:", input_folder)
    logger_function("output folder:", output_folder)
    logger_function("first block:", first_block)
    logger_function("blocks per run:", blocks_per_run)
    logger_function("run number:", run_number)
    logger_function("stations excluded:", imager_utility.stations_to_exclude)
    logger_function("prefered station:", imager_utility.prefered_station)
    
    
    
    if block_override is None:
        logger_function("processing step:", run_number)
    else:
        logger_function("block override:", block_override)
    
    
    
    #### open files, save header if necisary ####
    imager_utility.open_files(logger_function, True)
    if run_number==0:
        header_outfile = h5py.File(data_dir + "/header.h5", "w")
        imager_utility.save_header( header_outfile )
        header_outfile.close()    
    
    #### run the algorithm!! ####
    for block_index in range(run_number*blocks_per_run+first_block, (run_number+1)*blocks_per_run+first_block):
        if block_override is not None:
            block_index = block_override
        
        out_fname = data_dir + "/block_"+str(block_index)+".h5"
        tmp_fname = data_dir + "/tmp_"+str(block_index)+".h5"
        
        if skip_blocks_done and isfile(out_fname):
            logger_function("block:", block_index, "already completed. Skipping")
            continue
        
        search_locations = []
        for search_block in range(block_index-half_blocks_to_search, block_index+half_blocks_to_search+1):
            if search_block in IPSE_by_block:
                search_locations += [IPSE.loc for IPSE in IPSE_by_block[search_block]]
        
        if len(search_locations) > 0:
            logger_function("starting processing block:", block_index, len(search_locations), 'search locations')
            start_time = time.time()
            
            block_outfile = h5py.File(tmp_fname, "w")
            block_start = initial_datapoint + imager_utility.active_block_length*block_index
            imager_utility.process_block( block_start, block_index, search_locations, block_outfile, log_func=logger_function )
            
            block_outfile.close()
            rename(tmp_fname, out_fname)
    
            logger_function("block done. took:", (time.time() - start_time), 's')
            logger_function()
            
        else:
            logger_function("skipping block", block_index, 'no search locations')
            
        
        if block_override is not None:
            break
        
    
    logger_function("done processing")
    
    
    
    
    
    
    
            
            