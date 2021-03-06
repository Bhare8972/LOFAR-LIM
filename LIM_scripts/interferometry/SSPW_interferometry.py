#!/usr/bin/env python3

from os import mkdir, rename
from os.path import isdir, isfile
import time

import numpy as np
from matplotlib import pyplot as plt

import h5py

from scipy.optimize import minimize

from LoLIM.utilities import v_air, RTD, processed_data_dir, logger
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.findRFI import FindRFI, window_and_filter
#from LoLIM.IO.metadata import getClockCorrections
from LoLIM.read_pulse_data import read_antenna_delays, read_station_delays, read_bad_antennas, read_antenna_pol_flips
from LoLIM.signal_processing import half_hann_window, remove_saturation, num_double_zeros
from LoLIM.antenna_response import LBA_antenna_model

from LoLIM.interferometry import PW_imager_tools as II_tools

def do_nothing(*A, **B):
    pass

def pairData_evenAnts(input_file,  bad_antennas=[], max_antennas=np.inf):
    even_bad_antennas = [A[0] for A in bad_antennas]
    
    antennas = []
    ant_names = input_file.get_antenna_names()
    num_antennas_in_station = len( ant_names )
    for x in range( min( int(num_antennas_in_station/2) ,max_antennas) ):
        if ant_names[x*2] not in even_bad_antennas:
            antennas.append( x*2 )
            
    return np.array( antennas, dtype=int )

def stochastic_minimizer(function, converg_num=5, converg_rad=0.005, max_itters=50, initial_result=None, options={}, test_spot=None):
    
    def rads_apart(cosAlphacosBeta1, cosAlphacosBeta2):
        dz1 = cosAlphacosBeta1[0]*cosAlphacosBeta1[0] + cosAlphacosBeta1[1]*cosAlphacosBeta1[1]
        dz2 = cosAlphacosBeta2[0]*cosAlphacosBeta2[0] + cosAlphacosBeta2[1]*cosAlphacosBeta2[1]
        
        dz1 = np.sqrt(1 - dz1 )
        dz2 = np.sqrt(1 - dz2 )
        
        D = cosAlphacosBeta1[0]*cosAlphacosBeta2[0]
        D += cosAlphacosBeta1[1]*cosAlphacosBeta2[1]
        D += dz1*dz2
        
        return np.arccos( D )
    
    if initial_result is None:
        best_itter = None
        num_itters = 0
    else:
        best_itter = initial_result
        num_itters = 1
        
        
        
    if test_spot is not None:
        res = minimize(function, test_spot, method="Nelder-Mead", options=options)
        
        if best_itter is None:
            best_itter = res
            num_itters = 1
        elif rads_apart( best_itter.x, res.x ) < converg_rad:
            num_itters += 1
            if res.fun < best_itter.fun:
                best_itter = res
        elif res.fun < best_itter.fun:
            best_itter = res
            num_itters = 1
        
        
    tst_pnt = np.zeros(2)
    for i in range(max_itters):
        R = np.random.random()*0.99
        T = np.random.random()*2*np.pi
        
        tst_pnt[0] = R*np.cos(T)
        tst_pnt[1] = R*np.sin(T)
        
        res = minimize(function, tst_pnt, method="Nelder-Mead", options=options)
        
        if best_itter is None:
            best_itter = res
            num_itters = 1
        elif rads_apart( best_itter.x, res.x ) < converg_rad:
            num_itters += 1
            if res.fun < best_itter.fun:
                best_itter = res
        elif res.fun < best_itter.fun:
            best_itter = res
            num_itters = 1
            
        if num_itters == converg_num:
            break
        
    if num_itters < converg_num:
        best_itter.success = False
        
    return best_itter, i+1, num_itters
        
class interferometric_locator:
    def __init__(self, timeID, additional_antenna_delays_fname, bad_antennas_fname, pol_flips_fname, 
                 pulse_length, initial_RFI_block=None, RFI_num_blocks=None, RFI_max_blocks=None):
        self.timeID = timeID
        self.pulse_length = pulse_length
               
        self.additional_antenna_delays_fname = additional_antenna_delays_fname
        self.bad_antennas_fname = bad_antennas_fname
        self.pol_flips_fname = pol_flips_fname
        
        self.do_RFI_filtering = True
        self.use_saved_RFI_info= True
        self.initial_RFI_block = initial_RFI_block
        self.RFI_num_blocks = RFI_num_blocks
        self.RFI_max_blocks = RFI_max_blocks
        
        #### other settings  that may be changed ####
        self.block_size = 2**16
        self.max_num_antennas = np.inf
        
        self.upsample_factor = 8
        self.max_events_perBlock = 100
        self.min_pref_ant_amplitude = 10
        self.min_pulse_amplitude = 2
        
        self.positive_saturation = 2046
        self.negative_saturation = -2047
        self.saturation_removal_length = 50
        self.saturation_hann_window_length = 50
        
        self.hann_window_fraction = 0.1
        
        self.converg_num = 5
        self.converg_radius = 0.01
        self.max_itters = 500
        
    def open_files(self, station, log_func=do_nothing, force_metadata_ant_pos=True):
        processed_data_loc = processed_data_dir(self.timeID)
        
        #### open callibration data files ####
        ### TODO: fix these so they are accounted for in a more standard maner
        self.additional_ant_delays = read_antenna_delays( processed_data_loc+'/'+self.additional_antenna_delays_fname )
        self.bad_antennas = read_bad_antennas( processed_data_loc+'/'+self.bad_antennas_fname )
        self.pol_flips = read_antenna_pol_flips( processed_data_loc+'/'+self.pol_flips_fname )
        
        #### open data files, and find RFI ####
        self.station = station
        raw_fpaths = filePaths_by_stationName(self.timeID)[station]
        self.input_file = MultiFile_Dal1( raw_fpaths, force_metadata_ant_pos=force_metadata_ant_pos )
        self.input_file.set_polarization_flips( self.pol_flips )
        
        if self.do_RFI_filtering and self.use_saved_RFI_info:
            self.RFI_filters.append( window_and_filter(timeID=self.timeID, sname=station) )
        
        elif self.do_RFI_filtering:
            RFI_result = FindRFI(self.input_file, self.block_size, self.initial_RFI_block, self.RFI_num_blocks, self.RFI_max_blocks, verbose=False, figure_location=None)
            self.RFI_filters.append( window_and_filter(find_RFI=RFI_result) )
            
        else: ## only basic filtering
            self.RFI_filters.append( window_and_filter(blocksize=self.block_size) )
        
        
        
        #### find antenna pairs ####
        self.antennas_to_use = pairData_evenAnts(self.input_file, self.bad_antennas, self.max_num_antennas )
        self.num_antennas = len(self.antennas_to_use)
        
        
        #### get antenna locations and delays ####
        self.antenna_locations = np.zeros((self.num_antennas, 3), dtype=np.double)
        self.antenna_delays = np.zeros(self.num_antennas, dtype=np.double)
        
        for ant_i, station_ant_i in enumerate(self.antennas_to_use):
            
            ant_name = self.input_file.get_antenna_names()[ station_ant_i ]
            
            self.antenna_locations[ant_i] = self.input_file.get_LOFAR_centered_positions()[ station_ant_i ]
            self.antenna_delays[ant_i] = self.input_file.get_timing_callibration_delays()[ station_ant_i ]
            
            ## add additional timing delays
            ##note this only works for even antennas!
            if ant_name in self.additional_ant_delays:
                self.antenna_delays[ant_i] += self.additional_ant_delays[ ant_name ][0]
        
        
        
        ##### find window offsets and lengths ####
        self.half_antenna_data_length = np.zeros(self.num_antennas, dtype=np.long)
                
        for ant_i, station_ant_i in enumerate(self.antennas_to_use):
            
            ### find max duration to any of the prefered antennas
            max_distance = 0.0
            for ant_j, station_ant_j in enumerate(self.antennas_to_use):
                if ant_j == ant_i:
                    continue
                
                distance = np.linalg.norm( self.antenna_locations[ ant_j ]-self.antenna_locations[ant_i] )
                if distance > max_distance:
                    max_distance = distance
                    
            self.half_antenna_data_length[ant_i] = int(self.pulse_length/2) + int(max_distance/(v_air*5.0E-9))
        
        self.block_exclusion_length = int(0.1*self.block_size) + np.max( self.half_antenna_data_length ) + 1
        self.active_block_length = self.block_size - 2*self.block_exclusion_length##the length of block used for looking at data
        
        self.trace_length = 2*np.max(self.half_antenna_data_length )
        self.trace_length = 2**( int(np.log2( self.trace_length )) + 1 )
        
        
        #### allocate some memory ####
        self.data_block = np.empty((self.num_antennas,self.block_size), dtype=np.complex)
        self.hilbert_envelope_tmp = np.empty(self.block_size, dtype=np.double)
        
        self.stage_1_imager = II_tools.image_data(self.antenna_locations, self.antenna_delays, self.trace_length, self.upsample_factor)


        
    def save_header(self, h5_header_file):
        header_group = h5_header_file.create_group("header")
        header_group.attrs["timeID"] = self.timeID
        header_group.attrs["pulse_length"] = self.pulse_length
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
        header_group.attrs["converg_num"] = self.converg_num
        header_group.attrs["converg_radius"] = self.converg_radius
        header_group.attrs["max_itters"] = self.max_itters
        header_group.attrs["trace_length"] = self.trace_length
        header_group.attrs["max_num_antennas"] = self.max_num_antennas
        header_group.attrs["block_exclusion_length"] = self.block_exclusion_length
        header_group.attrs["station"] = self.station
        
        
#        header_group.attrs["station_timing_offsets"] = self.station_timing_offsets
#        header_group.attrs["additional_antenna_delays"] = self.additional_ant_delays
        header_group.attrs["bad_antennas"] = np.array(self.bad_antennas, dtype='S')
        header_group.attrs["polarization_flips"] = np.array(self.pol_flips, dtype='S')
        
        for ant_i, station_ant_i in enumerate(self.antennas_to_use):
            
            ant_name = self.input_file.get_antenna_names()[ station_ant_i ]
            
            antenna_group = header_group.create_group( str(ant_i) )
            antenna_group.attrs["antenna_name"] = ant_name
            antenna_group.attrs["location"] = self.antenna_locations[ant_i]
            antenna_group.attrs["timing_delay"] = self.antenna_delays[ant_i] 
            antenna_group.attrs["half_window_length"] = self.half_antenna_data_length[ant_i]
            antenna_group.attrs["station_antenna_i"] = station_ant_i
            antenna_group.attrs["with_core_ant_i"] = ant_i
                
I AM HERE. FIX STUFF BELOW. CHECK STUFF ABOVE
        
    def process_block(self, start_index, block_index, h5_groupobject, log_func=do_nothing):
        
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
                   
            stage_1_result, num_itter, num_stage1_itters = stochastic_minimizer(self.stage_1_imager.intensity, self.bounding_box, converg_num=self.stage_1_converg_num, 
                                                                       converg_rad=self.stage_1_converg_radius, max_itters=self.stage_1_max_itters)
            
            log_func("   stoch. itters:", num_itter, num_stage1_itters)
            
            
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
            source_dataset.attrs["stage_1_success"] = (num_stage1_itters==self.stage_1_converg_num)
            source_dataset.attrs["stage_1_num_itters"] = num_stage1_itters
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
    output_folder = "interferometry_out4_no_erase"
    
    skip_blocks_done = True
   
    block_size = 2**16
    initial_datapoint = 229376000 
    first_block = 0
    blocks_per_run = 140
    run_number = 4#5
    
    block_override = None#338 339 340 341 
    
    station_delays = "station_delays2.txt"
    additional_antenna_delays = "ant_delays.txt"
    bad_antennas = "bad_antennas.txt"
    polarization_flips = "polarization_flips.txt"
    
    bounding_box = np.array([[-20000.,  -14000.], [  7000.,  12000.], [     0. ,  6000.]], dtype=np.double)
    
    pulse_length = 50
    num_antennas_per_station = 6
    
    
    
    imager_utility = interferometric_locator(timeID,   station_delays, additional_antenna_delays, bad_antennas, polarization_flips,
                            bounding_box, pulse_length, num_antennas_per_station)
    
    imager_utility.stations_to_exclude = ['CS026', 'CS028', 'RS106', 'RS305', 'RS205', 'CS201', 'RS407'] 
    imager_utility.block_size = block_size
    
    imager_utility.prefered_station = None
    imager_utility.use_core_stations_S1 = True
    imager_utility.use_core_stations_S2 = False
    
    imager_utility.do_RFI_filtering = True
    imager_utility.use_saved_RFI_info = True
    imager_utility.initial_RFI_block= 5
    imager_utility.RFI_num_blocks = 20
    imager_utility.RFI_max_blocks = 100
    
    imager_utility.upsample_factor = 8
    imager_utility.max_events_perBlock = 100
    
    imager_utility.stage_1_converg_num = 100
    imager_utility.stage_1_max_itters = 1500
    
    imager_utility.erase_pulses = False
    
    
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
    logger_function.set( fname )
    logger_function.take_stdout()
    logger_function.take_stderr()
    
    
    #### TODO!#### improve log all options
    logger_function("timeID:", timeID)
    logger_function("output folder:", output_folder)
    logger_function("block size:", block_size)
    logger_function("initial data point:", initial_datapoint)
    logger_function("first block:", first_block)
    logger_function("blocks per run:", blocks_per_run)
    logger_function("run number:", run_number)
    logger_function("block override:", block_override)
    logger_function("station delay file:", station_delays)
    logger_function("additional antenna delays file:", additional_antenna_delays)
    logger_function("bad antennas file:", bad_antennas)
    logger_function("pol flips file:", polarization_flips)
    logger_function("bounding box:", bounding_box)
    logger_function("pulse length:", pulse_length)
    logger_function("num antennas per station:", num_antennas_per_station)
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
        
        logger_function("starting processing block:", block_index)
        start_time = time.time()
        
        block_outfile = h5py.File(tmp_fname, "w")
        block_start = initial_datapoint + imager_utility.active_block_length*block_index
        imager_utility.process_block( block_start, block_index, block_outfile, log_func=logger_function )
        block_outfile.close()
        rename(tmp_fname, out_fname)

        logger_function("block done. took:", (time.time() - start_time), 's')
        logger_function()
        
        if block_override is not None:
            break
        
    
    logger_function("done processing")
    
    
    
    
    
    
    
            
            