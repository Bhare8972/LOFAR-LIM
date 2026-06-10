#!/usr/bin/env python3

from os import mkdir, listdir
from os.path import isdir, isfile, join
import time
import datetime

import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import brute, least_squares

import h5py

from LoLIM.utilities import processed_data_dir, logger, RTD
from LoLIM import utilities as utils

#from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1
from LoLIM.IO import raw_tbb_IO

from LoLIM.getTrace_fromLoc import dataLoaderHelper

#from LoLIM.findRFI import window_and_filter
#from LoLIM.signal_processing import remove_saturation, data_cut_inspan, locate_data_loss


from LoLIM.iterativeMapper.cython_utils import parabolic_fitter, planewave_locator_helper, autoadjusting_upsample_and_correlate, \
    pointsource_locator, abs_max, GSL_error_to_string

def do_nothing(*A, **B):
    pass

############# HEADER TOOLS ###################
    
from LoLIM.iterativeMapper.mapper_header import make_header, read_header 


############# DATA MANAGER ####################

class raw_data_manager:
    def __init__(self, header, print_func=do_nothing):

        ### useful variables ###
        self.blocksize = header.blocksize

        ### data loader ###
        self.TBB_dict = {  station:raw_tbb_IO.MultiFile_Dal1( fpaths, force_metadata_ant_pos=True, only_complete_pairs=False, total_cal=header.total_cal_object) \
         for station, fpaths in header.raw_fpaths.items()}

        self.dataLoader = dataLoaderHelper( self.TBB_dict )
        self.dataLoader.set_simple_freqFilter(blocksize=self.blocksize, lower_filter=header.lowerFrequency, upper_filter=header.upperFrequency, \
            half_window_percent=header.hann_window_fraction, filter_roll_width = header.filter_roll_width)

        if header.RFI_data != None:
            self.dataLoader.set_RFIFilter( header.RFI_data, is_adv=True )


        self.sample_frequency = self.dataLoader.get_outputSampleFrequency()
        self.sample_time = 1.0/ self.sample_frequency
        self.usable_blocksize = self.dataLoader.get_maxTraceLength()
        self.edgeSize = self.dataLoader.get_window_edge_size()



#### get data from files ####        
        ## objects that are length of number of stations
        self.station_names = header.station_order
        self.num_stations = len(self.station_names)
        self.station_locations = np.empty( [self.num_stations,3], dtype=np.double )
        self.station_to_antSpan = np.empty( [self.num_stations,2], dtype=int )
        self.is_RS = np.empty( self.num_stations, dtype=bool )
        
        ## objects by number of antennas
        self.antenna_expected_RMS = np.empty(header.num_total_antennas, dtype=np.double)
        self.antenna_locations = np.empty( [header.num_total_antennas,3], dtype=np.double )
        self.antenna_names = []

#self.station_antenna_index = np.empty(header.num_total_antennas, dtype=int)
#        self.antI_to_station = np.empty(header.num_total_antennas, dtype=int)
        #self.antenna_delays = np.empty(header.num_total_antennas, dtype=np.double)

        current_AntSpan_startIndex = 0        
        for station_index,sname in enumerate(self.station_names):
            tbbFile = self.TBB_dict[ sname ]
            antenna_names_toUse = header.antenna_name_dict[ sname ]
            numAnts = len(antenna_names_toUse)

            current_AntSpan_endIndex = current_AntSpan_startIndex+len(antenna_names_toUse)
            self.station_to_antSpan[station_index] = [ current_AntSpan_startIndex,current_AntSpan_endIndex ]

            self.is_RS[station_index] = sname[:2]=='RS'

            station_ant_names = tbbFile.get_antenna_names()
            station_ant_locs = tbbFile.get_LOFAR_centered_positions()
            for locali, antName in enumerate( antenna_names_toUse ):

                if antName not in station_ant_names:
                    print_func("ERROR: requested antenna not in available antenna names")
                    print("  antenna:", antName)
                    print("  available antennas:", station_ant_names)
                    raise Exception("requested antenna not in available antenna names")

                self.antenna_expected_RMS[ current_AntSpan_startIndex+locali ] = header.antenna_timing_RMS
                self.antenna_locations[ current_AntSpan_startIndex+locali ] = station_ant_locs[ station_ant_names.index( antName ) ]
                self.antenna_names.append( antName )


            self.station_locations[ station_index ] = np.average( self.antenna_locations[ current_AntSpan_startIndex:current_AntSpan_endIndex ], axis=0 )

            current_AntSpan_startIndex = current_AntSpan_endIndex

        self.data_deltaT_dist = []

        #### allocate memory ####
        self.raw_data = np.empty( [header.num_total_antennas,self.usable_blocksize], dtype=complex )
        #self.tmp_workspace = np.empty( self.blocksize, dtype=complex )
        self.antenna_data_loaded = np.zeros( header.num_total_antennas, dtype=bool )
        self.starting_time = np.empty(header.num_total_antennas, dtype=np.double) ## real time of the first sample in raw data
            
#        self.referance_ave_delay = np.average( self.antenna_delays[self.station_to_antSpan[0,0]:self.station_to_antSpan[0,1]] )
#        self.staturation_removals = [ [] ]*header.num_total_antennas
#        self.data_loss_spans = [ [] ]*header.num_total_antennas



###################

#         #### variables ####
#         self.num_dataLoss_zeros = header.num_zeros_dataLoss_Threshold
#         self.positive_saturation = header.positive_saturation
#         self.negative_saturation = header.negative_saturation
#         self.saturation_removal_length = header.saturation_removal_length
#         self.saturation_half_hann_length  = header.saturation_half_hann_length
#         self.half_hann_safety_region = int(header.hann_window_fraction*self.blocksize) + 5
#         self.remove_saturation = header.remove_saturation
#         self.remove_RFI =        header.remove_RFI
        
#         raw_fpaths = filePaths_by_stationName( header.timeID )
        
#         #### get data from files ####
#         ## objects that are length of number of stations
#         self.station_data_files = []
#         self.RFI_filters = []
#         self.station_to_antSpan = np.empty( [header.num_stations,2], dtype=int )
#         self.station_locations = np.empty( [header.num_stations,3], dtype=np.double )
#         self.is_RS = np.empty( header.num_stations, dtype=bool )
        
#         ## objects by number of antennas
#         self.station_antenna_index = np.empty(header.num_total_antennas, dtype=int)
#         self.antenna_expected_RMS = np.empty(header.num_total_antennas, dtype=np.double)
#         self.antI_to_station = np.empty(header.num_total_antennas, dtype=int)
#         self.antenna_locations = np.empty( [header.num_total_antennas,3], dtype=np.double )
#         self.antenna_delays = np.empty(header.num_total_antennas, dtype=np.double)
#         self.antenna_names = []
        
#         next_ant_i = 0
#         for sname, antenna_list in zip(header.station_names, header.antenna_info):
#             print_func('opening', sname)
#             stat_i = len( self.station_data_files )
            
#             raw_data_file = MultiFile_Dal1(raw_fpaths[sname], force_metadata_ant_pos=True, total_cal = header.total_cal_data )
#             data_filter = window_and_filter(timeID=header.timeID, sname=sname, half_window_percent=header.hann_window_fraction)
#             self.station_data_files.append( raw_data_file )
#             self.RFI_filters.append( data_filter )
            
#             if data_filter.blocksize != self.blocksize:
#                 print_func("RFI info from station", sname, "has wrong block size of", data_filter.blocksize, '. expected', self.blocksize)
#                 quit()
                
#             first_ant_i = next_ant_i
            
#             station_antenna_names = raw_data_file.get_antenna_names()
#             for ant in antenna_list:
#                 self.antenna_names.append( ant.ant_name )
#                 self.station_antenna_index[ next_ant_i ] = station_antenna_names.index( ant.ant_name )
#                 self.antenna_expected_RMS[ next_ant_i ] = ant.planewave_RMS
#                 self.antenna_locations[ next_ant_i ] = ant.location
#                 self.antenna_delays[ next_ant_i ] = ant.delay
#                 next_ant_i += 1
            
#             self.antI_to_station[ first_ant_i:next_ant_i ] = stat_i
#             self.station_locations[ stat_i ] = np.average( self.antenna_locations[ first_ant_i:next_ant_i ], axis=0 )
#             self.station_to_antSpan[ stat_i, 0] = first_ant_i
#             self.station_to_antSpan[ stat_i, 1] = next_ant_i
#             self.is_RS[ stat_i ] = sname[:2]=='RS'       
        
#         #### allocate memory ####
#         self.raw_data = np.empty( [header.num_total_antennas,self.blocksize], dtype=complex )
#         self.tmp_workspace = np.empty( self.blocksize, dtype=complex )
#         self.antenna_data_loaded = np.zeros( header.num_total_antennas, dtype=bool )
#         self.starting_time = np.empty(header.num_total_antennas, dtype=np.double) ## real time of the first sample in raw data
            
#         self.referance_ave_delay = np.average( self.antenna_delays[self.station_to_antSpan[0,0]:self.station_to_antSpan[0,1]] )
# #        self.antenna_delays -= self.referance_ave_delay
#         self.staturation_removals = [ [] ]*header.num_total_antennas
#         self.data_loss_spans = [ [] ]*header.num_total_antennas

#####################

# I think not needed
#def get_station_location(self, station_i):
#    return self.station_locations[ station_i ]
    
#def antI_to_sname(self, ant_i):
#    station_i = self.antI_to_station[ ant_i ]
#    stationFile = self.station_data_files[ station_i ]
 #   return stationFile.get_station_name()
    
    def antI_to_antName(self, ant_i):
        return self.antenna_names[ ant_i ]

    def reset_dataReloadDt_dist(self):
        self.data_deltaT_dist = []

    def get_dataReloadDt_dist(self):
        return self.data_deltaT_dist
    
    def statI_to_sname(self, stat_i):
        #return self.station_data_files[ stat_i ].get_station_name()
        return self.station_names[ stat_i ]
    
    def get_antI_span(self, station_i):
        return self.station_to_antSpan[ station_i ]
    
    def get_antenna_locations(self, station_i=None):
        if station_i is None:
            return self.antenna_locations
        else:
            span = self.station_to_antSpan[ station_i ]
            return self.antenna_locations[ span[0]:span[1] ]
    
    def get_antenna_timingRMS(self):
        return self.antenna_expected_RMS

    def get_sample_frequency(self):
        return self.sample_frequency

    def get_station_locations(self):
        return self.station_locations

    def open_antenna_SampleNumber(self, antenna_i, start_time):
        """opens the data for an antenna, where antenna starts within a sample of the given time. Returns the data block for this station, start time, and number of data-loss samples"""
        
#        print('opening:', antenna_i)
 
        start_sample = int(start_time/self.sample_time)-self.edgeSize

        real_start_time = start_time - self.edgeSize*self.sample_time

        data, dataloss = self.dataLoader.get_SingleProcessed_block( real_start_time, self.antenna_names[ antenna_i ], initial_sample_mode='t', perform_ifft=True)

        data = data[self.edgeSize:-self.edgeSize]
        
        old_start_time = self.starting_time[ antenna_i ]
        self.starting_time[ antenna_i ] = start_time #(start_sample+self.edgeSize)*self.sample_time

        if self.antenna_data_loaded[ antenna_i ]:
            block_frac_move = (self.starting_time[ antenna_i ] - old_start_time) / ( self.sample_time*self.usable_blocksize )  
            self.data_deltaT_dist.append( self.starting_time[ antenna_i ] - old_start_time )
            
        self.antenna_data_loaded[ antenna_i ] = True

        self.raw_data[ antenna_i,: ] = data
        
        return self.raw_data[ antenna_i ], self.starting_time[ antenna_i ], dataloss


        
        # station_i = self.antI_to_station[ antenna_i ]
        # stationFile = self.station_data_files[ station_i ]
        # data_filter = self.RFI_filters[ station_i ]
        
        # station_antenna_index = self.station_antenna_index[ antenna_i ]
        # TMP = stationFile.get_data(sample_number, self.blocksize, antenna_index=station_antenna_index  )
            
        # if len(TMP) != self.blocksize:
        #     MSG = 'data length wrong. is: ' + str(len(TMP)) + " should be " + str(self.blocksize) + ". sample: " + \
        #       str(sample_number) + ". antenna: " + str(station_antenna_index) + ". station: " + stationFile.get_station_name()
              
        #     raise Exception( MSG )
        
        # self.data_loss_spans[ antenna_i ], DL = locate_data_loss(TMP, self.num_dataLoss_zeros)
            
        # self.tmp_workspace[:] = TMP

        # if self.remove_saturation:
        #     self.staturation_removals[antenna_i] = remove_saturation( self.tmp_workspace, self.positive_saturation, self.negative_saturation, self.saturation_removal_length, self.saturation_half_hann_length )
        # else:
        #     self.staturation_removals[antenna_i] = []
    
        # if self.remove_RFI:
        #     self.raw_data[ antenna_i ] = data_filter.filter( self.tmp_workspace )
        # else:
        #     self.raw_data[ antenna_i ] = self.tmp_workspace
            
        
        
    # def open_station_SampleNumber(self, station_i, sample_number):
    #     """opens the data for a station, where every antenna starts at the given sample number. Returns the data block for this station, and the start times for each antenna"""
        
    #     antI_span = self.station_to_antSpan[ station_i ]
        
    #     for antI in range(antI_span[0],antI_span[1]):
    #         self.open_antenna_SampleNumber( antI, sample_number )
            
    #     return self.raw_data[ antI_span[0]:antI_span[1] ],   self.starting_time[antI_span[0]:antI_span[1]]


    def open_station_SampleNumber(self, station_i, start_time):
        """opens the data for a station, where every antenna starts at the given sample number. Returns the data block for this station, the start times for each antenna, and data-loss"""

        antI_span = self.station_to_antSpan[ station_i ]
        data_losses = []
        for antI in range(antI_span[0],antI_span[1]):
            data, starttime, dl = self.open_antenna_SampleNumber( antI, start_time )
            data_losses.append( dl )

        return self.raw_data[ antI_span[0]:antI_span[1] ],   self.starting_time[antI_span[0]:antI_span[1]],   data_losses
                
            
    # def get_station_safetySpan(self, station_i, start_time, stop_time):
    #     """returns the data and start times for a station (same as open_station_SampleNumber), insuring that the data between start_time and stop_time is valid for
    #     all antennas in this station. (assume the duration is smaller then a block). If new data is opened, then start_time is aligned to just after the beginning of the file."""
        
    #     antI_span = self.station_to_antSpan[ station_i ]
        
    #     do_load = False
    #     if not self.station_data_loaded[ station_i ]:
    #         do_load = True
    #     else:
    #         present_start_time = np.max( self.starting_time[antI_span[0]:antI_span[1]] ) + self.half_hann_safety_region*5.0E-9
    #         is_safe = present_start_time < start_time
            
    #         present_end_time = np.min( self.starting_time[antI_span[0]:antI_span[1]] ) + self.blocksize*5.0E-9 - self.half_hann_safety_region*5.0E-9
    #         is_safe = is_safe and present_end_time>stop_time
            
    #         do_load = not is_safe
        
    #     if do_load:
    #         antenna_delays = self.antenna_delays[ antI_span[0]:antI_span[1] ]
    #         start_sample_number = int( (start_time + min( antenna_delays ))/5.0E-9 ) + 1
    #         start_sample_number -= self.half_hann_safety_region
            
    #         self.open_station_SampleNumber( station_i, start_sample_number )
            
    #     return self.raw_data[ antI_span[0]:antI_span[1] ], self.starting_time[antI_span[0]:antI_span[1]]
        



    def get_antenna_safetySpan(self,  antenna_i, start_T, number_samples ):
        """returns the data and start time for an antenna. WHere the time of first sample is close as possible to start_T"""

        do_load = False
        if not self.antenna_data_loaded[ antenna_i ]:
            do_load = True
        else:
            present_start_time = self.starting_time[ antenna_i ]
            is_safe = present_start_time < start_T
            
            stop_time = start_T + number_samples*self.sample_time
            present_end_time = present_start_time + self.usable_blocksize*self.sample_time
            is_safe = is_safe and (present_end_time > stop_time)
            
            do_load = not is_safe
            
                    
        if do_load:
            self.open_antenna_SampleNumber(antenna_i, start_T - 0.25*self.usable_blocksize*self.sample_time)


        num_samples_beginning = int((start_T - self.starting_time[ antenna_i ])/self.sample_time)
        num_samples_end = num_samples_beginning + number_samples

        data = self.raw_data[ antenna_i,  num_samples_beginning:num_samples_end ]



        return data, self.starting_time[ antenna_i ]+num_samples_beginning*self.sample_time
            
    # def check_saturation_span(self, ant_i, first_index, last_index):
    #     return data_cut_inspan( self.staturation_removals[ant_i], first_index,last_index )
        
    # def check_dataLoss_span(self, ant_i, first_index, last_index):
    #     return data_cut_inspan( self.data_loss_spans[ant_i], first_index,last_index )
    
    # def get_station_dataloss(self, stat_i, out_array=None):
    #     antI_span = self.station_to_antSpan[ stat_i ]
        
    #     if out_array is None:
    #         out_array = np.zeros(antI_span[1]-antI_span[0], dtype=int)
    #     else:
    #         out_array[:] = 0
            
    #     for ant_i in range(antI_span[0], antI_span[1]):
    #         for span in self.data_loss_spans[ant_i]:
    #             out_array[ant_i] += span[1]-span[0]
                
    #     return out_array
    
    # def get_station_saturation(self, stat_i, out_array=None):
    #     antI_span = self.station_to_antSpan[ stat_i ]
        
    #     if out_array is None:
    #         out_array = np.zeros(antI_span[1]-antI_span[0], dtype=int)
    #     else:
    #         out_array[:] = 0
            
    #     for ant_i in range(antI_span[0], antI_span[1]):
    #         for span in self.staturation_removals[ant_i]:
    #             out_array[ant_i] += span[1]-span[0]
                
    #     return out_array

######### LOCATORS!!! #########
        
class planewave_locator:
    def __init__(self, header, data_manager, half_window, v_air_ground, sample_time):
        self.data_manager = data_manager
        self.antenna_locations = self.data_manager.get_antenna_locations( 0 )
        self.min_amp = header.min_amplitude
        self.upsample_factor = header.upsample_factor
        self.sample_time = sample_time
        self.half_window = half_window
        self.num_antennas = len(self.antenna_locations)
        
        self.max_delta_matrix = np.empty( (self.num_antennas,self.num_antennas), dtype=int )
        for i, iloc in enumerate(self.antenna_locations):
            for j, jloc in enumerate(self.antenna_locations):
                R = np.linalg.norm( iloc-jloc )
                self.max_delta_matrix[i,j] = int( R/(v_air_ground*self.sample_time) ) + 1

    
        self.max_half_length = half_window + np.max( self.max_delta_matrix )
        self.correlator = autoadjusting_upsample_and_correlate(2*self.max_half_length, self.upsample_factor)
        self.max_CC_size = self.correlator.get_current_output_length()
        
        ## setup memory
        self.relative_ant_locations = np.array( self.antenna_locations )
        
        self.cross_correlation_storage = np.empty( (self.num_antennas, self.max_CC_size ), dtype=np.double )
        self.cal_delta_storage = np.empty( self.num_antennas , dtype=np.double )
        
        self.workspace = np.empty( self.num_antennas, dtype=np.double )
        self.measured_dt = np.empty( self.num_antennas, dtype=np.double )
        self.antenna_mask = np.ones( self.num_antennas, dtype=int)
        self.CC_lengths = np.ones( self.num_antennas, dtype=int)
        
        ## CHECK THESE!
        self.locator = planewave_locator_helper(self.upsample_factor, self.num_antennas, self.max_CC_size, 50, v_air=v_air_ground, sample_time=self.sample_time )

        self.locator.set_memory( self.cross_correlation_storage, self.relative_ant_locations, self.cal_delta_storage, 
                                self.workspace, self.measured_dt, self.antenna_mask, self.CC_lengths )
        
        self.max_itters = header.max_minimize_itters
        self.ftol = header.minimize_ftol
        self.xtol = header.minimize_xtol
        self.gtol = header.minimize_gtol
        
        self.ZeAz = np.empty(2, dtype=np.double)
        
    def set_data(self, ref_ant_i, data_block, start_times):
        self.ref_ant_i = ref_ant_i
        
        self.relative_ant_locations[:] = self.antenna_locations
        self.relative_ant_locations[:] -= self.antenna_locations[ self.ref_ant_i ]
        
        self.data_block = data_block
        self.start_times = start_times
        
    def locate(self, peak_index, do_plot):
        
        ## extract trace on ref antenna
        ref_trace = self.data_block[ self.ref_ant_i, peak_index-self.half_window:peak_index+self.half_window  ]
        self.correlator.set_referance( ref_trace )
        ref_time = self.start_times[ self.ref_ant_i ]
        
        ## get data from all other antennas
        traces = [None for i in range(len(self.start_times))]
        traces[ self.ref_ant_i ] = ref_trace
        self.cal_delta_storage[ self.ref_ant_i ] = 0.0
        self.measured_dt[ self.ref_ant_i ] = 0.0 ## ??
        self.workspace[ self.ref_ant_i ] = 0.0 ##??

        for i, (data,start_time) in enumerate(zip(self.data_block,self.start_times)):
            if i == self.ref_ant_i:
                self.antenna_mask[i] = 0
                continue
            
            half_length = self.max_delta_matrix[self.ref_ant_i, i] + self.half_window
            trace = data[ peak_index-half_length:peak_index+half_length ]
            HEmax = abs_max( trace )
            
            if HEmax < self.min_amp:
                self.antenna_mask[i] = 0
                continue
            
            #has_saturation = self.data_manager.check_saturation_span(i, peak_index-half_length, peak_index+half_length )
            #if  has_saturation:
            #    self.antenna_mask[i] = False
             #   continue
            
            #if self.data_manager.check_dataLoss_span(i, peak_index-half_length, peak_index+half_length ):
            #    self.antenna_mask[i] = False
            #    continue
            
            traces[i] = trace
            
            cross_corelation = self.correlator.correlate( trace )
            CC_length = len(cross_corelation)
            np.abs(cross_corelation, out = self.cross_correlation_storage[i, :CC_length])
            
            self.CC_lengths[i] = CC_length
            self.cal_delta_storage[i] = start_time-ref_time  - self.max_delta_matrix[self.ref_ant_i, i]*self.sample_time
            self.antenna_mask[i] = 1
            
        N = np.sum( self.antenna_mask ) 
        
        if N<2:
            return 0, [0,0], 0.0, self.measured_dt, self.antenna_mask, N
        
        ## get brute guess
        self.locator.run_brute(self.ZeAz)
        
        ## now run minimizer
        succsses_code, RMS = self.locator.run_minimizer( self.ZeAz, self.max_itters, self.xtol, self.gtol, self.ftol)

    
        return succsses_code, self.ZeAz, RMS, self.measured_dt, self.antenna_mask, N
    
    def num_itters(self):
        return self.locator.get_num_iters()
    
class timer_acclimator:
    def __init__(self):
        self.current_time = 0
        self.CT = None

    def start_timer(self):
        self.CT = time.process_time_ns()

    def stop_timer(self):
        dt = time.process_time_ns() - self.CT 
        self.current_time += dt    

    def get_time(self):
        return self.current_time 

class iterative_mapper:
    def __init__(self, header, print_func=do_nothing):
        self.header = header
        
        ## open raw data files
        self.data_manager = raw_data_manager( header, print_func )
        self.antenna_XYZs = self.data_manager.get_antenna_locations()
        self.sample_time = 1.0/self.data_manager.get_sample_frequency()
        self.stop_chisquared_factor = header.stop_chisquared_factor


        self.block_data_size = self.data_manager.usable_blocksize
        self.nonTukey_windowEdgeSize = self.header.nonTukey_windowEdgeSize
        self.analyzable_blocksize = self.header.analyzable_blocksize
        
        ## find best station order
        self.station_locs = self.data_manager.get_station_locations()
        self.station_order = np.arange( len(self.station_locs) )


        ## load helpers:
        self.half_refPulse_length_samples = int(header.min_pulse_length_samples/2)
        self.half_otherPulse_length_samples = self.half_refPulse_length_samples + self.header.min_timeUncertainty_samples

        self.atmosphere = header.total_cal_object.atmosphere
        
        v_air_ground = self.atmosphere.get_effective_lightSpeed(np.array([0,0,0]), self.station_locs[0])
        self.planewave_manager = planewave_locator( header, self.data_manager, int(header.min_pulse_length_samples/2), v_air_ground=v_air_ground, sample_time=self.sample_time )



    
        self.correlator = autoadjusting_upsample_and_correlate( self.half_otherPulse_length_samples*2, header.upsample_factor   )
        self.peak_fitter = parabolic_fitter()
        self.pointsource_manager = pointsource_locator( self.header.num_total_antennas )
    
        ### LOAD MEMORY ###
        self.antenna_mask = np.zeros( header.num_total_antennas, dtype=int )
        self.measured_dt = np.zeros( header.num_total_antennas, dtype=np.double )
        self.weights = np.zeros( header.num_total_antennas, dtype=np.double )
        self.v_air_inverses = np.zeros( header.num_total_antennas, dtype=np.double )
        
        self.pointsource_manager.set_memory( self.antenna_XYZs, self.measured_dt, self.antenna_mask, self.weights, self.v_air_inverses )


        self.len_ref_station = self.data_manager.station_to_antSpan[0,1]-self.data_manager.station_to_antSpan[0,0]

        self.ref_ant_workspace = np.empty( self.analyzable_blocksize, dtype=np.double )
        self.refAnt_peakIndeces = np.empty( header.max_events_perBlock, dtype=int )
        
        self.XYZc = np.zeros( 4, dtype=np.double )
        self.XYZc_old = np.zeros( 4, dtype=np.double )
        self.station_mode = np.zeros( self.data_manager.num_stations, dtype=int ) ## current fitting mode for all stations. 0: do not fit, 1: reload if needed, 2: always reload
        
        self.covariance_matrix = np.empty( (4,4), dtype=np.double )

        self.found_sources = {
        "ID":np.empty(self.header.max_events_perBlock, dtype=int),
        "easting":np.empty(self.header.max_events_perBlock, dtype=float),
        "northing":np.empty(self.header.max_events_perBlock, dtype=float),
        "height":np.empty(self.header.max_events_perBlock, dtype=float),
        "time":np.empty(self.header.max_events_perBlock, dtype=float),
        "RMS":np.empty(self.header.max_events_perBlock, dtype=float),
        "reduced_chi_squared":np.empty(self.header.max_events_perBlock, dtype=float),
        "reference_amplitude":np.empty(self.header.max_events_perBlock, dtype=float),
        "loc_error":np.empty(self.header.max_events_perBlock, dtype=float),
        "station_mask":np.empty(self.header.max_events_perBlock, dtype=int),
        "num_RS":np.empty(self.header.max_events_perBlock, dtype=int),
        }

        ### note, if station mask is a string like s="0110", then i=int(s,2) converts it to an integer that is stored as mask and x=bin(i)[2:].zfill( num_stations ) converts back

    def load_atmosphere(self, XYZ):
        """Given a source location of XYZ, load average v_air into correct array per antenna"""

        for stat_i in self.station_order:
            station_location = self.station_locs
            antenna_span = self.data_manager.get_antI_span( stat_i )
            v_air = self.atmosphere.get_effective_lightSpeed( XYZ, station_location )

            self.v_air_inverses[ antenna_span[0] : antenna_span[1] ] = 1.0/v_air



    def reload_data(self, print_func, diagnostic_prints=False):
        """this is where the magic REALLY happens"""

## NOTE: linearlized error estiamate is J C J^T.  Where J is jacobian, C is covarience matrix, and J^T is transpose of Jacobian

        if diagnostic_prints:
            print_func('loading data from location:', self.XYZc)
        
        for stat_i in self.station_order:
            if self.station_mode[ stat_i ] == 0:
                continue
            
            ant_span = self.data_manager.station_to_antSpan[ stat_i ]
            for ant_i in range( ant_span[0], ant_span[1] ):
                predicted_dt = self.pointsource_manager.relative_arrival_time(ant_i, self.XYZc)

                if diagnostic_prints:
                    print_func("prdict DT. stat:", stat_i, 'ant:', ant_i, "dt:", predicted_dt)
                
                if self.station_mode[ stat_i ] != 2 and self.antenna_mask[ ant_i ]:
                    if diagnostic_prints:  print_func("  known err[ns]:", ( predicted_dt-self.measured_dt[ant_i] )/(1e-9) )

                    if np.abs( predicted_dt-self.measured_dt[ant_i] ) < self.sample_time*self.header.min_timeUncertainty_samples:
                        continue ## do not re-load info!
                        
                ## get data
                half_data_length_samples = self.half_otherPulse_length_samples
                start_T = self.referance_peakTime + predicted_dt - half_data_length_samples*self.sample_time
                antenna_data, start_time = self.data_manager.get_antenna_safetySpan( ant_i, start_T, number_samples = half_data_length_samples*2 )
                 
                #sample_index = int( (start_T - start_time)/5.0E-9 )
                #antenna_trace = antenna_data[ sample_index:sample_index + self.header.min_pulse_length_samples ]
                #trace_startTime = start_time + sample_index*5.0E-9
                
                ## check quality
                if len(antenna_data) != half_data_length_samples*2: ## something went wrong, I dobn't think this should ever really happen
                    self.antenna_mask[ ant_i ] = False
                    # if self.station_mode[stat_i] == 2:
                        # print_func('A', ant_i)
                    continue
                    
                if abs_max( antenna_data ) < self.header.min_amplitude:
                    self.antenna_mask[ ant_i ] = False
                    # if self.station_mode[stat_i] == 2:
                        # print_func('B', ant_i)
                    continue
                
#                has_saturation = self.data_manager.check_saturation_span(ant_i, sample_index, sample_index+self.header.min_pulse_length_samples )
#                if  has_saturation:
#                    self.antenna_mask[ ant_i ] = False
#                    # if self.station_mode[stat_i] == 2:
#                        # print_func('C', ant_i)
#                    continue
#                
#                if self.data_manager.check_dataLoss_span(ant_i, sample_index, sample_index+self.header.min_pulse_length_samples ):
#                    self.antenna_mask[ ant_i ] = False
#                    # if self.station_mode[stat_i] == 2:
#                        # print_func('D', ant_i)
#                    continue
                
            
                ## do cross correlation and get peak location
                cross_correlation = self.correlator.correlate( antenna_data )
                CC_length = len(cross_correlation)
                workspace_slice = self.ref_ant_workspace[:CC_length]
                np.abs(cross_correlation, out=workspace_slice)
                
                CC_peak = self.peak_fitter.fit( workspace_slice )
                
                if CC_peak > CC_length/2:
                    CC_peak -= CC_length
                    
                peak_location = CC_peak*self.sample_time/self.header.upsample_factor
                peak_location += start_time - self.referance_startTime

                self.measured_dt[ ant_i ] = peak_location
                if diagnostic_prints:  print_func("  new err[ns]:", ( predicted_dt-self.measured_dt[ant_i] )/(1e-9) )

                self.antenna_mask[ ant_i ] = True

    def plot_process(self, showPlot=True, saveFigLoc=None):
        """Make curtain plot of current data"""


        plotting_data = []

        total_data_height = 0
        for stat_i in self.station_order:
            if self.station_mode[ stat_i ] == 0:
                continue
            
            antenna_dataList = []

            antenna_height = 0
            ant_span = self.data_manager.station_to_antSpan[ stat_i ]


            half_data_length_samples = self.half_otherPulse_length_samples*5 ## NOTE!!

            for ant_i in range( ant_span[0], ant_span[1] ):
                if self.antenna_mask[ ant_i ]:
                    predicted_dt = self.pointsource_manager.relative_arrival_time(ant_i, self.XYZc)
                    predicted_arrival_time = self.referance_peakTime + predicted_dt

                    measured_arrivalTime = self.measured_dt[ ant_i ] 

                    antenna_data, start_time = self.data_manager.get_antenna_safetySpan( ant_i, predicted_arrival_time-half_data_length_samples*self.sample_time, number_samples = half_data_length_samples*2 )

                    HE = np.abs(antenna_data)

                    antenna_height = max(antenna_height, np.max(HE) )

                    data_pack = {
                    "data":HE,
                    "data_startTime": start_time-predicted_arrival_time,
                    "measured_Time": measured_arrivalTime - predicted_dt
                    }

                    antenna_dataList.append(data_pack)

            plotting_data.append(  {"sname":self.data_manager.statI_to_sname( stat_i ),  "antennas":antenna_dataList} )
            total_data_height += antenna_height*1.1


        inches_per_AntennaSaturation = 2/2000
        H_inches = inches_per_AntennaSaturation * total_data_height

        if H_inches < 4:
            H_inches = 4

        fig, ax = plt.subplots(  figsize=(6.4,H_inches) )

        current_height = 0
        for stationData in plotting_data:
            sname = stationData["sname"]
            antennas = stationData["antennas"]

            max_ant = 0
            for antData in antennas:
                data           = antData["data"]
                data_startTime = antData["data_startTime"]
                #measured_Time  = antData["measured_Time"]

                antMax = np.max( data )
                max_ant = max( max_ant, antMax )

                data_T = np.arange( len(data), dtype=float )
                data_T *= self.sample_time
                data_T += data_startTime

                data = data+current_height

                ax.plot(data_T, data )

            for antData in antennas:
                #data           = antData["data"]
                #data_startTime = antData["data_startTime"]
                measured_Time  = antData["measured_Time"]

                ax.plot( [measured_Time,measured_Time], [current_height, current_height+max_ant] )

            ax.annotate(sname, (-self.half_otherPulse_length_samples*self.sample_time, current_height) )

            current_height += max_ant*1.1

        if saveFigLoc != None:
            fig.savefig(fname=saveFigLoc)

        if showPlot:
            plt.show()

        plt.close(fig)
        del fig





        
    def process(self, block_i, print_func=do_nothing, plotSource=None):
        """block_i is defined as blocks since initial_time. E.G. starts at 0. If block_i starts after final_time, than an error is raised.
        print_func should be a function to call for logging. (e.g.  print, or a logging function).
        plotSource is for diagnostics. If None then nothing happens. If an integer, than diagnositic plots for that source will be made."""

        loading_data_timer =  timer_acclimator()
        planewave_timer =  timer_acclimator()
        fitter_timer =  timer_acclimator()

        total_timer =  timer_acclimator()
        total_timer.start_timer()


        start_time = self.header.initial_time + block_i*self.analyzable_blocksize*self.sample_time

        if start_time > self.header.final_time:
            raise("Exception: requested block in process is after final_time. (block: "+str(block_i)+')')
        
        #### get initial data and identify referance antenna ####

        loading_data_timer.start_timer()
        referanceStat_blockData, startTimes, dataLosses = self.data_manager.open_station_SampleNumber(0, start_time-self.nonTukey_windowEdgeSize*self.sample_time)
        loading_data_timer.stop_timer()


        self.referance_antI = np.argmin( dataLosses )
        self.referance_ant_name = self.data_manager.antI_to_antName( self.referance_antI )
        
        self.planewave_manager.set_data(self.referance_antI, referanceStat_blockData, startTimes)
        self.pointsource_manager.set_ref_antenna( self.referance_antI )
        self.ref_XYZ = self.antenna_XYZs[ self.referance_antI ]  ### does this need to be self?
        
        #### set the appropriate weights ####
        self.weights[:] = self.data_manager.antenna_expected_RMS # load the expected RMS
        self.weights *= self.weights                             # convert to covariance
        self.weights += self.weights[ self.referance_antI ]      # add covariances
        np.sqrt(self.weights, out=self.weights)                  # back to RMS
        
        
        #### find and sort the top peaks in ref antennas ####
        A = referanceStat_blockData[self.referance_antI, self.nonTukey_windowEdgeSize:self.nonTukey_windowEdgeSize+self.analyzable_blocksize]
        np.abs( referanceStat_blockData[self.referance_antI, self.nonTukey_windowEdgeSize:self.nonTukey_windowEdgeSize+self.analyzable_blocksize], out=self.ref_ant_workspace )
        num_events = 0
        half_erasure_length = int( self.header.erasure_length/2 )
        for event_i in range( self.header.max_events_perBlock ):
            peak_index = np.argmax( self.ref_ant_workspace )
            amplitude = self.ref_ant_workspace[ peak_index ]
            
            if amplitude < self.header.min_amplitude:
                break
            else:                
                self.refAnt_peakIndeces[ event_i ] = self.nonTukey_windowEdgeSize + peak_index
                num_events += 1
                
                if peak_index < half_erasure_length:
                    self.ref_ant_workspace[ 0 : peak_index+half_erasure_length ] = 0.0
                elif peak_index >= len(self.ref_ant_workspace)-half_erasure_length:
                    self.ref_ant_workspace[ peak_index-half_erasure_length : ] = 0.0
                else:
                    self.ref_ant_workspace[ peak_index-half_erasure_length : peak_index+half_erasure_length ] = 0.0
                
        ref_indeces = self.refAnt_peakIndeces[:num_events]
        ref_indeces.sort()
        
        #### find events!! ####
        print_func("block", block_i, "at", start_time, "ref. antenna ID:", self.referance_antI)
        print_func("fitting", num_events, 'events')
        
        out_i = 0
        for event_i, event_ref_index in enumerate( ref_indeces ):

            do_diagnostics = False
            if plotSource == event_i:
                do_diagnostics = True
            
            #### setup ref info
            referance_trace = referanceStat_blockData[ self.referance_antI, 
                                   event_ref_index-self.half_refPulse_length_samples:event_ref_index+self.half_refPulse_length_samples ]

            self.referance_peakTime = startTimes[ self.referance_antI ] + self.sample_time*event_ref_index
            self.referance_startTime = startTimes[ self.referance_antI ] + (event_ref_index-self.half_refPulse_length_samples)*self.sample_time
            referance_amplitude = np.abs( referanceStat_blockData[ self.referance_antI, event_ref_index] )
            
            self.correlator.set_referance( referance_trace )
        
            print_func()
            print_func('event:', event_i, '/', num_events, event_ref_index, ". amplitude:", referance_amplitude)
            print_func("    block index:", event_ref_index, "time:", start_time+event_ref_index*self.sample_time, 'on antenna',  self.data_manager.antenna_names[ self.referance_antI ] )
        
        
            #### try to fit planewave
            planewave_timer.start_timer()
            GSL_code, zenith_azimuth, planewave_RMS, planewave_dt, planewave_mask, planewave_num_ants = self.planewave_manager.locate( 
                event_ref_index, False)
            planewave_timer.stop_timer()
            
            print_func('planewave RMS:', planewave_RMS, 'num antennas:', planewave_num_ants, 'num iters:', self.planewave_manager.num_itters() )
            print_func('  zenith, azimuth:', zenith_azimuth[0]*RTD, zenith_azimuth[1]*RTD, '[deg]')

            if planewave_num_ants<2:
                print_func("  too few antennas")
                print_func()
                print_func()
                continue
            elif planewave_RMS > 10e-9:
                print_func("  planewave RMS is too high")
                print_func()
                print_func()
                continue

            if (GSL_code!=0):
                print('WARNING: planewave GSL error', GSL_code)
                print("   err:", GSL_error_to_string(GSL_code) )

#            print_func()
            
            
            self.XYZc[0] = np.sin( zenith_azimuth[0] )*np.cos( zenith_azimuth[1] )*self.header.guess_distance + self.ref_XYZ[0]
            self.XYZc[1] = np.sin( zenith_azimuth[0] )*np.sin( zenith_azimuth[1] )*self.header.guess_distance + self.ref_XYZ[1]
            self.XYZc[2] = np.abs( np.cos( zenith_azimuth[0] )*self.header.guess_distance) + self.ref_XYZ[2]
            self.XYZc[3] = 0.0
            
            
            #### prep for fitting
            self.antenna_mask[:] = 0
            self.antenna_mask[:self.len_ref_station] = planewave_mask
            self.measured_dt[:self.len_ref_station]  = planewave_dt
#            success, chi_squared = self.pointsource_manager.run_minimizer( self.XYZc, self.header.max_minimize_itters, 
#                        self.header.minimize_xtol, self.header.minimize_gtol, self.header.minimize_ftol )
            
#            print_func('initial fit chi-squared:', chi_squared)
            print_func('  initial planewave XYZ', self.XYZc[:3])
            print_func()
            
#            self.XYZc[2] = np.abs( self.XYZc[2] )
#            self.XYZc[:3] *= self.header.guess_distance/np.linalg.norm( self.XYZc[:3] )
            
            
            #### now do the thing!!
            self.station_mode[:] = 0 ## 0 means do not use
            is_good = True
            current_red_chi_sq = np.inf #chi_squared
            num_throws = 0
            for stat_i in self.station_order:
                self.station_mode[ stat_i ] = 2 ## 2 means force-load data

                # print_func('doing station', stat_i,  self.header.station_names[stat_i])

                
                self.load_atmosphere( self.XYZc[0:3] )
                loading_data_timer.start_timer()
                self.reload_data( print_func, diagnostic_prints=do_diagnostics )
                loading_data_timer.stop_timer()

                if do_diagnostics:
                    file_name = "DiagCurten_Block:"+str(block_i)+"_Source:"+str(event_i)+'_Station:'+str(stat_i)+":"+self.header.station_order[stat_i]+'.pdf'
                    plot_output_loc = join( self.header.logging_folder, file_name)
                    self.plot_process(showPlot=False, saveFigLoc=plot_output_loc)

                
                self.station_mode[ stat_i ] = 1 ## 1 means load data if needed
                
                num_ants = np.sum( self.antenna_mask )
                if  num_ants < 3:
                    print_func('station', stat_i, '(', self.header.station_order[stat_i] ,') too few antennas' )
                    continue
                
                fitter_timer.start_timer()
                GSL_err_cd, new_chi_squared = self.pointsource_manager.run_minimizer( self.XYZc, self.header.max_minimize_itters, 
                        self.header.minimize_xtol, self.header.minimize_gtol, self.header.minimize_ftol )
                fitter_timer.stop_timer()
                
                if GSL_err_cd!=0:
                    print_func("WARNING: GSL error in pointsource fitter!", GSL_err_cd)
                    print_func("   GSL err:", GSL_error_to_string(GSL_err_cd) )
                    print_func()


                if (stat_i != 0) and ( (current_red_chi_sq > 1 and new_chi_squared > self.stop_chisquared_factor*current_red_chi_sq) or \
                    (current_red_chi_sq<1 and new_chi_squared>5) ):
                    self.station_mode[ stat_i ] = 0
                    ant_span = self.data_manager.station_to_antSpan[ stat_i ]
                    self.antenna_mask[ ant_span[0]:ant_span[1] ] = 0
                    
                    print_func("  throwing station:", stat_i, '(', self.header.station_order[stat_i] ,') had red. chi-squared of:', new_chi_squared, 'previous:', current_red_chi_sq)
                    self.XYZc[:] = self.XYZc_old ## keep the old chi-squared and location
                    num_throws += 1
                    
                elif (stat_i != 0) and (new_chi_squared > self.header.stop_chi_squared) and num_ants>5:
                    print_func("chi-squared too high:", new_chi_squared)
                    print_func()
                    print_func()
                    is_good = False
                    break
                else:
                    current_red_chi_sq = new_chi_squared
                    self.XYZc[2] = np.abs( self.XYZc[2] )
                    self.XYZc_old[:] = self.XYZc


                    ant_span = self.data_manager.station_to_antSpan[ stat_i ]
                    num_ant = np.sum( self.antenna_mask[ ant_span[0]:ant_span[1] ] )
                    if num_ant == 0:
                        print_func("  warning station:", stat_i, '(', self.header.station_order[stat_i], ') has no active antennas')
                
                if stat_i == 0:
                    self.XYZc[:3] *= self.header.guess_distance/np.linalg.norm( self.XYZc[:3] )

                if do_diagnostics:
                    print("station num", stat_i, "final chi2:", new_chi_squared, 'err:', GSL_err_cd)
                    # used_jacobian = self.pointsource_manager.get_used_jacobian()
                    # ana_jacobian = self.pointsource_manager.get_analytical_jacobian( self.XYZc )
                    # difJac = used_jacobian-ana_jacobian
                    # DN = np.linalg.norm(difJac)
                    # print('   norm( usedJac - anaJac ):',  DN )
                    # print('   rank usedJac:', np.linalg.matrix_rank(used_jacobian) )
                    # print('   rank anaJac: ', np.linalg.matrix_rank(ana_jacobian) )

                    # if DN > 1e6:
                    #     fname = "DiagJacobian_Block:"+str(block_i)+"_Source:"+str(event_i)+'_Station:'+str(stat_i)+":"+self.header.station_order[stat_i]+'.txt'
                    #     fname = join( self.header.logging_folder, fname)

                    #     write_three_jacs_to_file(fname, used_jacobian, ana_jacobian, difJac, 'used', 'analytic', 'used - ana')


#if self.header.total_cal_data:

#    self.current_v_air = self.header.total_cal_data.atmosphere.get_effective_lightSpeed(self.XYZc[:3], np.zeros(3))  #assume same index of refraction for all antennas
#    set_c_air_inverse( 1.0/self.header.v_air )                
                    
            if not is_good:
                continue
            
            #### get final info
            self.load_atmosphere( self.XYZc[0:3] )

            fitter_timer.start_timer()
            GSL_err_cd, current_red_chi_sq = self.pointsource_manager.run_minimizer( self.XYZc, self.header.max_minimize_itters, 
                    self.header.minimize_xtol, self.header.minimize_gtol, self.header.minimize_ftol )

            if GSL_err_cd!=0:
                print_func("WARNING: GSL error in final pointsource fitter!", GSL_err_cd)
                print_func("   GSL err:", GSL_error_to_string(GSL_err_cd) )
                print_func()

            fitter_timer.stop_timer()



## calculate some other things (error bars, time, and whatnot)

            self.load_atmosphere( self.XYZc[0:3] )
            
            RMS = self.pointsource_manager.get_RMS()
            
            num_remote_stations = 0

            has_station = []
            for stat_i in self.station_order:
                ant_span = self.data_manager.station_to_antSpan[ stat_i ]
                num_ant = np.sum( self.antenna_mask[ ant_span[0]:ant_span[1] ] )
                if num_ant > 0:
                    has_station.append( '1' )
                    if self.data_manager.is_RS[ stat_i ]:
                        num_remote_stations += 1
                else:
                    has_station.append( '0' )
            station_mask_string = ''.join( has_station )
            station_mask_int = int(station_mask_string,2)

                      
            self.pointsource_manager.get_covariance_matrix( cov_out=self.covariance_matrix )

            if current_red_chi_sq > 1:
                self.covariance_matrix *= current_red_chi_sq
            
            try:
                eigvalues, eigenvectors = np.linalg.eigh( self.covariance_matrix[:3, :3] )

                location_error = np.sqrt( np.max( np.abs( eigvalues ) ) )
            except:
                print_func('eigenvalues did not converge???')
                location_error = 1.0
        

            T = self.referance_peakTime - np.linalg.norm( self.XYZc[:3]-self.ref_XYZ )*self.v_air_inverses[0] ## use v air of 0th antenna on 0th station
                   
            #### output!
            print_func("successful fit :", out_i)
            print_func("  RMS:", RMS, 'red. chi-square:', current_red_chi_sq)
            print_func("  XYZT:", self.XYZc[:3], T)
            print_func("  num RS:", num_remote_stations)
            print_func('  est. loc. Err:', location_error)
            print_func()
            print_func()

            self.found_sources['ID'][out_i] = block_i*self.header.max_events_perBlock + out_i
            self.found_sources['easting'][out_i] = self.XYZc[0]
            self.found_sources['northing'][out_i] = self.XYZc[1]
            self.found_sources['height'][out_i] = self.XYZc[2]
            self.found_sources['time'][out_i] = T
            self.found_sources['RMS'][out_i] = RMS
            self.found_sources['reduced_chi_squared'][out_i] = current_red_chi_sq
            self.found_sources['reference_amplitude'][out_i] = referance_amplitude
            self.found_sources['loc_error'][out_i] = location_error
            self.found_sources['station_mask'][out_i] = station_mask_int
            self.found_sources['num_RS'][out_i] = num_remote_stations
            
            out_i += 1
            

        minimizing_time = fitter_timer.get_time()
        loadingData_time = loading_data_timer.get_time()
        planewave_time = planewave_timer.get_time()
        total_timer.stop_timer()
        total_time = total_timer.get_time()


        filtered_data = { k:d[:out_i] for k,d in self.found_sources.items() }, planewave_time, loadingData_time, minimizing_time, total_time

        return filtered_data
    
    def process_blocks(self, initial_block, final_block, print_func=do_nothing, skip_blocks_done=True):
        """process many blocks from initial_block to final_block (not included), and save results to a file"""

        for block_i in range(initial_block,final_block):

            out_fname = join( self.header.tmp_out_data, str(block_i))

            if skip_blocks_done and isfile(out_fname):
                print_func("block:", block_i, "already completed. Skipping")
                continue

            self.data_manager.reset_dataReloadDt_dist()

            data, planewave_time, loadingData_time, minimizing_time, total_time = self.process( block_i , print_func )

            dt_dist = self.data_manager.get_dataReloadDt_dist()

            with open(out_fname, 'w') as fout:
                N = len( data['ID'] )

                fout.write('v1\n')
                fout.write('planewave_time[ns] ')
                fout.write(str(planewave_time))
                fout.write(' ')

                fout.write('loadingData_time[ns] ')
                fout.write(str(loadingData_time))
                fout.write(' ')

                fout.write('minimizing_time[ns] ')
                fout.write(str(minimizing_time))
                fout.write(' ')

                fout.write('total_time[ns] ')
                fout.write(str(total_time))
                fout.write('\n')

                fout.write('numPts ')
                fout.write(str(N))
                fout.write('\n')

                fout.write("ID easting northing height time RMS reduced_chi_squared reference_amplitude loc_error station_mask num_RS\n")

                for i in range(N):
                    fout.write(str(data['ID'][i]) )
                    fout.write( " " )
                    fout.write(str(data['easting'][i]) )
                    fout.write( " " )
                    fout.write(str(data['northing'][i]) )
                    fout.write( " " )
                    fout.write(str( data['height'][i]) )
                    fout.write( " " )
                    fout.write(str( data['time'][i]) )
                    fout.write( " " )
                    fout.write(str( data['RMS'][i]) )
                    fout.write( " " )
                    fout.write(str( data['reduced_chi_squared'][i]) )
                    fout.write( " " )
                    fout.write(str( data['reference_amplitude'][i]) )
                    fout.write( " " )
                    fout.write(str( data['loc_error'][i]) )
                    fout.write( " " )
                    fout.write(str( data['station_mask'][i]) )
                    fout.write( " " )
                    fout.write(str( data['num_RS'][i]) )
                    fout.write( "\n" )


                fout.write('reloadDt ')
                fout.write(str(len(dt_dist)))
                fout.write('\n')
                for d in dt_dist:
                    fout.write(str(d))
                    fout.write(' ')
                fout.write("\n")

    def DiagnositicBlock(self,  block_i, source_i,   print_func=do_nothing):
        """Process a single block and produce diagnositic plots for a specific source. Does not save resulting data file"""

   
        print_func('Diagnositic block', block_i)

        self.data_manager.reset_dataReloadDt_dist()

        data, planewave_time, loadingData_time, minimizing_time, total_time = self.process( block_i , print_func, plotSource=source_i)



    def getBlocksToProcess(self, thread_number, number_threads, num_Consecutive_Blocks=100):
        """where  0 <= threadnumber < number_threads. Return a list of block ranges to process. Where the two items in each range can be passed to process_blocks"""

        total_num_blocks = (self.header.final_time - self.header.initial_time)/( self.sample_time*self.header.analyzable_blocksize  )
        int_total_num_blocks = int( total_num_blocks )
        if (total_num_blocks-int_total_num_blocks) > 0.2:
            total_num_blocks = int_total_num_blocks + 1
        else:
            total_num_blocks = int_total_num_blocks


        ranges_return = []
        current_block = thread_number*num_Consecutive_Blocks
        while current_block < total_num_blocks:

            end_block = current_block + num_Consecutive_Blocks

            if end_block > total_num_blocks:
                end_block = total_num_blocks

            ranges_return.append( [current_block,end_block] )

            current_block += number_threads*num_Consecutive_Blocks

        return ranges_return



                            
                    
# if __name__ == "__main__":
#     from LoLIM import utilities
#     utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
#     utilities.default_processed_data_loc = "/home/brian/processed_files"
    
#     out_folder = 'iterMapper_50_CS002_TST2'
    
#     timeID = 'D20170929T202255.000Z'
#     stations_to_exclude = [ 'RS407', 'RS409']
                
#     outHeader = make_header(timeID, 3000*(2**16), station_delays_fname = 'station_delays4.txt', 
#                 additional_antenna_delays_fname = 'ant_delays.txt', bad_antennas_fname = 'bad_antennas.txt', 
#                 pol_flips_fname = 'polarization_flips.txt')
# #            
#     outHeader.stations_to_exclude = stations_to_exclude    
# #    outHeader.max_events_perBlock = 100
#     outHeader.run( out_folder )
    
# #    read_header('iterMapper_50_CS002', timeID).resave_header( out_folder )
    
#     inHeader = read_header(out_folder, timeID)
# #    inHeader.print_all_info()
    
#     log_fname = inHeader.next_log_file()

#     logger_function = logger()
#     logger_function.set( log_fname, True )
#     logger_function.take_stdout()
#     logger_function.take_stderr()
    
    
#     mapper = iterative_mapper(inHeader, print)
# #    for i in range(1113, 1113+10):
#     mapper.process_block(816, print, False)
            
            
            
            
            
def write_three_jacs_to_file(fname, jacA, jacB, jacC, jacA_name, jacB_name, jacC_name):

    with open(fname, 'w') as fout:
        fout.write(jacA_name)
        fout.write("    |    ")
        fout.write(jacB_name)
        fout.write("    |    ")
        fout.write(jacC_name)
        fout.write("\n")


        fout.write('norms\n')
        fout.write(str(np.linalg.norm(jacA)))
        fout.write("    |    ")
        fout.write(str(np.linalg.norm(jacB)))
        fout.write("    |    ")
        fout.write(str(np.linalg.norm(jacC)))
        fout.write("\n")


        for i in range(len(jacA)):
            S1 = '{0:.2e}  {1:.2e}  {2:.2e}  {3:.2e}  |  '.format( jacA[i,0],  jacA[i,1], jacA[i,2], jacA[i,3])
            S2 = '{0:.2e}  {1:.2e}  {2:.2e}  {3:.2e}  |  '.format( jacB[i,0],  jacB[i,1], jacB[i,2], jacB[i,3] )
            S3 = '{0:.2e}  {1:.2e}  {2:.2e}  {3:.2e}\n'.format( jacC[i,0],  jacC[i,1], jacC[i,2], jacC[i,3] )

            fout.write(S1)
            fout.write(S2)
            fout.write(S3)
            
            
            
            
    