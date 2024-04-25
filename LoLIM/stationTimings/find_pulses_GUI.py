#!/usr/bin/env python3

""" NEED TO ADD GOOD DOC STRING. Press 'h' after running for info on controls. """

## TODO: impove documentation, add ability to search for events

#### future improvements
## add a button that alligns all pulses from all stations
## goto a station
## include the stochastic fitter somehow
## try to have a guided pulse-finder
## metric to estimate "goodness" metric of the pulse shape?
## essentually, try to automate things more, but one small step at a time

from os import listdir

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import hilbert
from scipy.signal import resample
from scipy.signal import decimate

from matplotlib.widgets import RectangleSelector
from matplotlib.transforms import blended_transform_factory
import matplotlib.patches as patches

import h5py

from LoLIM.utilities import processed_data_dir, v_air
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
from LoLIM.signal_processing import remove_saturation
from LoLIM.findRFI import window_and_filter
from LoLIM.signal_processing import parabolic_fit
from LoLIM.func_curry import cur


class clock_offsets:
    """this is a class to track the clock offsets, since there are different ways they can be defined"""
    
    def __init__(self, referance_station, input_offsets, flash_location=None, TBB_dictionary=None):
        
        
        self.referance_station = referance_station
        self.input_offsets = dict(input_offsets)
        
        for sname in TBB_dictionary.keys():
            if sname not in self.input_offsets:
                self.input_offsets[sname] = 0.0
        
        self.visual_offsets = dict( self.input_offsets )
        if flash_location is not None:
            flash_location = np.array(flash_location)
            
            for sname, offset in self.visual_offsets.items():
                station_location = TBB_dictionary[sname].get_LOFAR_centered_positions()[0]
                dt = np.linalg.norm( station_location -  flash_location)/v_air
                self.visual_offsets[ sname ] += dt
                
        self.visual_offsets = {sname:T-self.visual_offsets[referance_station] for sname,T in self.visual_offsets.items()}
        
    def get_visual_offsets(self):
        return self.visual_offsets 
    
    def print_delays(self):

        fout = open('stationDelayOut.txt', 'w')

        for sname, offset in self.input_offsets.items():
            print("'"+sname+"' :", offset, ',')

            fout.write("'"+sname+"' : ")
            fout.write(str(offset))
            fout.write(' ,\n')
            

        print()
        print(' written to stationDelayOut.txt')
        
    def shift_data_right(self, station, shift):
        """apply a shift to the station delays that moves the data to the right. (which makes the delay more negative). Returns new visual offsets, and a boolean.
        If boolean is True, then ALL stations need to be reloaded. If False, then only the one changed. (True if referance station was shifted). Also prints the new input offsets."""
        
        all_shifted = False
        
        self.visual_offsets[station] -= shift
        self.input_offsets[station] -= shift
        
        if station == self.referance_station:
            self.visual_offsets = {sname:T-self.visual_offsets[self.referance_station] for sname,T in self.visual_offsets.items()}
            self.input_offsets = {sname:T-self.input_offsets[self.referance_station] for sname,T in self.input_offsets.items()}
            
            all_shifted = True
            
        self.print_delays()
        return self.visual_offsets, all_shifted
    
class frames:
    """The frame is the set of stations and blocks that are presently being displayed. This class manages that"""
    
    def __init__(self, initial_block, block_shift, station_names, max_num_stations, referance_station, display_block_size, num_blocks,
        TBB_files, RFI_filters, edge_width, block_size, clock_offset_manager, antenna_time_corrections, ave_ref_ant_delay,
        positive_saturation, negative_saturation, saturation_post_removal_length, saturation_half_hann_length, do_remove_saturation,
        do_remove_RFI, decimation_factor):
        self.current_block = initial_block
        self.current_set = 0
        self.block_shift = block_shift
        self.display_block_size = display_block_size
        self.num_blocks = num_blocks
        self.station_names = station_names

        self.TBB_files = TBB_files
        self.RFI_filters = RFI_filters
        self.edge_width = edge_width
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.clock_offset_manager  = clock_offset_manager
        self.antenna_time_corrections = antenna_time_corrections

        self.display_block_size = self.block_size - 2*self.edge_width
        self.ave_ref_ant_delay = ave_ref_ant_delay

        self.positive_saturation = positive_saturation
        self.negative_saturation = negative_saturation
        self.saturation_post_removal_length = saturation_post_removal_length
        self.saturation_half_hann_length = saturation_half_hann_length
        self.do_remove_saturation = do_remove_saturation
        self.do_remove_RFI = do_remove_RFI

        self.decimation_factor = decimation_factor
        if self.decimation_factor is not None:
            # self.decimated_blocksize = int(self.display_block_size/self.decimation_factor)

            EMPTY = np.empty( self.display_block_size )
            demiated_empty = decimate( EMPTY, self.decimation_factor )
            self.decimated_blocksize = len(demiated_empty)

        else:
            self.decimated_blocksize = self.display_block_size
            self.decimation_factor = 1.0

        self.total_data_length = self.decimated_blocksize*self.num_blocks
        
        self.station_sets = []
        current_set = None
        self.current_data = {}
        self.max_num_antennas_per_station = 0 
        for sname in station_names:


            num_antennas = len( self.antenna_time_corrections[sname] )

            self.current_data[ sname ] = np.empty( (num_antennas, self.total_data_length), dtype=np.double )
            self.max_num_antennas_per_station = max( self.max_num_antennas_per_station, num_antennas )


            if sname == referance_station:
                continue
            
            if current_set is None:
                current_set = []
                current_set.append( referance_station )
                
            current_set.append( sname )
            
            if len(current_set) == max_num_stations:
                self.station_sets.append( current_set )
                current_set = None



                
        if current_set is not None:
            self.station_sets.append( current_set )
            
        self.num_sets = len( self.station_sets )
        
        ## plotting things
        self.transform = blended_transform_factory(plt.gca().transAxes, plt.gca().transData)


        self.is_station_loaded = {sname:False for sname in self.station_names}
        self.known_saturation = {sname:None for sname in self.station_names}
        self.station_TOffset = {sname:0.0 for sname in self.station_names}

        self.station_to_height = {}
        for S in self.station_sets:
            for i, sname in enumerate(S):
        
                self.station_to_height[sname] = i 


        self.tmp_block = np.empty( self.block_size , dtype=complex )



    def increment_block(self):
        self.current_block += self.block_shift
        self.is_station_loaded = {sname:False for sname in self.station_names}
        
    def decrement_block(self):
        self.current_block -= self.block_shift
        self.is_station_loaded = {sname:False for sname in self.station_names}
        
    def set_block(self, i):
        self.current_block = i
        self.is_station_loaded = {sname:False for sname in self.station_names}
        
    def get_block(self):
        return self.current_block
    
    def get_T_bounds(self):
        return self.current_block*5.0E-9*self.display_block_size,  (self.current_block+self.num_blocks)*5.0E-9*self.display_block_size

    def increment_station_set(self):
        self.current_set += 1
        if self.current_set == self.num_sets:
            self.current_set = 0
            
    def decrement_station_set(self):
        self.current_set -= 1
        if self.current_set == -1:
            self.current_set = self.num_sets-1
            
    def itter(self):
        return enumerate( self.station_sets[self.current_set] )
            
    def plt_sname_annotations(self):
        for height, sname in self.itter():
            plt.annotate(sname, (0.0, height), textcoords=self.transform, xycoords=self.transform)
    
    def sname_from_index(self, i):
        set = self.station_sets[self.current_set]
        if i>= len(set):
            return None
        elif i<0:
            return None
        else:
            return set[i]

    def reload_station(self, station):

        print('reloading station', station)
        
        station_clock_offset = self.clock_offset_manager.get_visual_offsets()[station]
        antenna_delays = self.antenna_time_corrections[station] - self.ave_ref_ant_delay
        TBB_file = self.TBB_files[station]
        RFI_filter = self.RFI_filters[station]
        initial_block = self.get_block()
        
        #### open data, track saturation ####
        saturated_ranges = {block_i:[] for block_i in range(self.num_blocks)}
        current_data_block = self.current_data[station]


        station_max = 0.0
        for even_ant_i in range(0, len(antenna_delays), 2):
            delay = ( antenna_delays[even_ant_i] + antenna_delays[even_ant_i+1] )*0.5
            total_delay = delay + station_clock_offset
            delay_points = int(total_delay/5.0E-9)
            
            for block_i in range(self.num_blocks):
                #block_start = (block_i+initial_block)*self.block_size
                block_start = (initial_block + block_i)*self.display_block_size


                ##### even #### 
                self.tmp_block[:] = TBB_file.get_data(block_start+delay_points, self.block_size, antenna_index=even_ant_i)
                
                if self.do_remove_saturation:
                    sat_ranges = remove_saturation(self.tmp_block[:], self.positive_saturation, self.negative_saturation, self.saturation_post_removal_length, self.saturation_half_hann_length)
                    saturated_ranges[block_i] += sat_ranges
                if self.do_remove_RFI:
                    filtered_data = RFI_filter.filter( self.tmp_block[:] )
                else:
                    filtered_data = hilbert( self.tmp_block[:] )

                data = np.abs(filtered_data)[ self.edge_width:-self.edge_width ]
                if  self.decimation_factor > 1:
                    current_data_block[even_ant_i, block_i*self.decimated_blocksize:(block_i+1)*self.decimated_blocksize] = decimate( data, self.decimation_factor )
                else:
                    current_data_block[even_ant_i, block_i*self.decimated_blocksize:(block_i+1)*self.decimated_blocksize] = data
                


                
                ##### odd #### 
                self.tmp_block[:] = TBB_file.get_data(block_start+delay_points, self.block_size, antenna_index=even_ant_i+1)
                
                if self.do_remove_saturation:
                    sat_ranges = remove_saturation(self.tmp_block[:], self.positive_saturation, self.negative_saturation, self.saturation_post_removal_length, self.saturation_half_hann_length)
                    saturated_ranges[block_i] += sat_ranges
                if self.do_remove_RFI:
                    filtered_data = RFI_filter.filter( self.tmp_block[:] )
                else:
                    filtered_data = hilbert( self.tmp_block[:] )


                data = np.abs(filtered_data)[ self.edge_width:-self.edge_width ]
                if  self.decimation_factor > 1:
                    current_data_block[even_ant_i+1, block_i*self.decimated_blocksize:(block_i+1)*self.decimated_blocksize] = decimate( data, self.decimation_factor )
                else:
                    current_data_block[even_ant_i+1, block_i*self.decimated_blocksize:(block_i+1)*self.decimated_blocksize] = data
                

            peak_odd = np.max( current_data_block[even_ant_i+1, :] )
            peak_even = np.max( current_data_block[even_ant_i, :] )
            peak = max( peak_odd, peak_even )
            if peak > station_max:
                station_max = peak


        current_data_block /= station_max
        # current_data_block += self.station_to_height[ station ]

        self.known_saturation[station] = saturated_ranges
        self.is_station_loaded[station] = True
        self.station_TOffset[station] = initial_block*self.display_block_size*5.0E-9  + station_clock_offset


    def plotTime_to_StationTime(self, plotT, sname):
        plot_start_t = self.get_T_bounds()[0]
        station_start_t = self.get_block()*self.display_block_size*5.0e-9 + self.clock_offset_manager.get_visual_offsets()[ sname ] + self.edge_width*5.0e-9  - self.ave_ref_ant_delay

        return plotT - plot_start_t + station_start_t



    def get_station_data(self, station):

        if not self.is_station_loaded[station]:
            self.reload_station( station )

        T_array = np.arange( self.total_data_length, dtype=np.double )
        T_array *= self.decimation_factor*5.0e-9
        T_array += self.station_TOffset[ station ] - self.clock_offset_manager.get_visual_offsets()[station]

        return T_array, self.current_data[station], self.known_saturation[station], self.station_to_height[ station ]

                
class pulses:
    """this is a class to manage the selected events and pulses"""
    def __init__(self, pulse_length, block_size, out_folder, TBB_data_dict, used_delay_dict, upsample_factor, min_ant_amp,
                 referance_station, other_folder=None, starting_event_index=None, RFI_filter_dict=None, remove_saturation_curry=None):
        self.half_pulse_length = int(0.5*pulse_length)
        self.pulse_length = 2*self.half_pulse_length
        self.current_event_index = starting_event_index
        self.TBB_data_dict = TBB_data_dict
        self.antenna_delays = used_delay_dict
        self.block_size = block_size
        self.RFI_filter_dict = RFI_filter_dict
        self.out_folder = out_folder
        self.remove_saturation_curry = remove_saturation_curry
        self.upsample_factor = upsample_factor
        self.min_ant_amp = min_ant_amp
        self.referance_station = referance_station
        
        self.temp_data = {sname:np.empty((len(data.get_antenna_names()),pulse_length), dtype=np.double) for sname,data in TBB_data_dict.items()}
        self.temp_data_block = np.empty( (2,self.block_size), dtype=np.double)
        
        self.editable_pulses = self.open_pulse_files( out_folder )
        self.fixed_event_info = {}
        if other_folder is not None:
            self.fixed_event_info = self.open_pulse_files( other_folder )
        
        if self.current_event_index is None:
            self.select_unused_index()
            
    def open_pulse_files(self, folder):
        
        out_dict = {}
        
        for filename in listdir(folder):
            if filename.endswith(".h5"):
                
                index = int(filename[10:-3])
                out_file = h5py.File(folder+'/'+filename, "r")
                event_dict = {}
                
                for station, stat_group in out_file.items():
                    
                    even_times = []
                    odd_times = []
                    for ant_dataset in stat_group.values():
                        even_time = ant_dataset.attrs['PolE_peakTime']
                        odd_time = ant_dataset.attrs['PolO_peakTime']
                        
                        if np.isfinite(even_time):
                            even_times.append( even_time )
                        if np.isfinite(odd_time):
                            odd_times.append( odd_time )
                            
                    event_dict[ station ] = ( np.array(even_times), np.array(odd_times) )
                out_dict[index] = event_dict
                
        return out_dict
        
    def select_unused_index(self):
        i = 0
        while (i in self.fixed_event_info) or (i in self.editable_pulses):
            i += 1
            
        self.set_current_event_index( i )
        
    def set_current_event_index(self, new_event_index):
#        if new_event_index in self.fixed_event_info:
#            print("EVENT", new_event_index, "cannot be edited")
#        else:
        self.current_event_index = new_event_index
        self.print_current_event_index()
            
    def print_current_event_index(self):
        print("Current EVENT to edit is:", self.current_event_index)
        
    def decrement_pulse(self):
        if self.current_event_index > 0:
            self.set_current_event_index( self.current_event_index-1 )
        
    def increment_pulse(self):
        self.set_current_event_index( self.current_event_index+1 )
        
    def set_unique_pulse_ID(self):
        current_IDs = list( self.editable_pulses.keys() )
        i = 0
        while i in current_IDs:
            i += 1
        self.set_current_event_index( i )
        
    def print_pulse_info(self):
        print()
        print('current pulse:', self.current_event_index)

        for ID, data in self.editable_pulses.items():
            print()
            print()
            print("pulse", ID)
            stations = list( data.keys() )
            print("  ", len(stations), "stations saved:")
            print("  ", stations)
        
    def select_pulse(self, station, minT, maxT):
        TBB_data = self.TBB_data_dict[station]
        antenna_names = TBB_data.get_antenna_names()
        ant_delays = self.antenna_delays[station]
        
        out_fname = self.out_folder + "/potSource_"+str(self.current_event_index)+".h5"
        try:
            out_file = h5py.File(out_fname, "r+")
        except:
            out_file = h5py.File(out_fname, "w")
        h5_statGroup = out_file.require_group( station )
        
        mid_T= (minT+maxT)*0.5
        half_width_T = (maxT-minT)*0.5
        half_block = int(self.block_size/2)
        half_width_T_points = int(half_width_T/5.0E-9)
        
        if self.RFI_filter_dict is not None:
            RFI_filter = self.RFI_filter_dict[station]
        else:
            RFI_filter = None
        
        
        even_pulse_time_list = []
        odd_pulse_time_list = []
        
        print()
        print(station)
        for even_ant_i in range(0,len(antenna_names),2):
            even_delay = ant_delays[even_ant_i]
            odd_delay = ant_delays[even_ant_i+1]
            
            ave_delay = (even_delay + odd_delay)*0.5
            time_points = int( (mid_T + ave_delay)/5.0E-9 )
            
            ## even bit
            self.temp_data_block[0,:] = TBB_data.get_data(time_points-half_block, self.block_size, antenna_index=even_ant_i)
            
            if self.remove_saturation_curry is not None:
                self.remove_saturation_curry(self.temp_data_block[0,:])

            if RFI_filter is not None:
                filtered_data = RFI_filter.filter( self.temp_data_block[0,:] )
            else:
                filtered_data = hilbert( self.temp_data_block[0,:] )
                
            even_real_all = np.real(filtered_data)
            np.abs(filtered_data, out=self.temp_data_block[0,:])
            
            ## odd bit
            self.temp_data_block[1,:] = TBB_data.get_data(time_points-half_block, self.block_size, antenna_index=even_ant_i+1)
            
            if self.remove_saturation_curry is not None:
                self.remove_saturation_curry(self.temp_data_block[1,:])

            if RFI_filter is not None:
                filtered_data = RFI_filter.filter( self.temp_data_block[1,:] )
            else:
                filtered_data = hilbert( self.temp_data_block[1,:] )
                
            odd_real_all = np.real(filtered_data)
            np.abs(filtered_data, out=self.temp_data_block[1,:])
            
            
            even_peak_index = np.argmax( self.temp_data_block[0,  half_block-half_width_T_points : half_block+half_width_T_points  ] )   + (half_block-half_width_T_points)
            odd_peak_index  = np.argmax( self.temp_data_block[1,  half_block-half_width_T_points : half_block+half_width_T_points  ] )   + (half_block-half_width_T_points)
            
            peak_index = odd_peak_index
            if self.temp_data_block[0,even_peak_index] > self.temp_data_block[1,odd_peak_index]:
                peak_index = even_peak_index
                
                
            initial_index = peak_index-self.half_pulse_length
            even_HE = self.temp_data_block[0, initial_index : initial_index+self.pulse_length]
            even_real = even_real_all[        initial_index : initial_index+self.pulse_length]
            odd_HE = self.temp_data_block[1,  initial_index : initial_index+self.pulse_length]
            odd_real = odd_real_all[          initial_index : initial_index+self.pulse_length]
            
            ##### now we save the data ####

            print( antenna_names[even_ant_i], 'amp:', np.max(even_HE) )
            print( antenna_names[even_ant_i+1], 'amp:', np.max(odd_HE) )
            
            h5_Ant_dataset = h5_statGroup.require_dataset(antenna_names[even_ant_i], (4, self.pulse_length), dtype=np.double, exact=True)
            
            h5_Ant_dataset[0] = even_real
            h5_Ant_dataset[1] = even_HE
            h5_Ant_dataset[2] = odd_real
            h5_Ant_dataset[3] = odd_HE
            
            starting_index = (time_points-half_block) + initial_index
            h5_Ant_dataset.attrs['starting_index'] = starting_index
            h5_Ant_dataset.attrs['PolE_timeOffset'] = -even_delay ## NOTE: due to historical reasons, there is a sign flip here
            h5_Ant_dataset.attrs['PolO_timeOffset'] = -odd_delay ## NOTE: due to historical reasons, there is a sign flip here
            h5_Ant_dataset.attrs['PolE_timeOffset_CS'] = even_delay
            h5_Ant_dataset.attrs['PolO_timeOffset_CS'] = odd_delay 
#            ave_delay = (even_delay+odd_delay)*0.5
            
            sample_time = 5.0E-9
            PolE_HE = even_HE
            PolO_HE = odd_HE
            if self.upsample_factor > 1:
                PolE_HE = resample(even_HE, len(even_HE)*self.upsample_factor )
                PolO_HE = resample(odd_HE, len(odd_HE)*self.upsample_factor )
                
                sample_time /= self.upsample_factor
            
            
            if np.max(PolE_HE)> self.min_ant_amp:
                
                
                PolE_peak_finder = parabolic_fit( PolE_HE )
                h5_Ant_dataset.attrs['PolE_peakTime'] =  starting_index*5.0E-9 - even_delay + PolE_peak_finder.peak_index*sample_time
                even_pulse_time_list.append( h5_Ant_dataset.attrs['PolE_peakTime'] )
            else:
                h5_Ant_dataset.attrs['PolE_peakTime'] = np.nan
            
            if np.max(PolO_HE)> self.min_ant_amp:
                
                
                PolO_peak_finder = parabolic_fit( PolO_HE )
                h5_Ant_dataset.attrs['PolO_peakTime'] =  starting_index*5.0E-9 - odd_delay + PolO_peak_finder.peak_index*sample_time
                odd_pulse_time_list.append( h5_Ant_dataset.attrs['PolO_peakTime'] )
            else:
                h5_Ant_dataset.attrs['PolO_peakTime'] =  np.nan
                
            print("  ", antenna_names[even_ant_i], h5_Ant_dataset.attrs['PolE_peakTime'], h5_Ant_dataset.attrs['PolO_peakTime'])
                
            
        if self.current_event_index not in self.editable_pulses:
            self.editable_pulses[ self.current_event_index ] = {}
        self.editable_pulses[ self.current_event_index ][station] = (np.array(even_pulse_time_list), np.array(odd_pulse_time_list))
        
        
    def erase_pulse(self, station, minT, maxT):
        
        out_fname = self.out_folder + "/potSource_"+str(self.current_event_index)+".h5"
        try:
            out_file = h5py.File(out_fname, "r+")
        except:
            print("cannot erase")
            return
        h5_statGroup = out_file[ station ]
        
        even_pulse_time_list = []
        odd_pulse_time_list = []
        for ant_dataset in h5_statGroup.values():
            
            polE_time = ant_dataset.attrs['PolE_peakTime']
            if np.isfinite( polE_time ) and polE_time>minT and polE_time<maxT:
                ant_dataset.attrs['PolE_peakTime'] = np.nan
            elif np.isfinite( polE_time ):
                even_pulse_time_list.append( polE_time )
            
            polO_time = ant_dataset.attrs['PolO_peakTime']
            if np.isfinite( polE_time ) and polO_time>minT and polO_time<maxT:
                ant_dataset.attrs['PolO_peakTime'] = np.nan
            elif np.isfinite( polO_time ):
                odd_pulse_time_list.append( polO_time )
                
        self.editable_pulses[ self.current_event_index ][station] = (np.array(even_pulse_time_list), np.array(odd_pulse_time_list) )
                

    def plot_lines(self, station, height, min_T, max_T, delay):
        
        #is_ref_stat = (station == self.referance_station )
        
        for pulse_I, eventData in self.editable_pulses.items():
            if station in eventData:
                even_times, odd_times = eventData[station]
                
                even_times = even_times - delay
                odd_times = odd_times - delay
                
                for T in even_times:
                    if T>min_T and T<max_T:
                        plt.plot([T,T], [height,height+1], 'g')
                for T in odd_times:
                    if T>min_T and T<max_T:
                        plt.plot([T,T], [height,height+1], 'm')
                    
                if (len(even_times)>0 or len(odd_times)>0):
                    ave = ( np.sum(even_times)+np.sum(odd_times) ) / (len(even_times)+len(odd_times))
                    
                    
                    if ave>min_T and ave<max_T:
                        
                        plt.plot([ave,ave], [height,height+1], 'tab:brown')
                        
                        #if is_ref_stat:
                        plt.annotate(str(pulse_I), (ave, height+0.5), fontsize=15)
                        
        
        for pulse_I, eventData in self.fixed_event_info.items():
            if station in eventData:
                even_times, odd_times = eventData[station]
                
                even_times = even_times - delay
                odd_times = odd_times - delay
                
                for T in even_times:
                    if T>min_T and T<max_T:
                        plt.plot([T,T], [height,height+1], 'g')
                for T in odd_times:
                    if T>min_T and T<max_T:
                        plt.plot([T,T], [height,height+1], 'm')
                    
                if (len(even_times)>0 or len(odd_times)>0):
                    ave = ( np.sum(even_times)+np.sum(odd_times) ) / (len(even_times)+len(odd_times))
                    
                    if ave>min_T and ave<max_T:
                        plt.plot([ave,ave], [height,height+1], 'k')
                        
                        #if is_ref_stat:
                        plt.annotate(str(pulse_I), (ave, height+0.5), fontsize=15)
                        
                        
    def search_for_pulse(self, sname, I):
        if I in self.editable_pulses:
            return np.average( self.editable_pulses[I][sname][0] )
        else:
            print("cannot find pulse ", I)
            return None
                        
                        
        
    
class plot_stations:
    def __init__(self, timeID, guess_delays, block_size, initial_block, num_blocks, working_folder, other_folder=None, max_num_stations=np.inf, guess_location = None, 
                bad_stations=[], total_cal_file=None, polarization_flips="polarization_flips.txt", bad_antennas = "bad_antennas.txt", additional_antenna_delays = None,
                do_remove_saturation = True, do_remove_RFI = True, positive_saturation = 2046, negative_saturation = -2047, saturation_post_removal_length = 50,
                saturation_half_hann_length = 50, referance_station = "CS002", pulse_length=50, upsample_factor=4, min_antenna_amplitude=5, fix_polarization_delay=True,
                decimation_factor=4):
        
        #### disable matplotlib shortcuts
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.fullscreen'] = ''
        plt.rcParams['keymap.home'] = ''
        plt.rcParams['keymap.back'] = ''
        plt.rcParams['keymap.forward'] = ''
        plt.rcParams['keymap.pan'] = ''
        plt.rcParams['keymap.zoom'] = ''
        plt.rcParams['keymap.save'] = ''
        plt.rcParams['keymap.quit_all'] = ''
        plt.rcParams['keymap.grid'] = ''
        plt.rcParams['keymap.grid_minor'] = ''
        plt.rcParams['keymap.yscale'] = ''
        plt.rcParams['keymap.xscale'] = ''
        #plt.rcParams['keymap.all_axes'] = ''
        plt.rcParams['keymap.copy'] = ''

#keymap.help : f1                    ## display help about active tools
#keymap.quit : ctrl+w, cmd+w, q      ## close the current figure
        
        
        
        
        self.pressed_data = None
        #guess_location = np.array(guess_location[:3])
        
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        self.positive_saturation = positive_saturation
        self.negative_saturation = negative_saturation
        self.saturation_post_removal_length = saturation_post_removal_length
        self.saturation_half_hann_length = saturation_half_hann_length
        self.do_remove_saturation = do_remove_saturation
        self.do_remove_RFI = do_remove_RFI
        self.referance_station = referance_station
        
        #### open data ####

        processed_data_folder = processed_data_dir(timeID)
                
        raw_fpaths = filePaths_by_stationName(timeID)
        station_names = [sname for sname in raw_fpaths.keys() if sname not in bad_stations]


        if total_cal_file:
            self.TBB_files = {sname:MultiFile_Dal1(raw_fpaths[sname], total_cal=processed_data_folder + '/' +total_cal_file) \
                          for sname in station_names}


            for sname, file in self.TBB_files.items(): ## need to ignore station delays in total cal
                file.set_station_delay(0)

        else:
            polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
            bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
            if additional_antenna_delays is not None:
                additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )


            self.TBB_files = {sname:MultiFile_Dal1(raw_fpaths[sname], polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays) \
                              for sname in station_names}
            
            if fix_polarization_delay:
                for sname, file in self.TBB_files.items():
                    
                    file.find_and_set_polarization_delay()

            
        self.RFI_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in station_names}
        self.edge_width = int( self.block_size*0.1 )
        self.display_block_size = self.block_size - 2*self.edge_width
    
        #### some data things ####
        self.antenna_time_corrections = {sname:TBB_file.get_total_delays() for sname,TBB_file in self.TBB_files.items()}
        self.ave_ref_ant_delay = np.average( self.antenna_time_corrections[ referance_station ] )
        
#        ref_antenna_delay = self.antenna_time_corrections[ referance_station ][0]
#        for station_corrections in self.antenna_time_corrections.values():
#            station_corrections -= ref_antenna_delay
            
        max_num_antennas = np.max( [len(delays) for delays in self.antenna_time_corrections.values()] )
        self.temp_data_block = np.empty( (max_num_antennas, num_blocks, block_size), dtype=np.double )
        self.time_array = np.arange(block_size)*5.0E-9
        
        self.figure = plt.gcf()
        self.axes = plt.gca()
        plt.tight_layout()
        
        
        
        
        #### make managers
        guess_delays = {sname:delay for sname,delay in guess_delays.items() if sname not in bad_stations}
        self.clock_offset_manager = clock_offsets(referance_station, guess_delays, guess_location, self.TBB_files)

        self.frame_manager = frames(initial_block, int(num_blocks/2), station_names, max_num_stations, referance_station, self.display_block_size, num_blocks,
            self.TBB_files, self.RFI_filters, self.edge_width, self.block_size, self.clock_offset_manager, self.antenna_time_corrections, self.ave_ref_ant_delay,
            self.positive_saturation, self.negative_saturation, self.saturation_post_removal_length, self.saturation_half_hann_length, do_remove_saturation,
            do_remove_RFI, decimation_factor )
        
        rem_sat_curry = cur( remove_saturation, 5 )(positive_saturation=positive_saturation, negative_saturation=negative_saturation, post_removal_length=saturation_post_removal_length, half_hann_length=saturation_half_hann_length)
        self.pulse_manager = pulses(pulse_length=pulse_length, block_size=block_size, out_folder=working_folder, other_folder=other_folder, TBB_data_dict=self.TBB_files, used_delay_dict=self.antenna_time_corrections, 
                                    upsample_factor=upsample_factor, min_ant_amp=min_antenna_amplitude,referance_station=referance_station, 
                                    RFI_filter_dict=(self.RFI_filters if self.do_remove_RFI  else None), remove_saturation_curry=(rem_sat_curry if self.do_remove_saturation  else None))
        
        

        self.mode = 0 ##0 is change delay, 1 is set peak, etc...
        self.input_mode = 0 ## 0 is normal. press 's' to enter 1, where numbers are used to enter event. press 's' again to go back to 0, and search for event
        self.antenna_display_mode = 0 ## 0 means all antennas on top of each other. 1 means seperated
        self.search_string = ''

        self.plot()
        
        self.rect_selector = RectangleSelector(plt.gca(), self.rectangle_callback, useblit=True, 
                                               props=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
        
        self.mouse_press = self.figure.canvas.mpl_connect('button_press_event', self.mouse_press) 
        self.mouse_release = plt.gcf().canvas.mpl_connect('button_release_event', self.mouse_release)
        
        self.key_press = plt.gcf().canvas.mpl_connect('key_press_event', self.key_press_event) 
        
        self.home_lims = [ self.axes.get_xlim(), self.axes.get_ylim() ]
            
        plt.show()
        
        
    def plot(self):
        self.axes.cla()
        
        for height, sname in self.frame_manager.itter():
            self.plot_a_station(sname, height)
        self.frame_manager.plt_sname_annotations()
        
        self.axes.autoscale()
        self.home_lims = [ self.axes.get_xlim(), self.axes.get_ylim() ]
   
    # def plot_a_station(self, sname, height):
    #     print("plotting", sname)
        
    #     station_clock_offset = self.clock_offset_manager.get_visual_offsets()[sname]
    #     antenna_delays = self.antenna_time_corrections[sname] - self.ave_ref_ant_delay
    #     TBB_file = self.TBB_files[sname]
    #     RFI_filter = self.RFI_filters[sname]
    #     initial_block = self.frame_manager.get_block()
        
    #     #### open data, track saturation ####
    #     saturated_ranges = {block_i:[] for block_i in range(self.num_blocks)}
    #     station_max = 0.0
        
    #     for even_ant_i in range(0, len(antenna_delays), 2):
    #         delay = ( antenna_delays[even_ant_i] + antenna_delays[even_ant_i+1] )*0.5
    #         total_delay = delay + station_clock_offset
    #         delay_points = int(total_delay/5.0E-9)
            
    #         for block_i in range(self.num_blocks):
    #             #block_start = (block_i+initial_block)*self.block_size
    #             block_start = (initial_block + block_i)*self.display_block_size
                
    #             ##### even #### 
    #             self.temp_data_block[even_ant_i, block_i, :] = TBB_file.get_data(block_start+delay_points, self.block_size, antenna_index=even_ant_i)
                
    #             if self.do_remove_saturation:
    #                 sat_ranges = remove_saturation(self.temp_data_block[even_ant_i, block_i, :], self.positive_saturation, self.negative_saturation, self.saturation_post_removal_length, self.saturation_half_hann_length)
    #                 saturated_ranges[block_i] += sat_ranges
    #             if self.do_remove_RFI:
    #                 filtered_data = RFI_filter.filter( self.temp_data_block[even_ant_i, block_i, :] )
    #             else:
    #                 filtered_data = hilbert( self.temp_data_block[even_ant_i, block_i, :] )
    #             self.temp_data_block[even_ant_i, block_i, :] = np.abs(filtered_data)
                
    #             peak = np.max( self.temp_data_block[even_ant_i, block_i, :] )
    #             if peak > station_max:
    #                 station_max = peak
                
    #             ##### odd #### 
    #             self.temp_data_block[even_ant_i+1, block_i, :] = TBB_file.get_data(block_start+delay_points, self.block_size, antenna_index=even_ant_i+1)
                
    #             if self.do_remove_saturation:
    #                 sat_ranges = remove_saturation(self.temp_data_block[even_ant_i+1, block_i, :], self.positive_saturation, self.negative_saturation, self.saturation_post_removal_length, self.saturation_half_hann_length)
    #                 saturated_ranges[block_i] += sat_ranges
    #             if self.do_remove_RFI:
    #                 filtered_data = RFI_filter.filter( self.temp_data_block[even_ant_i+1, block_i, :] )
    #             else:
    #                 filtered_data = hilbert( self.temp_data_block[even_ant_i+1, block_i, :] )
    #             self.temp_data_block[even_ant_i+1, block_i, :] = np.abs(filtered_data)
                
    #             peak = np.max( self.temp_data_block[even_ant_i+1, block_i, :] )
    #             if peak > station_max:
    #                 station_max = peak



    #     time_by_block = {block_i: (self.time_array + (initial_block + block_i) * self.display_block_size * 5.0E-9) for
    #                      block_i in range(self.num_blocks)}
    #     #### plot data ####
    #     for pair_i in range( int(len(antenna_delays)/2) ):
    #         even_ant = pair_i*2
    #         odd_odd = pair_i*2 + 1
            
    #         for block_i in range(self.num_blocks):
    #             block_start = (initial_block + block_i)*self.display_block_size 
                
    #             even_trace = self.temp_data_block[even_ant, block_i][ self.edge_width:-self.edge_width  ]
    #             odd_trace = self.temp_data_block[odd_odd, block_i][ self.edge_width:-self.edge_width ]
                
    #             even_trace /= station_max
    #             odd_trace /= station_max
                
    #             even_trace += height
    #             odd_trace += height
                
    #             T = time_by_block[block_i][ self.edge_width:-self.edge_width ]
    #             plt.plot(T, even_trace, 'g')
    #             plt.plot(T, odd_trace, 'm')

    #     #### plot saturated bits ####
    #     for block_i, ranges in saturated_ranges.items():
    #         T = time_by_block[block_i]
    #         for imin, imax in ranges:
    #             Tmin = T[imin]
    #             width = T[imax - 1] - Tmin  ## range does not include end point

    #             rect = patches.Rectangle((Tmin, height), width, 1, linewidth=1, edgecolor='r', facecolor='r')
    #             self.axes.add_patch(rect)
                
    #     minT, maxT = self.frame_manager.get_T_bounds()
    #     self.pulse_manager.plot_lines( sname, height, minT, maxT, station_clock_offset - self.ave_ref_ant_delay)


   
    def plot_a_station(self, sname, height):
        print("plotting", sname)

        time_array, data, saturated_ranges, bottom = self.frame_manager.get_station_data( sname )

        num_pairs = int(len(data)/2)
        #### plot data ####
        for pair_i in range( num_pairs ):
            even_ant = pair_i*2
            odd_odd = pair_i*2 + 1

            if self.antenna_display_mode == 0:
                plt.plot(time_array, data[even_ant] + bottom, 'g')
                plt.plot(time_array, data[odd_odd] + bottom, 'm')
            else:
                A = data[odd_odd]/num_pairs 
                A += pair_i/num_pairs + bottom
                B = data[even_ant]/num_pairs 
                B += pair_i/num_pairs + bottom

                plt.plot(time_array, A, 'm')
                plt.plot(time_array, B, 'g')


        #### plot saturated bits ####
        try:
            for block_i, ranges in saturated_ranges.items():
                print('saturated ranges in block', block_i, 'N:', len(ranges) )

                time_by_block = time_array[ block_i*self.frame_manager.decimated_blocksize : (block_i+1)*self.frame_manager.decimated_blocksize ]
                # T = time_by_block[block_i]
                for imin, imax in ranges:
                    Tmin = time_by_block[imin]
                    width = time_by_block[imax - 1] - Tmin  ## range does not include end point

                    rect = patches.Rectangle((Tmin, height), width, 1, linewidth=1, edgecolor='r', facecolor='r')
                    self.axes.add_patch(rect)
        except Exception as e:
            print('ERROR when plotting saturated areas:')
            print(e)

                

        minT, maxT = self.frame_manager.get_T_bounds()

        # station_clock_offset = self.clock_offset_manager.get_visual_offsets()[sname]
        # self.pulse_manager.plot_lines( sname, height, minT, maxT, station_clock_offset)# - self.ave_ref_ant_delay)

        self.pulse_manager.plot_lines( sname, height, minT, maxT,  self.frame_manager.plotTime_to_StationTime(0, sname) )

                
    def key_press_event(self, event):
        ### modes ###
        print("letter:", event.key)
        if self.input_mode == 0:
            if event.key == '0':
                self.mode=0
                print("MODE: OFF")
            elif event.key == '1':
                self.mode=1
                print("MODE: positive shift")
                
            elif event.key == '2':
                self.mode=2
                print("MODE: negative shift")
                
            elif event.key == 'z':
                self.mode =3
                print("MODE: zoom in")
                
            elif event.key == 'c':
                self.mode =4
                print("MODE: zoom out")
                
            elif event.key == '3':
                self.mode=5
                print("MODE: find pulse index", self.pulse_manager.current_event_index)
                
            ### change frame ###
            elif event.key == 'w':
                self.frame_manager.increment_station_set()
                self.plot()
                plt.draw()
                
            elif event.key == 'x':
                self.frame_manager.decrement_station_set()
                self.plot()
                plt.draw()
                
            elif event.key == 'd':
                self.frame_manager.increment_block()
                self.plot()
                plt.draw()
                print('t bounds', self.frame_manager.get_T_bounds())
                
            elif event.key == 'a':
                self.frame_manager.decrement_block()
                self.plot()
                plt.draw()
                print('t bounds', self.frame_manager.get_T_bounds())
                
            elif event.key == 'e':
                self.mode=6
                print("MODE: erase pulse")
                
            elif event.key == 'i':
                self.input_mode = 1
                self.search_string = ''
                print("WARNING: entering input mode, normal keys will not help you here!")
                
            ### other commands ###
            elif event.key == 'j':
                self.axes.set_xlim( self.home_lims[0][0], self.home_lims[0][1] )
                self.axes.set_ylim( self.home_lims[1][0], self.home_lims[1][1] )
                plt.draw()
                
            elif event.key == 'p':
                self.clock_offset_manager.print_delays()
                
            elif event.key == 'b':
                print('current block is', self.frame_manager.current_block)
                
            elif event.key == 'o':
                self.pulse_manager.print_pulse_info()
                
            elif event.key == '+':
                self.pulse_manager.increment_pulse()
                
            elif event.key == '-':
                self.pulse_manager.decrement_pulse()

            elif event.key == 'r':
                for i,sname in self.frame_manager.itter():
                    self.frame_manager.reload_station(sname)
                self.plot()
                plt.draw()
                
            elif event.key == 'n':
                self.pulse_manager.set_unique_pulse_ID()

            elif event.key == 't':
                if self.antenna_display_mode == 0:
                    self.antenna_display_mode = 1
                    print()
                    print('now plotting antennas seperately')
                    print()
                elif self.antenna_display_mode == 1:
                    self.antenna_display_mode = 0
                    print()
                    print('now plotting antennas together')
                    print()
                self.plot()
                plt.draw()
            
            elif event.key == 'h':
                print("how to use:")
                print(" press 'w' and 'x' to change group of stations. Referance station always stays at bottom")
                print(" press 'a' and 'd' to view later or earlier blocks")
                print(" 'p' prints current delays")
                print(" 'b' prints current block")
                print(" 'o' prints current pulse info")
                print(" 'r' forces a reload of the data")
                print(" hold middle mouse button and drag to translate view")
                print(" j resets view (zoom and translate) back to default.")
                print()
                print(" t switches between plotting antennas together or seperately. Note, only whole stations can still be selected. Not individual antennas.")

                print()
                print(" there are multiple modes, that define what happens when main mouse button is dragged")
                print("   '+' and '-' increase and decrease the pulse number that is being saved to.")
                print("   'n' will set the pulse number to a new pulse.") ## TODO
                print("   'z' and 'c' enter zoom-in and zoom-out modes")
                print("   '1' and '2' shift stations right or left, and adjust timing accordingly. (they shift the station below the mouse, so draw a horizontal line).")
                print("   '3' selects pulses for all stations that the box crosses")
                print("   'e' erases pulses, for the station immediatly below the mouse")
                print("   '0' is 'off', nothing happens if you drag mouse")
                print()
                print(" every selected pulse gets a red line, and an annotation on the referance station.")
                print(" pulses from the other_folder get a black line")
                print(" areas of the traces affected by saturation are covered by a red box")
                print()
                print(" in order to search for a particular pulse, enter 'input mode' by pressing 'i'")
                print("    then type the index of the event you want to search for")
                print("    press 'i' again to finish and center on the event")
                print(" in order to jump to a different block, enter 'input mode' by pressing 'i'")
                print("    then type the number of the block you want to jump to")
                print("    then press 'j' to do the jump")
                
            else:
                print("pressed:", event.key, ". press 'h' for help")
                
        elif self.input_mode == 1:
            if event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                self.search_string += event.key
                print('input number:', self.search_string)
                
            elif event.key == 'i':
                try:
                    index = int(self.search_string)
                except:
                    print("not a valid index")
                else:
                    T = self.pulse_manager.search_for_pulse(  self.referance_station, index )
                    if T is not None:
                        ## the "delay" for ref-station is zero, so do not delete it
                        print('showning event:', index)
                        T += self.ave_ref_ant_delay
                        block = int(T/(self.display_block_size*5.0E-9)) - 1
                        self.frame_manager.set_block(block)
                        self.plot()
                        plt.draw()
                self.search_string = ''
                self.input_mode = 0
            elif event.key == 'j':
                try:
                    index = int(self.search_string)
                except:
                    print("not a block number")
                else:
                    print('going to block:', index)
                    self.frame_manager.set_block(index)
                    self.plot()
                    plt.draw()
                self.search_string = ''
                self.input_mode = 0
            
            
            
    def rectangle_callback(self, eclick, erelease):
        
        
        
#        if self.mode==0:
#          minH = min(eclick.ydata, erelease.ydata)  
#          maxH = min(eclick.ydata, erelease.ydata)  
          
          
        click_H = int(eclick.ydata)
        if eclick.ydata<0:
            click_H = -1
        relea_H = int(erelease.ydata)
        if erelease.ydata<0:
            relea_H = -1
        
        if self.mode==1 and click_H==relea_H:
            ## add positive shift
            station = self.frame_manager.sname_from_index( click_H )
            offset = erelease.xdata - eclick.xdata
            new_offsets, replot_all = self.clock_offset_manager.shift_data_right(station, np.abs(offset) )
            
            Xlims, Ylims = self.axes.get_xlim(), self.axes.get_ylim()
            self.plot()
            self.axes.set_xlim(Xlims[0], Xlims[1])
            self.axes.set_ylim(Ylims[0], Ylims[1])
            plt.draw()
            
        elif self.mode==2 and click_H==relea_H:
            ## add negative shift
            station = self.frame_manager.sname_from_index( click_H )
            offset = erelease.xdata - eclick.xdata
            new_offsets, replot_all = self.clock_offset_manager.shift_data_right(station, -np.abs(offset) )
            
            Xlims, Ylims = self.axes.get_xlim(), self.axes.get_ylim()
            self.plot()
            self.axes.set_xlim(Xlims[0], Xlims[1])
            self.axes.set_ylim(Ylims[0], Ylims[1])
            plt.draw()
            
        elif self.mode==3:
            ## zoom in 
            minX = min(eclick.xdata, erelease.xdata)
            maxX = max(eclick.xdata, erelease.xdata)
            minY = min(eclick.ydata, erelease.ydata)
            maxY = max(eclick.ydata, erelease.ydata)
            self.axes.set_xlim( minX, maxX )
            self.axes.set_ylim( minY, maxY )
            plt.draw()
            
        elif self.mode==4:
            ## zoom out
            minX = min(eclick.xdata, erelease.xdata)
            maxX = max(eclick.xdata, erelease.xdata)
            minY = min(eclick.ydata, erelease.ydata)
            maxY = max(eclick.ydata, erelease.ydata)
            
            Xl, Xu = self.axes.get_xlim()
            pXl = (minX - Xl)/( Xu-Xl )
            pXu = (maxX - Xl)/( Xu-Xl )
            
            WXl = (pXl*Xu - pXu*Xl)/(pXl-pXu)
            WXu = (Xl-WXl)/pXl + WXl
            
            Yl, Yu = self.axes.get_ylim()
            pYl = (minY - Yl)/( Yu-Yl )
            pYu = (maxY - Yl)/( Yu-Yl )
            
            WYl = (pYl*Yu - pYu*Yl)/(pYl-pYu)
            WYu = (Yl-WYl)/pYl + WYl
            
            self.axes.set_xlim( WXl, WXu )
            self.axes.set_ylim( WYl, WYu )
            plt.draw()
            
        elif self.mode==5:
            ## select pulse
            min_H = min(click_H, relea_H )
            max_H = max(click_H, relea_H )
            minT = min(eclick.xdata, erelease.xdata)
            maxT = max(eclick.xdata, erelease.xdata)
            for H in range(min_H+1, max_H+1):
                station = self.frame_manager.sname_from_index( H )
                if station is None:
                    continue
                print(station, H)

                # delay = self.clock_offset_manager.get_visual_offsets()[station] - self.ave_ref_ant_delay
                # self.pulse_manager.select_pulse(station, minT+delay, maxT+delay)


                min_station_T = self.frame_manager.plotTime_to_StationTime( minT, station)
                max_station_T = self.frame_manager.plotTime_to_StationTime( maxT, station)
                self.pulse_manager.select_pulse(station, min_station_T, max_station_T)

                
            Xlims, Ylims = self.axes.get_xlim(), self.axes.get_ylim()
            self.plot()
            self.axes.set_xlim(Xlims[0], Xlims[1])
            self.axes.set_ylim(Ylims[0], Ylims[1])
            plt.draw()
            
        elif self.mode==6 and click_H==relea_H:
            ## erase pulse
            station = self.frame_manager.sname_from_index( click_H )
            minT = min(eclick.xdata, erelease.xdata)
            maxT = max(eclick.xdata, erelease.xdata)
            if station is None:
                return
            
            delay = self.clock_offset_manager.get_visual_offsets()[station] - self.ave_ref_ant_delay
            self.pulse_manager.erase_pulse(station, minT+delay, maxT+delay)
                
            Xlims, Ylims = self.axes.get_xlim(), self.axes.get_ylim()
            self.plot()
            self.axes.set_xlim(Xlims[0], Xlims[1])
            self.axes.set_ylim(Ylims[0], Ylims[1])
            plt.draw()
                
            
            
    def mouse_press(self, data):
        if data.button == 2:
            self.pressed_data = data
            
    def mouse_release(self, data):
        if data.button == 2 and self.pressed_data != None:
            drag_X = data.xdata - self.pressed_data.xdata
            drag_Y = data.ydata - self.pressed_data.ydata
        
            Xlims = self.axes.get_xlim()
            self.axes.set_xlim(Xlims[0]-drag_X, Xlims[1]-drag_X)
            
            Ylims = self.axes.get_ylim()
            self.axes.set_ylim(Ylims[0]-drag_Y, Ylims[1]-drag_Y)
            
            plt.draw()
            
        
        
        
        
        
        
        
        
        
        
            