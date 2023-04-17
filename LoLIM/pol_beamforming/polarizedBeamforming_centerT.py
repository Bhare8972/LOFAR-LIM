#!/usr/bin/env python3

import numpy as np

from matplotlib import pyplot as plt

from scipy.optimize import minimize
from scipy.signal import hann
from scipy import fft

from LoLIM.utilities import v_air, RTD, natural_sort
import LoLIM.utilities as util
from LoLIM.antenna_response import getGalaxyCalibrationData, load_default_antmodel
from LoLIM.signal_processing import locate_data_loss, data_cut_inspan


from LoLIM.pol_beamforming.intensity_plotting import image_plotter
from LoLIM.pol_beamforming import cython_beamforming_tools_centerT as cyt


##TODO:
# currently written so that calculations are not run twice if imager is run twice on same spot. This however costs memory
# rewrite so this is not done. Never run twice in same spot, so memory is wasted


### TODO: increase wing-size with han-width

###### tools for picking antennas #####

class antenna_picker_parent:
    def choose(self, TBB_file, X_antNames, X_XYZs, X_antStartTimes, X_cal,
                               Y_antNames, Y_XYZs, Y_antStartTimes, Y_cal):
        """ return two lists. First list is indeces of X_antNames to use, second list is indeces of Y_antNames to use"""
        print("NOT IMPLEMENTED")
        quit()
               
class outermost_antPicker( antenna_picker_parent ):
    """pick N outermost antennas. Insures equal number X and Y""" 
    
    def __init__(self, N):
        self.N = N
        if N == 0:
            print('no')
            quit()
    
    def choose(self, TBB_file, X_antNames, X_XYZs, X_antStartTimes, X_cal, 
                               Y_antNames, Y_XYZs, Y_antStartTimes, Y_cal):
        
        X_out = []
        if len(X_antNames) > 0:
            sorter = np.argsort( [int(n) for n in X_antNames] )
            i = min( len(X_antNames), self.N )
            X_out = sorter[-i:]
        
        Y_out = []
        if len(Y_antNames) > 0:
            sorter = np.argsort( [int(n) for n in Y_antNames] )
            i = min( len(Y_antNames), self.N )
            Y_out = sorter[-i:]
        
        N_X = len(X_out)
        N_Y = len(Y_out)
        
        if N_X != N_Y:
            N = min(N_X, N_Y)
            X_out = X_out[N]
            Y_out = Y_out[N]
            
        return X_out, Y_out


class beamformer_3D:
    def __init__(self, center_XYZ, voxelDelta_XYZ, numVoxels_XYZ, minTraceLength_samples, imaging_half_hann_length_samples,
                 TBB_file_dict, RFI_data_filters_dict, frequency_width_factor=None, antenna_picker=None, store_antenna_data=False, use_antenna_delays=True):

        ## basic input options setting
        center_XYZ = np.array( center_XYZ, dtype=np.double )
        voxelDelta_XYZ = np.array( voxelDelta_XYZ, dtype=np.double )
        numVoxels_XYZ = np.array( numVoxels_XYZ, dtype=np.int )
        
        # if minTraceLength_samples < 100:
        #     print("WARNING: should probably not beamform less than 100 points at a time")
        
        self.center_XYZ = center_XYZ
        self.voxelDelta_XYZ = voxelDelta_XYZ
        self.numVoxels_XYZ = numVoxels_XYZ
        self.minTraceLength_samples = minTraceLength_samples
        self.half_minTrace = int(round( self.minTraceLength_samples/2 ))
        self.imaging_half_hann_length_samples = imaging_half_hann_length_samples
        self.TBB_file_dict = TBB_file_dict
        self.RFI_data_filters_dict = RFI_data_filters_dict

        if frequency_width_factor != None:
            print('WARNING: frequency_width_factor should probably be left to None!')
        else:
            frequency_width_factor = 0.0
        self.frequency_width_factor = frequency_width_factor
        self.store_antenna_data = store_antenna_data
        self.use_antenna_delays = use_antenna_delays
        
        
        ## set location arrays
        self.X_array = np.arange(numVoxels_XYZ[0], dtype=np.double)
        self.X_array -= int(numVoxels_XYZ[0]/2)
        centerX_voxel = np.where(self.X_array==0)[0][0]
        self.X_array *= voxelDelta_XYZ[0]
        self.X_array += center_XYZ[0]
        
        
        self.Y_array = np.arange(numVoxels_XYZ[1], dtype=np.double)
        self.Y_array -= int(numVoxels_XYZ[1]/2)
        centerY_voxel = np.where(self.Y_array==0)[0][0]
        self.Y_array *= voxelDelta_XYZ[1]
        self.Y_array += center_XYZ[1]
        
        
        self.Z_array = np.arange(numVoxels_XYZ[2], dtype=np.double)
        self.Z_array -= int(numVoxels_XYZ[2]/2)
        centerZ_voxel = np.where(self.Z_array==0)[0][0]
        self.Z_array *= voxelDelta_XYZ[2]
        self.Z_array += center_XYZ[2]
        
        self.center_voxel = np.array([centerX_voxel, centerY_voxel, centerZ_voxel], dtype=np.int)
        
        
        if antenna_picker is None:
            antenna_picker = outermost_antPicker(3)
        elif isinstance(antenna_picker, int):
            antenna_picker = outermost_antPicker( antenna_picker )

        
        
    ### organize antennas and stations ###
        self.station_names = natural_sort( [ sname for sname in TBB_file_dict.keys()] )
        self.station_TBBfiles = [ TBB_file_dict[sname] for sname in self.station_names ]
        self.station_filters = [ RFI_data_filters_dict[ sname ] for sname in  self.station_names]
        self.num_stations = len( self.station_names )
        
        # collect antenna names
        self.stationi_to_antRange = []
        self.stationi_to_anti = [0]
        
        self.anti_to_stati = []
        self.all_antnames = []
        self.all_antXYZs = []
        self.all_antStartTimes  = []
        self.antenna_polarization = [] ## 0 for X-dipole, 1 for Y-dipole
        self.amplitude_calibrations = []
        
        for stat_i, stat_TBB in enumerate(self.station_TBBfiles):
            ant_names = stat_TBB.get_antenna_names()
            ant_times = stat_TBB.get_time_from_second() 
            ant_locs = stat_TBB.get_LOFAR_centered_positions()
            freq_filter_info = self.station_filters[stat_i].RFI_data
            
            early_N = len(self.all_antnames)
            
            
            cal_antenna_names = freq_filter_info["antenna_names"]
            cleaned_power = freq_filter_info["cleaned_power"]
            timestamp = freq_filter_info["timestamp"]
            # analyzed_blocksize = freq_filter_info["blocksize"]
            
            even_cal_factors, odd_cal_factors = getGalaxyCalibrationData(cleaned_power,  timestamp, antenna_type="outer" )
            
            new_X_antNames = []
            new_X_XYZs = []
            new_X_antStartTimes = []
            new_X_cal = []
            
            new_Y_antNames = []
            new_Y_XYZs = []
            new_Y_antStartTimes = []
            new_Y_cal = []
            
            
            for cal_ant_i in range(0, int(len(cal_antenna_names)/2)):
                even_ant_name = cal_antenna_names[ cal_ant_i*2 ]
                odd_ant_name = cal_antenna_names[ cal_ant_i*2 + 1 ]
                
                
                if np.isfinite( even_cal_factors[cal_ant_i] ) and (even_ant_name in ant_names):
                    even_ant_i = ant_names.index( even_ant_name )

                    new_Y_antNames.append( even_ant_name )
                    new_Y_XYZs.append( ant_locs[even_ant_i] )
                    new_Y_antStartTimes.append( ant_times[even_ant_i] )
                    new_Y_cal.append( even_cal_factors[cal_ant_i] )
                    
                
                if np.isfinite( odd_cal_factors[cal_ant_i] ) and (odd_ant_name in ant_names):
                    odd_ant_i = ant_names.index( odd_ant_name )

                    new_X_antNames.append( odd_ant_name )
                    new_X_XYZs.append( ant_locs[odd_ant_i] )
                    new_X_antStartTimes.append( ant_times[odd_ant_i] )
                    new_X_cal.append( odd_cal_factors[cal_ant_i] )
            

            X_indeces, Y_indeces = antenna_picker.choose( TBB_file=stat_TBB, 
                           X_antNames=new_X_antNames, X_XYZs=new_X_XYZs, X_antStartTimes=new_X_antStartTimes, X_cal=new_X_cal, 
                           Y_antNames=new_Y_antNames, Y_XYZs=new_Y_XYZs, Y_antStartTimes=new_Y_antStartTimes, Y_cal=new_Y_cal)
            
            
            self.anti_to_stati += [stat_i]*( len(X_indeces) + len(Y_indeces) )
            self.antenna_polarization += [0]*len(X_indeces)
            self.antenna_polarization += [1]*len(Y_indeces)
            
            for i in X_indeces:
                self.all_antnames.append( new_X_antNames[i] )
                self.all_antXYZs.append( new_X_XYZs[i] )
                self.all_antStartTimes.append( new_X_antStartTimes[i] )
                self.amplitude_calibrations.append( new_X_cal[i] )
            
            for i in Y_indeces:
                self.all_antnames.append( new_Y_antNames[i] )
                self.all_antXYZs.append( new_Y_XYZs[i] )
                self.all_antStartTimes.append( new_Y_antStartTimes[i] )
                self.amplitude_calibrations.append( new_Y_cal[i] )
                    
            self.stationi_to_antRange.append( slice(early_N, len(self.all_antnames) ) )
            self.stationi_to_anti.append( len(self.all_antnames) )
            
        self.all_antXYZs = np.array(self.all_antXYZs , dtype=np.double)

        if self.use_antenna_delays: ## normal mode, uses calibrations
            self.all_antStartTimes = np.array(self.all_antStartTimes , dtype=np.double)
        else: ## all calibrations 0
            self.all_antStartTimes = np.zeros( len(self.all_antStartTimes), dtype=np.double )

        self.antenna_polarization = np.array(self.antenna_polarization , dtype=np.intc)
        self.anti_to_stati = np.array(self.anti_to_stati, dtype=np.int)
## turn off amplitude calibrations
        self.amplitude_calibrations = np.array(self.amplitude_calibrations , dtype=np.double)
        # self.amplitude_calibrations = np.ones(len(self.amplitude_calibrations) , dtype=np.double)
        self.stationi_to_anti = np.array(self.stationi_to_anti, dtype=np.int)
            
        self.num_antennas = len(self.all_antnames)
        print(self.num_antennas, 'antennas')
        
        
        
    #### initial engine setup
        self.geometric_delays = np.empty( self.num_antennas, dtype=np.double )  ## time delay from center voxel to antenna (positive)
        self.min_DTs = np.empty( self.num_antennas, dtype=np.double ) ## minimum difference of geo delay over all voxels from center (negative!)
        self.max_DTs = np.empty( self.num_antennas, dtype=np.double ) ## maximum difference of geo delay over all voxels from center (positive!)
        self.index_shifts = np.empty(self.num_antennas, dtype=np.int) ## index shifts such that if a pulse from center arrives at this index difference on all antennas
        self.cal_shifts = np.empty( self.num_antennas, dtype=np.double )  
        self.reference_XYZ = np.array([0.0 ,0.0, 0.0], dtype=np.double)

        self.engine = cyt.beamform_engine3D(
            X_array=self.X_array, Y_array=self.Y_array, Z_array=self.Z_array, center_XYZ=center_XYZ, reference_XYZ=self.reference_XYZ,
            antenna_locs=self.all_antXYZs, ant_startTimes=self.all_antStartTimes,
            antenna_polarizations=self.antenna_polarization, anti_to_stat_i=self.anti_to_stati,  stati_to_anti=self.stationi_to_anti,
            geometric_delays_memory=self.geometric_delays, min_DTs_memory=self.min_DTs,  max_DTs_memory=self.max_DTs,
            index_shifts_memory=self.index_shifts,
            cal_shifts_memory=self.cal_shifts)
        
        
        earliest_ant_i = np.where( self.index_shifts==0 )[0][0]
        self.center_delay = self.all_antStartTimes[earliest_ant_i] - self.geometric_delays[earliest_ant_i]
        # defined so that arrival_index = (emisstion_T-self.center_delay)/5.0e-9  +   self.index_shifts
        
        
        print('setup frequencies')
    #### calculate trace lengths
        self.earlyHalf_lengths = np.empty( self.num_antennas, dtype=np.int )
        self.lateHalf_lengths = np.empty( self.num_antennas, dtype=np.int )
        for ant_i in range(self.num_antennas):
            self.earlyHalf_lengths[ ant_i ] = int(abs( self.min_DTs[ant_i]/(5.0e-9) )) + 1
            self.lateHalf_lengths[ ant_i ] = int(abs( self.max_DTs[ant_i]/(5.0e-9) )) + 1
    
        self.max_earlyHalf_length = np.max( self.earlyHalf_lengths )
        self.max_lateHalf_length = np.max( self.lateHalf_lengths )
        
        self.total_trace_length = fft.next_fast_len( self.max_earlyHalf_length + self.max_lateHalf_length + minTraceLength_samples + 2*imaging_half_hann_length_samples )
        self.starting_edge_length = self.max_earlyHalf_length + imaging_half_hann_length_samples

        print('total trace length', self.total_trace_length)
        self.trace_loadBuffer_length = self.total_trace_length # this is buffer before arrival sample. this is a little long, probably only need half this!
        self.frequencies = np.fft.fftfreq(self.total_trace_length, d=5.0e-9)
        
        
        
        print('Jones Matrices')
    #### jones matrices
        ## first used JM pointing upwards to get frequency range
        antenna_model = load_default_antmodel()
        
        upwards_JM = antenna_model.Jones_Matrices(self.frequencies, zenith=0.0, azimuth=0.0)
        
        half_F = int( len(self.frequencies)/2 )
        lowest_Fi = np.where( self.frequencies[:half_F]>30e6 )[0][0]
        highest_Fi = np.where( self.frequencies[:half_F]<80e6 )[0][-1]
        self.F30MHZ_i = lowest_Fi
        self.F80MHZ_i = highest_Fi
        
        # posFreq_amps = np.abs( upwards_JM[lowest_Fi:highest_Fi, 0,0] ) 
        posFreq_amps = np.array( [ np.linalg.norm(upwards_JM[fi,:,:],ord=2) for fi in range(lowest_Fi,highest_Fi) ] )

        
        max_freq_index = np.argmax( posFreq_amps ) + lowest_Fi
        self.max_freq_index = max_freq_index
        ref_amp = np.max(posFreq_amps)*frequency_width_factor

        if posFreq_amps[0] <= ref_amp:
            self.start_freq_index = np.where( np.logical_and( posFreq_amps[:-1]<=ref_amp,    posFreq_amps[1:]>ref_amp)  )[0][0]
        else:
            self.start_freq_index = 0

        if posFreq_amps[-1] <= ref_amp:
            self.end_freq_index = np.where( np.logical_and( posFreq_amps[:-1]>=ref_amp,    posFreq_amps[1:]<ref_amp)  )[0][0]
        else:
            self.end_freq_index = highest_Fi - lowest_Fi

        self.antenna_norms_in_range = np.array( posFreq_amps[self.start_freq_index:self.end_freq_index ] )
        self.start_freq_index += lowest_Fi
        self.end_freq_index += lowest_Fi

        self.beamformed_freqs = self.frequencies[ self.start_freq_index:self.end_freq_index ]
        self.num_freqs = self.end_freq_index-self.start_freq_index

        print('frequency range:', self.frequencies[self.start_freq_index], self.frequencies[self.end_freq_index])
        print('  response amps (start, peak, end)', posFreq_amps[self.start_freq_index-lowest_Fi], np.max(posFreq_amps), posFreq_amps[self.end_freq_index-lowest_Fi-1 ])
        print('  number frequency points:', self.num_freqs )
    
        ## ALL jones matrices!
        self.cut_jones_matrices = np.empty( (self.num_stations, self.num_freqs,2,2), dtype=np.cdouble )
        self.JM_condition_numbers = np.empty(self.num_stations, dtype=np.double)  ## both at peak frequency
        self.JM_magnitudes = np.empty(self.num_stations, dtype=np.double)
        self.station_R = np.empty(self.num_stations, dtype=np.double) ## distance to center pixel
        for stat_i in range(self.num_stations):
            ant_XYZs = self.all_antXYZs[ self.stationi_to_antRange[ stat_i ] ]
            stat_XYZ = np.average( ant_XYZs, axis=0 )
            
            ## from station to source!
            delta_XYZ = center_XYZ - stat_XYZ 
            
            center_R = np.linalg.norm( delta_XYZ )
            center_zenith = np.arccos(delta_XYZ[2]/center_R)*RTD
            center_azimuth = np.arctan2( delta_XYZ[1], delta_XYZ[0] )*RTD
            
            self.cut_jones_matrices[stat_i, :,:,:] = antenna_model.Jones_Matrices(self.beamformed_freqs, zenith=center_zenith, azimuth=center_azimuth)
            # self.cut_jones_matrices[stat_i, :,:,:] = antenna_model.Jones_ONLY(self.beamformed_freqs, zenith=center_zenith, azimuth=center_azimuth)

            self.JM_condition_numbers[stat_i] = np.linalg.cond( self.cut_jones_matrices[stat_i, max_freq_index-self.start_freq_index,  :,:]  )
        
            self.JM_magnitudes[stat_i] = np.linalg.norm( self.cut_jones_matrices[stat_i, max_freq_index-self.start_freq_index,  :,:], ord=2 )
    
            self.station_R[stat_i] = center_R
    
    #### windowing matrices!
        self.engine.set_antenna_functions( self.total_trace_length, self.start_freq_index, self.end_freq_index, 
                                      self.frequencies, self.cut_jones_matrices)
        
        self.engine.turn_on_all_antennas()
        self.set_weights_by_station()
    
    ### some memory
        self.blocksize = self.station_filters[0].blocksize
        self.hann_sample_length = int( self.station_filters[0].half_window_percent * self.blocksize )
    
    
        ## loading
        self.loading_temp = np.empty(self.blocksize, dtype=np.double)
        self.loaded_data = np.empty( (self.num_antennas, self.blocksize-2*self.hann_sample_length), dtype=np.cdouble )
        self.loaded_samples = np.empty( self.num_antennas, dtype=np.int )
        
        self.data_loss_spans = [ [] ]*self.num_antennas
        self.loaded_indexRange = [np.inf, -np.inf]
        
        ## windowing
        self.temp_window = np.empty( self.total_trace_length, dtype=np.cdouble )
        self.antenna_windowed = np.empty( self.num_antennas, dtype=np.int ) ## false if data loss, true otherwise
        self.imaging_hann = hann(2*imaging_half_hann_length_samples)
        
        ## this will include the data that was loaded into the imager
        if self.store_antenna_data :
            self.antenna_data = np.zeros( (self.num_antennas,self.total_trace_length), dtype=np.cdouble )
        
        self.correction_matrix = None


        self.temp_inversion_matrix = np.empty((3, 3), dtype=np.cdouble)
        self.inverted_matrix = np.empty((3, 3), dtype=np.cdouble)
        self.invertrix = cyt.SVD_psuedoinversion(3, 3)



        self.ifft_full_tmp = self.get_empty_partial_inverse_FFT()

        self.TMP_oPol_matrix = None

### weights and condition numbers
    def set_weights_by_station(self, station_weights=None):
        
        if station_weights is None:
            station_weights = np.ones( self.num_stations, dtype=np.double )
        
        station_weights /= np.sum(station_weights)
        station_weights *= self.num_stations
        
        self.used_station_weights = station_weights
        
        for stat_i in range( self.num_stations ):
            for ant_i in range(self.stationi_to_anti[stat_i], self.stationi_to_anti[stat_i+1]):
                self.engine.set_antennaWeight(ant_i,  station_weights[stat_i] )
    
    def calc_CN(self, station_weights=None):
        # print('  r')
        if station_weights is not None:
            self.set_weights_by_station( station_weights )
        
        self.TMP_oPol_matrix = self.engine.get_correctionMatrix( self.TMP_oPol_matrix )
        
        ACN = np.linalg.cond( self.TMP_oPol_matrix )
        A_mag = np.linalg.norm(self.TMP_oPol_matrix , ord=2)
        
        
        part_B = 0
        part_C = 0
        for stat_i in range( self.num_stations ):
            B = self.JM_condition_numbers[stat_i]*self.used_station_weights[stat_i]/( self.JM_magnitudes[stat_i]*self.station_R[stat_i] )
            part_B += B*B
            
            C = self.JM_magnitudes[stat_i]/self.station_R[stat_i]
            part_C += C*C
            
        return ACN*np.sqrt(part_B/self.num_stations)*np.sqrt(part_C)/A_mag
            

        
    def calc_set_weights(self):
        print('S')
        F = self.calc_CN
        self.TMP_oPol_matrix = None ## temporary memory needed for the function
        
        station_weights_guess = np.ones( self.num_stations, dtype=np.double )
        bound = [[0,np.inf] for i in range(self.num_stations)]
        
        ret = minimize( F, station_weights_guess, method='powell', bounds=bound,
                options={'maxiter': 1000,  'xtol':1e-30, 'ftol':1e-30})
        
        self.set_weights_by_station( ret.x )
        self.correction_matrix = self.engine.get_correctionMatrix()

        print('E')
        return ret
    

            
    
### for loading and manipulating data ###
    def load_raw_data(self, sky_T):
        
        print('loading')
        
        self.loaded_skyT = sky_T
        first_index = int(round( (sky_T - self.center_delay)/5.0e-9 )) - (self.hann_sample_length + self.trace_loadBuffer_length)
        self.loaded_indexRange = [ first_index+self.hann_sample_length, first_index+self.blocksize-self.hann_sample_length ]
        
        for ant_i in range(self.num_antennas):
            ant_name = self.all_antnames[ ant_i ]
            stat_i = self.anti_to_stati[ ant_i ]
            TBB_file = self.station_TBBfiles[ stat_i ]
            freq_filter = self.station_filters[ stat_i ]
            
            start_sample = self.index_shifts[ant_i] + first_index
            
            self.loading_temp[:] = TBB_file.get_data(start_sample, self.blocksize, antenna_ID=ant_name  )
            dataLoss, number = locate_data_loss( self.loading_temp[ self.hann_sample_length:-self.hann_sample_length ], 5 )
            self.data_loss_spans[ ant_i ] = dataLoss
            
            self.loaded_data[ant_i, :] = freq_filter.filter( self.loading_temp )[ self.hann_sample_length:-self.hann_sample_length ]
            self.loaded_data[ant_i, :] *= self.amplitude_calibrations[ant_i]
            self.loaded_samples[ant_i] = start_sample+self.hann_sample_length
    
    def window_data(self, sky_T, average_station=None):
        if average_station is not None:
            ave_stat_i = self.station_names.index( average_station )
        amp_ave = 0
        num_amp_ave = 0
        
        
        sample_center = int(round( (sky_T - self.center_delay)/5.0e-9 ))
        
        earliest_sample = sample_center - self.max_earlyHalf_length - self.half_minTrace - self.imaging_half_hann_length_samples
        latest_sample = sample_center + self.max_lateHalf_length  + self.half_minTrace + self.imaging_half_hann_length_samples
        if earliest_sample<self.loaded_indexRange[0] or latest_sample>self.loaded_indexRange[1]:
            self.load_raw_data( sky_T )

        # print('windowing')
        n_han_samp = self.imaging_half_hann_length_samples
        for ant_i in range(self.num_antennas):
            ant_center_sample = sample_center + self.index_shifts[ant_i] - self.loaded_samples[ant_i]
            ant_first_sample = ant_center_sample - self.earlyHalf_lengths[ant_i] - self.half_minTrace - n_han_samp
            ant_final_sample = ant_center_sample + self.lateHalf_lengths[ant_i] + self.half_minTrace + n_han_samp
            width = ant_final_sample - ant_first_sample
            
            has_data_loss = data_cut_inspan( self.data_loss_spans[ ant_i ], ant_first_sample, ant_final_sample )
            if has_data_loss:
                # self.windowed_data[ant_i] = 0.0
                self.engine.set_antennaData_zero( ant_i )
                self.antenna_windowed[ ant_i ] = 0
                continue
            self.antenna_windowed[ant_i] = 1
            
            
            delay_samples = self.max_earlyHalf_length - self.earlyHalf_lengths[ant_i]
            self.temp_window[:] = 0.0
            self.temp_window[delay_samples:delay_samples+width] = self.loaded_data[ant_i, ant_first_sample:ant_final_sample]
            
            self.temp_window[delay_samples: delay_samples+n_han_samp] *= self.imaging_hann[:n_han_samp]
            self.temp_window[delay_samples+width-n_han_samp: delay_samples+width] *= self.imaging_hann[n_han_samp:]
            
            
            if (average_station is not None) and self.anti_to_stati[ant_i] == ave_stat_i:
                amp_ave += np.max( np.abs( self.temp_window ) )/self.amplitude_calibrations[ant_i] ## DE-calibrate
                num_amp_ave += 1
            
            if self.store_antenna_data :
                self.antenna_data[ant_i, :] = self.temp_window
            self.engine.set_antennaData(ant_i, self.temp_window )#.view(np.double) )
            
        if (average_station is not None):
            self.ave_stat_i = ave_stat_i
            self.station_ave_amp = amp_ave/num_amp_ave
        else:
            self.ave_stat_i = None
            
    def plot_data(self, sky_T, source_XYZT=None):
        print('PLOT DATA IS PROBABLY WRONG! check tukey window??')
        self.plotted_sky_T = sky_T
        
        sample_center = int(round( (sky_T - self.center_delay)/5.0e-9 ))
        
        earliest_sample = sample_center - self.max_earlyHalf_length - self.half_minTrace
        latest_sample = sample_center + self.max_lateHalf_length  + self.half_minTrace
        # if earliest_sample<self.loaded_indexRange[0] or latest_sample>self.loaded_indexRange[1]:
            # self.load_raw_data( sky_T )

        # n = self.imaging_half_hann_length_samples
        for stat_i in range( self.num_stations ):
            signal_dt = []
            
            # cal_sum = 0
            # n_ants = 0
            max_amp = 0
            for ant_i in range(self.stationi_to_anti[stat_i], self.stationi_to_anti[stat_i+1]):
                ant_center_sample = sample_center + self.index_shifts[ant_i] - self.loaded_samples[ant_i]
                ant_first_sample = ant_center_sample - self.earlyHalf_lengths[ant_i] - self.half_minTrace
                    
                
                data = self.antenna_data[ant_i, :]
                
                abs_window = np.abs( data )
                max_ant_amp = np.max( abs_window )
                
                if max_ant_amp > max_amp:
                    max_amp = max_ant_amp
                    
                # cal_sum += self.amplitude_calibrations[ant_i]
                # n_ants += 1
                
                
                if source_XYZT is not None:
                    ant_XYZ = self.all_antXYZs[ant_i]
                    reception_time = np.linalg.norm(ant_XYZ - source_XYZT[:3])/v_air
                    first_sample_time = self.all_antStartTimes[ant_i] + (self.loaded_samples[ant_i] + ant_first_sample)*(5.0e-9)
                    
                    if not np.isfinite(source_XYZT[3]):
                        signal_t = np.argmax( abs_window )*(5.0e-9)
                        source_XYZT[3] = first_sample_time+signal_t - reception_time
                    reception_time += source_XYZT[3]
                    
                    signal_dt.append( reception_time-first_sample_time )  
                else:
                    signal_dt.append( None )
                    
                    
            # print(stat_i, max_amp*n_ants/cal_sum)
            for ant_i, sdt in zip(range(self.stationi_to_anti[stat_i], self.stationi_to_anti[stat_i+1]),   signal_dt):
                p = self.antenna_polarization[ant_i]
                
                data = np.array( self.antenna_data[ant_i, :] )
                
                data *= 1.0/max_amp
                offset = stat_i*3  + p*0.75
                plt.plot( np.abs(data) + offset )
                plt.plot( np.real(data) + offset )
                
                if source_XYZT is not None:
                    sdt_samples = sdt/(5.0e-9)
                    plt.plot( [sdt_samples,sdt_samples], [offset,offset+1] )
                    
            plt.annotate( self.station_names[stat_i], (0, stat_i*3) )
            
    def plt_sourceLines(self, source_XYZT, color):
         
        sample_center = int(round( (self.plotted_sky_T - self.center_delay)/5.0e-9 ))
        
        # earliest_sample = sample_center - self.max_earlyHalf_length - self.half_minTrace
        # latest_sample = sample_center + self.max_lateHalf_length  + self.half_minTrace

        # n = self.imaging_half_hann_length_samples
        for stat_i in range( self.num_stations ):
            for ant_i in range(self.stationi_to_anti[stat_i], self.stationi_to_anti[stat_i+1]):
                
                ant_center_sample = sample_center + self.index_shifts[ant_i] - self.loaded_samples[ant_i]
                ant_first_sample = ant_center_sample - self.earlyHalf_lengths[ant_i] - self.half_minTrace
                # ant_final_sample = ant_center_sample + self.lateHalf_lengths[ant_i] + self.half_minTrace
                # width = ant_final_sample - ant_first_sample
            
                # has_data_loss = data_cut_inspan( self.data_loss_spans[ ant_i ], ant_first_sample, ant_final_sample )
                # if has_data_loss:
                #     continue
                
                abs_window = np.abs( self.antenna_data[ant_i, :] )

                
                ant_XYZ = self.all_antXYZs[ant_i]
                reception_time = np.linalg.norm(ant_XYZ - source_XYZT[:3])/v_air
                first_sample_time = self.all_antStartTimes[ant_i] + (self.loaded_samples[ant_i] + ant_first_sample)*(5.0e-9)
                
                if not np.isfinite(source_XYZT[3]):
                    signal_t = np.argmax( abs_window )*(5.0e-9)
                    source_XYZT[3] = first_sample_time+signal_t - reception_time
                reception_time += source_XYZT[3]
                
                sdt = reception_time-first_sample_time 
                
                sdt_samples = sdt/(5.0e-9)
                offset = stat_i*3  + self.antenna_polarization[ant_i] *0.75
                plt.plot( [sdt_samples,sdt_samples], [offset,offset+1], c=color )
                
        
    def load_PSF(self, X_val, Y_val, Z_val, average_station=None):
        """ make a point source at center voxel with XYZ polarization. average_station calculates the average peak amplitude for that station"""

        
        stat_X_dipole = np.empty( len(self.beamformed_freqs), dtype=np.cdouble )
        stat_Y_dipole = np.empty( len(self.beamformed_freqs), dtype=np.cdouble )
        
        TMP = np.zeros( len(self.frequencies), dtype=np.cdouble )
        shifter_TMP = np.zeros( len(self.beamformed_freqs), dtype=np.cdouble )
        PSF_systematic_shift = (self.max_earlyHalf_length + self.half_minTrace + self.imaging_half_hann_length_samples)*5.0e-9 ## puts the PSF in the center of the image
        
        
        if (average_station is not None) and isinstance(average_station, str):
            ave_stat_i = self.station_names.index( average_station )
        amp_ave = 0
        num_amp_ave = 0
        
        for stat_i in range( self.num_stations ):
            ant_range = self.stationi_to_antRange[stat_i]
            # print('station', stat_i, 'Fi:', self.max_freq_index-self.start_freq_index)
            
            
        ### get jones matrices
            
            J_00 = self.cut_jones_matrices[stat_i, :, 0,0]
            J_01 = self.cut_jones_matrices[stat_i, :, 0,1]
            J_10 = self.cut_jones_matrices[stat_i, :, 1,0]
            J_11 = self.cut_jones_matrices[stat_i, :, 1,1]
        
        ### get angles
            ## from station to source!!
            stat_X = self.center_XYZ[0] - np.average( self.all_antXYZs[ant_range, 0] )
            stat_Y = self.center_XYZ[1] - np.average( self.all_antXYZs[ant_range, 1] )
            stat_Z = self.center_XYZ[2] - np.average( self.all_antXYZs[ant_range, 2] )
        
            stat_R = np.sqrt( stat_X*stat_X + stat_Y*stat_Y + stat_Z*stat_Z )
            
            stat_zenith = np.arccos( stat_Z/stat_R )
            stat_azimuth = np.arctan2( stat_Y, stat_X)
            
            sin_stat_azimuth = np.sin( stat_azimuth )
            cos_stat_azimuth = np.cos( stat_azimuth )
            sin_stat_zenith = np.sin( stat_zenith )
            cos_stat_zenith = np.cos( stat_zenith )
            
            stat_X_dipole[:] = 0.0
            stat_Y_dipole[:] = 0.0
            
    ## X dipole
        ## X-orriented field
            ## zenithal 
            T = cos_stat_azimuth*cos_stat_zenith*J_00
            ## azimuthal
            T += -sin_stat_azimuth*J_01
            stat_X_dipole += T*X_val
            
        ## Y-orriented field
            ## zenithal 
            T = cos_stat_zenith*sin_stat_azimuth*J_00
            ## azimuthal
            T += cos_stat_azimuth*J_01
            stat_X_dipole += T*Y_val
            
        ## Z-orriented field
            ## zenithal 
            T = -sin_stat_zenith*J_00
            ## no azimuthal!!
            stat_X_dipole += T*Z_val
            
    ## Y dipole
        ## X-orriented field
            ## zenithal 
            T = cos_stat_azimuth*cos_stat_zenith*J_10
            ## azimuthal
            T += -sin_stat_azimuth*J_11
            stat_Y_dipole += T*X_val
            
        ## Y-orriented field
            ## zenithal 
            T = cos_stat_zenith*sin_stat_azimuth*J_10
            ## azimuthal
            T += cos_stat_azimuth*J_11
            stat_Y_dipole += T*Y_val
            
        ## Z-orriented field
            ## zenithal 
            T = -sin_stat_zenith*J_10
            ## no azimuthal!!
            stat_Y_dipole += T*Z_val

            # datums = []
            # pol = []
            # signal_dt = []
            # max_amp = 0.0
            
            if (average_station is not None) and ave_stat_i==stat_i:
                do_amp_average = True
            else:
                do_amp_average = False
                
            
            for ant_i in range(self.stationi_to_anti[stat_i], self.stationi_to_anti[stat_i+1] ):
                
                sub_sample_shift = self.geometric_delays[ant_i] + self.cal_shifts[ant_i]
                
                shifter_TMP[:] = self.beamformed_freqs
                shifter_TMP *= -2j*np.pi*(sub_sample_shift + PSF_systematic_shift)
                np.exp( shifter_TMP, out=shifter_TMP )
                
                R = np.linalg.norm( self.all_antXYZs[ant_i] - self.center_XYZ )
                if self.antenna_polarization[ant_i] == 0: ## X-dipole
                    shifter_TMP *= stat_X_dipole
                else:                                     ## Y-dipole
                    shifter_TMP *= stat_Y_dipole
                    
                                
                shifter_TMP *= 1.0/R
                TMP[self.start_freq_index:self.end_freq_index] = shifter_TMP
                ifft = np.fft.ifft( TMP )

                if self.store_antenna_data:
                    self.antenna_data[ant_i, :] = ifft
                
                if do_amp_average:
                    amp_ave += np.max(np.abs(ifft))
                    num_amp_ave += 1
                
                self.engine.set_antennaData(ant_i, ifft) ## note this modifies in place!!

        if self.store_antenna_data:
            self.antenna_data_magnitude = np.sqrt(  cyt.total_2dnorm_squared( self.antenna_data )  )
        
        if average_station is not None:
            self.ave_stat_i = ave_stat_i
            self.station_ave_amp = amp_ave/num_amp_ave
        else:
            self.ave_stat_i = None
            
    def inject_noise(self, ratio, ratio_mode=0, station_to_plot=None, plot_to_file=None):
        if ratio_mode == 0:
            noise_sigma = 2*self.antenna_data_magnitude*ratio/np.sqrt( self.antenna_data.shape[0]*self.antenna_data.shape[1] )
        elif ratio_mode == 1:
            noise_sigma = self.station_ave_amp*ratio

        if (station_to_plot is not None) and isinstance(station_to_plot, str):
            station_to_plot = self.station_names.index( station_to_plot )

        total_norm_sq = 0.0
        ave_amp = 0
        num_ave_amp = 0
        
        plot_Y = 0
        
        FTMP = np.zeros( len(self.frequencies) , dtype=np.cdouble)
        
        for ant_i in range(self.num_antennas):
            
            rand = np.random.normal(size=2*( self.F80MHZ_i - self.F30MHZ_i ), scale=noise_sigma).view(np.cdouble)
            FTMP[:] = 0.0
            FTMP[self.F30MHZ_i:self.F80MHZ_i] = rand
            rand = np.fft.ifft( FTMP )
            
            total_norm_sq += cyt.total_1dnorm_squared( rand )
            
            
            
            if (self.ave_stat_i is not None) and self.anti_to_stati[ant_i]==self.ave_stat_i:
                ave_amp += np.average( np.abs( rand ) )
                num_ave_amp += 1
            
            rand += self.antenna_data[ant_i]
            
            
            if (station_to_plot is not None) and self.anti_to_stati[ant_i]==station_to_plot:
                A = np.array( rand )
                ABS = np.abs(A)
                ABS_max = np.max(ABS)
                plt.plot(ABS+plot_Y)
                plt.plot(A.real + plot_Y)
                plot_Y += ABS_max
            
            self.engine.set_antennaData(ant_i, rand)


        if (station_to_plot is not None) :
            if self.ave_stat_i is not None:
                plt.axhline(ave_amp/num_ave_amp, c='r')
                plt.axhline(self.station_ave_amp, c='b')
            if plot_to_file:
                plt.savefig(plot_to_file)
            else:
                plt.show()
            plt.clf()

        total_norm = np.sqrt( total_norm_sq )
        
        if (self.ave_stat_i is None):
            return total_norm/self.antenna_data_magnitude
        else:
            return total_norm/self.antenna_data_magnitude,   self.station_ave_amp/(ave_amp/num_ave_amp)
    
    
    def get_empty_image(self):
       return np.empty( (self.numVoxels_XYZ[0], self.numVoxels_XYZ[1], self.numVoxels_XYZ[2], self.num_freqs, 3) , dtype=np.cdouble)
    
    def get_image(self, out_image=None, print_progress=False, weighting=None):

        if weighting is not None:
            if weighting is True:
                weighting = self.antenna_norms_in_range

        A = self.engine.full_image( out_image, print_progress, frequency_weights=weighting )


        return A

    def get_empty_chuncked_image(self, num_chunks):
        if num_chunks == 0: ## see behavior below
            num_chunks = 1
        return np.empty( (self.numVoxels_XYZ[0], self.numVoxels_XYZ[1], self.numVoxels_XYZ[2], num_chunks) , dtype=np.cdouble)

    def get_chunked_intesity_image(self, num_chunks, out_image=None, do_ifft=True, print_progress=False, weighting=None, do_matrix=None):
        """ set num_chunks to 0 to sum over whole length. """

        if weighting is not None:
            if weighting is True:
                weighting = self.antenna_norms_in_range

        if not do_ifft:
            if num_chunks > 1:
                print('WARNING: inconsistent inputs! do_ifft=False and num_chunks>1!')

            starting_i = self.start_freq_index
            chunck_size = self.end_freq_index-self.start_freq_index
            num_chunks = 1
            _do_ifft = 0

        elif num_chunks == 0:
            starting_i = 0
            chunck_size = self.total_trace_length
            num_chunks = 1
            _do_ifft = 1
        else:
            starting_i = self.starting_edge_length
            chunck_size = int(self.minTraceLength_samples/num_chunks)
            _do_ifft = 1

        if do_matrix is not None:
            if do_matrix is True:


                self.temp_inversion_matrix[:,:] = self.get_correctionMatrix(out=self.temp_inversion_matrix)

                # self.temp_inversion_matrix[:, :] = self.correction_matrix

                ret_info = self.invertrix.set_matrix(self.temp_inversion_matrix)
                if ret_info>0:
                    print('unable to invert!', ret_info)

                self.invertrix.get_psuedoinverse(self.inverted_matrix, override_cond=0)
                do_matrix = self.inverted_matrix
            if do_matrix is False:
                do_matrix = None

        RET = self.engine.ChunkedIntensity_Image( starting_i, chunck_size, num_chunks, do_ifft=_do_ifft,
                image = out_image, print_progress = print_progress, frequency_weights = weighting, matrix=do_matrix)

        return RET

    def get_timeDiff_at_loc(self, XYZloc):
        """given a source at XYZ, return time diff, that should be subtracted from sky_T, to get the time at that location"""

        return ( np.linalg.norm( self.reference_XYZ - XYZloc ) - np.linalg.norm(  self.reference_XYZ - self.center_XYZ) )/v_air

    def get_correctionMatrix(self, out=None, loc_to_use=None ):
        return self.engine.get_correctionMatrix(out, loc_to_use)

    def get_empty_partial_inverse_FFT(self):
        return np.empty(self.total_trace_length, dtype=np.cdouble)
    
    def partial_inverse_FFT(self, in_data, out_data=None):
        return self.engine.partial_inverse_FFT( in_data, out_data)

    def get_empty_full_inverse_FFT(self, mode='wingless'):
        """ mode can be 'full' or 'wingless'. wingless has time traces minTraceLength_samples long.
        full is total_trace_length long.
         hannless simply cuts-off the tukey windows."""

        if mode == 'full':
            T = self.total_trace_length
        elif mode == 'wingless':
            T = self.minTraceLength_samples
        elif mode == 'hannless':
            T = self.total_trace_length-2*self.imaging_half_hann_length_samples


        return np.empty( ( self.numVoxels_XYZ[0], self.numVoxels_XYZ[1], self.numVoxels_XYZ[2], T, 3),
                             dtype = np.cdouble)

    def full_inverse_FFT(self, in_image, out_data=None, mode='wingless'):
        """ mode can be 'full', or 'wingless'. wingless has time traces minTraceLength_samples long, else full is total_trace_length long"""

        if out_data is None:
            out_data = self.get_empty_full_inverse_FFT(mode)
        #out_data[:] = 0.0

        # TMP = np.empty(self.total_trace_length, dtype=np.cdouble)

        if mode == 'wingless':
            # dN = self.total_trace_length - self.minTraceLength_samples
            # hdN = int(dN/2)
            hdN = self.starting_edge_length
            L = self.minTraceLength_samples
            

        for xi in range(self.numVoxels_XYZ[0]):
            for yi in range(self.numVoxels_XYZ[1]):
                for zi in range(self.numVoxels_XYZ[2]):
                    for pi in range(3):
                        self.engine.partial_inverse_FFT(in_image[xi,yi,zi,:,pi], self.ifft_full_tmp)
                        if mode == 'full':
                            out_data[xi,yi,zi,:,pi] = self.ifft_full_tmp
                        else:
                            out_data[xi, yi, zi, :, pi] = self.ifft_full_tmp[ hdN : hdN+L ]

        return out_data


    def get_empty_SpotImage(self):
        return np.empty((self.num_freqs, 3), dtype=np.cdouble)

    def get_SpotImage(self, loc, out_image=None, weighting=None, do_matrix=None):

        if weighting is not None:
            if weighting is True:
                weighting = self.antenna_norms_in_range

        if do_matrix is not None:
            if do_matrix is True:
                self.temp_inversion_matrix = self.get_correctionMatrix(out=self.temp_inversion_matrix,
                                                                              loc_to_use=loc)
                self.invertrix.set_matrix(self.temp_inversion_matrix)
                self.invertrix.get_psuedoinverse(self.inverted_matrix, override_cond=0)
                do_matrix = self.inverted_matrix

        A = self.engine.Image_at_Spot(loc[0], loc[1], loc[2], out_image,
                                             frequency_weights=weighting, freq_mode=1, matrix=do_matrix)

        return A

    def get_empty_polarized_inverse_FFT(self, mode='full'):
        if mode == 'full':
            return np.empty((self.total_trace_length, 3), dtype=np.cdouble)
        elif mode == 'wingless':
            return np.empty((self.minTraceLength_samples, 3), dtype=np.cdouble)

    def polarized_inverse_FFT(self, in_data, out_data=None, mode='full'):
        if out_data is None:
            out_data = self.get_empty_polarized_inverse_FFT(mode)

        for pi in range(3):
            TMP = self.engine.partial_inverse_FFT(in_data[:, pi], self.ifft_full_tmp, freq_mode=1)

            if mode == 'full':
                out_data[:, pi] = TMP

            elif mode == 'wingless':
                out_data[:, pi] = TMP[
                                  self.starting_edge_length: self.starting_edge_length + self.minTraceLength_samples]

        return out_data


    def get_secondaryLength_beamformer(self, trace_length):
        if trace_length > self.minTraceLength_samples:
            print('ERROR: secondary length must be smaller than initial length')
            quit()

        return self.secondary_length_beamformer( trace_length, self )

    class secondary_length_beamformer:
        def __init__(self, trace_length, parent):
            self.minTraceLength_samples = trace_length
            self.parent = parent
            self.half_minTrace = int(round(self.minTraceLength_samples / 2))

            self.total_trace_length = fft.next_fast_len( parent.max_earlyHalf_length + parent.max_lateHalf_length + trace_length +  + 2*parent.imaging_half_hann_length_samples )
            self.starting_edge_length = self.parent.max_earlyHalf_length + parent.imaging_half_hann_length_samples

            # self.trace_loadBuffer_length = self.total_trace_length # this is buffer before arrival sample. this is a little long, probably only need half this!
            self.frequencies = np.fft.fftfreq(self.total_trace_length, d=5.0e-9)

            antenna_model = load_default_antmodel()
            upwards_JM = antenna_model.Jones_Matrices(self.frequencies, zenith=0.0, azimuth=0.0)

            half_F = int(len(self.frequencies) / 2)
            lowest_Fi = np.where(self.frequencies[:half_F] > 30e6)[0][0]
            highest_Fi = np.where(self.frequencies[:half_F] < 80e6)[0][-1]
            self.F30MHZ_i = lowest_Fi
            self.F80MHZ_i = highest_Fi

            posFreq_amps = np.array(
                [np.linalg.norm(upwards_JM[fi, :, :], ord=2) for fi in range(lowest_Fi, highest_Fi)])

            self.max_freq_index = np.argmax(posFreq_amps) + lowest_Fi
            ref_amp = np.max(posFreq_amps) * parent.frequency_width_factor

            if posFreq_amps[0] <= ref_amp:
                self.start_freq_index = \
                np.where(np.logical_and(posFreq_amps[:-1] <= ref_amp, posFreq_amps[1:] > ref_amp))[0][0]
            else:
                self.start_freq_index = 0

            if posFreq_amps[-1] <= ref_amp:
                self.end_freq_index = \
                np.where(np.logical_and(posFreq_amps[:-1] >= ref_amp, posFreq_amps[1:] < ref_amp))[0][0]
            else:
                self.end_freq_index = len(posFreq_amps)

            self.antenna_norms_in_range = np.array(posFreq_amps[self.start_freq_index:self.end_freq_index])
            self.start_freq_index += lowest_Fi
            self.end_freq_index += lowest_Fi

            self.beamformed_freqs = self.frequencies[self.start_freq_index:self.end_freq_index]
            self.num_freqs = self.end_freq_index - self.start_freq_index

            ## ALL jones matrices!
            self.cut_jones_matrices = np.empty((self.parent.num_stations, self.num_freqs, 2, 2), dtype=np.cdouble)
            self.JM_condition_numbers = np.empty(self.parent.num_stations, dtype=np.double)  ## both at peak frequency
            self.JM_magnitudes = np.empty(self.parent.num_stations, dtype=np.double)
            # self.station_R = np.empty(self.num_stations, dtype=np.double)  ## distance to center pixel
            for stat_i in range(self.parent.num_stations):
                ant_XYZs = parent.all_antXYZs[parent.stationi_to_antRange[stat_i]]
                stat_XYZ = np.average(ant_XYZs, axis=0)

                ## from station to source!
                delta_XYZ = parent.center_XYZ - stat_XYZ

                center_R = np.linalg.norm(delta_XYZ)
                center_zenith = np.arccos(delta_XYZ[2] / center_R) * RTD
                center_azimuth = np.arctan2(delta_XYZ[1], delta_XYZ[0]) * RTD

                # self.cut_jones_matrices[stat_i, :,:,:] = antenna_model.Jones_Matrices(self.beamformed_freqs, zenith=center_zenith, azimuth=center_azimuth)
                self.cut_jones_matrices[stat_i, :, :, :] = antenna_model.Jones_Matrices(self.beamformed_freqs,
                                                                                    zenith=center_zenith,
                                                                                    azimuth=center_azimuth)

                self.JM_condition_numbers[stat_i] = np.linalg.cond(
                    self.cut_jones_matrices[stat_i, self.max_freq_index - self.start_freq_index, :, :])

                self.JM_magnitudes[stat_i] = np.linalg.norm(
                    self.cut_jones_matrices[stat_i, self.max_freq_index - self.start_freq_index, :, :], ord=2)

                # self.station_R[stat_i] = center_R

            #### windowing matrices!
            self.parent.engine.set_antenna_functions(self.total_trace_length, self.start_freq_index, self.end_freq_index,
                                              self.frequencies, self.cut_jones_matrices, freq_mode=2)
            ### memory
            self.temp_window = np.empty(self.total_trace_length, dtype=np.cdouble)

            self.temp_inversion_matrix = np.empty((3, 3), dtype=np.cdouble)
            self.inverted_matrix = np.empty((3, 3), dtype=np.cdouble)
            self.invertrix = parent.invertrix#cyt.SVD_psuedoinversion(3, 3)

            self.ifft_full_tmp = self.get_empty_partial_inverse_FFT()


        def window_data(self, sky_T, average_station=None):
            # if average_station is not None:
            #     ave_stat_i = self.parent.station_names.index(average_station)
            amp_ave = 0
            num_amp_ave = 0

            sample_center = int(round((sky_T - self.parent.center_delay) / 5.0e-9))

            earliest_sample = sample_center - self.parent.max_earlyHalf_length - self.half_minTrace
            latest_sample = sample_center + self.parent.max_lateHalf_length + self.half_minTrace
            if earliest_sample < self.parent.loaded_indexRange[0] or latest_sample > self.parent.loaded_indexRange[1]:
                self.parent.load_raw_data(sky_T)

            # print('windowing')
            n = self.parent.imaging_half_hann_length_samples
            for ant_i in range(self.parent.num_antennas):
                ant_center_sample = sample_center + self.parent.index_shifts[ant_i] - self.parent.loaded_samples[ant_i]
                ant_first_sample = ant_center_sample - self.parent.earlyHalf_lengths[ant_i] - self.half_minTrace
                ant_final_sample = ant_center_sample + self.parent.lateHalf_lengths[ant_i] + self.half_minTrace
                width = ant_final_sample - ant_first_sample

                has_data_loss = data_cut_inspan(self.parent.data_loss_spans[ant_i], ant_first_sample, ant_final_sample)
                if has_data_loss:
                    # self.windowed_data[ant_i] = 0.0
                    self.parent.engine.set_antennaData_zero(ant_i)
                    # self.antenna_windowed[ant_i] = 0
                    continue
                # self.antenna_windowed[ant_i] = 1

                delay_samples = self.parent.max_earlyHalf_length - self.parent.earlyHalf_lengths[ant_i]
                self.temp_window[:] = 0.0
                self.temp_window[delay_samples:delay_samples + width] = self.parent.loaded_data[ant_i,
                                                                        ant_first_sample:ant_final_sample]

                self.temp_window[delay_samples: delay_samples + n] *= self.parent.imaging_hann[:n]
                self.temp_window[delay_samples + width - n: delay_samples + width] *= self.parent.imaging_hann[n:]

                # if (average_station is not None) and self.anti_to_stati[ant_i] == ave_stat_i:
                #     amp_ave += np.max(np.abs(self.temp_window)) / self.amplitude_calibrations[ant_i]  ## DE-calibrate
                #     num_amp_ave += 1

                # if self.store_antenna_data:
                #     self.antenna_data[ant_i, :] = self.temp_window
                self.parent.engine.set_antennaData(ant_i, self.temp_window, freq_mode=2)  # .view(np.double) )

            # if (average_station is not None):
            #     self.ave_stat_i = ave_stat_i
            #     self.station_ave_amp = amp_ave / num_amp_ave
            # else:
            #     self.ave_stat_i = None



        def get_empty_SpotImage(self):
            return np.empty(( self.num_freqs, 3), dtype=np.cdouble)


        def get_SpotImage(self, loc, out_image=None, weighting=None, do_matrix=None):

            if weighting is not None:
                if weighting is True:
                    weighting = self.antenna_norms_in_range

            if do_matrix is not None:
                if do_matrix is True:
                    self.temp_inversion_matrix = self.parent.get_correctionMatrix(out=self.temp_inversion_matrix, loc_to_use=loc )
                    self.invertrix.set_matrix(self.temp_inversion_matrix )
                    self.invertrix.get_psuedoinverse(self.inverted_matrix, override_cond=0)
                    do_matrix = self.inverted_matrix

            A = self.parent.engine.Image_at_Spot(loc[0], loc[1], loc[2], out_image,
                frequency_weights=weighting, freq_mode=2,  matrix=do_matrix)

            return A

        def get_empty_partial_inverse_FFT(self):
            return np.empty(self.total_trace_length, dtype=np.cdouble)

        def partial_inverse_FFT(self, in_data, out_data=None):
            return self.parent.engine.partial_inverse_FFT(in_data, out_data, freq_mode=2)

        def get_empty_polarized_inverse_FFT(self, mode='full'):
            if mode == 'full':
                return np.empty( (self.total_trace_length,3), dtype=np.cdouble)
            elif mode == 'wingless':
                return np.empty( (self.minTraceLength_samples,3), dtype=np.cdouble)

        def polarized_inverse_FFT(self,  in_data, out_data=None, mode='full'):
            if out_data is None:
                out_data = self.get_empty_polarized_inverse_FFT( mode )

            for pi in range(3):
                TMP = self.parent.engine.partial_inverse_FFT(in_data[:,pi], self.ifft_full_tmp, freq_mode=2)

                if mode == 'full':
                    out_data[:,pi] = TMP

                elif mode == 'wingless':
                    out_data[:, pi] = TMP[ self.starting_edge_length : self.starting_edge_length + self.minTraceLength_samples ]

            return out_data



#### now we need code to interpret the beamformer

## first, 3D stokes

def simple_coherency( vec ):
    return np.outer( vec, np.conj( vec ) )

class stokes_3D:
    def __init__(self, coherency_matrix):
        self.coherency_matrix = np.array( coherency_matrix )
        
        self.Rreal_eigvals, self.Rreal_eigvecs = np.linalg.eig( np.real(self.coherency_matrix) )
        sorter = np.argsort(self.Rreal_eigvals)[::-1]
        self.Rreal_eigvals = self.Rreal_eigvals[sorter]
        self.Rreal_eigvecs = self.Rreal_eigvecs[:, sorter]

        for i in range(len(self.Rreal_eigvecs )):
            self.Rreal_eigvecs[:, i] /= np.linalg.norm( self.Rreal_eigvecs[:, i] )
        
        self.R_total_eigvals = None
        
        
        
        self.Rreal_eigvecs_inverse = np.linalg.inv( self.Rreal_eigvecs )
        
        self.transformed_coherency_matrix = np.dot(self.Rreal_eigvecs_inverse,  np.dot( self.coherency_matrix,  self.Rreal_eigvecs)  )
        
        SM = np.zeros((3,3), dtype=np.double)

        SM[0,0] = np.real( self.transformed_coherency_matrix[0,0] + self.transformed_coherency_matrix[1,1] + self.transformed_coherency_matrix[2,2] )
        # SM[0,1] = np.real( self.transformed_coherency_matrix[0,1] + self.transformed_coherency_matrix[1,0] )
        # SM[0,2] = np.real( self.transformed_coherency_matrix[0,2] + self.transformed_coherency_matrix[2,0] ) 
        
        SM[1,0] = np.real( 1j*(self.transformed_coherency_matrix[0,1] - self.transformed_coherency_matrix[1,0]) )
        SM[1,1] = np.real( self.transformed_coherency_matrix[0,0] - self.transformed_coherency_matrix[1,1] )
        # SM[1,2] = np.real( self.transformed_coherency_matrix[0,2] + self.transformed_coherency_matrix[2,0] )
        
        SM[2,0] = np.real( 1j*(self.transformed_coherency_matrix[0,2] - self.transformed_coherency_matrix[2,0]) )
        SM[2,1] = np.real( 1j*(self.transformed_coherency_matrix[1,2] - self.transformed_coherency_matrix[2,1]) )
        SM[2,2] = np.real( (self.transformed_coherency_matrix[0,0] + self.transformed_coherency_matrix[1,1] - 2*self.transformed_coherency_matrix[2,2])/np.sqrt(3) )

        SM[0, 1] = SM[1,0]
        SM[0, 2] = SM[2,0]
        SM[1, 2] = SM[2,1]

        # print('SM')
        # print(SM)
        
        self.stokes_matrix = SM
        
        self.intensity = SM[0,0]
        self.linear_polarization = SM[1,1]
        self.degree_of_directionality = SM[2,2] = SM[2,2]*np.sqrt(3)/self.intensity
        
        self.angular_momentum = np.array( [ SM[2,1]*0.5,  -SM[2,0]*0.5,  SM[1,0]*0.5  ] )
        
    def get_axis(self, i=0):
        """return axis of the 3D elipse. axis 0 (default) is direction of linear polarization"""
        return self.Rreal_eigvecs[:,i]
        
    def get_intensity(self):
        """return total intensity"""
        return self.intensity
    
    def get_linear_intensity(self):
        """return linear polarization intensity (in direction of axis 0)"""
        return self.linear_polarization
        
    def get_circular_intensity(self):
        """return intensity of ciruclar polarization"""
        return np.linalg.norm(self.angular_momentum)*2
    
    def get_angular_momentum_normal(self):
        """return angular momentum vector. Has maximum amplitude of 0.5."""
        R =  np.dot( self.Rreal_eigvecs, self.angular_momentum)
        R *= 1.0/self.intensity
        return R
    
    def get_degrees_polarized(self):
        """return fraction linear polarized, circular polarized, and fraction directional. Closely related to degree of polarmetric purity"""
        A = np.array([ self.linear_polarization/self.intensity,  self.get_circular_intensity()/self.intensity, self.degree_of_directionality   ])
        return A
    
    def get_degree_of_directionality(self):
        return self.degree_of_directionality
     
    def get_degree_of_polarimetric_purity(self):
        S =  self.stokes_matrix[1,0]*self.stokes_matrix[1,0]
        S += self.stokes_matrix[1,1]*self.stokes_matrix[1,1]
        
        S += self.stokes_matrix[2,0]*self.stokes_matrix[2,0]
        S += self.stokes_matrix[2,1]*self.stokes_matrix[2,1]
        S += self.stokes_matrix[2,2]*self.stokes_matrix[2,2]
        
        return np.sqrt(3*S)/(2*self.intensity)
    
    def get_degree_of_polarization(self):
        if self.R_total_eigvals is None:
            self.R_total_eigvals = np.linalg.eigvals( self.coherency_matrix )
            sorter = np.argsort(self.R_total_eigvals)[::-1]
            self.R_total_eigvals = self.R_total_eigvals[sorter]
        
        return np.real( (self.R_total_eigvals[0] - self.R_total_eigvals[1])/self.intensity )
    
    def get_indeces_of_polarametric_purity(self):
        """return array of three values. First is ration of power of completely polarized wave over total power, i.e., amount of polarized power (could be degree of polarization)
        No idea what the second one is. Probably something about directionality.
        Last index is the degree of polarimetric purity, and is a combination of first two. It includes polarized energy, and how much the polarization plane wobbles"""
        
        P1 = self.get_degree_of_polarization()
        
        P2 = np.real( (self.R_total_eigvals[0] + self.R_total_eigvals[1] - 2*self.R_total_eigvals[2])/self.intensity )
        
        return np.array( [P1,P2, self.get_degree_of_polarimetric_purity()] )
    
    
### 3D parabolic fitter
def barycenter(image, half_width, Xarray, Yarray, Zarray, peak_indeces=None):
    if peak_indeces is None:
        peak_indeces = cyt.get_peak_loc(image)

    total = 0.0
    Xave = 0.0
    Yave = 0.0
    Zave = 0.0
    is_good = True
    for xi in range(-half_width, half_width+1):
        xi += peak_indeces[0]
        if xi < 0 or xi >= image.shape[0]:
            is_good=False
        if not is_good:
            break

        for yi in range(-half_width, half_width + 1):
            yi += peak_indeces[1]
            if yi < 0 or yi >= image.shape[1]:
                is_good=False
            if not is_good:
                break

            for zi in range(-half_width, half_width + 1):
                zi += peak_indeces[2]
                if zi < 0 or zi >= image.shape[2]:
                    is_good=False
                if not is_good:
                    break

                f = image[xi,yi,zi]
                total += f
                Xave += Xarray[xi]*f
                Yave += Yarray[yi]*f
                Zave += Zarray[zi]*f

    ret_loc = np.array([Xave/total, Yave/total, Zave/total]) if is_good else np.zeros(3, dtype=np.double)
    return is_good, peak_indeces, ret_loc

class parabola_3D:
    def __init__(self, half_N, X_array, Y_array, Z_array):
        self.X_array = X_array
        self.Y_array = Y_array
        self.Z_array = Z_array
        self.dx = X_array[1] - X_array[0]
        self.dy = Y_array[1] - Y_array[0]
        self.dz = Z_array[1] - Z_array[0]
        self.half_N = half_N
        self.N_1D = 2*half_N + 1
        self.num_points = self.N_1D*self.N_1D*self.N_1D
        
        self.matrix = np.empty( (self.num_points,10),dtype=np.double )

        total_i = -1
        for xi in range( self.N_1D ):
            for yi in range( self.N_1D ):
                for zi in range( self.N_1D ):
                    total_i += 1
                    
                    x_shift = self.dx*( xi - half_N )
                    y_shift = self.dy*( yi - half_N )
                    z_shift = self.dz*( zi - half_N )
                    
                    self.matrix[total_i, 0] = 1
                    self.matrix[total_i, 1] = x_shift
                    self.matrix[total_i, 2] = y_shift
                    self.matrix[total_i, 3] = z_shift
                    self.matrix[total_i, 4] = x_shift*x_shift
                    self.matrix[total_i, 5] = x_shift*y_shift
                    self.matrix[total_i, 6] = x_shift*z_shift
                    self.matrix[total_i, 7] = y_shift*y_shift
                    self.matrix[total_i, 8] = y_shift*z_shift
                    self.matrix[total_i, 9] = z_shift*z_shift
                    
        self.b_tmp = np.empty( self.num_points, dtype=np.double )
        self.psuedo_inverse = np.linalg.pinv( self.matrix )
        self.solved_A = None
        
        self.hessian = np.empty((3,3), dtype=np.double)
        self.constants = np.empty(3, dtype=np.double)
        
        self.peak_loc = None
        self.cond_warning_level = 40
        
    def solve(self, image, peak_indeces=None):

        if peak_indeces is None:
            peak_indeces = cyt.get_peak_loc( image )

        self.current_image = image
        self.peak_indeces = peak_indeces
        self.peak_intensity = image[ peak_indeces[0], peak_indeces[1], peak_indeces[2]]

        total_i = -1
        for xi in range( self.N_1D ):
            for yi in range( self.N_1D ):
                for zi in range( self.N_1D ):
                    total_i += 1
                    
                    x_total = ( xi - self.half_N ) + peak_indeces[0]
                    y_total = ( yi - self.half_N ) + peak_indeces[1]
                    z_total = ( zi - self.half_N ) + peak_indeces[2]
                    
                    if x_total < 0 or x_total>=image.shape[0]:
                        return False
                    
                    if y_total < 0 or y_total>=image.shape[1]:
                        return False
                    
                    if z_total < 0 or z_total>=image.shape[2]:
                        return False
                    
                    self.b_tmp[ total_i ] = image[ x_total, y_total, z_total]
                    
        self.solved_A = np.dot( self.psuedo_inverse, self.b_tmp )

        # print('values')
        # print(self.b_tmp)
        # print('solution')
        # print(self.solved_A)
        
        self.hessian[0,0] = 2*self.solved_A[4]
        self.hessian[0,1] =   self.solved_A[5]
        self.hessian[0,2] =   self.solved_A[6]

        self.hessian[1,0] =   self.solved_A[5]
        self.hessian[1,1] = 2*self.solved_A[7]
        self.hessian[1,2] =   self.solved_A[8]

        self.hessian[2,0] =   self.solved_A[6]
        self.hessian[2,1] =   self.solved_A[8]
        self.hessian[2,2] = 2*self.solved_A[9]

        self.constants[0] = -self.solved_A[1]
        self.constants[1] = -self.solved_A[2]
        self.constants[2] = -self.solved_A[3]
        
        self.peak_loc = np.linalg.solve( self.hessian, self.constants )
        # if self.get_hessian_cond() > self.cond_warning_level:
        #     print('warning! parabola is too flat')

        return True
    
    def get_hessian_cond(self):
        return np.linalg.cond(self.hessian)
        
    def get_fit_quality(self):
        """returns RMS/peak amp"""
        
        image_values = np.dot( self.matrix, self.solved_A )
        image_values -= self.b_tmp
        image_values *= image_values
        RMS = np.sqrt( np.average( image_values ) )
        return RMS/self.peak_intensity#self.get_intensity_at_loc()
        
    def get_loc(self, cap=True):
        RET = np.array(self.peak_loc)
        if cap:
            if (RET[0]>self.dx) or (RET[1]>self.dy) or (RET[2]>self.dz):
                RET[:] = 0.0

        RET[0] += self.X_array[ self.peak_indeces[0] ]
        RET[1] += self.Y_array[ self.peak_indeces[1] ]
        RET[2] += self.Z_array[ self.peak_indeces[2] ]
        return RET
    
    def get_indeces_loc(self):
        RET = np.array(self.peak_loc)
        RET[0] /= self.dx
        RET[1] /= self.dy
        RET[2] /= self.dz
        RET[0] += self.peak_indeces[0]
        RET[1] += self.peak_indeces[1]
        RET[2] += self.peak_indeces[2]
        return RET
    
    def get_intensity_at_loc(self):
        R = self.solved_A[0] 
        R += self.solved_A[1]*self.peak_loc[0]  +  self.solved_A[2]*self.peak_loc[1]  +  self.solved_A[3]*self.peak_loc[2]
        R += self.peak_loc[0]*(  self.solved_A[4]*self.peak_loc[0] + self.solved_A[5]*self.peak_loc[1] +  self.solved_A[6]*self.peak_loc[2]  )
        R += self.peak_loc[1]*(  self.solved_A[7]*self.peak_loc[1] +  self.solved_A[8]*self.peak_loc[2]  )
        R += self.peak_loc[2]*self.solved_A[9]*self.peak_loc[2]
        return R


class beamformer_driver:
    def __init__(self, beamformer, timeID, parbolic_half_size=2, chunk_size=None):
        self.beamformer = beamformer
        self.empty_image = beamformer.get_empty_image()
        self.emistivity_image = np.empty( (self.empty_image.shape[0], self.empty_image.shape[1], self.empty_image.shape[2]), dtype=np.double )
    
        self.parabolic_fitter = parabola_3D(parbolic_half_size, self.beamformer.X_array, self.beamformer.Y_array, self.beamformer.Z_array )
        self.timeID = timeID
        self.parbolic_half_size = parbolic_half_size

        self.chunk_size = chunk_size
        if self.chunk_size  is None:
            self.data_TMP = np.empty( (self.beamformer.num_freqs, 3), dtype=np.cdouble )
        else:
            self.realT_space = self.beamformer.get_empty_full_inverse_FFT()
            self.data_TMP = np.empty( (self.chunk_size, 3), dtype=np.cdouble )

        self.unique_index = None

        self.A_matrix_TMP = np.empty((3,3), dtype=np.cdouble)
        self.A_inverse_TMP = np.empty((3,3), dtype=np.cdouble)
        self.invertrix = cyt.SVD_psuedoinversion(3,3)

    def run_image(self, start_t, end_t, fout_name, overlap=0):
        self.unique_index = 0

        if self.chunk_size is None:
            self.__run_image_unchunked__(start_t, end_t, fout_name, overlap)
        else:
            if overlap != 0:
                print('WARNING: cannot presently overlap AND chunk')
            self.__run_image_chunking_(start_t, end_t, fout_name, self.chunk_size)

    def __run_image_unchunked__(self, start_t, end_t, fout_name, overlap=0):
        """make multiple images between two times. overlap should be less than 1. 0.5 gives 50% overlap between different images. Negative overlap gives deadtime between images"""



        time_between_images = self.beamformer.minTraceLength_samples*(5e-9)*(1-overlap)
        
        num_images = int( ( (end_t-start_t)/time_between_images ) ) + 1
        total_image_time = (num_images-1)*time_between_images + self.beamformer.minTraceLength_samples*(5e-9)
        first_time = start_t + ((end_t-start_t) - total_image_time)/2 + self.beamformer.minTraceLength_samples*(5e-9)/2


        fout = open(fout_name, 'w')
        self.write_header(fout, overlap, first_time, num_images, None)

        tmp_weights = np.ones(self.empty_image.shape[3], dtype=np.double)


        for image_i in range(num_images):
            image_time = first_time + image_i*time_between_images
            
            print('running', image_i, '/', num_images, 'at T=', image_time, "(",fout_name,')' )

            self.beamformer.window_data(image_time, 'CS002')

            print('  imaging')

            image = self.beamformer.get_image( self.empty_image, print_progress=True, weighting=True)

            self.process_image(fout, image_time, image, weights=tmp_weights )

    def __run_image_chunking_(self, start_t, end_t, fout_name, chunk_size):
        image_length = self.beamformer.minTraceLength_samples
        image_duration = image_length*5.e-9

        num_chunks_per_image = int( image_length/chunk_size )

        if num_chunks_per_image*chunk_size != image_length:
            print('WARNING: chunk size is not a multiple of image length. WIll be gap at end of image!')
        elif chunk_size > image_length:
            print('WARNING: chunk size is larger than image length. WHAT ARE YOU DOING?  Chunk size will be set to image length')
            chunk_size = image_length
            num_chunks_per_image = 1

        num_images = int( ( (end_t-start_t)/image_duration ) ) + 1
        total_image_time = num_images*image_duration
        first_time = start_t + ((end_t-start_t) - total_image_time)/2 + image_duration/2

        num_sources = num_images*num_chunks_per_image

        fout = open(fout_name, 'w')
        self.write_header(fout, 0.0, first_time, num_sources, chunk_size)

        tmp_weights = np.ones( chunk_size, dtype=np.double )

        for image_i in range(num_images):
            image_time = first_time + image_i*image_duration
            first_chunk_time = image_time - image_duration/2 + chunk_size*(5e-9)/2
            print('running', image_i, '/', num_images, 'at T=', image_time, "(",fout_name,')' )

            self.beamformer.window_data(image_time, 'CS002')
            print('  imaging')
            image = self.beamformer.get_image( self.empty_image, print_progress=True, weighting=True )

            print('  inverse FFT')
            ## TODO: weight this
            self.beamformer.full_inverse_FFT( image, self.realT_space )
            for chunk_i in range(num_chunks_per_image):
                print('   chunck', chunk_i)
                self.process_image(fout, first_chunk_time + chunk_i*chunk_size*(5e-9),
                                   self.realT_space[:,:,:, chunk_i*chunk_size:(chunk_i+1)*chunk_size, : ], weights = tmp_weights )


    def process_image(self, fout, image_time, image, weights=None):
        print('  processing. center T:', image_time)

        CS002_strength = self.beamformer.station_ave_amp

        emistivity, peak_indeces = cyt.get_total_emistivity(image, self.emistivity_image)
        print('  peak index', peak_indeces, 'p:', emistivity[peak_indeces[0], peak_indeces[1], peak_indeces[2]])
        can_find_location = self.parabolic_fitter.solve(emistivity, peak_indeces)
        if not can_find_location:
            print('  too close to edge')
            print('  CS002 strength:', CS002_strength)
            # image_plotter(self.beamformer.X_array, self.beamformer.Y_array, self.beamformer.Z_array, emistivity)
            return False

        parabolic_fit_quality = self.parabolic_fitter.get_fit_quality()

        XYZ_loc = self.parabolic_fitter.get_loc()
        print('  at:', XYZ_loc)
        XYZ_index_loc = self.parabolic_fitter.get_indeces_loc()

#C
        # vector_at_loc = cyt.interpolate_image_atF(image,
        #    XYZ_index_loc[0], XYZ_index_loc[1], XYZ_index_loc[2],
        #    self.beamformer.max_freq_index - self.beamformer.start_freq_index )

        # coherency = simple_coherency(vector_at_loc)
#EC
        print('interp')
        cyt.interpolate_image_full(image,
                                   XYZ_index_loc[0], XYZ_index_loc[1], XYZ_index_loc[2],
                                   self.data_TMP)
        print('di')


        self.beamformer.get_correctionMatrix( self.A_matrix_TMP, XYZ_loc )
        self.invertrix.set_matrix( self.A_matrix_TMP )
        self.invertrix.get_psuedoinverse( self.A_inverse_TMP, override_cond=0 )
        cyt.apply_matrix_in_place( self.data_TMP, self.A_inverse_TMP )

        if weights is None:
            weights = self.beamformer.antenna_norms_in_range

        coherency = cyt.weighted_coherency(self.data_TMP, weights)

        stokes = stokes_3D(coherency)

        intensity = stokes.get_intensity()
        direction = stokes.get_axis()
        linear_intensity = stokes.get_linear_intensity()
        circular_intensity = stokes.get_circular_intensity()
        degree_of_polarization = stokes.get_degree_of_polarization()
        actual_T = image_time - self.beamformer.get_timeDiff_at_loc( XYZ_loc )

        print('  actual T', actual_T)
        print('  intensity:', intensity, 'CS002 strength:', CS002_strength)
        print('  parabolic fit quality', parabolic_fit_quality)
        print('  direction:', direction)
        print('  degree polarized:', degree_of_polarization, 'linear', linear_intensity / intensity,
              'circular', circular_intensity / intensity)

        # ##image_plotter( self.beamformer.X_array, self.beamformer.Y_array, self.beamformer.Z_array, emistivity  )

        # write to file
        # unique_id distance_east distance_north distance_up time_from_second intensity CS002_amp para_fit lin_pol circ_pol dir_east dir_north dir_up
        if fout is not None:
            fout.write(str(self.unique_index))
            fout.write(' ')
            fout.write(str(XYZ_loc[0]))
            fout.write(' ')
            fout.write(str(XYZ_loc[1]))
            fout.write(' ')
            fout.write(str(XYZ_loc[2]))
            fout.write(' ')
            fout.write(str(actual_T))

            fout.write(' ')
            fout.write(str(intensity))
            fout.write(' ')
            fout.write(str(CS002_strength))
            fout.write(' ')
            fout.write(str(parabolic_fit_quality))

            fout.write(' ')
            fout.write(str(degree_of_polarization))
            fout.write(' ')
            fout.write(str(linear_intensity / intensity))
            fout.write(' ')
            fout.write(str(circular_intensity / intensity))

            fout.write(' ')
            fout.write(str(direction[0]))
            fout.write(' ')
            fout.write(str(direction[1]))
            fout.write(' ')
            fout.write(str(direction[2]))

            fout.write('\n')
            fout.flush()
        self.unique_index += 1

        return True

    def write_header(self, fout, overlap, first_image_T, num_images, chunk_length):
        ### setup outfile
        fout.write('! v 1\n')
        fout.write('! timeid ')
        fout.write(self.timeID)
        fout.write('\n')

        fout.write('% XYZ_grid_bounds ')
        fout.write(str(self.beamformer.X_array[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.X_array[-1]))
        fout.write(' ')
        fout.write(str(self.beamformer.Y_array[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.Y_array[-1]))
        fout.write(' ')
        fout.write(str(self.beamformer.Z_array[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.Z_array[-1]))
        fout.write('\n')

        fout.write('% XYZ_voxel_delta ')
        fout.write(str(self.beamformer.voxelDelta_XYZ[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.voxelDelta_XYZ[1]))
        fout.write(' ')
        fout.write(str(self.beamformer.voxelDelta_XYZ[2]))
        fout.write('\n')

        fout.write('% min_width_samples ')
        fout.write(str(self.beamformer.minTraceLength_samples))
        fout.write(' overlap ')
        fout.write(str(overlap))
        fout.write('\n')

        fout.write('# simple beamformer')
        fout.write('\n')

        if chunk_length is not None:
            fout.write('% chunck_size ')
            fout.write(str(chunk_length))
            fout.write('\n')

        fout.write('% para_half_width ')
        fout.write(str(self.parbolic_half_size))
        fout.write('\n')

        fout.write('% first_image_T ')
        fout.write(str(first_image_T))
        fout.write(' num_images ')
        fout.write(str(num_images))
        fout.write('\n')

        fout.write('! max_num_data ')
        fout.write(str(num_images))
        fout.write('\n')

        fout.write('unique_id distance_east distance_north distance_up time_from_second intensity CS002_amp para_fit deg_pol lin_pol circ_pol dir_east dir_north dir_up\n')
        fout.flush()


class superchunking_beamformer_driver_unpol:
    def __init__(self, beamformer, timeID, num_chunks, parbolic_half_size=1):
        self.beamformer = beamformer
        self.timeID = timeID
        self.parbolic_half_size = parbolic_half_size

        self.do_chunk = True
        if num_chunks is None:
            num_chunks = 1
            self.do_chunk = False

        self.num_chunks = num_chunks

        self.valid_trace_length = self.beamformer.minTraceLength_samples
        self.chunk_size = int( self.valid_trace_length/num_chunks )

        if self.chunk_size*num_chunks != self.valid_trace_length:
            print('WARNING: cannot divide', num_chunks, 'into', self.valid_trace_length, 'samples' )


        self.emistivity_image = np.empty( (beamformer.numVoxels_XYZ[0],beamformer.numVoxels_XYZ[1],beamformer.numVoxels_XYZ[2],num_chunks), dtype=np.double )

        # HERE
        # self.image_at_spot = np.empty( (beamformer.numVoxels_XYZ[0],beamformer.numVoxels_XYZ[1],beamformer.numVoxels_XYZ[2],num_chunks), dtype=np.double )


        # self.emistivity_image = np.empty( (self.empty_image.shape[0], self.empty_image.shape[1], self.empty_image.shape[2]), dtype=np.double)

        self.parabolic_fitter = parabola_3D(parbolic_half_size, self.beamformer.X_array, self.beamformer.Y_array,
                                            self.beamformer.Z_array)
        self.timeID = timeID
        self.parbolic_half_size = parbolic_half_size

        # self.realT_space = self.beamformer.get_empty_full_inverse_FFT()
        # self.data_TMP = np.empty((self.chunk_size, 3), dtype=np.cdouble)

        self.unique_index = None

        # self.A_matrix_TMP = np.empty((3, 3), dtype=np.cdouble)
        # self.A_inverse_TMP = np.empty((3, 3), dtype=np.cdouble)
        # self.invertrix = cyt.SVD_psuedoinversion(3, 3)

    def run_image(self, start_t, end_t, fout_name):
        self.unique_index = 0

        image_length = self.beamformer.minTraceLength_samples
        image_duration = image_length * 5.e-9

        num_images = int((end_t - start_t) / image_duration) + 1
        total_image_time = num_images * image_duration
        first_time = start_t + ((end_t - start_t) - total_image_time) / 2 + image_duration / 2

        num_sources = num_images * self.num_chunks

        fout = open(fout_name, 'w')
        self.write_header(fout, num_sources)

        for image_i in range(num_images):
            image_time = first_time + image_i * image_duration
            first_chunk_time = image_time - image_duration / 2 + (self.chunk_size * 5e-9 / 2)
            print('running', image_i, '/', num_images, 'at T=', image_time, "(", fout_name, ')')

            self.beamformer.window_data(image_time, 'CS002')
            print('  imaging')

            C = self.num_chunks
            if not self.do_chunk:
                C = 0
            image = self.beamformer.get_chunked_intesity_image(C, out_image=self.emistivity_image, print_progress=True, weighting=None)

            for chunk_i in range(self.num_chunks):
                chunk_t = first_chunk_time + chunk_i*self.chunk_size*5e-9

                print('chunck', chunk_i, "T:", chunk_t)

                chunk_emistivity = image[:,:,:, chunk_i]
                can_find_location = self.parabolic_fitter.solve( chunk_emistivity )

                indexLoc = self.parabolic_fitter.peak_indeces
                image_at_peak = chunk_emistivity[ indexLoc[0],
                                                  indexLoc[1],
                                                  indexLoc[2] ]

                XYZ_loc = np.array( [self.beamformer.X_array[indexLoc[0]], self.beamformer.Y_array[indexLoc[1]],
                                     self.beamformer.Z_array[indexLoc[2]]], dtype=np.double)

                print(' peak emistivity:', image_at_peak)
                print('  at',  indexLoc)
                print("    :", XYZ_loc)

                if can_find_location:
                    XYZ_loc = self.parabolic_fitter.get_loc()
                    print(' parabolic peak:', self.parabolic_fitter.get_intensity_at_loc())
                    print('  at', self.parabolic_fitter.get_indeces_loc() )
                    print( '   :', XYZ_loc )
                    print(' para fit:', self.parabolic_fitter.get_fit_quality())

                    #print('Hess', np.dot(self.parabolic_fitter.hessian, self.parabolic_fitter.peak_loc) - self.parabolic_fitter.constants )

                else:
                    print(' no para loc!')
                    continue

                actual_T = chunk_t - self.beamformer.get_timeDiff_at_loc( XYZ_loc )


                #image_plotter(self.beamformer.X_array, self.beamformer.Y_array, self.beamformer.Z_array, chunk_emistivity)

                # write to file
                # unique_id distance_east distance_north distance_up time_from_second intensity CS002_amp para_fit lin_pol circ_pol dir_east dir_north dir_up
                fout.write(str(self.unique_index))
                fout.write(' ')
                fout.write(str(XYZ_loc[0]))
                fout.write(' ')
                fout.write(str(XYZ_loc[1]))
                fout.write(' ')
                fout.write(str(XYZ_loc[2]))
                fout.write(' ')
                fout.write(str(actual_T))

                fout.write(' ')
                fout.write(str(image_at_peak))
                fout.write(' ')
                fout.write(str(0.0))
                fout.write(' ')
                fout.write(str(0.0))

                fout.write(' ')
                fout.write(str(1.0))
                fout.write(' ')
                fout.write(str(1.0))
                fout.write(' ')
                fout.write(str(1.0))

                fout.write(' ')
                fout.write(str(1.0))
                fout.write(' ')
                fout.write(str(0.0))
                fout.write(' ')
                fout.write(str(0.0))

                fout.write('\n')
                fout.flush()

                self.unique_index += 1

    def write_header(self, fout, max_num_sources):
        ### setup outfile
        fout.write('! v 1\n')
        fout.write('! timeid ')
        fout.write(self.timeID)
        fout.write('\n')

        fout.write('# superchunk unpol ')
        fout.write('\n')

        fout.write('% XYZ_grid_bounds ')
        fout.write(str(self.beamformer.X_array[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.X_array[-1]))
        fout.write(' ')
        fout.write(str(self.beamformer.Y_array[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.Y_array[-1]))
        fout.write(' ')
        fout.write(str(self.beamformer.Z_array[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.Z_array[-1]))
        fout.write('\n')

        fout.write('% XYZ_voxel_delta ')
        fout.write(str(self.beamformer.voxelDelta_XYZ[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.voxelDelta_XYZ[1]))
        fout.write(' ')
        fout.write(str(self.beamformer.voxelDelta_XYZ[2]))
        fout.write('\n')

        fout.write('% min_width_samples ')
        fout.write(str(self.beamformer.minTraceLength_samples))
        fout.write('\n')

        fout.write('% chunck_size ')
        fout.write(str(self.chunk_size))
        fout.write('\n')

        fout.write('% num_chunks ')
        fout.write(str(self.num_chunks))
        fout.write('\n')

        fout.write('% para_half_width ')
        fout.write(str(self.parbolic_half_size))
        fout.write('\n')

        fout.write('! max_num_data ')
        fout.write(str(max_num_sources))
        fout.write('\n')

        fout.write(
            'unique_id distance_east distance_north distance_up time_from_second intensity CS002_amp para_fit deg_pol lin_pol circ_pol dir_east dir_north dir_up\n')
        fout.flush()

class superchunking_beamformer_driver:
    def __init__(self, beamformer, timeID, num_chunks, chunk_polarization=None, parbolic_half_size=1):
        self.beamformer = beamformer
        self.timeID = timeID
        self.parbolic_half_size = parbolic_half_size
        self.num_chunks = num_chunks

        self.valid_trace_length = self.beamformer.minTraceLength_samples

        if self.num_chunks is not None:
            self.do_chunk = True

            self.chunk_size = int( self.valid_trace_length/num_chunks )
            if self.chunk_size*num_chunks != self.valid_trace_length:
                print('WARNING: cannot divide', num_chunks, 'into', self.valid_trace_length, 'samples' )

            self.chunked_imager = beamformer.get_secondaryLength_beamformer(self.chunk_size)

            if chunk_polarization is None:
                chunk_polarization = True
        else:
            self.do_chunk = False
            self.num_chunks = 1

            self.chunked_imager = beamformer
            if chunk_polarization is None:
                chunk_polarization = False

        self.chunk_polarization = chunk_polarization
        self.emistivity_image = np.empty( (beamformer.numVoxels_XYZ[0],beamformer.numVoxels_XYZ[1],beamformer.numVoxels_XYZ[2],self.num_chunks), dtype=np.double )


        self.tmp_FFT_image = self.chunked_imager.get_empty_SpotImage()

        if self.chunk_polarization:
            self.tmp_T_image = self.chunked_imager.get_empty_polarized_inverse_FFT( mode='wingless' )
            # self.ones = np.ones(len(self.tmp_T_image), dtype=np.double)
        # else:
            # self.ones = np.ones(len(self.tmp_FFT_image), dtype=np.double)



        self.parabolic_fitter = parabola_3D(parbolic_half_size, self.beamformer.X_array, self.beamformer.Y_array,
                                            self.beamformer.Z_array)
        self.timeID = timeID
        self.parbolic_half_size = parbolic_half_size



        self.unique_index = None

    def run_image(self, start_t, end_t, fout_name):
        self.unique_index = 0

        image_length = self.beamformer.minTraceLength_samples
        image_duration = image_length * 5.e-9

        num_images = int((end_t - start_t) / image_duration) + 1
        total_image_time = num_images * image_duration
        first_time = start_t + ((end_t - start_t) - total_image_time) / 2 + image_duration / 2

        num_sources = num_images * self.num_chunks

        if fout_name is not None:
            fout = open(fout_name, 'w')
            self.write_header(fout, num_sources)

        for image_i in range(num_images):
            image_time = first_time + image_i * image_duration
            if self.do_chunk:
                first_chunk_time = image_time - image_duration / 2 + (self.chunk_size * 5e-9 / 2)

            if fout_name is not None:
                print('running', image_i, '/', num_images, 'at T=', image_time, "(", fout_name, ')')
            else:
                print('running', image_i, '/', num_images, 'at T=', image_time)

            self.beamformer.window_data(image_time, 'CS002')
            print('  imaging')

            image = self.beamformer.get_chunked_intesity_image(self.num_chunks, out_image=self.emistivity_image, do_ifft=self.do_chunk,
                                                               print_progress=True, weighting=True)

            for chunk_i in range(self.num_chunks):
                if self.do_chunk:
                    chunk_t = first_chunk_time + chunk_i*self.chunk_size*5e-9
                else:
                    chunk_t = image_time ## note: self.num_chunks is 1 in this case

                print('chunck', chunk_i, "T:", chunk_t)

                chunk_emistivity = image[:,:,:, chunk_i]
                can_find_location = self.parabolic_fitter.solve( chunk_emistivity )

                indexLoc = self.parabolic_fitter.peak_indeces
                image_at_peak = chunk_emistivity[ indexLoc[0],
                                                  indexLoc[1],
                                                  indexLoc[2] ]

                XYZ_loc = np.array( [self.beamformer.X_array[indexLoc[0]], self.beamformer.Y_array[indexLoc[1]],
                                     self.beamformer.Z_array[indexLoc[2]]], dtype=np.double)

                print(' peak emistivity:', image_at_peak)
                print('  at',  indexLoc)
                print("    :", XYZ_loc)

                if not can_find_location:
                    print(' no para loc!')
                    continue

                XYZ_loc = self.parabolic_fitter.get_loc()
                print(' parabolic peak:', self.parabolic_fitter.get_intensity_at_loc())
                print('  at', self.parabolic_fitter.get_indeces_loc() )
                print( '   :', XYZ_loc )
                parabolic_fit_quality = self.parabolic_fitter.get_fit_quality()
                print(' para fit:', parabolic_fit_quality)


                self.chunked_imager.window_data( chunk_t )
                image_at_spot = self.chunked_imager.get_SpotImage( XYZ_loc, self.tmp_FFT_image,  weighting=True, do_matrix=True)

                if self.chunk_polarization:
                    image_chunk = self.chunked_imager.polarized_inverse_FFT( image_at_spot, self.tmp_T_image, mode='wingless' )
                else:
                    image_chunk = image_at_spot


                actual_T = chunk_t - self.beamformer.get_timeDiff_at_loc( XYZ_loc )

                coherency = cyt.coherency(image_chunk)
                stokes = stokes_3D(coherency)

                intensity = stokes.get_intensity()
                direction = stokes.get_axis()
                linear_intensity = stokes.get_linear_intensity()
                circular_intensity = stokes.get_circular_intensity()
                circular_direction = stokes.get_angular_momentum_normal()
                circular_direction *= 0.5 ## so ranges from 0 to 1
                degree_of_polarization = stokes.get_degree_of_polarization()

                print('  actual T', actual_T)
                print('  intensity:', intensity)
                print('  parabolic fit quality', parabolic_fit_quality)
                print('  direction:', direction)
                print('  degree polarized:', degree_of_polarization, 'linear', linear_intensity / intensity,
                      'circular', circular_intensity / intensity)




                #image_plotter(self.beamformer.X_array, self.beamformer.Y_array, self.beamformer.Z_array, chunk_emistivity)

                if fout_name is not None:
                    # write to file
                    # unique_id distance_east distance_north distance_up time_from_second intensity CS002_amp para_fit lin_pol circ_pol dir_east dir_north dir_up circ_east circ_north circ_up
                    fout.write(str(self.unique_index))
                    fout.write(' ')
                    fout.write(str(XYZ_loc[0]))
                    fout.write(' ')
                    fout.write(str(XYZ_loc[1]))
                    fout.write(' ')
                    fout.write(str(XYZ_loc[2]))
                    fout.write(' ')
                    fout.write(str(actual_T))

                    fout.write(' ')
                    fout.write(str(intensity))
                    fout.write(' ')
                    fout.write(str(parabolic_fit_quality))

                    fout.write(' ')
                    fout.write(str(degree_of_polarization))
                    fout.write(' ')
                    fout.write(str(linear_intensity / intensity))
                    fout.write(' ')
                    fout.write(str(circular_intensity / intensity))

                    fout.write(' ')
                    fout.write(str(direction[0]))
                    fout.write(' ')
                    fout.write(str(direction[1]))
                    fout.write(' ')
                    fout.write(str(direction[2]))

                    fout.write(' ')
                    fout.write(str(circular_direction[0]))
                    fout.write(' ')
                    fout.write(str(circular_direction[1]))
                    fout.write(' ')
                    fout.write(str(circular_direction[2]))

                    fout.write('\n')
                    fout.flush()

                self.unique_index += 1

    def write_header(self, fout, max_num_sources):
        ### setup outfile
        fout.write('! v 1\n')
        fout.write('! timeid ')
        fout.write(self.timeID)
        fout.write('\n')

        fout.write('# superchunk')
        fout.write('\n')

        fout.write('% XYZ_grid_bounds ')
        fout.write(str(self.beamformer.X_array[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.X_array[-1]))
        fout.write(' ')
        fout.write(str(self.beamformer.Y_array[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.Y_array[-1]))
        fout.write(' ')
        fout.write(str(self.beamformer.Z_array[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.Z_array[-1]))
        fout.write('\n')

        fout.write('% XYZ_voxel_delta ')
        fout.write(str(self.beamformer.voxelDelta_XYZ[0]))
        fout.write(' ')
        fout.write(str(self.beamformer.voxelDelta_XYZ[1]))
        fout.write(' ')
        fout.write(str(self.beamformer.voxelDelta_XYZ[2]))
        fout.write('\n')

        fout.write('% min_width_samples ')
        fout.write(str(self.beamformer.minTraceLength_samples))
        fout.write('\n')

        if self.do_chunk:
            fout.write('% chunck_size ')
            fout.write(str(self.chunk_size))
            fout.write('\n')

            fout.write('% num_chunks ')
            fout.write(str(self.num_chunks))
            fout.write('\n')

        fout.write('% para_half_width ')
        fout.write(str(self.parbolic_half_size))
        fout.write('\n')

        fout.write('! max_num_data ')
        fout.write(str(max_num_sources))
        fout.write('\n')

        fout.write(
            'unique_id distance_east distance_north distance_up time_from_second intensity para_fit deg_pol lin_pol circ_pol dir_east dir_north dir_up circ_east circ_north circ_up\n')
        fout.flush()
