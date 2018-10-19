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
from LoLIM.signal_processing import parabolic_fit
#from RunningStat import RunningStat

from LoLIM.read_pulse_data import writeTXT_station_delays,read_station_info, curtain_plot_CodeLog

from LoLIM.planewave_functions import read_SSPW_timeID

#### some random utilities
def none_max(lst):
    """given a list of numbers, return maximum, ignoreing None"""
    
    ret = -np.inf
    for a in lst:
        if a is not None and a>ret:
            ret=a
            
    return ret

def combine_SSPW(SSPW_A, SSPW_B):
    """for two SSPW, first loads all data from file. Then combines antenna data from SSPW_B into SSPW_A"""
    
    SSPW_A.reload_data()
    SSPW_B.reload_data()

    for ant_name, ant_data in SSPW_B.ant_data.items():
        SSPW_A.ant_data[ant_name] = ant_data
                
    
def get_radius_ze_az( XYZ ):
    radius = np.linalg.norm( XYZ )
    ze = np.arccos( XYZ[2]/radius )
    az = np.arctan2( XYZ[1], XYZ[0] )
    return radius, ze, az
    

#### main code

#### source object ####
## represents a potential source
## keeps track of a SSPW on the prefered station, and SSPW on other stations that could correlate and are considered correlated
## contains utilities for fitting, and for finding RMS in total and for each station
## also contains utilities for plotting and saving info

## need to handle inseartion of random error, and that choosen SSPE can change

class source_object():
## assume: guess_location , ant_locs, station_to_antenna_index_list, station_to_antenna_index_dict, referance_station, station_order, SSPW_dict,
#    sorted_antenna_names, station_locations, SSPW_dict
    # are global
    
    def do_combine_SSPW(self, SSPW, SSPW_list):
        combine_list = []
        for combination in self.SSPW_to_combine:
            if SSPW.unique_index in combination:
                combine_list = combination
                break
            
        for SSPW_ID in combine_list:
            if SSPW_ID == SSPW.unique_index:
                continue
            
            for new_SSPW in SSPW_list:
                if new_SSPW.unique_index == SSPW_ID:
                    break
                
            combine_SSPW( SSPW, new_SSPW )
            self.source_exclusions.append( SSPW_ID ) ## so we don't use this SSPW in the future
    
    def __init__(self, ref_SSPW_index,  viable_SSPW_indeces, source_exclusions, location=None, SSPW_to_combine=[], suspicious_SSPW_list=[] ):
        self.suspicious_SSPW_list = suspicious_SSPW_list
        self.SSPW_to_combine = SSPW_to_combine
        self.source_exclusions = source_exclusions
        self.SSPW_in_use = {}
        
        
        #### first we need to find the SSPW on the referance station
        self.SSPW = None
        for SSPW in SSPW_dict[ referance_station ]:
            if SSPW.unique_index == ref_SSPW_index:
                self.SSPW = SSPW
                
                if not self.SSPW.timeseries_loaded:
                    self.SSPW.reload_data()
                    
                self.do_combine_SSPW( self.SSPW, SSPW_dict[ referance_station ]  )
                break
                
        if self.SSPW is None:
            print("cannot find SSPW")
            quit()
            
        
        #### now we find SSPW on other stations
        self.viable_SSPW = {referance_station:[self.SSPW]}
        self.num_stations_with_unique_viable_SSPW = 0
        for sname, SSPW_ids in viable_SSPW_indeces.items():
            SSPW_list = []
            
            for ID in SSPW_ids:
                if ID in source_exclusions:
                    continue
                
                for SSPW in SSPW_dict[ sname ]:
                    if SSPW.unique_index == ID:
                        if not SSPW.timeseries_loaded:
                            SSPW.reload_data()
                        SSPW_list.append( SSPW )
                        self.do_combine_SSPW( SSPW, SSPW_dict[ sname ]  )
                        break
                    
            self.viable_SSPW[sname] = SSPW_list
            if len(SSPW_list) == 1 and not SSPW_list[0].unique_index in self.suspicious_SSPW_list:
                self.num_stations_with_unique_viable_SSPW += 1
        
        if location is None:
            guess_time = SSPW.ZAT[2] - np.linalg.norm( station_locations[referance_station]-guess_location )/v_air
            self.guess_XYZT = np.append( guess_location, [guess_time] )
        else:
            self.guess_XYZT = np.array( location )

                 
            
    def prep_for_fitting(self, polarization):
        self.polarization = polarization
        
        self.pulse_times = np.empty( len(sorted_antenna_names) )
        self.pulse_times[:] = np.nan
        self.waveforms = [None]*len(self.pulse_times)
        self.waveform_startTimes = [None]*len(self.pulse_times)
        
        #### first add times from referance_station
        self.add_SSPW( self.SSPW )
                
        #### next we add in stations that only have 1 unique SSPW
        for sname in station_order:
            SSPW_list = self.viable_SSPW[sname]
            if len(SSPW_list) == 1  and not SSPW_list[0].unique_index in self.suspicious_SSPW_list:
                SSPW_to_add = SSPW_list[0]
                
                self.add_SSPW( SSPW_to_add )
        
                        
    def prep_for_fitting_knownFit(self, polarity, SSPW_associations ):
        self.polarization = polarity
        
        self.pulse_times = np.empty( len(sorted_antenna_names) )
        self.pulse_times[:] = np.nan
        self.waveforms = [None]*len(self.pulse_times)
        self.waveform_startTimes = [None]*len(self.pulse_times)
        
        for sname, SSPW_ID in SSPW_associations.items():
            SSPW_list = self.viable_SSPW[sname]
            for SSPW in SSPW_list:
                if SSPW.unique_index == SSPW_ID:
                    break## found correct SSPW
                
            self.add_SSPW( SSPW )
            
        self.perturbed_pulse_times = np.array(self.pulse_times)
                
                
    def remove_station(self, sname):
        if sname in self.SSPW_in_use:
            del self.SSPW_in_use[ sname ]
            
        antenna_index_range = station_to_antenna_index_dict[sname]
        self.pulse_times[ antenna_index_range[0]:antenna_index_range[1] ] = np.nan
        
    def has_station(self, sname):
        return (sname in self.SSPW_in_use)
         
    def add_SSPW(self, SSPW):
        
        self.remove_station( SSPW.sname )
        
        self.SSPW_in_use[ SSPW.sname ] = SSPW.unique_index
        
        if not SSPW.timeseries_loaded:
            SSPW.reload_data()
        
        antenna_index_range = station_to_antenna_index_dict[SSPW.sname]
        
        for ant_i in range(antenna_index_range[0], antenna_index_range[1]):
            ant_name = sorted_antenna_names[ant_i]
            if ant_name in SSPW.ant_data:
                ant_info = SSPW.ant_data[ant_name]
                start_time = ant_info.pulse_starting_index*5.0E-9
                
                if self.polarization != 3:
                    pt = ant_info.PolE_peak_time if self.polarization==0 else ant_info.PolO_peak_time
                    waveform = ant_info.PolE_hilbert_envelope if self.polarization==0 else ant_info.PolO_hilbert_envelope
                    start_time += ant_info.PolE_time_offset if self.polarization==0 else ant_info.PolO_time_offset
                    amp = np.max(waveform)
                    
                    if pt == np.inf:
                        pt = np.nan
                    if amp<min_antenna_amplitude:
                        pt = np.nan
                else:
                    PolE_data = ant_info.PolE_antenna_data
                    PolO_data = ant_info.PolO_antenna_data
                
                    if np.max( PolE_data ) < min_antenna_amplitude or np.max( PolO_data ) < min_antenna_amplitude:
                        pt = np.nan
                        waveform = None
                    else:
                        ant_loc = ant_locs[ ant_i ]
                        
                        radius, zenith, azimuth = get_radius_ze_az( self.guess_XYZT[:3]-ant_loc )
                
                        antenna_calibrator.FFT_prep(ant_name, PolE_data, PolO_data)
                        antenna_calibrator.apply_time_shift(0.0, ant_info.PolO_time_offset-ant_info.PolE_time_offset)
                        polE_cal, polO_cal = antenna_calibrator.apply_GalaxyCal()
                        
                        if not np.isfinite(polE_cal) or not np.isfinite(polO_cal):
                            pt = np.nan
                            waveform = None
                            
                        else:
                            antenna_calibrator.unravelAntennaResponce(zenith*RTD, azimuth*RTD)
                            zenith_data, azimuth_data = antenna_calibrator.getResult()
                
                            ZeD_R = np.real( zenith_data )
                            ZeD_I = np.imag( zenith_data )
                            Az_R = np.real( azimuth_data )
                            Az_I = np.imag( azimuth_data )
                            
                            total_amplitude_waveform = np.sqrt( ZeD_R*ZeD_R + ZeD_I*ZeD_I + Az_R*Az_R + Az_I*Az_I )
                            waveform = total_amplitude_waveform
                            
                            peak_finder = parabolic_fit( total_amplitude_waveform )
                            pt = (peak_finder.peak_index + ant_info.pulse_starting_index)*5.0E-9 + ant_info.PolE_time_offset
                            start_time += ant_info.PolE_time_offset
                
            
                self.pulse_times[ ant_i ] = pt
                self.waveforms[ ant_i ] = waveform
                self.waveform_startTimes[ ant_i ] = start_time
                
    def insert_noise(self, sigma):
        self.perturbed_pulse_times[:] = self.pulse_times
        self.perturbed_pulse_times += np.random.normal(scale=sigma, size=len(self.perturbed_pulse_times))
        
        
    def try_location_LS(self, delays, XYZT_location, out):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        delta_X_sq = ant_locs[:,0]-X
        delta_Y_sq = ant_locs[:,1]-Y
        delta_Z_sq = ant_locs[:,2]-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
            
        out[:] = T - self.perturbed_pulse_times
        
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
        
        
        out_loc[:,3] = T - self.perturbed_pulse_times
        out_loc[:,3] *= 2*v_air*v_air
        
        delay_i = 0
        for index_range, delay in zip(station_to_antenna_index_list,  delays):
            first,last = index_range
            
            out_loc[first:last,3] += delay*2*v_air*v_air
            out_delays[first:last,delay_i] = out_loc[first:last,3]
                
            delay_i += 1
            
    def num_DOF(self):
        return np.sum( np.isfinite(self.perturbed_pulse_times) ) - 3 ## minus three or four?
            
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
        
        workspace[:] -= self.perturbed_pulse_times*v_air ## this is now source time
        
        ##now account for delays
        for index_range, delay in zip(station_to_antenna_index_list,  delays):
            first,last = index_range
            workspace[first:last] += delay*v_air ##note the wierd sign
                
                
        ave_error = np.nanmean( workspace )
        return -ave_error/v_air
            
#    def SSqE_fit(self, delays, XYZT_location):
#        X,Y,Z,T = XYZT_location
#        Z = np.abs(Z)
#        
#        delta_X_sq = ant_locs[:,0]-X
#        delta_Y_sq = ant_locs[:,1]-Y
#        delta_Z_sq = ant_locs[:,2]-Z
#        
#        delta_X_sq *= delta_X_sq
#        delta_Y_sq *= delta_Y_sq
#        delta_Z_sq *= delta_Z_sq
#        
#        distance = delta_X_sq
#        distance += delta_Y_sq
#        distance += delta_Z_sq
#        
#        np.sqrt(distance, out=distance)
#        distance *= 1.0/v_air
#        
#        distance += T
#        distance -= self.pulse_times
#        
#        ##now account for delays
#        for index_range, delay in zip(station_to_antenna_index_list,  delays):
#            first,last = index_range
#            if first is not None:
#                distance[first:last] += delay ##note the wierd sign
#                
#        distance *= distance
#        return np.nansum(distance)
    
            
            
class Part1_input_manager:
    def __init__(self, input_files, inject_sources):
        self.input_files = input_files
        
        self.input_data = []
        for fname in input_files:
            input_fname = processed_data_dir + "/" + fname + '/out'
            with open(input_fname, 'rb') as fin:
                input_SSPW_sort = load(fin)
                self.input_data.append( input_SSPW_sort )
                
        if len(inject_sources)>0:
            self.input_data.append(  [ (data[referance_station][0], data) for data in inject_sources ] )
                
        self.indeces = np.zeros( len(self.input_data), dtype=int )
#        self.last_file_index = None
        
    def known_source(self, ID):
        
        for current_i, index in enumerate(self.indeces):
            ref_SSPW_index, viable_SSPW_indeces = self.input_data[ current_i ][ index ]
            
            if ref_SSPW_index in bad_sources:
                self.indeces[ current_i ] += 1
                return self.known_source( ID )
            
            if ref_SSPW_index==ID:
                self.indeces[ current_i ] += 1
                return ref_SSPW_index, viable_SSPW_indeces
            
        return None
        
    def next(self):
#        #### first check for sources that are well known
#        for current_i, index in enumerate(self.indeces):
#            ref_SSPW_index, viable_SSPW_indeces = self.input_data[ current_i ][ index ]
#            
#            if ref_SSPW_index in bad_sources:
#                self.indeces[ current_i ] += 1
#                return self.next()
#            
#            elif ref_SSPW_index in known_sources:
#                self.indeces[ current_i ] += 1
##                self.last_file_index = current_i
#                return ref_SSPW_index, viable_SSPW_indeces
            
        #### if we are here, no sources are well known. check for sources that are currently being fitted
        best_file_i = 0
        for current_i, index in enumerate(self.indeces):
            ref_SSPW_index, viable_SSPW_indeces = self.input_data[ current_i ][ index ]
            
            if ref_SSPW_index in planewave_exclusions:
                self.indeces[ current_i ] += 1
#                self.last_file_index = current_i
                return ref_SSPW_index, viable_SSPW_indeces
            
            if index < self.indeces[best_file_i]:
                best_file_i = current_i
                
        #### if we are here, then no fitted sources left, only fully uknown sources
        ## return the "highest" unkown source
        
#        self.last_file_index = best_file_i
        return self.input_data[best_file_i][ self.indeces[best_file_i] ]
            
        
if __name__ == "__main__":
    #### TODO: find error of arrival time on a station ####
    
    timeID = "D20170929T202255.000Z"
    output_folder = "autoCorrelator_unCal"
    
#    part1_input = "autoCorrelator_Part1"
    part1_inputs = ["autoCorrelator_Part1", "autoCorrelator_Part1_2"]
    
    
    SSPW_folder = 'SSPW'
    first_block = 3450
    num_blocks = 300
    
    
    #### fitter settings ####
    timing_error = 1E-9
    num_runs = 2000
    
    #### source quality requirments ####
    min_stations = 4
    max_station_RMS = 5.0E-9
    min_antenna_amplitude = 10
    
    #### initial guesses ####
    referance_station = "CS002"
    guess_location = np.array( [1.72389621e+04,   9.50496918e+03, 2.37800915e+03] )
    
    guess_timings = {
#        "CS002":0.0,
        "CS003":1.0E-6,
        "CS004":0.0,
        "CS005":0.0,
        "CS006":0.0,
        "CS007":0.0,
        "CS011":0.0,
        "CS013":-0.000003,
        "CS017":-7.0E-6,
        "CS021":-8E-7,
#        "CS026":-7E-6,
        "CS030":-5.5E-6,
        "CS032":-5.5E-6 + 2E-6 + 1E-7,
        "CS101":-7E-6,
        "CS103":-22.5E-6-20E-8,
#        "RS106":35E-6 +30E-8 +12E-7,
#        "CS201":-7.5E-6,
#        "RS205":25E-6,
        "RS208":8E-5+3E-6,
        "CS301":6.5E-7,
        "CS302":-3.0E-6-35E-7,
#        "RS305":-6E-6,
        "RS306":-7E-6,
        "RS307":175E-7+8E-7,
        "RS310":8E-5+6E-6,
        "CS401":-3E-6,
        "RS406":-25E-6,
#        "RS407":5E-6 -8E-6 -15E-7,
        "RS409":8E-6,
        "CS501":-12E-6,
        "RS503":-30.0E-8-10E-7,
        "RS508":6E-5+5E-7,
        "RS509":10E-5+15E-7,
        }
        
    guess_timing_error = 5E-6
    guess_is_from_curtain_plot = True ## if true, must take station locations into account in order to get true offsets
    
    if referance_station in guess_timings:
        del guess_timings[referance_station]
    
    
    #### these are sources whose location have been fitted, and SSPW are associated
    known_sources = [275988, 278749, 280199, 274969, 275426, 276467, 274359]
    bad_sources = [275989, 278750, 274968, 275427] ## these are sources that should not be fit for one reason or anouther
    
    ### locations of fitted sources
    known_source_locations = {
    275988 :[ -17214.596844 , 9068.79560213 , 2680.11907723 , 1.17337205368 ],
    278749 :[ -15589.7573812 , 8168.1684321 , 1779.95016001 , 1.2086121771 ],
    280199 :[ -15711.2644128 , 10805.73449 , 4870.33848556 , 1.22938993148 ],
    274969 :[ -16113.565593 , 9841.29254323 , 3103.83539507 , 1.16010479078 ],
    275426 :[ -15661.53842 , 9765.44819988 , 3497.38428455 , 1.16635812263 ],
    276467 :[ -15928.8326741 , 10160.5826651 , 3860.43326479 , 1.17974440268 ],
    274359 :[ -15903.2643789 , 9116.21999261 , 3605.43534373 , 1.1531768367 ],
    }
    
    ### polarization of fitted sources
    known_polarizations = {
    275988 : 3 ,
    278749 : 3 ,
    280199 : 3 ,
    274969 : 3 ,
    275426 : 3 ,
    276467 : 3 ,
    
    274359 : 3,
    }

    #### SSPW associated with known sources
    known_SSPW_associations = {
    275988 :{
      'CS002': 275988 ,
      'CS003': 24748 ,
      'CS004': 14200 ,
      'CS005': 88756 ,
      'CS006': 57214 ,
      'CS007': 146309 ,
      'CS011': 320165 ,
      'CS013': 242590 ,
      'CS017': 109146 ,
      'CS021': 157424 ,
      'CS030': 254161 ,
      'CS032': 212655 ,
      'CS101': 36666 ,
      'CS103': 177242 ,
      'RS208': 100043 ,
      'CS301': 78487 ,
      'CS302': 46855 ,
      'RS306': 223578 ,
      'RS307': 331331 ,
#      'RS310': 139072 ,
      'CS401': 234914 ,
      'CS501': 120972 ,
      'RS503': 264891 ,
    },
    278749 :{
      'CS002': 278749 ,
      'CS003': 27636 ,
      'CS004': 16653 ,
      'CS005': 91465 ,
      'CS006': 59926 ,
      'CS007': 148951 ,
      'CS011': 322779 ,
      'CS013': 245389 ,
      'CS017': 111986 ,
      'CS021': 159903 ,
      'CS030': 256660 ,
      'CS032': 215217 ,
      'CS101': 39074 ,
      'CS103': 179617 ,
      'CS301': 80884 ,
      'CS302': 49269 ,
      'RS307': 333870 ,
      'CS401': 236365 ,
#      'RS409': 203996 ,
      'CS501': 123427 ,
      'RS503': 267510 ,
    },
    280199 :{
      'CS002': 280199 ,
      'CS003': 29170 ,
      'CS004': 17941 ,
      'CS005': 92973 ,
      'CS006': 61367 ,
      'CS007': 150387 ,
      'CS011': 324197 ,
      'CS013': 246868 ,
      'CS017': 113547 ,
      'CS021': 161298 ,
      'CS030': 257980 ,
      'CS032': 216549 ,
      'CS101': 40362 ,
      'CS103': 180885 ,
      'CS302': 50561 ,
      'CS401': 237257 ,
      'CS501': 124716 ,
      'RS503': 268846 ,
      'RS307': 335211 ,
    },
    274969 :{
      'CS002': 274969 ,
      'CS003': 23683 ,
      'CS004': 13255 ,
      'CS005': 87738 ,
      'CS006': 56136 ,
      'CS007': 145284 ,
      'CS011': 319157 ,
      'CS013': 241518 ,
      'CS017': 108091 ,
      'CS021': 156470 ,
      'CS030': 253194 ,
      'CS032': 211661 ,
      'CS101': 35724 ,
      'CS103': 176359 ,
      'RS208': 99056 ,
      'CS302': 45962 ,
      'RS306': 222520 ,
      'RS307': 330365 ,
      'CS401': 234365 ,
#      'RS406': 186386 ,
      'RS503': 263923 ,
    },
    275426 :{
      'CS002': 275426 ,
      'CS003': 24178 ,
      'CS004': 13711 ,
      'CS005': 88213 ,
      'CS006': 56657 ,
      'CS011': 319648 ,
      'CS013': 242031 ,
      'CS017': 108580 ,
      'CS021': 156912 ,
      'CS030': 253646 ,
      'CS032': 212137 ,
      'CS101': 36177 ,
      'CS103': 176782 ,
      'CS301': 78006 ,
      'CS302': 46387 ,
      'RS307': 330823 ,
      'CS401': 234639 ,
      'RS409': 200775 ,
      'CS501': 120471 ,
      'RS503': 264392 ,
      'CS007': 145749 ,
      'RS208': 99549 ,
    },
    276467 :{
      'CS002': 276467 ,
      'CS003': 25287 ,
      'CS004': 14637 ,
      'CS005': 89274 ,
      'CS006': 57673 ,
      'CS007': 146788 ,
      'CS011': 320634 ,
      'CS013': 243118 ,
      'CS017': 109701 ,
      'CS021': 157874 ,
      'CS030': 254609 ,
      'CS032': 213131 ,
      'CS101': 37091 ,
      'CS103': 177678 ,
      'CS301': 78918 ,
      'CS302': 47306 ,
      'CS401': 235180 ,
      'CS501': 121428 ,
      'RS503': 265345 ,
      'RS208': 100559 ,
    },
    274359 :{
      'CS002': 274359 ,
      'CS003': 23074 ,
      'CS004': 12701 ,
      'CS005': 87161 ,
      'CS006': 55515 ,
      'CS007': 144697 ,
      'CS011': 318581 ,
      'CS013': 240917 ,
      'CS017': 107475 ,
      'CS021': 155974 ,
      'CS030': 252601 ,
      'CS103': 175823 ,
      'CS301': 77035 ,
      'CS302': 45417 ,
      'RS307': 329828 ,
      'CS401': 234180 ,
      'RS409': 199737 ,
      'CS501': 119469 ,
      'RS503': 263303 ,
      'RS508': 309188 ,
      'RS509': 129999 ,
      'RS208': 98543 ,
    },
        
    }

    inject_sources = [
        {
        "CS002":[274359],
        "CS003":[23074,23075],
        "CS004":[12701],
        "CS005":[87161,87162],
        "CS006":[55515],
        "CS007":[144697],
        "CS011":[318581,318582],
        "CS013":[240917,240918],
        "CS017":[107475,107476],
        "CS021":[155974],
        "CS030":[252601, 252602],
        "CS032":[211073],
        "CS101":[35148],
        "CS103":[175823],
        "RS208":[98543],
        "CS301":[77035,77036],
        "CS302":[45417],
        "RS306":[],
        "RS307":[329828,329829],
        "RS310":[],
        "CS401":[234180],
        "RS406":[],
        "RS409":[199737],
        "CS501":[119469],
        "RS503":[263303,263304],
        "RS508":[309188],
        "RS509":[129999],
        },
        {
        "CS002":[274360],
        "CS003":[],
        "CS004":[12702],
        "CS005":[87163,87164],
        "CS006":[55516],
        "CS007":[],
        "CS011":[],
        "CS013":[],
        "CS017":[107477],
        "CS021":[],
        "CS030":[],
        "CS032":[],
        "CS101":[35149],
        "CS103":[175824,175825],
        "RS208":[],
        "CS301":[77037],
        "CS302":[45418],
        "RS306":[],
        "RS307":[329830,329831,329832],
        "RS310":[137879],
        "CS401":[],
        "RS406":[185833],
        "RS409":[],
        "CS501":[],
        "RS503":[],
        "RS508":[309189],
        "RS509":[]
        }
    ]

    

    
    ### this is a dictionary that helps control which planewaves are correlated together
    ### Key is unique index of planewaves on referance station
    ### if the value is a list, then the items in the list should be unique_indeces of planewaves on other stations to NOT associate with this one
    planewave_exclusions = {
            275988:[223579, 100044, 131378],
            278749:[133731, 141099, 226374, 226375],#[133731, 141099, 226374, 226375, 333870, 333869],
            280199:[180886, 205404],#[180886,192177,192175,192174,192173,141874,14875, 205404, 124716, 314360, 314361, 134607],
            274969:[],
            275426:[],
            276467:[],
            
            274359:[],
            }
    
    ## these are SSPW that don't look right, but could be okay. They are treated as if there are mulitple SSPW are viable on that station
    suspicious_SSPW = {
            280199:[141874,141875,  335211, 134607, 314360,314361],
            
            275426:[99548,99549,18919],
            
            276467:[331787,331788,139439],
            
            274359:[98543],
            
            } 
    
    ### some SSPW are actually the same event, due to an error in previous processesing. This says which SSPW should be combined
    SSPW_to_combine = {
            275988: [ [275988,275989], [139072,139073], [131376,131377], [131379,131380] ],
            
            278749: [ [278749,278750], [148950,148951], [333869,333870], [203996,203997] ],
            
            280199: [ [141874,141875], [192173,192174,192175], [314360,314361]],
            
            274969: [ [274969,274968], [87738,87739],   [330364,330365], ],
            
            275426:[ [275426,275427], [24177,24178],  [14578,14579], [99548,99549], [330824,330823]],
            
            276467:[ [89274,89275], [331787,331788] ],
            
            274359:[ [23074,23075],  [87161,87162], [318581,318582], [240917,240918], [107475,107476], [77035,77036], [263303,263304], 
                    [252601, 252602], [329828,329829]],
            }

    antenna_calibrator = LBA_ant_calibrator(timeID)
    
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
    
    
    #### read SSPW ####
    print("reading SSPW")
    SSPW_data = read_SSPW_timeID(timeID, SSPW_folder, data_loc="/home/brian/processed_files", min_block=first_block, max_block=first_block+num_blocks, load_timeseries=False)
    SSPW_dict = SSPW_data["SSPW_dict"]
    ant_loc_dict = SSPW_data["ant_locations"]
    
    
    
    #### sort antennas and stations ####
    station_order = list(guess_timings.keys())## note this doesn't include reference station
    sorted_antenna_names = []
    station_to_antenna_index_dict = {}
    
    for sname in station_order + [referance_station]:
        s_id = Sname_to_SId_dict[ sname ]
        first_index = len(sorted_antenna_names)
        
        for ant_name in ant_loc_dict.keys():
            if int(ant_name[:3]) == s_id:
                sorted_antenna_names.append( ant_name )
                
        station_to_antenna_index_dict[sname] = (first_index, len(sorted_antenna_names))
    
    ant_locs = np.zeros( (len(sorted_antenna_names), 3))
    for i, ant_name in enumerate(sorted_antenna_names):
        ant_locs[i] = ant_loc_dict[ant_name]
    
    station_locations = {sname:ant_locs[station_to_antenna_index_dict[sname][0]] for sname in station_order + [referance_station]}
    station_to_antenna_index_list = [station_to_antenna_index_dict[sname] for sname in station_order]
    
    
    #### sort the delays guess, and account for station locations ####
    current_delays_guess = np.array([guess_timings[sname] for sname in station_order])
    if guess_is_from_curtain_plot:
        reference_propagation_delay = np.linalg.norm( guess_location - station_locations[referance_station] )/v_air
        for stat_i, sname in enumerate(station_order):
            propagation_delay =  np.linalg.norm( guess_location - station_locations[sname] )/v_air
            current_delays_guess[ stat_i ] -= propagation_delay - reference_propagation_delay
    original_delays = np.array( current_delays_guess )        
    
    
    
    
    #### open info from part 1 ####
#    input_fname = processed_data_dir + "/" + part1_input + '/out'
#    with open(input_fname, 'rb') as fin:
#        input_SSPW_sort = load(fin)

    input_manager = Part1_input_manager( part1_inputs, inject_sources )



    #### first we fit the known sources ####
    current_sources = []
    for knownID in known_sources:
        
        ref_SSPW_index, viable_SSPW_indeces = input_manager.known_source( knownID )
        
      
        
        print("prep fitting:", ref_SSPW_index)
            
        
        ## we have a source to add. Get info.
        source_exclusions = []
        if ref_SSPW_index in planewave_exclusions:
            source_exclusions = planewave_exclusions[ref_SSPW_index]
            
        SSPW_combine_list = []
        if ref_SSPW_index in SSPW_to_combine:
            SSPW_combine_list = SSPW_to_combine[ref_SSPW_index]
            
        suspicious_SSPW_list = []
        if ref_SSPW_index in suspicious_SSPW:
            suspicious_SSPW_list = suspicious_SSPW[ref_SSPW_index]
            
        location = known_source_locations[ref_SSPW_index]
            
            
        ## make source
        source_to_add = source_object(ref_SSPW_index,  viable_SSPW_indeces, source_exclusions, location, SSPW_combine_list, suspicious_SSPW_list )
        current_sources.append( source_to_add )


        polarity = known_polarizations[ref_SSPW_index]
        SSPW_associations = known_SSPW_associations[ref_SSPW_index]

            
        source_to_add.prep_for_fitting_knownFit(polarity, SSPW_associations )
        
        
    
    num_antennas = len(sorted_antenna_names)
    num_measurments =num_antennas*len(current_sources)
    num_delays = len(station_order)
        
    def objective_fun(sol):
        workspace_sol = np.zeros(num_measurments, dtype=np.double)
        delays = sol[:num_delays]
        ant_i = 0
        param_i = num_delays
        for PSE in current_sources:
            
            PSE.try_location_LS(delays, sol[param_i:param_i+4], workspace_sol[ant_i:ant_i+num_antennas])
            
            ant_i += num_antennas
            param_i += 4
            
        filter = np.logical_not( np.isfinite(workspace_sol) )
        workspace_sol[ filter ]  = 0.0
            
        return workspace_sol
    #        workspace_sol *= workspace_sol
    #        return np.sum(workspace_sol)
        
    
    def objective_jac( sol):
        workspace_jac = np.zeros((num_measurments, num_delays+4*len(current_sources)), dtype=np.double)
    
            
        delays = sol[:num_delays]
        ant_i = 0
        param_i = num_delays
        for PSE in current_sources:
            
            PSE.try_location_JAC(delays, sol[param_i:param_i+4],  workspace_jac[ant_i:ant_i+num_antennas, param_i:param_i+4],  
                                 workspace_jac[ant_i:ant_i+num_antennas, 0:num_delays])
            
            filter = np.logical_not( np.isfinite(workspace_jac[ant_i:ant_i+num_antennas, param_i+3]) )
            workspace_jac[ant_i:ant_i+num_antennas, param_i:param_i+4][filter] = 0.0
            workspace_jac[ant_i:ant_i+num_antennas, 0:num_delays][filter] = 0.0
            
            ant_i += num_antennas
            param_i += 4
        
        return workspace_jac
        
    
    solution = np.zeros(  num_delays+4*len(current_sources) )
    solution[:num_delays] = current_delays_guess
    param_i = num_delays
    for PSE in current_sources:
        solution[param_i:param_i+4] = PSE.guess_XYZT
        param_i += 4
        
    for source in current_sources:
        source.insert_noise( 0.0 )
            
    fit_res = least_squares(objective_fun, solution, jac=objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
    solution = fit_res.x
        
    
    solutions = np.empty( (num_runs, len(solution)) )
    for i in range(num_runs):
        print("step",i)
        
        for source in current_sources:
            source.insert_noise( timing_error )
            
        fit_res = least_squares(objective_fun, solution, jac=objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
        solutions[i] = fit_res.x
        
        
        
    print("station timing offset errors")
    for stat_i, sname in enumerate(station_order):
        print(sname, ":", np.std(solutions[:,stat_i]))
    
    
    
    print()
    print("average location error")
    param_i = num_delays
    ave_locs = np.zeros( (num_runs,4) )
    for PSE in current_sources:
        
        ave_locs[:,0] += solutions[:,param_i]
        ave_locs[:,1] += solutions[:,param_i+1]
        ave_locs[:,2] += solutions[:,param_i+2]
        ave_locs[:,3] += solutions[:,param_i+3]
        
        param_i += 4
        
    ave_locs /= len(current_sources)
    print( np.std(ave_locs[:,0]), np.std(ave_locs[:,1]), np.std(ave_locs[:,2]), np.std(ave_locs[:,3]) )
    
    print()
    print("relative location errors")
    param_i = num_delays
    for PSE in current_sources:
        solutions[:,param_i] -= ave_locs[:,0]
        solutions[:,param_i+1] -= ave_locs[:,1]
        solutions[:,param_i+2] -= ave_locs[:,2]
        solutions[:,param_i+3] -= ave_locs[:,3]
        
        print(PSE.SSPW.unique_index, ":", np.std(solutions[:,param_i]), np.std(solutions[:,param_i+1]), np.std(solutions[:,param_i+2]), np.std(solutions[:,param_i+3]) )
        param_i += 4
    
    
    
    
        
        
        
        
        
        
        
    
    
    
    
    