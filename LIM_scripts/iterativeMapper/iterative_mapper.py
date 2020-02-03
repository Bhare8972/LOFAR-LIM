#!/usr/bin/env python3

from os import mkdir, listdir
from os.path import isdir, isfile
#import time
from collections import deque
import datetime

import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import brute, least_squares

import h5py

from LoLIM.utilities import processed_data_dir, v_air, antName_is_even, BoundingBox_collision, logger
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1, read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
from LoLIM.findRFI import window_and_filter
from LoLIM.signal_processing import remove_saturation, data_cut_inspan, locate_data_loss


from LoLIM.iterativeMapper.cython_utils import parabolic_fitter, planewave_locator_helper, autoadjusting_upsample_and_correlate, \
    pointsource_locator, abs_max

def do_nothing(*A, **B):
    pass

############# HEADER TOOLS ####################

def read_planewave_fits(cal_folder, station_name):
    """returns a dictionary, key is antenna name, value is RMS fit value of planewaves. If folder doesn't exist, returns empty dictionary.
    If station_name ends with _secDer, then it returns the average second derivatives of the peaks instead"""
    
    fname = cal_folder + '/antenna_fits/' + station_name + '.txt'
    
    out = {}
    
    if not isfile( fname ):
        print("planewave RMS file not available:", fname )
        print("  using default values")
        return {}
    
    with open(fname, 'r') as fin:
        for line in fin:
            ant_name, RMSFit, *throw = line.split()
            RMSFit = float(RMSFit)
            out[ant_name] = RMSFit
    
    return out

class make_header:
    def __init__(self, timeID, initial_datapoint, 
                 station_delays_fname, additional_antenna_delays_fname, bad_antennas_fname, pol_flips_fname):
        
        self.timeID = timeID
        self.initial_datapoint = initial_datapoint
        
        self.max_antennas_per_station = 6
        self.referance_station = None
        
        self.station_delays_fname = station_delays_fname
        self.additional_antenna_delays_fname = additional_antenna_delays_fname
        self.bad_antennas_fname = bad_antennas_fname
        self.pol_flips_fname = pol_flips_fname
        
        self.stations_to_exclude = []
        
        self.blocksize = 2**16
        
        self.remove_saturation = True
        self.remove_RFI = True
        self.positive_saturation = 2046
        self.negative_saturation = -2047
        self.saturation_removal_length = 50
        self.saturation_half_hann_length = 50
        
        self.hann_window_fraction = 0.1
        
        self.num_zeros_dataLoss_Threshold = 10
        
        self.min_amplitude = 15
        
        self.upsample_factor = 8
        self.max_events_perBlock =  500# 100
        self.min_pulse_length_samples = 20
        self.erasure_length = 20 ## num points centered on peak to not image again
        
        self.guess_distance = 10000
        
        
        self.kalman_devations_toSearch = 3 # +/- on either side of prediction
        
        self.pol_flips_are_bad = True
        
        self.antenna_RMS_info = "find_calibration_out"
        self.default_expected_RMS = 1.0E-9
        self.max_planewave_RMS = 1.0E-9
        self.stop_chi_squared = 100.0
        
        self.max_minimize_itters = 100
        self.minimize_ftol = 3.0E-16
        self.minimize_xtol = 3.0E-16
        self.minimize_gtol = 3.0E-16
        
    def run(self, output_dir, dir_is_organized=True):
        
        processed_data_folder = processed_data_dir( self.timeID )
        
        if dir_is_organized:
            output_dir = processed_data_folder +'/' + output_dir
        
        polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + self.pol_flips_fname )
        bad_antennas = read_bad_antennas( processed_data_folder + '/' + self.bad_antennas_fname )
        additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + self.additional_antenna_delays_fname )
        station_timing_offsets = read_station_delays( processed_data_folder+'/'+ self.station_delays_fname )
        
        raw_fpaths = filePaths_by_stationName( self.timeID )

        self.station_data_files = []
        ref_station_i = None
        referance_station_set = None
        station_i = 0
        self.posix_timestamp = None
        
        for station, fpaths  in raw_fpaths.items():
            if (station in station_timing_offsets) and (station not in self.stations_to_exclude ):
                print('opening', station)
                
                raw_data_file = MultiFile_Dal1(fpaths, polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays, pol_flips_are_bad=self.pol_flips_are_bad)
                self.station_data_files.append( raw_data_file )
                
                raw_data_file.set_station_delay( station_timing_offsets[station] )
                raw_data_file.find_and_set_polarization_delay()
                
                if self.posix_timestamp is None:
                    self.posix_timestamp = raw_data_file.get_timestamp()
                elif self.posix_timestamp != raw_data_file.get_timestamp():
                    print("ERROR: POSIX timestamps are different")
                    quit()
                
                if self.referance_station is None:
                    if (referance_station_set is None) or ( int(station[2:]) < int(referance_station_set[2:]) ): ## if no referance station, use one with smallist number
                        ref_station_i = station_i
                        referance_station_set = station
                elif self.referance_station == station:
                    ref_station_i = station_i
                    referance_station_set = station
                        
                station_i += 1
                
        ## set reference station first
        tmp = self.station_data_files[0]
        self.station_data_files[0] = self.station_data_files[ ref_station_i ]
        self.station_data_files[ ref_station_i ] = tmp
        self.referance_station = referance_station_set
        
        ## organize antenana info
        self.station_antenna_indeces = [] ## 2D list. First index is station_i, second is a local antenna index, value is station antenna index
        self.station_antenna_RMS = [] ## same as above, but value is RMS
        self.num_total_antennas = 0
        current_ant_i = 0
        for station_i,stationFile in enumerate(self.station_data_files):
            ant_names = stationFile.get_antenna_names()
            num_evenAntennas = int( len(ant_names)/2 )
            
            ## get RMS planewave fits
            if self.antenna_RMS_info is not None:
                
                antenna_RMS_dict = read_planewave_fits(processed_data_folder+'/'+self.antenna_RMS_info, stationFile.get_station_name() )
            else:
                antenna_RMS_dict = {}
                
                
            antenna_RMS_array = np.array([ antenna_RMS_dict[ant_name] if ant_name in antenna_RMS_dict else self.default_expected_RMS  for ant_name in ant_names if antName_is_even(ant_name)  ])
            antenna_indeces = np.arange(num_evenAntennas)*2
            
            ## remove bad antennas and sort
            ant_is_good = np.isfinite( antenna_RMS_array )
            antenna_RMS_array = antenna_RMS_array[ ant_is_good ]
            antenna_indeces = antenna_indeces[ ant_is_good ]
            
            sorter = np.argsort( antenna_RMS_array )
            antenna_RMS_array = antenna_RMS_array[ sorter ]
            antenna_indeces = antenna_indeces[ sorter ]
            
            ## select only the best!
            antenna_indeces = antenna_indeces[:self.max_antennas_per_station]
            antenna_RMS_array = antenna_RMS_array[:self.max_antennas_per_station]
                
            ## fill info
            self.station_antenna_indeces.append( antenna_indeces )
            self.station_antenna_RMS.append( antenna_RMS_array )
            
            self.num_total_antennas += len(antenna_indeces)
            
            current_ant_i += len(antenna_indeces)
        
        ### output to header!
        if not isdir(output_dir):
            mkdir(output_dir)
            
        logging_folder = output_dir + '/logs_and_plots'
        if not isdir(logging_folder):
            mkdir(logging_folder)
            
        header_outfile = h5py.File(output_dir + "/header.h5", "w")
        header_outfile.attrs["timeID"] = self.timeID
        header_outfile.attrs["initial_datapoint"] = self.initial_datapoint
        header_outfile.attrs["max_antennas_per_station"] = self.max_antennas_per_station
        header_outfile.attrs["referance_station"] = self.station_data_files[0].get_station_name()
        header_outfile.attrs["station_delays_fname"] = self.station_delays_fname
        header_outfile.attrs["additional_antenna_delays_fname"] = self.additional_antenna_delays_fname
        header_outfile.attrs["bad_antennas_fname"] = self.bad_antennas_fname
        header_outfile.attrs["pol_flips_fname"] = self.pol_flips_fname
        header_outfile.attrs["blocksize"] = self.blocksize
        header_outfile.attrs["remove_saturation"] = self.remove_saturation
        header_outfile.attrs["remove_RFI"] = self.remove_RFI
        header_outfile.attrs["positive_saturation"] = self.positive_saturation
        header_outfile.attrs["negative_saturation"] = self.negative_saturation
        header_outfile.attrs["saturation_removal_length"] = self.saturation_removal_length
        header_outfile.attrs["saturation_half_hann_length"] = self.saturation_half_hann_length
        header_outfile.attrs["hann_window_fraction"] = self.hann_window_fraction
        header_outfile.attrs["num_zeros_dataLoss_Threshold"] = self.num_zeros_dataLoss_Threshold
        header_outfile.attrs["min_amplitude"] = self.min_amplitude
        header_outfile.attrs["upsample_factor"] = self.upsample_factor
        header_outfile.attrs["max_events_perBlock"] = self.max_events_perBlock
        header_outfile.attrs["min_pulse_length_samples"] = self.min_pulse_length_samples
        header_outfile.attrs["erasure_length"] = self.erasure_length
        header_outfile.attrs["guess_distance"] = self.guess_distance
        header_outfile.attrs["kalman_devations_toSearch"] = self.kalman_devations_toSearch
        header_outfile.attrs["pol_flips_are_bad"] = self.pol_flips_are_bad
        header_outfile.attrs["antenna_RMS_info"] = self.antenna_RMS_info
        header_outfile.attrs["default_expected_RMS"] = self.default_expected_RMS
        header_outfile.attrs["max_planewave_RMS"] = self.max_planewave_RMS
        header_outfile.attrs["stop_chi_squared"] = self.stop_chi_squared
        header_outfile.attrs["max_minimize_itters"] = self.max_minimize_itters
        header_outfile.attrs["minimize_ftol"] = self.minimize_ftol
        header_outfile.attrs["minimize_xtol"] = self.minimize_xtol
        header_outfile.attrs["minimize_gtol"] = self.minimize_gtol
        
        header_outfile.attrs["refStat_delay"] = station_timing_offsets[  self.station_data_files[0].get_station_name()  ]
        header_outfile.attrs["refStat_timestamp"] = self.posix_timestamp#self.station_data_files[0].get_timestamp()
        header_outfile.attrs["refStat_sampleNumber"] = self.station_data_files[0].get_nominal_sample_number()
        
        header_outfile.attrs["stations_to_exclude"] = np.array(self.stations_to_exclude, dtype='S')
        
        header_outfile.attrs["polarization_flips"] = np.array(polarization_flips, dtype='S')
        
        header_outfile.attrs["num_stations"] = len(self.station_data_files)
        header_outfile.attrs["num_antennas"] = self.num_total_antennas
        
        for stat_i, (stat_file, antenna_indeces, antenna_RMS) in enumerate(zip(self.station_data_files, self.station_antenna_indeces, self.station_antenna_RMS)):
            
            station_group = header_outfile.create_group( str(stat_i) )
            station_group.attrs["sname"] = stat_file.get_station_name()
            station_group.attrs["num_antennas"] = len( antenna_indeces )
            
            locations = stat_file.get_LOFAR_centered_positions()
            total_delays = stat_file.get_total_delays()
            ant_names = stat_file.get_antenna_names()
            
            for ant_i, (stat_index, RMS) in enumerate(zip(antenna_indeces, antenna_RMS)):
                
                antenna_group = station_group.create_group( str(ant_i) )
                antenna_group.attrs['antenna_name'] = ant_names[stat_index]
                antenna_group.attrs['location'] = locations[stat_index]
                antenna_group.attrs['delay'] = total_delays[stat_index]
                antenna_group.attrs['planewave_RMS'] = RMS
                
class read_header:
    class antenna_info_object:
        def __init__(self):
            self.sname = None
            self.ant_name = None
            self.delay = None
            self.planewave_RMS = None
            self.location = None
    
    def __init__(self, input_folder, timeID=None):
        
        if timeID is not None:
            processed_data_folder = processed_data_dir( timeID )
            input_folder = processed_data_folder + '/' + input_folder
        
        self.input_folder = input_folder
        
        header_infile = h5py.File(input_folder + "/header.h5", "r")
        
        #### input settings ####
        self.timeID = header_infile.attrs['timeID']
        self.initial_datapoint = header_infile.attrs["initial_datapoint"]
        self.max_antennas_per_station = header_infile.attrs["max_antennas_per_station"]
        if "referance_station" in header_infile.attrs:
            self.referance_station = header_infile.attrs["referance_station"]#.decode()
        
        self.station_delays_fname = header_infile.attrs["station_delays_fname"]#.decode()
        self.additional_antenna_delays_fname = header_infile.attrs["additional_antenna_delays_fname"]#.decode()
        self.bad_antennas_fname = header_infile.attrs["bad_antennas_fname"]#.decode()
        self.pol_flips_fname = header_infile.attrs["pol_flips_fname"]#.decode()
        
        self.blocksize = header_infile.attrs["blocksize"]
        self.remove_saturation = header_infile.attrs["remove_saturation"]
        self.remove_RFI = header_infile.attrs["remove_RFI"]
        self.positive_saturation = header_infile.attrs["positive_saturation"]
        self.negative_saturation = header_infile.attrs["negative_saturation"]
        self.saturation_removal_length = header_infile.attrs["saturation_removal_length"]
        self.saturation_half_hann_length = header_infile.attrs["saturation_half_hann_length"]
        self.hann_window_fraction = header_infile.attrs["hann_window_fraction"]
        
        self.num_zeros_dataLoss_Threshold = header_infile.attrs["num_zeros_dataLoss_Threshold"]
        self.min_amplitude = header_infile.attrs["min_amplitude"]
        self.upsample_factor = header_infile.attrs["upsample_factor"]
        self.max_events_perBlock = header_infile.attrs["max_events_perBlock"]
        self.min_pulse_length_samples = header_infile.attrs["min_pulse_length_samples"]
        self.erasure_length = header_infile.attrs["erasure_length"]
        
        self.guess_distance = header_infile.attrs["guess_distance"]
        
        self.kalman_devations_toSearch = header_infile.attrs["kalman_devations_toSearch"]
        
        self.pol_flips_are_bad = header_infile.attrs["pol_flips_are_bad"]
        
        self.antenna_RMS_info = header_infile.attrs["antenna_RMS_info"]
        self.default_expected_RMS = header_infile.attrs["default_expected_RMS"]
        self.max_planewave_RMS = header_infile.attrs["max_planewave_RMS"]
        self.stop_chi_squared = header_infile.attrs["stop_chi_squared"]
        
        self.max_minimize_itters = header_infile.attrs["max_minimize_itters"]
        self.minimize_ftol = header_infile.attrs["minimize_ftol"]
        self.minimize_xtol = header_infile.attrs["minimize_xtol"]
        self.minimize_gtol = header_infile.attrs["minimize_gtol"]
        
        self.refStat_delay = header_infile.attrs["refStat_delay"] 
        self.refStat_timestamp =  header_infile.attrs["refStat_timestamp"]
        self.refStat_sampleNumber = header_infile.attrs["refStat_sampleNumber"]
        
        
        self.stations_to_exclude = header_infile.attrs["stations_to_exclude"]
        self.stations_to_exclude = [sname.decode() for sname in self.stations_to_exclude]
        
        #### processed info ####
        self.pol_flips = header_infile.attrs["polarization_flips"]
        self.pol_flips = [name.decode() for name in self.pol_flips]
        
        self.num_stations = header_infile.attrs["num_stations"]
        self.num_total_antennas = header_infile.attrs["num_antennas"]
        
        self.station_names = [ None ]*self.num_stations
        self.antenna_info = [ None ]*self.num_stations ## each item will be a list of antenna info
        
        for stat_i, stat_group in header_infile.items():
            stat_i = int(stat_i)
            sname = stat_group.attrs["sname"]#.decode()
            num_antennas = stat_group.attrs["num_antennas"]
            
            ant_info = [ None ]*num_antennas
            for ant_i, ant_group in stat_group.items():
                ant_i = int(ant_i)
                
                new_antenna_object = self.antenna_info_object()
                new_antenna_object.sname = sname
                new_antenna_object.ant_name = ant_group.attrs["antenna_name"]#.decode()
                new_antenna_object.delay = ant_group.attrs["delay"]
                new_antenna_object.planewave_RMS = ant_group.attrs["planewave_RMS"]
                new_antenna_object.location = ant_group.attrs["location"]
                
                ant_info[ant_i] = new_antenna_object
                
            self.station_names[ stat_i ] = sname 
            self.antenna_info[ stat_i ] = ant_info
            
    def print_settings(self, print_func=print):
        print_func('timeID:', self.timeID)
        print_func('initial_datapoint:', self.initial_datapoint)
        print_func('max_antennas_per_station:', self.max_antennas_per_station)
        print_func('referance_station:', self.referance_station)
        
        print_func('station_delays_fname:', self.station_delays_fname)
        print_func('additional_antenna_delays_fname:', self.additional_antenna_delays_fname)
        print_func('bad_antennas_fname:', self.bad_antennas_fname)
        print_func('pol_flips_fname:', self.pol_flips_fname)
        
        print_func('blocksize:', self.blocksize)
        print_func('remove_saturation:', self.remove_saturation)
        print_func('remove_RFI:', self.remove_RFI)
        print_func('positive_saturation:', self.positive_saturation)
        print_func('negative_saturation:', self.negative_saturation)
        print_func('saturation_removal_length:', self.saturation_removal_length)
        print_func('saturation_half_hann_length:', self.saturation_half_hann_length)
        print_func('hann_window_fraction:', self.hann_window_fraction)
        
        print_func('num_zeros_dataLoss_Threshold:', self.num_zeros_dataLoss_Threshold)
        print_func('min_amplitude:', self.min_amplitude)
        print_func('upsample_factor:', self.upsample_factor)
        print_func('max_events_perBlock:', self.max_events_perBlock)
        print_func('min_pulse_length_samples:', self.min_pulse_length_samples)
        print_func('erasure_length:', self.erasure_length)
        
        print_func('guess_distance:', self.guess_distance)
        
        print_func('kalman_devations_toSearch:', self.kalman_devations_toSearch)
        
        print_func('pol_flips_are_bad:', self.pol_flips_are_bad)
        
        print_func('antenna_RMS_info:', self.antenna_RMS_info)
        print_func('default_expected_RMS:', self.default_expected_RMS)
        print_func('max_planewave_RMS:', self.max_planewave_RMS)
        print_func('stop_chi_squared:', self.stop_chi_squared)
        
        print_func('max_minimize_itters:', self.max_minimize_itters)
        print_func('minimize_ftol:', self.minimize_ftol)
        print_func('minimize_xtol:', self.minimize_xtol)
        print_func('minimize_gtol:', self.minimize_gtol)
        
        print_func('stations_to_exclude:', self.stations_to_exclude)
        
    def print_all_info(self, print_func=print):
        self.print_settings(print_func)
        
        print_func('refStat_delay:', self.refStat_delay)
        print_func('refStat_timestamp:', self.refStat_timestamp)
        print_func('refStat_sampleNumber:', self.refStat_sampleNumber)
        
        print_func('pol_flips:', self.pol_flips)
        
        print_func('num_stations:', self.num_stations)
        print_func('num_total_antennas:', self.num_total_antennas)
        
        print_func()
        for sname, antennas in zip( self.station_names, self.antenna_info ):
            print_func(sname)
            for ant in antennas:
                print_func( " ", ant.ant_name )
                print_func( "    delay:", ant.delay )
                print_func( "    RMS:", ant.planewave_RMS )
                print_func( "    loc:", ant.location )
                
    def resave_header(self, output_folder):
        """Used to adjust settings of mapper. Open header, edit settings, and use this method to re-save out header to a different folder for a new mapping run"""
        new_header = make_header( self.timeID, self.initial_datapoint, 
             self.station_delays_fname, self.additional_antenna_delays_fname, self.bad_antennas_fname, self.pol_flips_fname )
        
        new_header.max_antennas_per_station = self.max_antennas_per_station
        new_header.referance_station = self.referance_station
        new_header.stations_to_exclude =self.stations_to_exclude
        new_header.blocksize = self.blocksize
        new_header.remove_saturation = self.remove_saturation
        new_header.remove_RFI = self.remove_RFI
        new_header.positive_saturation = self.positive_saturation
        new_header.negative_saturation = self.negative_saturation
        new_header.saturation_removal_length = self.saturation_removal_length
        new_header.saturation_half_hann_length = self.saturation_half_hann_length
        new_header.hann_window_fraction = self.hann_window_fraction
        new_header.num_zeros_dataLoss_Threshold = self.num_zeros_dataLoss_Threshold
        new_header.min_amplitude = self.min_amplitude
        new_header.upsample_factor = self.upsample_factor
        new_header.max_events_perBlock = self.max_events_perBlock
        new_header.min_pulse_length_samples = self.min_pulse_length_samples
        new_header.erasure_length = self.erasure_length
        new_header.guess_distance = self.guess_distance 
        new_header.kalman_devations_toSearch = self.kalman_devations_toSearch
        new_header.pol_flips_are_bad = self.pol_flips_are_bad
        new_header.antenna_RMS_info = self.antenna_RMS_info
        new_header.default_expected_RMS = self.default_expected_RMS
        new_header.max_planewave_RMS = self.max_planewave_RMS
        new_header.stop_chi_squared = self.stop_chi_squared 
        new_header.max_minimize_itters = self.max_minimize_itters
        new_header.minimize_ftol = self.minimize_ftol
        new_header.minimize_xtol = self.minimize_xtol
        new_header.minimize_gtol = self.minimize_gtol
        
        new_header.run( output_folder )
        
    def next_log_file(self):
        """return the filename (including directory) of the next log file that should be saved to"""
        
        file_number = 0
        fname = self.input_folder + "/logs_and_plots/log_run_"+str(file_number)+".txt"
        while isfile(fname):
            file_number += 1
            fname = self.input_folder + "/logs_and_plots/log_run_"+str(file_number)+".txt"
            
        return fname
        
    
    ##### INFO TO LOAD DATA  ####
    
    def timestamp_as_datetime(self):
        return datetime.datetime.fromtimestamp( self.refStat_timestamp, tz=datetime.timezone.utc )
    
    class PSE_source:
        def __init__(self):
            self.ID = None
            self.uniqueID = None
            self.block = None
            self.XYZT = np.empty(4, dtype=np.double)
            self.RMS = None
            self.RedChiSquared = None
            self.numRS = None
            self.refAmp = None
            self.numThrows = None
            self.cov_matrix = np.empty( (3,3), dtype=np.double )
            self.__cov_eig__ = None
            
        def in_BoundingBox(self, BB):
            inX = BB[0,0] <= self.XYZT[0] <= BB[0,1]
            inY = BB[1,0] <= self.XYZT[1] <= BB[1,1]
            inZ = BB[2,0] <= self.XYZT[2] <= BB[2,1]
            inT = BB[3,0] <= self.XYZT[3] <= BB[3,1]
            return inX and inY and inZ and inT
        
        def cov_eig(self):
            """returns np.eig( cov_matrix ). Returns None if cov_matrix isn't finite"""
            if self.__cov_eig__ is None:
                if not np.all( np.isfinite(self.cov_matrix) ):
                    self.__cov_eig__ = None
                else:
                    self.__cov_eig__ = np.linalg.eig(self.cov_matrix)
            return self.__cov_eig__
        
        def max_sqrtEig(self):
            cov_eigs = self.cov_eig()
            if cov_eigs is None:
                return np.inf
            else:
                return np.sqrt(np.max(cov_eigs[0]))
        
    def load_data_as_sources(self, maxRMS=None, maxRChi2=None, minRS=None, minAmp=None, bounds=None ):
        """returns sources as deques"""
        out_sources = deque()
        
        folder = self.input_folder
        
        file_names = [fname for fname in listdir(folder) if fname.endswith(".h5") and not fname.startswith('header')]
        for fname in file_names:
            infile = h5py.File(folder+'/'+fname, "r")
        
            for block_dataset in infile.values():
                block_i = block_dataset.attrs['block']
                print('opening block', block_i)
                
                if bounds is not None:
                    block_bounds = block_dataset.attrs['bounds']
                    if not BoundingBox_collision(bounds, block_bounds):
                        continue
                    
                do_num_throws = 'numThrows' in block_dataset.dtype.fields.keys()
                    
                for source_info in block_dataset:
                    new_source = self.PSE_source()
                    new_source.ID = source_info['ID']
                    new_source.uniqueID = block_i*self.max_events_perBlock + new_source.ID
                    new_source.block = block_i
                    new_source.XYZT[0] = source_info['x']
                    new_source.XYZT[1] = source_info['y']
                    new_source.XYZT[2] = source_info['z']
                    new_source.XYZT[3] = source_info['t']
                    new_source.RMS = source_info['RMS']
                    new_source.RedChiSquared = source_info['RedChiSquared']
                    new_source.numRS = source_info['numRS']
                    new_source.refAmp = source_info['refAmp']
                    if do_num_throws:
                        new_source.numThrows = source_info['numThrows']
                    else:
                        new_source.numThrows = 0
                    
                    new_source.cov_matrix[0,0] = source_info['covXX']
                    new_source.cov_matrix[0,1] = source_info['covXY']
                    new_source.cov_matrix[0,2] = source_info['covXZ']
                    
                    new_source.cov_matrix[1,0] = new_source.cov_matrix[0,1]
                    new_source.cov_matrix[1,1] = source_info['covYY']
                    new_source.cov_matrix[1,2] = source_info['covYZ']
                    
                    new_source.cov_matrix[2,0] = new_source.cov_matrix[0,2]
                    new_source.cov_matrix[2,1] = new_source.cov_matrix[1,2]
                    new_source.cov_matrix[2,2] = source_info['covZZ']
                    
                    if maxRMS is not None and new_source.RMS > maxRMS:
                        continue
                    
                    if maxRChi2 is not None and new_source.RedChiSquared > maxRChi2:
                        continue
                    
                    if minRS is not None and new_source.numRS < minRS:
                        continue
                    
                    if minAmp is not None and new_source.refAmp < minAmp:
                        continue
                    
                    if bounds is not None and not new_source.in_BoundingBox( bounds ):
                        continue
                    
                    out_sources.append( new_source )
                    
        return out_sources

############# DATA MANAGER ####################
    
class raw_data_manager:
    def __init__(self, header, print_func=do_nothing):
        
        #### variables ####
        self.blocksize = header.blocksize
        self.num_dataLoss_zeros = header.num_zeros_dataLoss_Threshold
        self.positive_saturation = header.positive_saturation
        self.negative_saturation = header.negative_saturation
        self.saturation_removal_length = header.saturation_removal_length
        self.saturation_half_hann_length  = header.saturation_half_hann_length
        self.half_hann_safety_region = int(header.hann_window_fraction*self.blocksize) + 5
        self.remove_saturation = header.remove_saturation
        self.remove_RFI =        header.remove_RFI
        
        raw_fpaths = filePaths_by_stationName( header.timeID )
        
        #### get data from files ####
        ## objects that are length of number of stations
        self.station_data_files = []
        self.RFI_filters = []
        self.station_to_antSpan = np.empty( [header.num_stations,2], dtype=np.int )
        self.station_locations = np.empty( [header.num_stations,3], dtype=np.double )
        self.is_RS = np.empty( header.num_stations, dtype=bool )
        
        ## objects by number of antennas
        self.station_antenna_index = np.empty(header.num_total_antennas, dtype=np.int)
        self.antenna_expected_RMS = np.empty(header.num_total_antennas, dtype=np.double)
        self.antI_to_station = np.empty(header.num_total_antennas, dtype=np.int)
        self.antenna_locations = np.empty( [header.num_total_antennas,3], dtype=np.double )
        self.antenna_delays = np.empty(header.num_total_antennas, dtype=np.double)
        self.antenna_names = []
        
        next_ant_i = 0
        for sname, antenna_list in zip(header.station_names, header.antenna_info):
            print_func('opening', sname)
            stat_i = len( self.station_data_files )
            
            raw_data_file = MultiFile_Dal1(raw_fpaths[sname], polarization_flips=header.pol_flips, pol_flips_are_bad=header.pol_flips_are_bad)
            data_filter = window_and_filter(timeID=header.timeID, sname=sname, half_window_percent=header.hann_window_fraction)
            self.station_data_files.append( raw_data_file )
            self.RFI_filters.append( data_filter )
            
            if data_filter.blocksize != self.blocksize:
                print_func("RFI info from station", sname, "has wrong block size of", data_filter.blocksize, '. expected', self.blocksize)
                quit()
                
            first_ant_i = next_ant_i
            
            station_antenna_names = raw_data_file.get_antenna_names()
            for ant in antenna_list:
                self.antenna_names.append( ant.ant_name )
                self.station_antenna_index[ next_ant_i ] = station_antenna_names.index( ant.ant_name )
                self.antenna_expected_RMS[ next_ant_i ] = ant.planewave_RMS
                self.antenna_locations[ next_ant_i ] = ant.location
                self.antenna_delays[ next_ant_i ] = ant.delay
                next_ant_i += 1
            
            self.antI_to_station[ first_ant_i:next_ant_i ] = stat_i
            self.station_locations[ stat_i ] = np.average( self.antenna_locations[ first_ant_i:next_ant_i ], axis=0 )
            self.station_to_antSpan[ stat_i, 0] = first_ant_i
            self.station_to_antSpan[ stat_i, 1] = next_ant_i
            self.is_RS[ stat_i ] = sname[:2]=='RS'       
        
        #### allocate memory ####
        self.raw_data = np.empty( [header.num_total_antennas,self.blocksize], dtype=np.complex )
        self.tmp_workspace = np.empty( self.blocksize, dtype=np.complex )
        self.antenna_data_loaded = np.zeros( header.num_total_antennas, dtype=bool )
        self.starting_time = np.empty(header.num_total_antennas, dtype=np.double) ## real time of the first sample in raw data
            
        self.referance_ave_delay = np.average( self.antenna_delays[self.station_to_antSpan[0,0]:self.station_to_antSpan[0,1]] )
#        self.antenna_delays -= self.referance_ave_delay
        self.staturation_removals = [ [] ]*header.num_total_antennas
        self.data_loss_spans = [ [] ]*header.num_total_antennas

    def get_station_location(self, station_i):
        return self.station_locations[ station_i ]
    
    def antI_to_sname(self, ant_i):
        station_i = self.antI_to_station[ ant_i ]
        stationFile = self.station_data_files[ station_i ]
        return stationFile.get_station_name()
    
    def antI_to_antName(self, ant_i):
        return self.antenna_names[ ant_i ]
    
    def statI_to_sname(self, stat_i):
        return self.station_data_files[ stat_i ].get_station_name()
    
    def get_antI_span(self, station_i):
        return self.station_to_antSpan[ station_i ]
    
    def get_antenna_locations(self, station_i):
        span = self.station_to_antSpan[ station_i ]
        return self.antenna_locations[ span[0]:span[1] ]
    
    def open_antenna_SampleNumber(self, antenna_i, sample_number):
        """opens the data for a station, where every antenna starts at the given sample number. Returns the data block for this station, and the start times for each antenna"""
        
#        print('opening:', antenna_i)
        
        station_i = self.antI_to_station[ antenna_i ]
        stationFile = self.station_data_files[ station_i ]
        data_filter = self.RFI_filters[ station_i ]
        
        station_antenna_index = self.station_antenna_index[ antenna_i ]
        TMP = stationFile.get_data(sample_number, self.blocksize, antenna_index=station_antenna_index  )
            
        if len(TMP) != self.blocksize:
            MSG = 'data length wrong. is: ' + str(len(TMP)) + " should be " + str(self.blocksize) + ". sample: " + \
              str(sample_number) + ". antenna: " + str(station_antenna_index) + ". station: " + stationFile.get_station_name()
              
            raise Exception( MSG )
        
        self.data_loss_spans[ antenna_i ], DL = locate_data_loss(TMP, self.num_dataLoss_zeros)
            
        self.tmp_workspace[:] = TMP

        if self.remove_saturation:
            self.staturation_removals[antenna_i] = remove_saturation( self.tmp_workspace, self.positive_saturation, self.negative_saturation, self.saturation_removal_length, self.saturation_half_hann_length )
        else:
            self.staturation_removals[antenna_i] = []
    
        if self.remove_RFI:
            self.raw_data[ antenna_i ] = data_filter.filter( self.tmp_workspace )
        else:
            self.raw_data[ antenna_i ] = self.tmp_workspace
            
        self.starting_time[ antenna_i ] = sample_number*5E-9 - self.antenna_delays[ antenna_i ]
            
        self.antenna_data_loaded[ antenna_i ] = True
        
        return self.raw_data[ antenna_i ], self.starting_time[ antenna_i ]
        
    def open_station_SampleNumber(self, station_i, sample_number):
        """opens the data for a station, where every antenna starts at the given sample number. Returns the data block for this station, and the start times for each antenna"""
        
        antI_span = self.station_to_antSpan[ station_i ]
        
        for antI in range(antI_span[0],antI_span[1]):
            self.open_antenna_SampleNumber( antI, sample_number )
            
        return self.raw_data[ antI_span[0]:antI_span[1] ],   self.starting_time[antI_span[0]:antI_span[1]]
                
            
    def get_station_safetySpan(self, station_i, start_time, stop_time):
        """returns the data and start times for a station (same as open_station_SampleNumber), insuring that the data between start_time and stop_time is valid for
        all antennas in this station. (assume the duration is smaller then a block). If new data is opened, then start_time is aligned to just after the beginning of the file."""
        
        antI_span = self.station_to_antSpan[ station_i ]
        
        do_load = False
        if not self.station_data_loaded[ station_i ]:
            do_load = True
        else:
            present_start_time = np.max( self.starting_time[antI_span[0]:antI_span[1]] ) + self.half_hann_safety_region*5.0E-9
            is_safe = present_start_time < start_time
            
            present_end_time = np.min( self.starting_time[antI_span[0]:antI_span[1]] ) + self.blocksize*5.0E-9 - self.half_hann_safety_region*5.0E-9
            is_safe = is_safe and present_end_time>stop_time
            
            do_load = not is_safe
        
        if do_load:
            antenna_delays = self.antenna_delays[ antI_span[0]:antI_span[1] ]
            start_sample_number = int( (start_time + min( antenna_delays ))/5.0E-9 ) + 1
            start_sample_number -= self.half_hann_safety_region
            
            self.open_station_SampleNumber( station_i, start_sample_number )
            
        return self.raw_data[ antI_span[0]:antI_span[1] ], self.starting_time[antI_span[0]:antI_span[1]]
        
    def get_antenna_safetySpan(self, antenna_i, start_time, stop_time):
        """returns the data and start times for a station (same as open_station_SampleNumber), insuring that the data between start_time and stop_time is valid for
        all antennas in this station. (assume the duration is smaller then a block). If new data is opened, then start_time is aligned to just after the beginning of the file."""
        
        do_load = False
        if not self.antenna_data_loaded[ antenna_i ]:
            do_load = True
        else:
            present_start_time = self.starting_time[ antenna_i ] + self.half_hann_safety_region*5.0E-9
            is_safe = present_start_time < start_time
            
            present_end_time = self.starting_time[ antenna_i ] + self.blocksize*5.0E-9 - self.half_hann_safety_region*5.0E-9
            is_safe = is_safe and present_end_time>stop_time
            
            do_load = not is_safe
            
            
        
        if do_load:
            
            antenna_delay = self.antenna_delays[ antenna_i ]
            start_sample_number = int( (start_time +  antenna_delay )/5.0E-9 )
            start_sample_number -= self.half_hann_safety_region
            
            try:
                self.open_antenna_SampleNumber(antenna_i, start_sample_number)
            except Exception as e:
                print('info', antenna_i, start_time, antenna_delay)
                raise e
            
            present_start_time = self.starting_time[ antenna_i ] + self.half_hann_safety_region*5.0E-9
            present_end_time = self.starting_time[ antenna_i ] + self.blocksize*5.0E-9 - self.half_hann_safety_region*5.0E-9

        return self.raw_data[ antenna_i ], self.starting_time[ antenna_i ]
            
    def check_saturation_span(self, ant_i, first_index, last_index):
        return data_cut_inspan( self.staturation_removals[ant_i], first_index,last_index )
        
    def check_dataLoss_span(self, ant_i, first_index, last_index):
        return data_cut_inspan( self.data_loss_spans[ant_i], first_index,last_index )
    
    def get_station_dataloss(self, stat_i, out_array=None):
        antI_span = self.station_to_antSpan[ stat_i ]
        
        if out_array is None:
            out_array = np.zeros(antI_span[1]-antI_span[0], dtype=np.int)
        else:
            out_array[:] = 0
            
        for ant_i in range(antI_span[0], antI_span[1]):
            for span in self.data_loss_spans[ant_i]:
                out_array[ant_i] += span[1]-span[0]
                
        return out_array
    
    def get_station_saturation(self, stat_i, out_array=None):
        antI_span = self.station_to_antSpan[ stat_i ]
        
        if out_array is None:
            out_array = np.zeros(antI_span[1]-antI_span[0], dtype=np.int)
        else:
            out_array[:] = 0
            
        for ant_i in range(antI_span[0], antI_span[1]):
            for span in self.staturation_removals[ant_i]:
                out_array[ant_i] += span[1]-span[0]
                
        return out_array

######### LOCATORS!!! #########
        
class planewave_locator:
    def __init__(self, header, data_manager, half_window):
        self.data_manager = data_manager
        self.antenna_locations = self.data_manager.get_antenna_locations( 0 )
        self.min_amp = header.min_amplitude
        self.upsample_factor = header.upsample_factor
        self.half_window = half_window
        self.num_antennas = len(self.antenna_locations)
        
        self.max_delta_matrix = np.empty( (self.num_antennas,self.num_antennas), dtype=np.int )
        for i, iloc in enumerate(self.antenna_locations):
            for j, jloc in enumerate(self.antenna_locations):
                R = np.linalg.norm( iloc-jloc )
                self.max_delta_matrix[i,j] = int( R/(v_air*5.0E-9) ) + 1
        

    
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
        self.locator = planewave_locator_helper(self.upsample_factor, self.num_antennas, self.max_CC_size, 50 )
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
            
            has_saturation = self.data_manager.check_saturation_span(i, peak_index-half_length, peak_index+half_length )
            if  has_saturation:
                self.antenna_mask[i] = False
                continue
            
            if self.data_manager.check_dataLoss_span(i, peak_index-half_length, peak_index+half_length ):
                self.antenna_mask[i] = False
                continue
            
            traces[i] = trace
            
            cross_corelation = self.correlator.correlate( trace )
            CC_length = len(cross_corelation)
            np.abs(cross_corelation, out = self.cross_correlation_storage[i, :CC_length])
            
            self.CC_lengths[i] = CC_length
            self.cal_delta_storage[i] = start_time-ref_time  - self.max_delta_matrix[self.ref_ant_i, i]*5.0e-9
            self.antenna_mask[i] = 1
            
        N = np.sum( self.antenna_mask ) 
        
        if N<2:
            return [0,0], 0.0, self.measured_dt, self.antenna_mask, N
        
        ## get brute guess
        self.locator.run_brute(self.ZeAz)
        
        ## now run minimizer
        succeeded, RMS = self.locator.run_minimizer( self.ZeAz, self.max_itters, self.xtol, self.gtol, self.ftol)
    
        return self.ZeAz, RMS, self.measured_dt, self.antenna_mask, N
    
    def num_itters(self):
        return self.locator.get_num_iters()
    
class iterative_mapper:
    def __init__(self, header, print_func=do_nothing):
        self.header = header
        
        ## open raw data files
        self.data_manager = raw_data_manager( header, print_func )
        self.antenna_XYZs = self.data_manager.antenna_locations
        self.antenna_expected_RMS = self.data_manager.antenna_expected_RMS
        
        
        ## find best station order
        station_locs = self.data_manager.station_locations
        relative_distances = np.linalg.norm( station_locs - station_locs[0], axis=1 )
        self.station_order = np.argsort( relative_distances )
        
        
        ## load helpers:
        self.planewave_manager = planewave_locator( header, self.data_manager, int(header.min_pulse_length_samples/2) )
        
        self.half_min_pulse_length_samples = int(header.min_pulse_length_samples/2)
        self.half_min_pulse_width_time = self.half_min_pulse_length_samples*5.0E-9
        
        self.edge_exclusion_points = self.planewave_manager.max_CC_size+ self.data_manager.half_hann_safety_region 
        self.usable_block_length = self.data_manager.blocksize - 2*self.edge_exclusion_points
    
        self.correlator = autoadjusting_upsample_and_correlate( header.min_pulse_length_samples, header.upsample_factor  )
        self.peak_fitter = parabolic_fitter()
        self.pointsource_manager = pointsource_locator( self.header.num_total_antennas )
    
        ### LOAD MEMORY ###
        self.antenna_mask = np.zeros( header.num_total_antennas, dtype=int )
        self.measured_dt = np.zeros( header.num_total_antennas, dtype=np.double )
        self.weights = np.zeros( header.num_total_antennas, dtype=np.double )
        
        self.pointsource_manager.set_memory( self.antenna_XYZs, self.measured_dt, self.antenna_mask, self.weights )


        self.len_ref_station = self.data_manager.station_to_antSpan[0,1]-self.data_manager.station_to_antSpan[0,0]
        self.data_loss_workspace = np.empty( self.len_ref_station, dtype=np.int )
        self.ref_ant_workspace = np.empty( self.usable_block_length, dtype=np.double )
        self.refAnt_peakIndeces = np.empty( header.max_events_perBlock, dtype=np.int )
        
        self.XYZc = np.zeros( 4, dtype=np.double )
        self.XYZc_old = np.zeros( 4, dtype=np.double )
        self.station_mode = np.zeros( header.num_stations, dtype=np.int ) ## current fitting mode for all stations. 0: do not fit, 1: reload if needed, 2: always reload
        
        self.covariance_matrix = np.empty( (3,3), dtype=np.double )

        output_type = [("ID", 'i8'), ('x', 'd'), ('y', 'd'), ('z', 'd'), ('t', 'd'),
               ('RMS', 'd'),  ('RedChiSquared', 'd'), ('numRS', 'i8'), ('refAmp', 'i8'), ('numThrows', 'i8'), ('image_i', 'i8'),
               ('refAnt', 'S9'),
               ('covXX', 'd'), ('covXY', 'd'), ('covXZ', 'd'), ('covYY', 'd'), ('covYZ', 'd'), ('covZZ', 'd')]
        self.output_type = np.dtype( output_type )
        self.output_data = np.empty( self.header.max_events_perBlock, dtype=self.output_type )

    def reload_data(self):
        "this is where the magic REALLY happens"
        
        for stat_i in self.station_order:
            if self.station_mode[ stat_i ] == 0:
                continue
            
            ant_span = self.data_manager.station_to_antSpan[ stat_i ]
            for ant_i in range( ant_span[0], ant_span[1] ):
                predicted_dt = self.pointsource_manager.relative_arrival_time(ant_i, self.XYZc)
                
                if self.station_mode[ stat_i ] != 2 and self.antenna_mask[ ant_i ]:
                    if np.abs( predicted_dt-self.measured_dt[ant_i] ) < self.header.min_pulse_length_samples*5.0E-9/4:
                        continue ## do not re-load info!
                        
                ## get data
                start_T = self.referance_peakTime + predicted_dt - self.header.min_pulse_length_samples*5.0E-9/2
                antenna_data, start_time = self.data_manager.get_antenna_safetySpan( ant_i, start_T, start_T + self.header.min_pulse_length_samples*5.0E-9)
                 
                sample_index = int( (start_T - start_time)/5.0E-9 )
                antenna_trace = antenna_data[ sample_index:sample_index + self.header.min_pulse_length_samples ]
                trace_startTime = start_time + sample_index*5.0E-9
                
                
                ## check quality
                if len(antenna_trace) != self.header.min_pulse_length_samples: ## something went wrong, I dobn't think this should ever really happen
                    self.antenna_mask[ ant_i ] = False
                    continue
                    
                if abs_max( antenna_trace ) < self.header.min_amplitude:
                    self.antenna_mask[ ant_i ] = False
                    continue
                
                has_saturation = self.data_manager.check_saturation_span(ant_i, sample_index, sample_index+self.header.min_pulse_length_samples )
                if  has_saturation:
                    self.antenna_mask[ ant_i ] = False
                    continue
                
                if self.data_manager.check_dataLoss_span(ant_i, sample_index, sample_index+self.header.min_pulse_length_samples ):
                    self.antenna_mask[ ant_i ] = False
                    continue
                
            
                ## do cross correlation and get peak location
                cross_correlation = self.correlator.correlate( antenna_trace )
                CC_length = len(cross_correlation)
                workspace_slice = self.ref_ant_workspace[:CC_length]
                np.abs(cross_correlation, out=workspace_slice)
                
                CC_peak = self.peak_fitter.fit( workspace_slice )
                
                if CC_peak > CC_length/2:
                    CC_peak -= CC_length
                    
                peak_location = CC_peak*5.0E-9/self.header.upsample_factor
                peak_location += trace_startTime - (self.referance_peakTime-self.half_min_pulse_length_samples*5.0E-9)
                
                self.measured_dt[ ant_i ] = peak_location
                self.antenna_mask[ ant_i ] = True
                
    def reload_data_and_plot(self, chi_squared):
        print('start loading. red chi square:', chi_squared)
        
        for stat_i in self.station_order:
            if self.station_mode[ stat_i ] == 0:
                continue
            
            ant_span = self.data_manager.station_to_antSpan[ stat_i ]
            
            for ant_i in range( ant_span[0], ant_span[1] ):
                predicted_dt = self.pointsource_manager.relative_arrival_time(ant_i, self.XYZc)
                
#                if self.station_mode[ stat_i ] != 2 and self.antenna_mask[ ant_i ]:
#                    if np.abs( predicted_dt-self.measured_dt[ant_i] ) < self.header.min_pulse_length_samples*5.0E-9/4:
#                        continue ## do not re-load info!
                
                if self.station_mode[ stat_i ] == 2 and ant_i==ant_span[0]:
                    print('loading:', self.data_manager.statI_to_sname(stat_i))
                    
                    self.pointsource_manager.load_covariance_matrix( self.XYZc, self.covariance_matrix )
                    
                    ref_XYZ = self.data_manager.antenna_locations[ self.referance_antI ]
                    ant_XYZ = self.data_manager.antenna_locations[ ant_i ]
                    source_XYZ = self.XYZc[:3]
                    
                    diff_ref = source_XYZ-ref_XYZ
                    ref_R = np.sqrt( diff_ref[0]**2 + diff_ref[1]**2 + diff_ref[2]**2 )
                    
                    diff_ant = source_XYZ-ant_XYZ
                    ant_R = np.sqrt( diff_ant[0]**2 + diff_ant[1]**2 + diff_ant[2]**2 )
                    
                    ## now we need the derivative
                    diff_ref *= 1.0/ref_R
                    diff_ant *= 1.0/ant_R
                    diff_ant -= diff_ref
                    diff_ant *= 1.0/v_air
                    
                    ## now we make a sandwich!
                    prediction_variance =  np.dot(diff_ant, np.dot( self.covariance_matrix, diff_ant ))
                    ## SO TASTY!!
                    
                    print("  model std:", np.sqrt(prediction_variance), np.sqrt(prediction_variance*chi_squared))
                    
                    
                        
                ## get data
                start_T = self.referance_peakTime + predicted_dt - self.header.min_pulse_length_samples*5.0E-9/2
                antenna_data, start_time = self.data_manager.get_antenna_safetySpan( ant_i, start_T, start_T + self.header.min_pulse_length_samples*5.0E-9)
                 
                sample_index = int( (start_T - start_time)/5.0E-9 )
                antenna_trace = antenna_data[ sample_index:sample_index + self.header.min_pulse_length_samples ]
                trace_startTime = start_time + sample_index*5.0E-9
                
                
                ## check quality
                if len(antenna_trace) != self.header.min_pulse_length_samples: ## something went wrong, I dobn't think this should ever really happen
                    self.antenna_mask[ ant_i ] = False
                    continue
                    
                if abs_max( antenna_trace ) < self.header.min_amplitude:
                    self.antenna_mask[ ant_i ] = False
                    continue
                
                has_saturation = self.data_manager.check_saturation_span(ant_i, sample_index, sample_index+self.header.min_pulse_length_samples )
                if  has_saturation:
                    self.antenna_mask[ ant_i ] = False
                    continue
                
                if self.data_manager.check_dataLoss_span(ant_i, sample_index, sample_index+self.header.min_pulse_length_samples ):
                    self.antenna_mask[ ant_i ] = False
                    continue
                
            
                ## do cross correlation and get peak location
                cross_correlation = self.correlator.correlate( antenna_trace )
                CC_length = len(cross_correlation)
                workspace_slice = self.ref_ant_workspace[:CC_length]
                np.abs(cross_correlation, out=workspace_slice)
                
                CC_peak = self.peak_fitter.fit( workspace_slice )
                
                if CC_peak > CC_length/2:
                    CC_peak -= CC_length
                    
                peak_location = CC_peak*5.0E-9/self.header.upsample_factor
                peak_location += trace_startTime - (self.referance_peakTime-self.half_min_pulse_length_samples*5.0E-9)
                
                self.measured_dt[ ant_i ] = peak_location
                self.antenna_mask[ ant_i ] = True
                
#                if self.station_mode[ stat_i ] == 2:
#                    self.TEST_MEMORY[ ant_i ] = peak_location
                
                T = np.arange( len(antenna_trace), dtype=np.double )
                T -= len(antenna_trace)/2
                T *= 5.0E-9
                
                A = np.abs(antenna_trace)
                A /= np.max(A)
                A += stat_i
                plt.plot(T, A)
            plt.annotate( self.data_manager.statI_to_sname(stat_i), (0.0, stat_i) )
        plt.show()
        
    def process(self, start_index, print_func=do_nothing):
        
        #### get initial data and identify referance antenna ####
        referanceStat_blockData, startTimes = self.data_manager.open_station_SampleNumber(0, start_index)
        self.data_manager.get_station_dataloss(0, self.data_loss_workspace )
        self.referance_antI = np.argmin( self.data_loss_workspace )
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
        np.abs( referanceStat_blockData[self.referance_antI, self.edge_exclusion_points:-self.edge_exclusion_points], out=self.ref_ant_workspace )
        num_events = 0
        half_erasure_length = int( self.header.erasure_length/2 )
        for event_i in range( self.header.max_events_perBlock ):
            peak_index = np.argmax( self.ref_ant_workspace )
            amplitude = self.ref_ant_workspace[ peak_index ]
            
            if amplitude < self.header.min_amplitude:
                break
            else:                
                self.refAnt_peakIndeces[ event_i ] = self.edge_exclusion_points + peak_index
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
        print_func("block at", start_index, "ref. antenna ID:", self.referance_antI)
        print_func("fitting", num_events, 'events')
        
        out_i = 0
        for event_i, event_ref_index in enumerate( ref_indeces ):
            
            #### setup ref info
            referance_trace = referanceStat_blockData[ self.referance_antI, 
                                   event_ref_index-self.half_min_pulse_length_samples:event_ref_index+self.half_min_pulse_length_samples ]

            self.referance_peakTime = startTimes[ self.referance_antI ] + 5.0E-9*event_ref_index
            referance_amplitude = np.abs( referanceStat_blockData[ self.referance_antI, event_ref_index] )
            
            self.correlator.set_referance( referance_trace )
        
            print_func()
            print_func('event:', event_i, '/', num_events, event_ref_index, ". amplitude:", referance_amplitude)
            print_func("    total index:", start_index+event_ref_index, 'on antenna',  self.data_manager.antenna_names[ self.referance_antI ] )
        
        
            #### try to fit planewave
            zenith_azimuth, planewave_RMS, planewave_dt, planewave_mask, planewave_num_ants = self.planewave_manager.locate( 
                event_ref_index, False)
            
            print_func('planewave RMS:', planewave_RMS, 'num antennas:', planewave_num_ants, 'num iters:', self.planewave_manager.num_itters() )
            print_func('zenith, azimuth:', zenith_azimuth)

            if planewave_num_ants<2:
                print_func("  too few antennas")
                print_func()
                print_func()
                continue
#            print_func()
            
            
            self.XYZc[0] = np.sin( zenith_azimuth[0] )*np.cos( zenith_azimuth[1] )*self.header.guess_distance + self.ref_XYZ[0]
            self.XYZc[1] = np.sin( zenith_azimuth[0] )*np.sin( zenith_azimuth[1] )*self.header.guess_distance + self.ref_XYZ[1]
            self.XYZc[2] = np.cos( zenith_azimuth[0] )*self.header.guess_distance + self.ref_XYZ[2]
            self.XYZc[3] = 0.0
            
            
            #### prep for fitting
            self.antenna_mask[:] = 0
            self.antenna_mask[:self.len_ref_station] = planewave_mask
            self.measured_dt[:self.len_ref_station] = planewave_dt
#            success, chi_squared = self.pointsource_manager.run_minimizer( self.XYZc, self.header.max_minimize_itters, 
#                        self.header.minimize_xtol, self.header.minimize_gtol, self.header.minimize_ftol )
            
#            print_func('initial fit chi-squared:', chi_squared)
            print_func('  XYZ', self.XYZc[:3])
            
            print_func()
            
#            self.XYZc[2] = np.abs( self.XYZc[2] )
#            self.XYZc[:3] *= self.header.guess_distance/np.linalg.norm( self.XYZc[:3] )
            
            
            #### now do the thing!!
            self.station_mode[:] = 0
            is_good = True
            current_red_chi_sq = np.inf #chi_squared
            num_throws = 0
            for stat_i in self.station_order:
                self.station_mode[ stat_i ] = 2
                
#                if planewave_num_ants > 3:
                self.reload_data()
#                else:
#                self.reload_data_and_plot(current_red_chi_sq)
                
#                if 190 <= event_i <= 194:
#                    self.reload_data_and_plot( current_red_chi_sq )
#                else:
#                    self.reload_data()
                
                self.station_mode[ stat_i ] = 1
                
                num_ants = np.sum( self.antenna_mask )
                if  num_ants < 3:
                    continue
                
                success, new_chi_squared = self.pointsource_manager.run_minimizer( self.XYZc, self.header.max_minimize_itters, 
                        self.header.minimize_xtol, self.header.minimize_gtol, self.header.minimize_ftol )
                
                if (stat_i != 0) and ( (current_red_chi_sq > 1 and new_chi_squared > 5*current_red_chi_sq) or \
                    (current_red_chi_sq<1 and new_chi_squared>5) ):
                    self.station_mode[ stat_i ] = 0
                    ant_span = self.data_manager.station_to_antSpan[ stat_i ]
                    self.antenna_mask[ ant_span[0]:ant_span[1] ] = 0
                    
                    print_func("  throwing station:", stat_i, 'had red. chi-squared of:', new_chi_squared, 'previous:', current_red_chi_sq) 
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
                
                if stat_i == 0:
                    self.XYZc[:3] *= self.header.guess_distance/np.linalg.norm( self.XYZc[:3] )
                    
                    
            if not is_good:
                continue
            
            #### get final info
            success, current_red_chi_sq = self.pointsource_manager.run_minimizer( self.XYZc, self.header.max_minimize_itters, 
                    self.header.minimize_xtol, self.header.minimize_gtol, self.header.minimize_ftol )
            
            RMS = self.pointsource_manager.get_RMS()
            
            num_remote_stations = 0
            for stat_i in self.station_order:
                if self.data_manager.is_RS[ stat_i ]:
                    ant_span = self.data_manager.station_to_antSpan[ stat_i ]
                    num_ant = np.sum( self.antenna_mask[ ant_span[0]:ant_span[1] ] )
                    if num_ant > 0:
                        num_remote_stations += 1
                      
            self.pointsource_manager.load_covariance_matrix( self.XYZc, self.covariance_matrix )
            
            try:
                eigvalues, eigenvectors = np.linalg.eigh( self.covariance_matrix )
                np.sqrt(eigvalues, out=eigvalues)
            except:
                print('eigenvalues did not converge???')
                eigvalues = None
                eigenvectors = self.covariance_matrix
            
#            T = (event_ref_index+start_index)*5.0E-9 - np.linalg.norm( self.XYZc[:3]-self.ref_XYZ )/v_air - self.header.refStat_delay
            T = self.referance_peakTime - np.linalg.norm( self.XYZc[:3]-self.ref_XYZ )/v_air
                   
            
            #### output!
            print_func("successful fit :", out_i)
            print_func("  RMS:", RMS, 'red. chi-square:', current_red_chi_sq)
            print_func("  XYZT:", self.XYZc[:3], T)
            print_func("  num RS:", num_remote_stations)
            print_func('  eig. info:')
            print_func( eigvalues )
            print_func(eigenvectors)
            print_func()
            print_func()
            
            self.output_data[out_i]['ID'] = out_i
            self.output_data[out_i]['x'] = self.XYZc[0]
            self.output_data[out_i]['y'] = self.XYZc[1]
            self.output_data[out_i]['z'] = self.XYZc[2]
            self.output_data[out_i]['t'] = T
            
            self.output_data[out_i]['RMS'] = RMS
            self.output_data[out_i]['RedChiSquared'] = current_red_chi_sq
            self.output_data[out_i]['numRS'] = num_remote_stations
            self.output_data[out_i]['refAmp'] = referance_amplitude
            self.output_data[out_i]['numThrows'] = num_throws
            self.output_data[out_i]['image_i'] = event_i
            self.output_data[out_i]['refAnt'] = self.referance_ant_name
            
            self.output_data[out_i]['covXX'] = self.covariance_matrix[0,0]
            self.output_data[out_i]['covXY'] = self.covariance_matrix[0,1]
            self.output_data[out_i]['covXZ'] = self.covariance_matrix[0,2]
            self.output_data[out_i]['covYY'] = self.covariance_matrix[1,1]
            self.output_data[out_i]['covYZ'] = self.covariance_matrix[1,2]
            self.output_data[out_i]['covZZ'] = self.covariance_matrix[2,2]
            
            out_i += 1
            
        return self.output_data[:out_i]
    
    def process_block(self, block_i, print_func=do_nothing, skip_blocks_done=True):
            
        output_dir = self.header.input_folder
        out_fname = output_dir + "/block_"+str(block_i)+".h5"
            
        if skip_blocks_done and isfile(out_fname):
            print_func("block:", block_i, "already completed. Skipping")
            return
        
        print_func("block:", block_i)
        
        data = self.process( self.header.initial_datapoint + block_i*self.usable_block_length, print_func )
        
        if len(data) != 0:
            X_array = data['x']
            minX = np.min( X_array )
            maxX = np.max( X_array )
            Y_array = data['y']
            minY = np.min( Y_array )
            maxY = np.max( Y_array )
            Z_array = data['z']
            minZ = np.min( Z_array )
            maxZ = np.max( Z_array )
            T_array = data['t']
            minT = np.min( T_array )
            maxT = np.max( T_array )
            bounds = np.array([ [minX,maxX], [minY,maxY], [minZ,maxZ], [minT,maxT] ])
        else:
            bounds = np.array([ [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0] ])
            
        block_outfile = h5py.File(out_fname, "w")
        out_dataset = block_outfile.create_dataset(str(block_i), data=data )
        out_dataset.attrs['block'] = block_i
        out_dataset.attrs['bounds'] = bounds
                            
                    
if __name__ == "__main__":
    from LoLIM import utilities
    utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
    utilities.default_processed_data_loc = "/home/brian/processed_files"
    
    out_folder = 'iterMapper_50_CS002_TST2'
    
    timeID = 'D20170929T202255.000Z'
    stations_to_exclude = [ 'RS407', 'RS409']
                
    outHeader = make_header(timeID, 3000*(2**16), station_delays_fname = 'station_delays4.txt', 
                additional_antenna_delays_fname = 'ant_delays.txt', bad_antennas_fname = 'bad_antennas.txt', 
                pol_flips_fname = 'polarization_flips.txt')
#            
    outHeader.stations_to_exclude = stations_to_exclude    
#    outHeader.max_events_perBlock = 100
    outHeader.run( out_folder )
    
#    read_header('iterMapper_50_CS002', timeID).resave_header( out_folder )
    
    inHeader = read_header(out_folder, timeID)
#    inHeader.print_all_info()
    
    log_fname = inHeader.next_log_file()

    logger_function = logger()
    logger_function.set( log_fname, True )
    logger_function.take_stdout()
    logger_function.take_stderr()
    
    
    mapper = iterative_mapper(inHeader, print)
#    for i in range(1113, 1113+10):
    mapper.process_block(816, print, False)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    