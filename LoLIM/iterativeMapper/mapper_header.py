#!/usr/bin/env python3
from os import mkdir, listdir
from os.path import isdir, isfile
from collections import deque
import datetime

import numpy as np
import h5py

from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1, read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
from LoLIM.utilities import processed_data_dir, antName_is_even, BoundingBox_collision, SId_to_Sname

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
            
    def get_antenna_info(self, antenna_name):
        station_ID = antenna_name[:3]
        antenna_sname = SId_to_Sname[ int(station_ID) ]
        for sname, antennas in zip(self.station_names, self.antenna_info ):
            if antenna_sname == sname:
                for ant in antennas:
                    if ant.ant_name == antenna_name:
                        return ant
                return None
        return None
            
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
            self.refAnt = None
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
        
    def load_data_as_sources(self, maxRMS=None, maxRChi2=None, minRS=None, minAmp=None, bounds=None, blocks=None ):
        """returns sources as deques"""
        out_sources = deque()
        
        folder = self.input_folder
        
        file_names = [fname for fname in listdir(folder) if fname.endswith(".h5") and not fname.startswith('header')]
        for fname in file_names:
            infile = h5py.File(folder+'/'+fname, "r")
        
            for block_dataset in infile.values():
                block_i = block_dataset.attrs['block']
                if not ( blocks is None or block_i in blocks):
                    continue
                    
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
                    new_source.refAnt =  source_info['refAnt'].decode()
                    
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