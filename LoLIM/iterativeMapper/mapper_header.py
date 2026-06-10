#!/usr/bin/env python3
from os import mkdir, listdir
from os.path import isdir, isfile, abspath, join
from collections import deque
import datetime
import json

import numpy as np
import h5py

from LoLIM.IO import raw_tbb_IO
from LoLIM.utilities import processed_data_dir, SId_to_Sname
import LoLIM.utilities  as utils
from LoLIM import findRFI_adv

class make_header:
    def __init__(self, timeID, total_cal):
        """TimeID indicates flash. 
        total_cal is either a total_cal object or string (which indicates file name with complete-ish path)"""

        
        self.timeID = timeID
        self.total_cal_info = total_cal

        self.notes=""

        self.data_directory = None   ## data folder. Where sub-folders are year, and then flash timeID, which contain HDF5 datafiles. Defaults to utils.default_raw_data_loc
        self.output_directory = None ## folder to save output. will make this folder if needed.  defaults to join( utils.processed_data, "impulsiveImager")
        
        self.max_antennas_per_station = np.inf
        self.how_to_choose_antennas = None ## Will indicated how to choose which of saved antennas to use. parameter is currently unused, and is only placeholder for time being.

        self.referance_station = 'CS002'
        self.stations_to_exclude = []

        #initial_time and final_time are seconds past the second
        # default to 1 block after start and 1 block before the end on reference staiton
        self.initial_time = None
        self.final_time = None
        
        self.antenna_polarization = "Y" ## give which antennas to use. Currently can be "Y" or "X". In future this maybe could indicate combining antennas
        
        #self.remove_saturation = True
        #self.remove_RFI = True
        #self.positive_saturation = 2046
        #self.negative_saturation = -2047
        #self.saturation_removal_length = 50
        #self.saturation_half_hann_length = 50
        
        self.hann_window_fraction = 0.1
        self.blocksize = 2**16
        self.lowerFrequency = 30.0e6 ## THESE ARE NEW
        self.upperFrequency = 80.0e6
        self.filter_roll_width = 2.5E6

        self.RFI_info = "default" ## if this is None, than only RFI cleaning is bandpass filter. 
            ## If RFI_info is a string that says "default", then will be join( root(self.output_directory ), 'findRFI', 'advFindRFI_results.json')
            ## if is string, and not "default", then assume is full path to findRFI_adv
            ## else, assume is jsonable findRFI_adv results 

        #self.num_zeros_dataLoss_Threshold = 10
        
        self.min_amplitude = 15
        
        self.upsample_factor = 2#8
        self.max_events_perBlock =  500# 100
        self.min_pulse_length_samples = 20
        self.min_timeUncertainty_samples = 5 ## expected uncertaininty in predicting when pulse is on the next station
        self.erasure_length = 20 ## num points centered on peak to zero-out, only when search ref-antenna for which peaks to image. 
        
        self.guess_distance = 10000
        
        self.pol_flips_are_bad = True
        
        self.antenna_timing_RMS = 1.0E-9 ## THIS IS NEW

        self.stop_chi_squared = 100.0   ## if chi-squared exceeds this the event is skipped
        self.stop_chisquared_factor = 5 ## if a station increases the chi-square by thsi factor or more, than that station is skipped
        
        self.max_minimize_itters = 200
        self.minimize_ftol = 3.0E-16
        self.minimize_xtol = 3.0E-16
        self.minimize_gtol = 3.0E-16
        



    def run(self):

### folder organization
        if self.data_directory is None:
            self.data_directory = utils.default_raw_data_loc
        self.data_directory = abspath( self.data_directory )

        if  self.output_directory is None:
            self.output_directory = join( processed_data_dir( self.timeID ), 'impulsiveImager')
        self.output_directory = abspath( self.output_directory )

        if not isdir(self.output_directory ):
            mkdir(self.output_directory )
            
        self.logging_folder = join( self.output_directory, 'logs_and_plots')
        if not isdir(self.logging_folder):
            mkdir(self.logging_folder)


        self.tmp_out_data = join(self.output_directory, 'outData')

        if not isdir(self.tmp_out_data):
            mkdir(self.tmp_out_data)

        
### load calibrations
        if isinstance(self.total_cal_info, str):
            totalCal_object = raw_tbb_IO.read_cal_file( self.total_cal_info, pol_flips_are_bad=self.pol_flips_are_bad  )

        elif isinstance(self.total_cal_info, raw_tbb_IO.total_cal_object):
            totalCal_object = self.total_cal_info
            self.total_cal_info = "inputed" ## important becouse this assumes the data is not jsonable

        else:
            raise Exception("total cal must be string or total_cal_object")

        ## to a jsonable data object
        self.total_cal_data = totalCal_object.to_JSONable_object()


    ### RFI lines
        if self.RFI_info == None:
            self.RFI_data = None 

        elif isinstance( self.RFI_info, str ):
            if self.RFI_info == "default":
                rootDir = abspath( join(self.output_directory, '../'))
                self.RFI_info = join( rootDir, 'findRFI', 'advFindRFI_results.json')

            self.RFI_data = findRFI_adv.open_advFindRFI( fname=self.RFI_info )
                
        elif isinstance( self.RFI_info, dict ):
            self.RFI_data = self.RFI_info
            self.RFI_info = "inputed"

        else: 
            raise Exception("RFI_info must be None, string, or dictionary")

    
    

### Decide which stations to use, and their fpaths
        self.raw_fpaths = raw_tbb_IO.filePaths_by_stationName( self.timeID, raw_data_loc=self.data_directory )
        self.raw_fpaths = {sname:fp for sname,fp in self.raw_fpaths.items() if sname not in self.stations_to_exclude}

        if self.referance_station not in self.raw_fpaths:
            raise Exception("referance_station is not available")

## Open station and choose antennas
        self.num_total_antennas = 0
        self.antenna_name_dict = {} ## key is sname, value is list of antenna names


        station_locations = []
        station_names = []
        ref_station_location = None
        for station, fpaths in self.raw_fpaths.items():
            print('opening', station)

            raw_data_file = raw_tbb_IO.MultiFile_Dal1(fpaths, force_metadata_ant_pos=True, only_complete_pairs=False, total_cal=totalCal_object )
            all_antenna_names = raw_data_file.get_antenna_names()

            if self.antenna_polarization == 'Y':
                used_antenna_names = all_antenna_names[::2]
            elif self.antenna_polarization == 'X':
                used_antenna_names = all_antenna_names[1::2]
            else:
                raise Exception("antenna polarization must be X or Y")


            used_antenna_names = [an for an in used_antenna_names if raw_data_file.has_antenna(antenna_name=an)]

            if self.how_to_choose_antennas is not None:
                raise Exception("how_to_choose_antennas is not yet implemented")


            if len(used_antenna_names) < 1:
                continue

            stat_loc = np.average(  raw_data_file.get_LOFAR_centered_positions(), axis=0 )
            station_locations.append( stat_loc )
            station_names.append( station )

            if (station==self.referance_station):
                if len(used_antenna_names)<4:
                    raise Exception("reference station has too few antennas")

                dataStart_time = raw_data_file.get_nominal_sample_number()*(1/raw_data_file.get_sample_frequency())
                dataEnd_time   = dataStart_time + np.min(raw_data_file.get_nominal_data_lengths())*(1/raw_data_file.get_sample_frequency())

                if self.initial_time is None:
                    self.initial_time = dataStart_time + (2**16)*(1/raw_data_file.get_sample_frequency())
                if self.final_time is None:
                    self.final_time = dataEnd_time - (2**16)*(1/raw_data_file.get_sample_frequency())

                ref_station_location = stat_loc

            if len(used_antenna_names) > self.max_antennas_per_station:
                used_antenna_names = used_antenna_names[:self.max_antennas_per_station]

            self.num_total_antennas += len(used_antenna_names)
            self.antenna_name_dict[station] = used_antenna_names

        self.raw_fpaths = {sname:fp for sname,fp in self.raw_fpaths.items() if sname in self.antenna_name_dict}

## calc station order
        station_locations = np.array( station_locations )
        relative_distances = np.linalg.norm( station_locations - ref_station_location, axis=1 )
        self.station_order = [ station_names[i] for i in  np.argsort( relative_distances )]

## calculate usable blocksize
        self.nonTukey_windowEdgeSize =int(  max( self.min_pulse_length_samples, self.erasure_length  ) + 100*(3.4e-9)/(5e-9)  )
                                          ##    actual pulse size                                     station propagation time 

        usable_blocksize = self.blocksize - 2*self.nonTukey_windowEdgeSize - self.blocksize*self.hann_window_fraction*2 - 2 
                            ##                                                 tukey window edge                         1 sample for edge-issues
        self.analyzable_blocksize = int(usable_blocksize/2)*2

### output to header!
        self.headerCreation_time = str( datetime.datetime.now( datetime.timezone.utc ) )

        jsonable_dict = self.__dict__  ## I'm lazy, what can I say

        output_fname = join( self.output_directory, 'header.json' )

        json.dump( jsonable_dict, fp=open(output_fname, 'w'),  cls=utils.JSON_CustomEncoder, indent=4) 



class read_header:
    def __init__(self, input_folder):
        self.__dict__ = json.load( fp=open(join(input_folder, 'header.json'), 'r'), cls=utils.JSON_CustomDecoder_maker() )

        if "stop_chisquared_factor" not in self.__dict__:  ## a hack to remove after testing phase
            self.stop_chisquared_factor = 5


        self.total_cal_object = raw_tbb_IO.total_cal_object.from_JSONable_object( self.total_cal_data )


    def next_log_file(self):
        """return the filename (including directory) of the next log file that should be saved to"""
        
        file_number = 0
        fname = self.logging_folder+ "/log_run_"+str(file_number)+".txt"
        while isfile(fname):
            file_number += 1
            fname = self.logging_folder+ "/log_run_"+str(file_number)+".txt"
            
        return fname

