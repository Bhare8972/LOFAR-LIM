#!/usr/bin/env python3

"""This module implements an interface for reading LOFAR TBB data.

This module is strongly based on pyCRtools module tbb.py by Pim Schellart, Tobias Winchen, and others.
However, it has been completely re-written for use with LOFAR-LIM

Author: Brian Hare

Definitions: 
LOFAR is split into a number of different stations. There are three main types: Core Stations (CS), Remote Stations (RS), and international stations
Each station contains 96 low band antennas (LBA) and 48 high band antennas (HBA). Each antenna is dual polarized.

Each station is refered to by its name (e.g. "CS001"), which is a string, or its ID (e.g. 1), which is an integer. In general, these are different!
The mapping, however, is unique and is given in utilities.py

There are a few complications with reading the data. 
1) The data from each station is often spread over multiple files
    There is a class below that can combine multiple files (even from different stations)
2) It is entirely possible that one file could contain multiple stations
    This feature is not used, so I assume that it isn't a problem (for now)
3) Each Station has unknown clock offsets. Technically the core stations are all on one clock, but there are some unknown cable delays
    This is a difficult  problem, not handled here
4) Each Antenna doesn't necisarily start reading data at precisely the same time.
    The code below picks the latest start time so this problem can be ignored by the end user
5) The software that inserts metadata (namely antenna positions and callibrations) sometimes "forgets" to do its job
    The code below will automatically read the metadata from other files when necisary
6) LOFAR is constantly changing
    So..keeping code up-to-date and still backwards compatible will be an interesting challange
    
7) LOFAR only has 96 RCUs (reciever control units) per station (at the moment).
    Each RCU is essentually one digitizer. Each antenna needs two RCS to record both polarizations. The result is only 1/3 of the antennas can 
    be read out each time. 
    
    LOFAR keeps track of things with two ways. First, the data is all refered to by its RCUid. 0 is the 0th RCU, ect... However, which antenna
    corresponds to which RCU depends on the antennaSet. For LOFAR-LIM the antenna set will generally be "LBA_OUTER". This could change, and sometimes
    the antenna set is spelled wrong in the data files. (This should be handeld here though)
    
    In the code below each RCU is refered to by ANTENNA_NAME or antennaID. These are the same thing (I think). They are, however, a misnomer, as they
    actually refer to the RCU, not antenna. The specific antenna depends on the antenna set. For the same antenna set, however, the ANTENNA_NAME will
    always refer to the same antenna.
    
    Each ANTENNA_NAME is a string of 9 digits. First three is the station ID (not name!), next three is the group (no idea, don't ask), final 3 is the RCU id
    
    For LBA_INNER data set, even RCU ids refer to X-polarized dipoles and odd RCU ids refer to Y-polarized dipoles. This is flipped for LBA_OUTER antenna set.
    X-polarization is NE-SW, and Y-polarization is NW-SE. antenna_responce.py, which handles the antenna function, assumes the data is LBA_OUTER.
    
"""

import os

import numpy as np
import h5py

import LoLIM.IO.metadata as md
import LoLIM.utilities as util


#nyquist_zone = {'LBA_10_90' : 1, 'LBA_30_90' : 1, 'HBA_110_190' : 2, 'HBA_170_230' : 3, 'HBA_210_250' : 3}
conversiondict = {"": 1.0, "kHz": 1000.0, "MHz": 10.0 ** 6, "GHz": 10.0 ** 9, "THz": 10.0 ** 12}

#### helper functions ####

def filePaths_by_stationName(timeID, raw_data_loc=None):
    """Given a timeID, and a location of raw data (default set in utilities.py), return a dictionary.
    The keys of the dictionary are antenna names, the values are lists of file paths to data files that contain that station."""
    
    data_file_path = util.raw_data_dir(timeID, raw_data_loc)
    h5_files = [f for f in os.listdir(data_file_path) if f[-6:] == 'tbb.h5']
    
    ret = {}
    for fname in h5_files:
        Fpath = data_file_path + '/' + fname
        junk, sname, junk, junk = util.Fname_data(Fpath)

        if sname not in ret:
            ret[sname] = []
            
        ret[sname].append( Fpath )
        
    return ret

def eventData_filePaths(timeID, raw_data_loc=None):
    """Given a timeID, and a location of raw data (default set in utilities.py), return a list of file paths of data files"""
    data_file_path = util.raw_data_dir(timeID, raw_data_loc)
    return [f for f in os.listdir(data_file_path) if f[-6:] == 'tbb.h5']

#### data reading class ####
# Note: ASTRON will "soon" release a new DAL (data )
    
class TBBData_Dal1:
    """a class for reading one station from one file. However, since one station is often spread between different files, 
    use filePaths_by_stationName combined with MultiFile_Dal1 below""" 
    
    def __init__(self, filename, force_metadata_ant_pos=False):
        self.filename = filename
        self.force_metadata_ant_pos = force_metadata_ant_pos
        
        #### open file and set some basic info####
        self.file = h5py.File(filename, "r")
        
        stationKeys = [s for s in self.file.keys() if s.startswith('Station')]
        ## assume there is only one station in the file
        if len(stationKeys) != 1:
            print("WARNING! file", self.filename, "has more then one station")
        self.stationKey = stationKeys[0]
        
        self.antennaSet = self.file.attrs['ANTENNA_SET'][0].decode()
        self.dipoleNames = list( self.file[ self.stationKey ].keys() )
        self.StationID = self.file[ self.stationKey ][ self.dipoleNames[0] ].attrs["STATION_ID"][0]
        self.StationName = util.SId_to_Sname[ self.StationID ]
        ## assume all antennas have the same sample frequency
        
        self.SampleFrequency = self.file[ self.stationKey ][ self.dipoleNames[0] ].attrs["SAMPLE_FREQUENCY_VALUE"
            ][0]*conversiondict[ self.file[ self.stationKey ][ self.dipoleNames[0] ].attrs["SAMPLE_FREQUENCY_UNIT"][0].decode() ]
        ## filter selection is typically "LBA_10_90"
        self.FilterSelection = self.file.attrs['FILTER_SELECTION'][0].decode()
        
        
        #### check that all antennas start in the same second, and record the same number of samples ####
        self.Time = None
        self.DataLengths = np.zeros(len(self.dipoleNames), dtype=int)
        self.SampleNumbers = np.zeros(len(self.dipoleNames), dtype=int)
        for dipole_i, dipole in enumerate(self.dipoleNames):
            
            if self.Time is None:
                self.Time = self.file[ self.stationKey ][ dipole ].attrs["TIME"][0]
            else:
                if self.Time != self.file[ self.stationKey ][ dipole ].attrs["TIME"][0]:
                    raise IOError("antennas do not start at same time in "+self.filename)
                
            self.DataLengths[dipole_i] = self.file[ self.stationKey ][ dipole ].attrs["DATA_LENGTH"][0]
            self.SampleNumbers[dipole_i] = self.file[ self.stationKey ][ dipole ].attrs["SAMPLE_NUMBER"][0]

        
        #### get position and delay metadata...maybe####
        self.have_metadata = 'DIPOLE_CALIBRATION_DELAY_VALUE' in self.file[ self.stationKey ][ self.dipoleNames[0] ].attrs
        self.antenna_filter = md.make_antennaID_filter(self.dipoleNames)
        
        if self.have_metadata and not self.force_metadata_ant_pos:
            self.ITRF_dipole_positions = np.empty((len(self.dipoleNames), 3), dtype=np.double)
            
            for i,dipole in enumerate(self.dipoleNames):
                self.ITRF_dipole_positions[i] = self.file[ self.stationKey ][dipole].attrs['ANTENNA_POSITION_VALUE']
      
        else:
            ## get antenna positions from file.
            self.ITRF_dipole_positions = md.getItrfAntennaPosition(self.StationName, self.antennaSet)[ self.antenna_filter ]
            
            
            
        if self.have_metadata: # and not self.forcemetadata_delays
            self.calibrationDelays = np.empty( len(self.dipoleNames), dtype=np.double )
            
            for i,dipole in enumerate(self.dipoleNames):
                self.calibrationDelays[i] = self.file[ self.stationKey ][dipole].attrs['DIPOLE_CALIBRATION_DELAY_VALUE']
            
        
        
        #### get the offset, in number of samples, needed so that each antenna starts at the same time ####
        self.nominal_sample_number = np.max( self.SampleNumbers )
        self.nominal_DataLengths = self.DataLengths + (np.min( self.SampleNumbers ) - self.nominal_sample_number)
        self.sample_offsets = self.nominal_sample_number - self.SampleNumbers
            
    #### GETTERS ####
    def needs_metadata(self):
        """return true if this file does not have metadata"""
        return not self.have_metadata
    
    def get_station_name(self):
        """returns the name of the station, as a string"""
        return self.StationName
    
    def get_station_ID(self):
        """returns the ID of the station, as an integer. This is not the same as StationName. Mapping is givin in utilities"""
        return self.StationID
    
    def get_antenna_names(self):
        """return name of antenna as a list of strings. This is really the RCU id, and the physical antenna depends on the antennaSet"""
        return self.dipoleNames
    
    def get_antenna_set(self):
        """return the antenna set as a string. Typically "LBA_OUTER" """
        return self.antennaSet
    
    def get_sample_frequency(self):
        """gets samples per second. Typically 200 MHz."""
        return self.SampleFrequency
    
    def get_filter_selection(self):
        """return a string that represents the frequency filter used. Typically "LBA_10_90"""
        return self.FilterSelection
        
    def get_timestamp(self):
        """return the POSIX timestamp of the first data point"""
        return self.Time
    
    def get_full_data_lengths(self):
        """get the number of samples stored for each antenna. Note that due to the fact that the antennas do not start recording
        at the exact same instant (in general), this full data length is not all usable
        returns array of ints"""
        return self.DataLengths
    
    def get_all_sample_numbers(self):
        """return numpy array that contains the sample numbers of each antenna. Divide this by the sample frequency to get time
        since the timestame of the first data point. Note that since these are, in general, different, they do NOT refer to sample
        0 of "get_data" in general """
        return self.SampleNumbers
    
    def get_nominal_sample_number(self):
        """return the sample number of the 0th data sample returned by get_data.
        Divide by sample_frequency to get time from timestamp of the 0th data sample"""
        return self.nominal_sample_number
        
    def get_nominal_data_lengths(self):
        """return the number of data samples that are usable for each antenna, accounting for different starting sample numbers.
        returns array of ints"""
        return self.nominal_DataLengths
    
    def get_ITRF_antenna_positions(self, copy=False):
        """returns the ITRF positions of the antennas. Returns a 2D numpy array. If copy is False, then this just returns the internal array of values"""
        if copy:
            return np.array( self.ITRF_dipole_positions )
        else:
            return self.ITRF_dipole_positions
        
    def get_LOFAR_centered_positions(self, out=None):
        """returns the positions (as a 2D numpy array) of the antennas with respect to CS002. 
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated"""
        return md.convertITRFToLocal(self.ITRF_dipole_positions, out=out)
    
    def get_timing_callibration_delays(self):
        """return the timing callibration of the anntennas, as a 1D np array. If not included in the metadata, will look
        for a data file in the same directory as this file. Otherwise returns None"""
        
        if self.have_metadata:
            return self.calibrationDelays
        else:
            fpath = os.path.dirname(self.filename) + '/'+self.StationName
            phase_calibration = md.getStationPhaseCalibration(self.StationName, self.antennaSet, file_location=fpath  )
            phase_calibration = phase_calibration[ self.antenna_filter ]
            return md.convertPhase_to_Timing(phase_calibration, 1.0/self.SampleFrequency)
            
    def get_data(self, start_index, num_points, antenna_index=None, antenna_ID=None):
        """return the raw data for a specific antenna, as an 1D int16 numpy array, of length num_points. First point returned is 
        start_index past get_nominal_sample_number(). Specify the antenna by giving the antenna_ID (which is a string, same
        as output from get_antenna_names(), or as an integer antenna_index. An antenna_index of 0 is the first antenna in 
        get_antenna_names()."""
        
        if antenna_index is None:
            if antenna_ID is None:
                raise LookupError("need either antenna_ID or antenna_index")
            antenna_index = self.dipoleNames.index(antenna_ID)
        else:
            antenna_ID = self.dipoleNames[ antenna_index ]
            
        initial_point = self.sample_offsets[ antenna_index ] + start_index
        final_point = initial_point+num_points
        
        return self.file[ self.stationKey ][ antenna_ID ][initial_point:final_point]
                
class MultiFile_Dal1:
    """A class for reading the data from one station from multiple files"""
    def __init__(self, filename_list, force_metadata_ant_pos=False):
        self.files = [TBBData_Dal1(fname, force_metadata_ant_pos) for fname in filename_list]
        
        #### get some data that should be constant ###
        self.antennaSet = self.files[0].antennaSet
        self.StationID = self.files[0].StationID
        self.StationName = self.files[0].StationName
        self.SampleFrequency = self.files[0].SampleFrequency
        self.FilterSelection = self.files[0].FilterSelection
        self.Time = self.files[0].Time
        
        #### check consistancy of data ####
        self.SampleNumbers = []
        self.dipoleNames = []
        self.antenna_to_file = []
        self.DataLengths = []
        for TBB_file in self.files:
            if TBB_file.antennaSet != self.antennaSet:
                raise IOError("antenna set not the same between files for station: "+self.StationName)
            if TBB_file.StationID != self.StationID:
                raise IOError("station ID not the same between files for station: "+self.StationName)
            if TBB_file.StationName != self.StationName:
                raise IOError("station name not the same between files for station: "+self.StationName)
            if TBB_file.FilterSelection != self.FilterSelection:
                raise IOError("filter selection not the same between files for station: "+self.StationName)
            if TBB_file.Time != self.Time:
                raise IOError("antenna set not the same between files for station: "+self.StationName)
                
            self.dipoleNames += TBB_file.dipoleNames
            self.SampleNumbers += list( TBB_file.SampleNumbers )
            self.antenna_to_file += [TBB_file]*len(TBB_file.dipoleNames)
            self.DataLengths += list(TBB_file.DataLengths)
            
        self.SampleNumbers = np.array( self.SampleNumbers, dtype=int )
        self.DataLengths = np.array(self.DataLengths, dtype=int)        
        
        self.nominal_sample_number = np.max( self.SampleNumbers )
        self.nominal_DataLengths = self.DataLengths + (np.min(self.SampleNumbers) + self.nominal_sample_number)
        self.sample_offsets = self.nominal_sample_number - self.SampleNumbers
                
    #### GETTERS ####
    def needs_metadata(self):
        for TBB_file in self.files:
            if TBB_file.needs_metadata():
                return True
        return False
    
    def get_station_name(self):
        """returns the name of the station, as a string"""
        return self.StationName
    
    def get_station_ID(self):
        """returns the ID of the station, as an integer. This is not the same as StationName. Mapping is givin in utilities"""
        return self.StationID
    
    def get_antenna_names(self):
        """return name of antenna as a list of strings. This is really the RCU id, and the physical antenna depends on the antennaSet"""
        return self.dipoleNames
    
    def get_antenna_set(self):
        """return the antenna set as a string. Typically "LBA_OUTER" """
        return self.antennaSet
    
    def get_sample_frequency(self):
        """gets samples per second. Typically 200 MHz."""
        return self.SampleFrequency
    
    def get_filter_selection(self):
        """return a string that represents the frequency filter used. Typically "LBA_10_90"""
        return self.FilterSelection
        
    def get_timestamp(self):
        """return the POSIX timestamp of the first data point"""
        return self.Time
    
    def get_full_data_lengths(self):
        """get the number of samples stored for each antenna. Note that due to the fact that the antennas do not start recording
        at the exact same instant (in general), this full data length is not all usable
        returns array of ints"""
        return self.DataLengths
    
    def get_all_sample_numbers(self):
        """return numpy array that contains the sample numbers of each antenna. Divide this by the sample frequency to get time
        since the timestame of the first data point. Note that since these are, in general, different, they do NOT refer to sample
        0 of "get_data" """
        return self.SampleNumbers
    
    def get_nominal_sample_number(self):
        """return the sample number of the 0th data sample returned by get_data.
        Divide by sample_frequency to get time from timestamp of the 0th data sample"""
        return self.nominal_sample_number
        
    def get_nominal_data_lengths(self):
        """return the number of data samples that are usable for each antenna, accounting for different starting sample numbers.
        returns array of ints"""
        return self.nominal_DataLengths
    
    def get_ITRF_antenna_positions(self, out=None):
        """returns the ITRF positions of the antennas. Returns a 2D numpy array. 
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated"""
        if out is None:
            out = np.empty( (len(self.dipoleNames), 3) )
        
        i = 0
        for TBB_file in self.files:
            out[i:i+len(TBB_file.dipoleNames)] = TBB_file.ITRF_dipole_positions
            i += len(TBB_file.dipoleNames)
            
        return out
        
    def get_LOFAR_centered_positions(self, out=None):
        """returns the positions (as a 2D numpy array) of the antennas with respect to CS002. 
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated"""
        if out is None:
            out = np.empty( (len(self.dipoleNames), 3) )
        
        i = 0
        for TBB_file in self.files:
            md.convertITRFToLocal(TBB_file.ITRF_dipole_positions, out=out[i:i+len(TBB_file.dipoleNames)])
            i += len(TBB_file.dipoleNames)
            
        return out
    
    def get_timing_callibration_delays(self, out=None):
        """return the timing callibration of the anntennas, as a 1D np array. If not included in the metadata, will look
        for a data file in the same directory as this file. Otherwise returns None.
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated"""
        
        if out is None:
            out = np.empty( len(self.dipoleNames) )
        
        i = 0
        for TBB_file in self.files:
            ret = TBB_file.get_timing_callibration_delays()
            if ret is None:
                return None
            out[i:i+len(TBB_file.dipoleNames)] = ret
            i += len(TBB_file.dipoleNames)
            
        return out
            
    def get_data(self, start_index, num_points, antenna_index=None, antenna_ID=None):
        """return the raw data for a specific antenna, as an 1D int16 numpy array, of length num_points. First point returned is 
        start_index past get_nominal_sample_number(). Specify the antenna by giving the antenna_ID (which is a string, same
        as output from get_antenna_names(), or as an integer antenna_index. An antenna_index of 0 is the first antenna in 
        get_antenna_names()."""
        
        if antenna_index is None:
            if antenna_ID is None:
                raise LookupError("need either antenna_ID or antenna_index")
            antenna_index = self.dipoleNames.index(antenna_ID)
        else:
            antenna_ID = self.dipoleNames[ antenna_index ]
            
        initial_point = self.sample_offsets[ antenna_index ] + start_index
        final_point = initial_point+num_points
        
        TBB_file = self.antenna_to_file[antenna_index]
        
        return TBB_file.file[ TBB_file.stationKey ][ antenna_ID ][initial_point:final_point]
        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    timeID =  "D20170929T202255.000Z"
    station = "RS406"
    antenna_id = 0
    
    block_size = 2**16
    block_number = 3900
    
    raw_fpaths = filePaths_by_stationName(timeID)
    
    infile = MultiFile_Dal1(raw_fpaths[station])
    
    data = infile.get_data(block_number*block_size, block_size, antenna_index=antenna_id)
    
    plt.plot(data)
    plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
    