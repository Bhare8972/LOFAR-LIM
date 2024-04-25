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
    
    
    
This whole file suffers from being the oldest and most important file in LoLIM. As such, it maintains backwards compatibility with many 'old' ways of doing things.
"""

##### TODO:
## add a way to combine event that is spread across close timeID  (is this necisary?)
## add proper fitting phase vs frequency to lines, adn func to return the frequency independand phase offset




import os
import datetime

import numpy as np
import h5py

import LoLIM.IO.metadata as md
import LoLIM.utilities as util
import LoLIM.atmosphere as atmo


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

########  The following four functions read what I call "correction files" these are corrections made to improve the data ##########
## THESE are the "old" way of storing calibrations.  New best-practice is total-cal below ##

def read_antenna_pol_flips(fname):
    antennas_to_flip = []
    with open(fname) as fin:
        for line in fin:
            ant_name = line.split()[0]
            antennas_to_flip.append( ant_name )
    return antennas_to_flip

def read_bad_antennas(fname):
    bad_antenna_data = []
    
    def parse_line_v1(line):
        ant_name, pol = line.split()[0:2]
        # bad_antenna_data.append((ant_name,int(pol)))
        if pol: 
            bad_antenna_data.append(  util.even_antName_to_odd( ant_name )  )
        else:
            bad_antenna_data.append( ant_name )
    
    def parse_line_v2(line):
        ant_name = line.split()[0]
        bad_antenna_data.append(  ant_name  )
        # pol = 0
        # if not util.antName_is_even(ant_name):
        #     ant_name = util.even_antName_to_odd(ant_name)
        #     pol = 1
        # bad_antenna_data.append((ant_name,pol))
        
    version = 1
    with open(fname) as fin:
        is_line_0 = True
        for line in fin:
            if is_line_0 and line[:2] == 'v2':
                version = 2
            else:
                if version == 1:
                    parse_line_v1( line )
                elif version == 2:
                    parse_line_v2( line )
                
            if is_line_0:
                is_line_0 = False
                
                
            
    return bad_antenna_data

def read_antenna_delays(fname):
    additional_ant_delays = {}
    
    def parse_line_v1(line):
        ant_name, pol_E_delay, pol_O_delay = line.split()[0:3]
        additional_ant_delays[ant_name] = [float(pol_E_delay), float(pol_O_delay)]
        
    def parse_line_v2(line):
        ant_name, delay = line.split()[0:2]
        pol = 0
        if not util.antName_is_even(ant_name):
            ant_name = util.even_antName_to_odd(ant_name)
            pol = 1
            
        if ant_name not in additional_ant_delays:
            additional_ant_delays[ant_name] = [0.0, 0.0]
            
        additional_ant_delays[ant_name][pol] = float(delay)
    
    parse_function = parse_line_v1
    with open(fname) as fin:
        is_line_0=True
        for line in fin:
            if is_line_0 and line[0] == 'v':
                if line[:2]=='v1':
                    pass
                elif line[:2]=='v2':
                    parse_function = parse_line_v2
            else:
                parse_function(line)
                
            if is_line_0:
                is_line_0 = False
            
    return additional_ant_delays

def read_station_delays(fname):
    station_delays = {}
    with open(fname) as fin:
        for line in fin:
            sname, delay = line.split()[0:2]
            station_delays[sname] = float(delay)
    return station_delays

### this replaces the above four, and reads from one "ultimate" cal file
def read_cal_file(fname, pol_flips_are_bad, timeID=None):
    
    # station_delays = {}
    # bad_antenna_data = []
    # ant_delays = {}
    # polarization_flips = []
    # sign_flips = []

    if timeID is not None:
        dir = util.processed_data_dir( timeID )
        fname = os.path.join( dir, fname )



    RET = total_cal_object()
    
    with open(fname) as fin:
        version = fin.readline() ## version of type of file. Presently unused as we only have one version

        if version[0]=='O':
            return read_olaf_file(fname, pol_flips_are_bad, unCalAnts_are_bad=False)


        mode = 1 ## 1 = bad_antennas 2 = pol_flips 3 = station_delays 4 = antenna_delays
        
        for line in fin:

            line_data = line.split()
            
            if len(line_data) == 0: # empyt line
                continue
            
            if line_data[0][0] == '#':
                ## comment!
                continue
        
            ## first check mode
            
            if line_data[0] == "MDA": ## metadata adjusts
                if line_data[1] == 'ANTENNA_SET':
                    RET.metadata_adjusts['ANTENNA_SET'] = line_data[2]
                    
            elif line_data[0] == "bad_antennas":
                mode = 1
            elif line_data[0] == "pol_flips":
                mode = 2
            elif line_data[0] == "station_delays":
                mode = 3
            elif line_data[0] == "antenna_delays":  ## these should be AFTER pol_flips. I.E., do not flip with pol_flips
                mode = 4
            elif line_data[0] == "sign_flips":
                mode = 5
            elif line_data[0] == "vair":
                v = float(line_data[1])
                util.set_vair( v )
                RET.atmosphere = atmo.simple_atmosphere( v )
                
            ### now we parse
            elif mode == 1 : ## bad antennas
                RET.bad_antenna_data.append( line_data[0] )
            elif mode == 2:
                if pol_flips_are_bad:
                    RET.bad_antenna_data.append( util.antName_to_even( line_data[0] ) )
                    RET.bad_antenna_data.append( util.antName_to_odd( line_data[0] ) )
                else:
                    RET.polarization_flips.append( util.antName_to_even( line_data[0] ) )
            elif mode == 3:
                RET.station_delays[ line_data[0] ] = float( line_data[1] )
            elif mode == 4:
                RET.ant_delays[ line_data[0] ] = float( line_data[1] )
            elif mode == 5:
                RET.sign_flips.append( line_data[0] )

            ## err
            else:
                print('reading cal file error! in mode:', mode, 'line:', line)
        
    # return bad_antenna_data, polarization_flips, station_delays, ant_delays
    return RET

class total_cal_object:
    def __init__(self):
        self.atmosphere = atmo.default_atmosphere
        self.bad_antenna_data = []
        self.polarization_flips = []
        self.station_delays = {}
        self.ant_delays = {}
        self.sign_flips = []

        self.metadata_adjusts = {}


def read_olaf_file(fname, pol_flips_are_bad, unCalAnts_are_bad=True):

    def antSAI_to_antname( SAI ):
        if len(SAI)==9:
            return SAI

        ant_num = SAI[-3:]
        sname = SAI[:-3]
        sname = sname.zfill(3)
        RCU = str( int(int(ant_num)/8) ).zfill(3)
        return sname + RCU + ant_num

        
    
    RET = total_cal_object()
    RET.atmosphere = atmo.olaf_constant_atmosphere
    
    with open(fname) as fin:
        version = fin.readline() ## version of type of file. Presently unused as we only have one version

        mode = 1 ## 1 = antenna delays, 2 = station delays, 3 = bad antennas, 4 = sign flips 
        
        for line in fin:
            if len(line) == 0: # emptu line
                continue

            line_data = line.split()
            
          #  if line_data[0][0] == '#':
                ## comment!
          #      continue
        
            ## first check mode
            
            if line_data[0][0] == "=":
                mode = 2
            elif line_data[0] == "bad_antennas":
                mode = 3
            elif line_data[0] == "sign_flips":
                mode = 4
            #elif line_data[0] == "pol_flips":
            #    mode = 5
                
            ### now we parse
            elif mode == 1:
                antSAI, delay_samples, do = line_data
                antname = antSAI_to_antname( antSAI )

                if (not int(do)) and unCalAnts_are_bad:
                    RET.bad_antenna_data.append( antname )
                else:
                    RET.ant_delays[ antname ] = float( delay_samples )*5.0e-9


            elif mode == 2:
                RET.station_delays[ line_data[0] ] = float( line_data[1] )*5.0e-9


            elif mode == 3:
                antname = antSAI_to_antname( line_data[0] )
                RET.bad_antenna_data.append( antname )

            elif mode == 4:
                antname = antSAI_to_antname( line_data[0] )
                RET.sign_flips.append( antname )

            else:
                print('reading cal file error! in mode:', mode, 'line:', line)
        
    # return bad_antenna_data, polarization_flips, station_delays, ant_delays
    return RET


#### data reading class ####
# Note: ASTRON will "soon" release a new DAL (data )

def decode_if_needed(IN):
    if not isinstance(IN, str):
        return IN.decode()
    return IN
    

#### NOTE:  metadata_adjusts is a dictionary of fixes to the metadata.
  ## key is string, which is type of metadata to adjust
  ## value is the new metadata info. Depends of type.
  ## types:
    ## ANTENNA_SET:   value is string ("LBA_OUTER", "LBA_SPARSE", etc.)  use this value for antenna set instead of that in file

class TBBData_Dal1:
    """a class for reading one station from one file. However, since one station is often spread between different files, 
    use filePaths_by_stationName combined with MultiFile_Dal1 below.
    This class is kept simple, and MultiFile_Dal1 has all the fancy features"""
    
    def __init__(self, filename, force_metadata_ant_pos=False, forcemetadata_delays=True, metadata_adjusts={}):
        self.filename = filename
        self.force_metadata_ant_pos = force_metadata_ant_pos
        self.forcemetadata_delays = forcemetadata_delays
        
        #### open file and set some basic info####
        self.file = h5py.File(filename, "r")
        
        stationKeys = [s for s in self.file.keys() if s.startswith('Station')]
        ## assume there is only one station in the file
        if len(stationKeys) != 1:
            print("WARNING! file", self.filename, "has more then one station")
        self.stationKey = stationKeys[0]
        
        if 'ANTENNA_SET' in metadata_adjusts:
            self.antennaSet = metadata_adjusts['ANTENNA_SET']
        else:
            self.antennaSet = decode_if_needed( self.file.attrs['ANTENNA_SET'][0] )


        self.dipoleNames = list( self.file[ self.stationKey ].keys() )
        self.StationID = self.file[ self.stationKey ][ self.dipoleNames[0] ].attrs["STATION_ID"][0]
        self.StationName = util.SId_to_Sname[ self.StationID ]
        ## assume all antennas have the same sample frequency
        
        self.SampleFrequency = self.file[ self.stationKey ][ self.dipoleNames[0] ].attrs["SAMPLE_FREQUENCY_VALUE"
            ][0]*conversiondict[ decode_if_needed( self.file[ self.stationKey ][ self.dipoleNames[0] ].attrs["SAMPLE_FREQUENCY_UNIT"][0] ) ]
        ## filter selection is typically "LBA_10_90"
        self.FilterSelection = decode_if_needed( self.file.attrs['FILTER_SELECTION'][0] )
        
        
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
        
        
        # load antenna locations from metadata and from file. IF they are too far apart, then give warning, and use metadata
        self.ITRF_dipole_positions = md.getItrfAntennaPosition(self.StationName, self.antennaSet)[ self.antenna_filter ] ## load positions from metadata file
        if self.have_metadata and not self.force_metadata_ant_pos:
            
            use_TBB_positions = True
            TBB_ITRF_dipole_positions = np.empty((len(self.dipoleNames), 3), dtype=np.double)
            for i,dipole in enumerate(self.dipoleNames):
                TBB_ITRF_dipole_positions[i] = self.file[ self.stationKey ][dipole].attrs['ANTENNA_POSITION_VALUE']
                
                dif = np.linalg.norm( TBB_ITRF_dipole_positions[i]-self.ITRF_dipole_positions[i] )
                if dif > 1:
                    print("WARNING: station", self.StationName, "has suspicious antenna locations. Using metadata instead")
                    use_TBB_positions = False
                
            if use_TBB_positions:
                self.ITRF_dipole_positions = TBB_ITRF_dipole_positions
            
            
            
        self.calibrationDelays = np.zeros( len(self.dipoleNames), dtype=np.double ) ## defined as callibration values in file. Never from external metadata!
        if self.have_metadata:# and not self.forcemetadata_delays:
            
            for i,dipole in enumerate(self.dipoleNames):
                self.calibrationDelays[i] = self.file[ self.stationKey ][dipole].attrs['DIPOLE_CALIBRATION_DELAY_VALUE']
            
        
        
        #### get the offset, in number of samples, needed so that each antenna starts at the same time ####
        self.nominal_sample_number = np.max( self.SampleNumbers )
        self.sample_offsets = self.nominal_sample_number - self.SampleNumbers
        self.nominal_DataLengths = self.DataLengths - self.sample_offsets
        
    #### PICKLING ####
        ## this is for multiprocessing. Note that doing this KILLS this file. Otherwise bugs result (I think?)
        # it doesn't work
    # def __getstate__(self):
    #     if self.file is not None:
    #         self.file.close()
    #         self.file = None
    #     d = dict( self.__dict__ )
    #     # d['file'] = None
    #     return d
                        
    # def __setstate__(self, d):
    #     self.__dict__ = d
    #     self.file = h5py.File(self.filename, "r")
        
            
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
    
    def get_timing_callibration_phases(self):
        """only a test function for the moment, do not use"""
        fpath = os.path.dirname(self.filename) + '/'+self.StationName
        phase_calibration = md.getStationPhaseCalibration(self.StationName, self.antennaSet,self.FilterSelection, file_location=fpath  )
        phase_calibration = phase_calibration[ self.antenna_filter ]
        return phase_calibration
    
    def get_timing_callibration_delays(self, force_file_delays=False):
        """return the timing callibration of the anntennas, as a 1D np array. If not included in the metadata, will look
        for a data file in the same directory as this file. Otherwise returns None"""
        
        if (self.have_metadata and not self.forcemetadata_delays) or force_file_delays:
            return self.calibrationDelays
        else:
            fpath = os.path.dirname(self.filename) + '/'+self.StationName
            phase_calibration = md.getStationPhaseCalibration(self.StationName, self.antennaSet,self.FilterSelection, file_location=fpath  )
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
    def __init__(self, filename_list, force_metadata_ant_pos=True, total_cal=None,
                 polarization_flips=None, bad_antennas=[], additional_ant_delays=None, station_delay=0.0, 
                 only_complete_pairs=True, pol_flips_are_bad=False):
        """filename_list:  list of filenames for this station for this event.
            force_metadata_ant_pos -if True, then load antenna positions from a metadata file and not the raw data file. Default True
            total_cal              - ...many behaviors. If a string, then assumes total_cal is file name, and is opened by read_cal_file. Else, should be output of read_cal_file.
                                    if total_cal is boolean True, than following callibrations are assumed to be a total_cal (explained better below in additional_ant_delays).

            If total_cal is not string or read_cal_file output, then following parameters can be used to specify calibration. These should probably be avoided:

            polarization_flips     -list of even antennas where it is known that even and odd antenna names are flipped in file. This is assumed to apply both to data and timing calibration
            bad_antennas           -list of antenna names that should not be used. Assumed to be BEFORE antenna flips are accounted for.
            additional_ant_delays  -a dictionary. Should rarely be needed. Behavior depends on total_cal
                                      If total_cal is None or boolean False (wierd and depreciated):
                                        Each key is name of even antenna, each value is a tuple with additional even and odd antenna delays, added to delays in TBB file.
                                        assumed to be found BEFORE antenna flips are accounted for. (i.e. get flipped with pol_flips)
                                      If total_cal is boolean True (probably also rarely used):
                                        Each key is antenna name, and value is delay. Even/odd antennas have different entries.
                                        additional_ant_delays IS the antenna time calibration and values from TBB file are discarded.
                                        additional_ant_delays are assumed to be found AFTER antenna flips are accounted for. (ie. do NOT flip with pol-flips)
            station_delay          -a single number that represents the clock offset of this station, as a delay
            NOTE: polarization_flips, bad_antennas, additional_ant_delays, and station_delay can now be strings that are file names. If this is the case, they will be read automatically. This behavior is depreciated.

            OTHER settings:

            only_complete_pairs    -if True, discards antenna if the other in pair is not present or is bad. If False, keeps all good antennas.
                                      If False, needs to use 'has_antenna' method. To make sure antenna actually exists! (see doc for method).
            pol_flips_are_bad      -if True, antennas that are in pol-flips are included in 'bad_antennas' 
        """

        if isinstance(total_cal, str):
            total_cal = read_cal_file(total_cal, pol_flips_are_bad)

        if isinstance(total_cal, total_cal_object):
            self.files = [TBBData_Dal1(fname, force_metadata_ant_pos, metadata_adjusts=total_cal.metadata_adjusts) for fname in filename_list]
        else:
            self.files = [TBBData_Dal1(fname, force_metadata_ant_pos) for fname in filename_list]
                
        #### get some data that should be constant #### TODO: change code to make use of getters
        self.antennaSet = self.files[0].antennaSet
        self.StationID = self.files[0].StationID
        self.StationName = self.files[0].StationName
        self.SampleFrequency = self.files[0].SampleFrequency
        self.FilterSelection = self.files[0].FilterSelection
        self.Time = self.files[0].Time
        
        


        if total_cal is not None:
            self.using_total_cal = True
            
            if isinstance(total_cal, total_cal_object):
                self.total_cal = total_cal
            elif total_cal is True: # note the 'is' only works for literal True
                self.total_cal = total_cal_object()

                self.total_cal.bad_antenna_data = bad_antennas
                self.total_cal.polarization_flips = polarization_flips
                self.total_cal.ant_delays = additional_ant_delays
                self.total_cal.station_delays = {self.StationName:station_delay} ## hack to be compatible with below

                additional_ant_delays = None ## since this should really be a different thing
            else:
                print('argument "total_cal", wierd type:', type(total_cal)  )
                quit()

            ## TO BE COMPATIBLE WITH CODE BELOW
            bad_antennas = self.total_cal.bad_antenna_data
            polarization_flips = self.total_cal.polarization_flips
            if self.StationName in self.total_cal.station_delays:
                station_delay = self.total_cal.station_delays[ self.StationName ]
            else:
                station_delay = 0.0
            
        else:
            self.using_total_cal = False
            
            if isinstance(polarization_flips, str):
                polarization_flips = read_antenna_pol_flips( polarization_flips )
            if isinstance(bad_antennas, str):
                bad_antennas = read_bad_antennas( bad_antennas )
            if isinstance(additional_ant_delays, str):
                additional_ant_delays = read_antenna_delays( additional_ant_delays )
                
            if isinstance(station_delay, str):
                station_delay = read_station_delays( station_delay )[ self.StationName ]
                
            if polarization_flips is not None and pol_flips_are_bad:
                for even_ant in polarization_flips:
                    bad_antennas.append( even_ant )
                    bad_antennas.append( util.even_antName_to_odd( even_ant ) )
                    # bad_antennas.append( (even_ant,0) )
                    # bad_antennas.append( (even_ant,1) )
                polarization_flips = []
        


        self.bad_antennas = bad_antennas
        self.odd_pol_additional_timing_delay = 0.0 # anouther timing delay to add to all odd-polarized antennas. Should remain zero if using_total_cal
        self.station_delay = station_delay
        
        
        #### check consistancy of data ####
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
        
        ## check LBA outer antenna set
        if self.antennaSet != "LBA_OUTER":
            print("WARNING: antenna set on station", self.StationName, "is not LBA_OUTER. TBB reading only tested for LBA_OUTER")
        
        
        #### find best files to get antennas from ####
        ## require each antenna shows up once, and even pol is followed by odd pol
        
        self.dipoleNames = []
        self.antenna_to_file = [] ##each item is a tuple. First item is file object, second is antenna index in file
        
        unused_antenna_names = []
        unused_antenna_to_file = []
        for TBB_file in self.files:
            file_ant_names = TBB_file.get_antenna_names()
            
            for ant_i,ant_name in enumerate(file_ant_names):
                if ant_name in bad_antennas:
                    continue
                
                if (ant_name in self.dipoleNames):
                    continue
                
                if util.antName_is_even(ant_name):
                    
                    odd_ant_name = util.even_antName_to_odd( ant_name )
                    if odd_ant_name in unused_antenna_names: ## we have the odd antenna
                        self.dipoleNames.append(ant_name)
                        self.dipoleNames.append(odd_ant_name)
                        
                        self.antenna_to_file.append( (TBB_file, ant_i)  )
                        odd_unused_index = unused_antenna_names.index( odd_ant_name )
                        self.antenna_to_file.append( unused_antenna_to_file[ odd_unused_index ]  )
                        
                        unused_antenna_names.pop( odd_unused_index )
                        unused_antenna_to_file.pop( odd_unused_index )
                    else: ## we haven't found the odd antenna, so store info for now
                        unused_antenna_names.append(ant_name)
                        unused_antenna_to_file.append( (TBB_file, ant_i)  )
                        
                else: ## antenna is odd
                    even_ant_name =  util.odd_antName_to_even( ant_name )
                    
                    if even_ant_name in unused_antenna_names: ## we have the even antenna
                        self.dipoleNames.append(even_ant_name)
                        self.dipoleNames.append(ant_name)
                        
                        even_unused_index = unused_antenna_names.index( even_ant_name )
                        self.antenna_to_file.append( unused_antenna_to_file[ even_unused_index ]  )
                        
                        unused_antenna_names.pop( even_unused_index )
                        unused_antenna_to_file.pop( even_unused_index )
                        
                        self.antenna_to_file.append( (TBB_file, ant_i)  )
                        
                    else: ## we haven't found the odd antenna, so store info for now
                        unused_antenna_names.append(ant_name)
                        unused_antenna_to_file.append( (TBB_file, ant_i)  )
                        
        if not only_complete_pairs:
            for ant_name, to_file in zip(unused_antenna_names, unused_antenna_to_file):
                if util.antName_is_even(ant_name): ##check if antenna is even
                    
                    self.dipoleNames.append( ant_name )
                    self.antenna_to_file.append( to_file )
                    
                    self.dipoleNames.append( util.even_antName_to_odd( ant_name ) ) ## add the odd antenna
                    # self.dipoleNames.append( None ) ## add the odd antenna
                    self.antenna_to_file.append( None ) ## doesn't exist in a file
                    
                else:
                    
                    self.dipoleNames.append( util.odd_antName_to_even( ant_name ) ) ## add the even antenna
                    # self.dipoleNames.append( None) ## add the even antenna
                    self.antenna_to_file.append( None ) ## doesn't exist in a file
                    
                    self.dipoleNames.append( ant_name )
                    self.antenna_to_file.append( to_file )
                    
                    
        if len(self.dipoleNames) == 0:
            print('station', self.StationName, 'has no antennas')
            return
                        
        self.index_adjusts = np.arange( len(self.antenna_to_file) )
        ### when given an antnna index to open data, use this index instead to open the correct data location
        ## internal_index = self.index_adjusts[ external_index ]
        ## note, this does NOT apply to the name of the antenna
        ## main job is to keep order of antenans as Y-X. This is for pol-flips and different modes

        self.name_adjusts = np.arange( len(self.antenna_to_file) )
        ### this is the same as above, but only applies to names of antennas. Job is to keep order of antennas to be Y-X


        if self.antennaSet == "LBA_INNER":
            for pair_i in range( int(len(self.antenna_to_file)/2) ):
                self.index_adjusts[pair_i*2]   += 1
                self.index_adjusts[pair_i*2+1] -= 1
                self.name_adjusts[pair_i*2]   += 1
                self.name_adjusts[pair_i*2+1] -= 1
        elif self.antennaSet == "LBA_SPARSE_EVEN":
##0 X    inner
# 1 Y    inner
# 2 Y    outer
# 3 X    outer
# 4 X    inner
# 5 Y    inner
# 6 Y    outer
# 7 X    outer

            for pair_i in range( int(len(self.antenna_to_file)/2) ):
                antName_Y = self.dipoleNames[ pair_i*2 ]
                NamePairI = int( int(antName_Y)/2 )

                EvenIsX = (NamePairI%2)==0   ## antenna pair index (of the name) is even
                AntNameIsEven = (int(antName_Y)%2)==0  ## should always be true?
                if EvenIsX and AntNameIsEven:
                    ## then this is an X antenna, and not a Y antenna, and they need a flipie-flipie
                    self.index_adjusts[pair_i*2]   += 1
                    self.index_adjusts[pair_i*2+1] -= 1
                    self.name_adjusts[pair_i*2]   += 1
                    self.name_adjusts[pair_i*2+1] -= 1

        
                        
        #### get sample numbers and offsets and lengths and other related stuff ####
        self.SampleNumbers = []   ## both set to -1 for antennas that do not exist
        self.DataLengths = []
        for F in self.antenna_to_file:
            if F is None:
                self.SampleNumbers.append( -1 )
                self.DataLengths.append( -1 )
            else:
                TBB_file, file_ant_i = F
                self.SampleNumbers.append( TBB_file.SampleNumbers[file_ant_i] )
                self.DataLengths.append( TBB_file.DataLengths[file_ant_i]  )
            
        self.SampleNumbers = np.array( self.SampleNumbers, dtype=int )
        self.DataLengths = np.array(self.DataLengths, dtype=int)    
        
        self.nominal_sample_number = np.max( self.SampleNumbers )
        self.sample_offsets = self.nominal_sample_number - self.SampleNumbers
        self.nominal_DataLengths = self.DataLengths - self.sample_offsets
        
        self.ant_pol_flips = None
        if polarization_flips is not None:
            self.set_polarization_flips( polarization_flips )
        self.additional_ant_delays = additional_ant_delays

    def set_polarization_flips(self, antenna_names):
        """given a set of names(IDs) of even antennas, flip the data between the even and odd antennas"""
        self.ant_pol_flips = antenna_names
        for ant_name in antenna_names:
            if ant_name in self.dipoleNames:
                antenna_index = self.dipoleNames.index(ant_name)
                even_antenna_index = int(antenna_index/2)*2  ## assumes adjusts are always flips!!

                self.index_adjusts[even_antenna_index] += 1
                self.index_adjusts[even_antenna_index+1] -= 1
                
    # def set_odd_polarization_delay(self, new_delay):
    #     self.odd_pol_additional_timing_delay = new_delay
        
    def set_station_delay(self, station_delay):
        """ set the station delay, should be a number"""
        self.station_delay = station_delay
                
    ## this should be depreciated
    # def find_and_set_polarization_delay(self, verbose=False, tolerance=1e-9):
    #     if self.using_total_cal:
    #         print('warning: calibration probably already accounts for polarized delay. IN: find_and_set_polarization_delay')
    #         print('   note find_and_set_polarization_delay does NOT use totalcal timing data, but ASTRON-provided data')
        
    #     fpath = os.path.dirname(self.files[0].filename) + '/'+self.StationName
    #     phase_calibration = md.getStationPhaseCalibration(self.StationName, self.antennaSet,self.FilterSelection, file_location=fpath  )
    #     all_antenna_calibrations = md.convertPhase_to_Timing(phase_calibration, 1.0/self.SampleFrequency) 
        
    #     even_delays = all_antenna_calibrations[::2]
    #     odd_delays = all_antenna_calibrations[1::2]
    #     odd_offset = odd_delays-even_delays
    #     median_odd_offset = np.median( odd_offset )
    #     if verbose:
    #         print("median offset is:", median_odd_offset)
    #     below_tolerance = np.abs( odd_offset-median_odd_offset ) < tolerance
    #     if verbose:
    #         print(np.sum(below_tolerance), "antennas below tolerance.", len(below_tolerance)-np.sum(below_tolerance), "above.")
    #     ave_best_offset = np.average( odd_offset[below_tolerance] )
    #     if verbose:
    #         print("average of below-tolerance offset is:", ave_best_offset)
    #     self.set_odd_polarization_delay( -ave_best_offset )
        
    #     above_tolerance = np.zeros( len(all_antenna_calibrations), dtype=bool )
    #     above_tolerance[::2] = np.logical_not( below_tolerance )
    #     above_tolerance[1::2] = above_tolerance[::2]
    #     above_tolerance = above_tolerance[ md.make_antennaID_filter(self.get_antenna_names()) ]
    #     return [AN for AN, AT in zip(self.get_antenna_names(),above_tolerance) if AT]
        
                
    #### GETTERS ####
    def needs_metadata(self):
        for TBB_file in self.files:
            if TBB_file.needs_metadata():
                return True
        return False
    
    def get_station_name(self):
        """returns the name of the station, as a string"""
        return self.StationName


    def get_station_delay(self):
        """ return the station delay"""
        return self.station_delay
                
    
    def get_station_ID(self):
        """returns the ID of the station, as an integer. This is not the same as StationName. Mapping is givin in utilities"""
        return self.StationID
    
    def get_antenna_names(self):
        """return name of antenna as a list of strings. This is really the RCU id, and the physical antenna depends on the antennaSet.
        Note that antenna_id used elsewhere in this class is the index of this list (order stays the same!).
        However, order and such is nto garunteed between different class instances, thus if you need to save info, save antenna_name, NOT antenna_ID.
        even indeces are always Y-oriented dipole, which are even-named antennas only for LBA_OUTER.
        Thus, if only_complete_pairs was false, some antennas may be here but not actually exist. Thus you need method "has_antenna"."""
        ret = []
        for i in self.name_adjusts:
            ret.append( self.dipoleNames[i] )
        return ret
        #return self.dipoleNames
    
    def has_antenna(self, antenna_name=None, antenna_index=None):
        """if only_complete_pairs is False, then we could have antenna names without the data.
        Give either antenna_name or antenna_index.
        Return True if we actually have the antenna, False otherwise. This accounts for polarization flips."""
        if antenna_name is not None:
            if antenna_name in self.dipoleNames:
                internal_name_index = self.dipoleNames.index(antenna_name)
                antenna_index = self.name_adjusts.index( internal_name_index )
            else:
                return False

        internal_antenna_index = self.index_adjusts[ antenna_index ]


        if self.antenna_to_file[ internal_antenna_index ] is None:
            return False
        else:
            return True

    def get_XYdipole_indeces(self):
        """Return 2D [i,j] array of integers. i is the index of the "full" dual-polarizated antenna (half of length of get_antenna_names). j is 0 for X-dipoles (odd for LBA_OUTER), 1 for Y-dipoles.
        Value is antenna_id (index of get_antenna_names). Will be -1 if antenna does not exist
        NOTE: this method may be phased out. As in future, order will always be Y-X"""

        n_pairs = int(len(self.dipoleNames)/2)
        ret = np.full( (n_pairs,2), -1, dtype=np.int)
        for pair_i in range(n_pairs):
            #if self.antennaSet == "LBA_OUTER":
            X_index = 2*pair_i + 1
            Y_index = 2*pair_i
            #elif self.antennaSet == "LBA_INNER":
                #X_index = 2*pair_i
                #Y_index = 2*pair_i + 1
            #else:
            #    print('unknown antenna set in get_XYdipole_indeces:', self.antennaSet)
            #    return None

            if self.has_antenna(antenna_ID=X_index):
                ret[pair_i, 0] = X_index
            if self.has_antenna(antenna_ID=Y_index):
                ret[pair_i, 1] = Y_index

        return ret


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
    
    def get_timestamp_as_datetime(self):
        """return the POSIX timestampe of the first data point as a python datetime localized to UTC"""
        return datetime.datetime.fromtimestamp( self.get_timestamp(), tz=datetime.timezone.utc )
    
    def get_full_data_lengths(self):
        """get the number of samples stored for each antenna. Note that due to the fact that the antennas do not start recording
        at the exact same instant (in general), this full data length is not all usable
        returns array of ints. Value is -1 if antenna does not exist."""

        ret = np.empty(len(self.DataLengths), dtype=int)
        for outI, correct_i in enumerate(self.order_adjusts):
            ret[outI] = self.DataLengths[correct_i]
        return ret
    
    def get_all_sample_numbers(self):
        """return numpy array that contains the sample numbers of each antenna. Divide this by the sample frequency to get time
        since the timestame of the first data point. Note that since these are, in general, different, they do NOT refer to sample
        0 of "get_data". Value is -1 if antenna does not exist """
        ##return self.SampleNumbers

        ret = np.empty(len(self.SampleNumbers), dtype=int)
        for external_index, internal_index in enumerate(self.index_adjusts):
            ret[external_index] = self.SampleNumbers[internal_index]
        return ret
    
    
    def get_nominal_sample_number(self):
        """return the sample number of the 0th data sample returned by get_data.
        Divide by sample_frequency to get time from timestamp of the 0th data sample"""
        return self.nominal_sample_number
        
    def get_nominal_data_lengths(self):
        """return the number of data samples that are usable for each antenna, accounting for different starting sample numbers.
        returns array of ints"""
        #return self.nominal_DataLengths

        ret = np.empty(len(self.nominal_DataLengths), dtype=int)
        for external_index, internal_index in enumerate(self.index_adjusts):
            ret[ external_index ] = self.nominal_DataLengths[ internal_index ]
        return ret
    
    def get_ITRF_antenna_positions(self, out=None):
        """returns the ITRF positions of the antennas. Returns a 2D numpy array. 
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated.
        Does not account for polarization flips, but shouldn't need too."""
        if out is None:
            out = np.empty( (len(self.dipoleNames), 3) )
        
        #for ant_i, (TBB_file,station_ant_i) in enumerate(self.antenna_to_file):
            #out[ant_i] = TBB_file.ITRF_dipole_positions[station_ant_i]
        
        for external_index, internal_index in enumerate(self.index_adjusts):
            TBB_file,station_ant_i = self.antenna_to_file[ internal_index ]

            out[ external_index ] = TBB_file.ITRF_dipole_positions[station_ant_i]
            
        return out
        
    def get_LOFAR_centered_positions(self, out=None):
        """returns the positions (as a 2D numpy array) of the antennas with respect to CS002. 
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated.
        Does not account for polarization flips, but shouldn't need too."""
        if out is None:
            out = np.empty( (len(self.dipoleNames), 3) )
        
        md.convertITRFToLocal( self.get_ITRF_antenna_positions(), out=out )
            
        return out
    
    ##def get_timing_callibration_phases(self):
        """only a test function for the moment, do not use"""
        
        # out = [None for i in range(len(self.dipoleNames))]
        
        # for TBB_file in self.files:
        #     ret = TBB_file.get_timing_callibration_phases()
        #     if ret is None:
        #         return None
            
        #     #for ant_i, (TBB_fileA,station_ant_i) in enumerate(self.antenna_to_file):
        #     #    if TBB_fileA is TBB_file:
        #     #        out[ant_i] = ret[station_ant_i]
            
        #     for ant_i, adjust_i in enumerate(self.order_adjusts):
        #         TBB_fileA,station_ant_i = self.antenna_to_file[ adjust_i ]
        #         if TBB_fileA is TBB_file:
        #             out[ant_i] = ret[station_ant_i]
                    
        # return np.array(out)
    
    def get_timing_callibration_delays(self, out=None, force_file_delays=False):
        """return the timing callibration of the anntennas, as a 1D np array. If not included in the TBB_metadata, will look
        for a data file in the same directory as this file. Otherwise returns None.
        if out is a numpy array, it is used to store the antenna delays, otherwise a new array is allocated. 
        This takes polarization flips, and additional_ant_delays into account.
        Also can account for a timing difference between even and odd antennas, if it is set.
        if force_file_delays is True, then will only return calibration values from TBB metadata"""
        
        if out is None:
            out = np.zeros( len(self.dipoleNames), dtype=np.double )
        
        if self.using_total_cal and not force_file_delays:
            #for i,ant_name in enumerate( self.dipoleNames ):
            #    if ant_name in self.total_cal.ant_delays:
            #        out[i] = self.total_cal.ant_delays[ant_name]
            for external_index, internal_index in enumerate( self.name_adjusts):
                ant_name = self.dipoleNames[internal_index]
                if ant_name in self.total_cal.ant_delays:
                    out[ external_index ] = self.total_cal.ant_delays[ant_name]
                else:
                    print("WARNING: cal. for antenna missing:", ant_name)
                    out[ external_index ] = 0.0
            
        else:## this whole clunky thing is purly historical. SHould be depreciated!!

            for TBB_file in self.files:
                ret = TBB_file.get_timing_callibration_delays(force_file_delays)
                if ret is None:
                    return None
                
                for external_index, internal_data_index in enumerate( self.index_adjusts ):
                    internal_name_index = self.name_adjusts[ external_index ]

                    TBB_fileA,station_ant_i = self.antenna_to_file[ internal_data_index ]
                    
                    if TBB_fileA is TBB_file:
                        out[ external_index ] = ret[station_ant_i]
                        
                        if self.additional_ant_delays is not None:
                            print('antenna delay code probably not working right')

                            ## additional_ant_delays stores only even antenna names for historical reasons. so we need to be clever here
                            ## this is probably completely wrong
                            antenna_polarization = 0 if (external_index%2==0) else 1 
                            even_ant_name = self.dipoleNames[ internal_name_index-antenna_polarization ]
                            if even_ant_name in self.additional_ant_delays:
                                if even_ant_name in self.ant_pol_flips:
                                    antenna_polarization = int(not antenna_polarization)
                                out[ant_i] += self.additional_ant_delays[ even_ant_name ][ antenna_polarization ]


                # for ant_i, adjust_i_A in enumerate(self.order_adjusts):
                #     adjust_i = self.index_adjusts[adjust_i_A]

                #     TBB_fileA,station_ant_i = self.antenna_to_file[adjust_i]
                    
                #     if TBB_fileA is TBB_file:
                #         out[ant_i] = ret[station_ant_i]
                        
                #         if self.additional_ant_delays is not None:
                #             ## additional_ant_delays stores only even antenna names for historical reasons. so we need to be clever here
                #             antenna_polarization = 0 if (ant_i%2==0) else 1 
                #             even_ant_name = self.dipoleNames[ ant_i-antenna_polarization ]
                #             if even_ant_name in self.additional_ant_delays:
                #                 if even_ant_name in self.even_ant_pol_flips:
                #                     antenna_polarization = int(not antenna_polarization)
                #                 out[ant_i] += self.additional_ant_delays[ even_ant_name ][ antenna_polarization ]
                        
        out[1::2] += self.odd_pol_additional_timing_delay
            
        return out
    
    def get_total_delays(self, out=None, force_file_delays=False):
        """Return the total delay for each antenna, accounting for all antenna delays, polarization delay, station clock offsets, and trigger time offsets (nominal sample number).
        This function should be prefered over 'get_timing_callibration_delays', but the offsets can have a large average. It is recomended to pick one antenna (on your referance station)
        and use it as a referance antenna so that it has zero timing delay. Note: this creates two defintions of T=0. I will call 'uncorrected time' is when the result of this function is
        used as-is, and a referance antenna is not choosen. (IE, the referance station can have a large total_delay offset), 'corrected time' will be otherwise.
        This function is literally the negative of get_time_from_second"""
        
        delays = self.get_timing_callibration_delays(out,force_file_delays)
        delays += self.station_delay - self.get_nominal_sample_number()*(1/self.get_sample_frequency())
        
        return delays
    
    def get_time_from_second(self, out=None, force_file_delays=False):
        """ return the time (in units of seconds) since the second of each antenna (which should be get_timestamp). accounting for delays. This is literally just the negative of get_total_delays"""
        out = self.get_total_delays(out,force_file_delays)
        out *= -1
        return out
    
    def get_geometric_delays(self, source_location, out=None, antenna_locations=None, atmosphere_override=None):
        """
        Calculate travel time from a XYZ location to each antenna. out can be an array of length equal to number of antennas.
        antenna_locations is the table of antenna locations, given by get_LOFAR_centered_positions(). If None, it is calculated. Note that antenna_locations CAN be modified in this function.
        If antenna_locations is less then all antennas, then the returned array will be correspondingly shorter.
        The output of this function plus get_total_delays plus emission time of the source, times sample frequency, is the data index the source is seen on each antenna.
        Use atmosphere_override if given.
        Else, use atmosphere_override from total_cal, or default atmosphere otherwise
        """
        
        if antenna_locations is None:
            antenna_locations = self.get_LOFAR_centered_positions()
        
        if out is None:
            out = np.empty( len(antenna_locations), dtype=np.double )
            
        if len(out) != len(antenna_locations):
            print("ERROR: arrays are not of same length in geometric_delays()")
            return None

        atmo_to_use = atmosphere_override
        if atmo_to_use is None:
            if self.using_total_cal:
                atmo_to_use = self.total_cal.atmosphere
            else:
                atmo_to_use = atmo.default_atmosphere
                
        v_airs = atmo_to_use.get_effective_lightSpeed(source_location, antenna_locations)

        antenna_locations -= source_location
        antenna_locations *= antenna_locations
        np.sum(antenna_locations, axis=1, out=out)
        np.sqrt(out, out=out)
        out /= v_airs
        return out
            

    def get_data(self, start_index, num_points, antenna_index=None, antenna_name=None):
        """return the raw data for a specific antenna, as an 1D int16 numpy array, of length num_points. First point returned is 
        start_index past get_nominal_sample_number(). Specify the antenna by giving the antenna_name (which is a string, same
        as output from get_antenna_names(), or as an integer antenna_index. An antenna_index of 0 is the first antenna in 
        get_antenna_names()."""
        
        if antenna_index is None:
            if antenna_name is None:
                raise LookupError("need either antenna_name or antenna_index")
            internal_name_index = self.dipoleNames.index(antenna_name)
            antenna_index = self.name_adjusts.index( internal_name_index )

        antenna_index = self.index_adjusts[antenna_index] ##incase of polarization flips
            
        initial_point = self.sample_offsets[ antenna_index ] + start_index
        final_point = initial_point+num_points
        
        to_file = self.antenna_to_file[antenna_index]
        if to_file is None:
            raise LookupError("do not have data for this antenna")
        TBB_file,station_antenna_index = to_file
        antenna_name = self.dipoleNames[ antenna_index ]
        
        if final_point >= len( TBB_file.file[ TBB_file.stationKey ][ antenna_name ] ):
            print("WARNING! data point", final_point, "is off end of file", len( TBB_file.file[ TBB_file.stationKey ][ antenna_name ] ))

        try:
            RET = TBB_file.file[ TBB_file.stationKey ][ antenna_name ][initial_point:final_point]

            if self.using_total_cal and (antenna_name in self.total_cal.sign_flips):
                RET *= -1

        except BaseException as error:

            print('error reading HDF5 TBB file. file:',  TBB_file.filename)
            print('   station', TBB_file.stationKey, 'internal ant name', antenna_name)
            print('   init n final point', initial_point, final_point)
            print('   error msg:', repr(error) )
            print('RERAISING')
            raise error

        return RET
        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    timeID =  "D20170929T202255.000Z"
    station = "RS406"
    antenna_id = 0
    
    block_size = 2**16
    block_number = 30#3900
    
    raw_fpaths = filePaths_by_stationName(timeID)
    
    infile = MultiFile_Dal1(raw_fpaths[station])
#    infile = MultiFile_Dal1(["./new_file.h5"])
    
    print( infile.get_LOFAR_centered_positions() )
    
    data = infile.get_data(block_number*block_size, block_size, antenna_index=antenna_id)
    
    plt.plot(data)
    plt.show()
#        
        
        
        
        
        
        
        
        
        
        
        
        
    