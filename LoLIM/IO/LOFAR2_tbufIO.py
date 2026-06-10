#!/usr/bin/env python3

import os
import datetime
import json

from copy import copy

import numpy as np
import h5py

import LoLIM.IO.metadata as md
import LoLIM.utilities as util
import LoLIM.atmosphere as atmo



from LoLIM.IO.calFiles import total_cal_object, read_cal_file

def antenna_L2_name_to_L1_name(L2_name):
    """LOFAR2.0 has a different nameing convention than LOFAR1.0. Therefore to keep code from breaking, we create a 1-1 map from LOFAR2.0 antenna names to LOFAR1.0 antenna names. Such that:
         It is nine digits long. The first three digits is the station ID, middle three digits are not important, and last three digits identify the antnena, and are even for Y-dipoles.
         Typical L2 name is like DIPOLE_CS001_LBA40X, which maps to a L1 name of 001999401 """


    is_LBA = L2_name[13:16]=='LBA'
    station = L2_name[7:12]
    number =  L2_name[16:18]
    is_Y = L2_name[18:19]=='Y'

    station_ID = str(util.Sname_to_SId_dict[station])
    station_ID = '0'*(3-len(station_ID)) + station_ID

    if is_LBA:
        if is_Y:
            end_digit = '0'
        else:
            end_digit = '1'
    else:
        if is_Y:
            end_digit = '2'
        else:
            end_digit = '3'

    return station_ID + '999' + number + end_digit

def antenna_name_to_L2_name(ant_name):
    """This is the inverse of L2_name_to_L1_name, but only if ant_name is in L1 style. If name is L2 style, then just return it. Throws an error if you actually give it a true L1 antenna."""

    if ant_name[0] == 'D':
        ## is already a L2 name
        return ant_name

    if ant_name[3:6] != '999':
        print('ERROR! a true LOFAR1.0 antenna name given to antenna_name_to_L2_name. True LOFAR1.0 antenna cannot be mapped to LOFAR2.0')
        quit()

    station_ID = ant_name[0:3]
    antNumber = ant_name[6:8]
    end_digit = ant_name[9]

    station_name = util.SId_to_Sname( int(station_ID) )

    if end_digit =='0':
        ant_type = 'LBA'
        pol = 'Y'
    elif end_digit =='1':
        ant_type = 'LBA'
        pol = 'X'
    elif end_digit =='2':
        ant_type = 'HBA'
        pol = 'Y'
    elif end_digit =='3':
        ant_type = 'HBA'
        pol = 'X'

    return 'DIPOLE_' + station_name + '_' + ant_type + antNumber + pol
    


def totalCal_to_L2( total_cal ):
    """This takes a total_cal object, and returns a new one. Such that the new one is insured that all antenna names are in L2 style. Will just return same object if all antennas are in L2 style"""
    

    ## step one : check if any antennas are in L1 style

    ## chk bad_antenna_data
    any_L1_antNames = False

    for ba in total_cal.bad_antenna_data:
        if ba[0]!='D':
            any_L1_antNames = True
            break

    ## chk polarization_flips if needed
    if not any_L1_antNames:
        for pf in total_cal.polarization_flips:
            if pf[0]!='D':
                any_L1_antNames = True
                break

    ## chk ant_delays if needed
    if not any_L1_antNames:
        for ant in total_cal.ant_delays.keys():
            if ant[0]!='D':
                any_L1_antNames = True
                break

    ## chk sign_flips if needed
    if not any_L1_antNames:
        for ant in total_cal.sign_flips:
            if ant[0]!='D':
                any_L1_antNames = True
                break

    ## if no L1 ants found, then we just return the total_cal
    if not any_L1_antNames:
        return total_cal

    ## if we are here, than we have to make a copy of total_cal, and convert all antenna names to L2 style

    new_cal = total_cal_object()
    ## first, just copy unchanged data.
    new_cal.atmosphere = total_cal.atmosphere
    new_cal.station_delays = total_cal.station_delays
    new_cal.metadata_adjusts = total_cal.metadata_adjusts

    ## now for antenna stuff
    ##  bad antnenas
    for ant in total_cal.bad_antenna_data:
        new_cal.bad_antenna_data.append( antenna_name_to_L2_name(ant) )

    ## polarization_flips
    for ant in total_cal.polarization_flips:
        new_cal.polarization_flips.append( antenna_name_to_L2_name(ant) )

    ## ant_delays
    for ant, delay in total_cal.ant_delays.items():
        new_cal.ant_delays[ antenna_name_to_L2_name(ant) ] = delay

    ## sign_flips
    for ant in total_cal.sign_flips:
        new_cal.sign_flips.append( antenna_name_to_L2_name(ant) )

    return new_cal




def freq_to_s( freq_unit ):

    if freq_unit == 'MHz':
        return 1.0e6
    else:
        print('unit not recognized:', freq_to_s)
        quit()


class LOFAR2_tbuf_reader:
    """A class for reading the data from one station from multiple files"""
    def __init__(self, file_list,  total_cal=None,
                 only_complete_pairs=True, pol_flips_are_bad=False,
                 antenna_mode='all'):
        """file_list:  list of filenames for this station for this event, or list of already-opened h5py file objects

            total_cal              - ...many behaviors. If a string, then assumes total_cal is file name, and is opened by read_cal_file. Else, should be output of read_cal_file.

            only_complete_pairs    -if True, discards antenna if the other in pair is not present or is bad. If False, keeps all good antennas.
                                      If False, needs to use 'has_antenna' method. To make sure antenna actually exists! (see doc for method).
                                      Currently it is expected L2 will always have both pairs, thus this setting only matters if one dipole is considered "bad"
            pol_flips_are_bad      -if True, antennas that are in pol-flips are included in 'bad_antennas' 

            antenna_mode           -values can be 'LBA', or 'all' (currently).  Indicates which antenna types can be read. Needs to be improved for HBA, but needs to indicate HBA bands, which I don't remember at moment.
        """


        self.file_list = file_list
        self.files = [ filename if isinstance(filename,h5py.File) else h5py.File(filename, "r") for filename in file_list]

        ## dbl check not L1!!

        good = True
        for f in self.files:    
            if not 'TELESCOPE_VERSION' in f.attrs:
                good = False 
                break 
            if f.attrs['TELESCOPE_VERSION'] != '2.0':
                good = False 
                break 

        if not good: 
            print('L1 data is given to L2 data reader!')
            quit()



        if total_cal != None:

            if isinstance(total_cal, str):
                self.total_cal = read_cal_file(total_cal, pol_flips_are_bad)
            else:
                self.total_cal = total_cal

            self.total_cal = totalCal_to_L2( self.total_cal ) ## this can copy total cal, thus may slightly increase memory usage. 

        else:
            self.total_cal = total_cal_object()  ## make an empty total cal to just make life easier

        self.only_complete_pairs = only_complete_pairs
        self.pol_flips_are_bad = pol_flips_are_bad
        self.antenna_mode = antenna_mode

        if not (self.antenna_mode in ['LBA', 'all']):
            print('ERROR: antenna_mode must be LBA or all')
            quit()



        self.station_name = None
        self.available_antenna_type = None    ## this indicates if all available antennas are same or different type. Currently values can be LBA or multiple
        ## HBA is currently a problem. Namely bcz I don't know how to handle multiple filters (yet)

        self.SampleFrequency = None ## we assume this will be the same between all antennas and doesn't need checking. 
        self.filter = None ## this will stay None if available_antenna_type is multiple
        self.Time = None 

        ### first we open all antennas to decide what order to put them in. 
        polFreeAntNames = []
        xant_metadata = []
        yant_metadata = []
        xant_DataGroup = []
        yant_DataGroup = []
        for file in self.files:
            for G_l1 in file.values():
                for antenna_field in G_l1.values():
                    ## attributes:
                    ## ['ANTENNA_FIELD_NAME', 'ANTENNA_FIELD_POSITION', 'ANTENNA_FIELD_POSITION_EPOCH', 'ANTENNA_FIELD_POSITION_FRAME', 'ANTENNA_FIELD_POSITION_UNIT', 'ANTENNA_SET', 'CLOCK_SOURCE', 'GROUPTYPE', 'SAMPLE_FREQUENCY', 'SAMPLE_FREQUENCY_UNIT', 'STATION_NAME']>
                    ## childs:
                    ## ['dipoles', 'dipoles_data']

                    if antenna_field.attrs['ANTENNA_FIELD_NAME'] == 'LBA' and ( (self.antenna_mode=='LBA') or (self.antenna_mode=='all') ):
                        pass ## we good
                    elif antenna_field.attrs['ANTENNA_FIELD_NAME'] != 'LBA':
                        print('HBA currently not supported by LOFAR2_tbuf_reader')
                        quit()
                    else:
                        ## we not good
                        continue


                    if self.station_name is None:
                        self.station_name = antenna_field.attrs['STATION_NAME']
                        self.SampleFrequency = antenna_field.attrs['SAMPLE_FREQUENCY']*freq_to_s( antenna_field.attrs['SAMPLE_FREQUENCY_UNIT'] )
                    else:
                        if self.station_name != antenna_field.attrs['STATION_NAME']:
                            print('ERROR! all files/antenna groups must be from same station;', self.station_name)


                    if self.available_antenna_type is None:
                        self.available_antenna_type = antenna_field.attrs['ANTENNA_FIELD_NAME'] ## is this correct for HBA? which has different filters
                    elif self.available_antenna_type == 'multiple':
                        pass  ## I guess it's alright then!
                    elif self.available_antenna_type != antenna_field.attrs['ANTENNA_FIELD_NAME']:
                        self.available_antenna_type = 'multiple'
                        self.filter = None


                    metaDataGroup = antenna_field['dipoles']
                    DataGroup = antenna_field['dipoles_data']
                    
                    for dipoleName, dipoleMetaGroup in metaDataGroup.items():

                        #antenna_field['dipoles_data'][dipoleName].attrs.keys()
                        # is 
                        # ['ADC2VOLTAGE', 'ANTENNA_ID', 'ANTENNA_NORMAL_VECTOR', 'ANTENNA_POSITION', 'ANTENNA_POSITION_EPOCH', 'ANTENNA_POSITION_FRAME', 'ANTENNA_POSITION_UNIT', 'ANTENNA_ROTATION_MATRIX', 'CABLE_DELAY', 'CABLE_DELAY_UNIT', 'CABLE_LOSS', 'CABLE_LOSS_UNIT', 'CLOCK_OFFSET_DELAY', 'CLOCK_OFFSET_DELAY_UNIT', 'CLOCK_OFFSET_PHASE_ZERO', 'CLOCK_OFFSET_PHASE_ZERO_UNIT', 'DATA_LENGTH', 'DIPOLE_CALIBRATION_GAIN_CURVE', 'DIPOLE_CALIBRATION_GAIN_CURVE_CROSSTALK', 'FILTER_SELECTION', 'NYQUIST_ZONE', 'POLARIZATION', 'SAMPLES_PER_FRAME', 'SAMPLE_NUMBER', 'STATION_NAME', 'TIME']

                        ### skip bad antennas
                        if dipoleName in self.total_cal.bad_antenna_data:
                            continue ## skip this antenna

                        ## chk metadata things

                        if self.Time is None:
                            self.Time = dipoleMetaGroup.attrs['TIME']
                        elif self.Time != dipoleMetaGroup.attrs['TIME']:
                            print('ERROR: differetn antennas have different TIME attribute. Immorality has been detected, and you are now being reported to the Pope.')
                            quit()


                        if self.available_antenna_type != 'multiple':
                            if self.filter is None:
                                self.filter = dipoleMetaGroup.attrs['FILTER_SELECTION']
                            elif self.filter != dipoleMetaGroup.attrs['FILTER_SELECTION']:
                                print('antennas have different filters! this is not understood, we are now paniking.')
                                quit()


                        ## append info to correct arrays
                        polFreeName = dipoleName[:-1]

                        if polFreeName in polFreeAntNames:
                            anti = polFreeAntNames.index( polFreeName )
                        else:
                            polFreeAntNames.append( polFreeName )
                            xant_metadata.append( None )
                            yant_metadata.append( None )
                            xant_DataGroup.append( None )
                            yant_DataGroup.append( None )
                            anti = len(polFreeAntNames)-1

                        if dipoleName[-1]=='X':
                            if not xant_metadata[anti] is None:
                                print('Odd err A in LOFAR2_tbuf_reader. This should not happen')
                                quit()

                            xant_metadata[anti] = dict(dipoleMetaGroup.attrs)
                            xant_DataGroup[anti] = DataGroup[dipoleName]
                        else:
                            if not yant_metadata[anti] is None:
                                print('Odd err B in LOFAR2_tbuf_reader. This should not happen')
                                quit()

                            yant_metadata[anti] = dict(dipoleMetaGroup.attrs) 
                            yant_DataGroup[anti] =  DataGroup[dipoleName] 

        ## now we sort antenna names
        self.antenna_names = []
        self.antenna_names_L1 = []
        self.antenna_metadata = []
        self.antenna_dataGroups = []

        sorter = np.argsort( polFreeAntNames )
        for anti in sorter:
            YantName = polFreeAntNames[anti] + 'Y'
            XantName = polFreeAntNames[anti] + 'X'

            hasYant = not (yant_metadata[anti] is None)
            hasXant = not (yant_metadata[anti] is None)

            if not (hasYant and hasXant): 
                if only_complete_pairs:
                    continue

            self.antenna_names.append( YantName )
            self.antenna_names.append( XantName )

            self.antenna_names_L1.append( antenna_L2_name_to_L1_name( YantName ) )
            self.antenna_names_L1.append( antenna_L2_name_to_L1_name( XantName ) )

            self.antenna_metadata.append( yant_metadata[anti] ) ## this can be none
            self.antenna_metadata.append( xant_metadata[anti] )

            ## flip dataGroup if there is a pol flip
            ydata = yant_DataGroup[anti]
            xdata = xant_DataGroup[anti]
            if ( YantName in self.total_cal.polarization_flips) or ( XantName in self.total_cal.polarization_flips):
                tmp = ydata
                ydata = xdata
                xdata = tmp

            self.antenna_dataGroups.append( ydata )
            self.antenna_dataGroups.append( xdata )


    #### need to handle sample numbers and data lengths   ####

        self.data_lengths = np.empty(len(self.antenna_names), dtype=int)
        self.data_lengths[:] = -1

        self.sample_numbers = np.empty(len(self.antenna_names), dtype=int)
        self.sample_numbers[:] = -1

        for ant_i in range(len(self.antenna_names)):
            metadata = self.antenna_metadata[ ant_i ]

            if metadata is not None:
                print('WARNING: hack implimented where data_length is divded by 2')
                self.data_lengths[ant_i] = int(metadata['DATA_LENGTH']/2)  
                self.sample_numbers[ant_i] = metadata['SAMPLE_NUMBER']


        self.nominal_sample_number = np.max( self.sample_numbers )


        if len(self.antenna_names) == 0:
            print('WARNING!: station', self.station, 'has no antennas.')


        self.station_delay = 0.0
        if self.station_name in self.total_cal.station_delays:
            self.station_delay = self.total_cal.station_delays[self.station_name]

        ## other housekeeping

        self.antennaTimings = None ## this is the complete antenna delay, include all info from Tbuf, plus total cal, include station delays. Unlike L1 reader, we will only use "total" delays
        self.zeroFreqPhases = None ## these are filled by get_timing_callibration_delays

        self.geo_delay_temp = None
        
        
    def set_station_delay(self, station_delay):
        """ set the station delay, should be a number"""
        self.station_delay = station_delay
                
        
    #### GETTERS ####

    def getSaturationValues(self):
        """return the largest value and smallest value data saturates at. Note, this can't be deduced from data type, since that may not be the native bitdepht output by the system"""
        return 8190, -8191

    def needs_metadata(self):
        return False
    
    def get_station_name(self):
        """returns the station_name of the station, as a string"""
        return self.station_name


    def get_station_delay(self):
        """ return the station delay"""
        return self.station_delay
                
    
    def get_station_ID(self):
        """returns the ID of the station, as an integer. This is not the same as StationName. Mapping is givin in utilities"""
        return util.Sname_to_SId_dict[ self.station_name ]
    
    def get_antenna_names(self, name_style=None):
        """return name of antenna as a list of strings.
        Note that antenna_id used elsewhere in this class is the index of this list (order stays the same!).
        However, order and such is not garunteed between different class instances, thus if you need to save info, save antenna_name, NOT antenna_ID.
        Even indeces are always Y-oriented dipole.
        Thus, if only_complete_pairs was false, some antennas may be here but not actually exist. Thus you need method "has_antenna".
        There are two kinds of antenna name styles: 'L2' or 'L1'. 
            'L2' style is the actual name of the antenna in LOFAR2.0 era. E.g.: DIPOLE_CS001_LBA40X
            However, some code may still rely on antennas having L1 style names. Thus there is 1-1 mapping to a L1 style antenna name.
            A L1 styple name has nine digits. First three are the station id (not name!), last three uniquelly identify the antenna (where Y dipole is even valued). and middle three digits are superflouous.
        name_style can be 'L2' 'L1' or none. If none (default) than 'L2' is used.
        L2 style is strongly prefered when the code works."""

        if name_style not in ['L1', 'L2', None]:
            print('error in LOFAR2_tbuf_reader.get_antenna_names: invalid name_style:', name_style)

        if name_style is None:
            name_style = 'L2'

        if name_style == 'L2':
            return copy( self.antenna_names )
        else:  ## is L1 style
            return copy( self.antenna_names_L1 )


    def get_antenna_metadata(self, antname, meta_value):
        """an experemental method for testing only"""

        ant_i = self.antenna_names.index(antname)
        return self.antenna_metadata[ant_i][meta_value]

    
    def has_antenna(self, antenna_name=None, antenna_index=None):
        """if only_complete_pairs is False, then we could have antenna names without the data.
        Give either antenna_name or antenna_index.
        Return True if we actually have the antenna, False otherwise. This accounts for polarization flips."""

        if antenna_name is not None:
            antenna_name = antenna_name_to_L2_name( antenna_name )

            if antenna_name in self.antenna_names:
                internal_antenna_index = self.antenna_names.index(antenna_name)
            else:
                return False
        else:
            internal_antenna_index = antenna_index

        if self.antenna_metadata[ internal_antenna_index ] is None:
            return False
        else:
            return True

    def get_XYdipole_indeces(self):
        """Return 2D [i,j] array of integers. i is the index of the "full" dual-polarizated antenna (half of length of get_antenna_names). j is 0 for X-dipoles (odd for LBA_OUTER), 1 for Y-dipoles.
        Value is antenna_id (index of get_antenna_names). Will be -1 if antenna does not exist
        NOTE: this method may be phased out. As in future, order will always be Y-X"""

        print('get_XYdipole_indeces is depreciated')
        print('    just like you should be')
        quit()

        # n_pairs = int(len(self.dipoleNames)/2)
        # ret = np.full( (n_pairs,2), -1, dtype=np.int)
        # for pair_i in range(n_pairs):
        #     #if self.antennaSet == "LBA_OUTER":
        #     X_index = 2*pair_i + 1
        #     Y_index = 2*pair_i
        #     #elif self.antennaSet == "LBA_INNER":
        #         #X_index = 2*pair_i
        #         #Y_index = 2*pair_i + 1
        #     #else:
        #     #    print('unknown antenna set in get_XYdipole_indeces:', self.antennaSet)
        #     #    return None

        #     if self.has_antenna(antenna_ID=X_index):
        #         ret[pair_i, 0] = X_index
        #     if self.has_antenna(antenna_ID=Y_index):
        #         ret[pair_i, 1] = Y_index

        # return ret


    def get_antenna_set(self):
        """return the antenna set as a string. Currently can be 'LBA' or 'multiple'. Work still needed for HBA """
        return self.available_antenna_type
    
    def get_sample_frequency(self):
        """gets samples per second. Typically 200 MHz."""
        return self.SampleFrequency
    
    def get_filter_selection(self):
        """return a string that represents the frequency filter used. e.g. "LBA_10_90
        will return None if get_antenna_set is multiple. In this case filter selection is differetn for different antennas and a different method (not currently implemented) should be used."""

        return self.filter

        
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

        return self.data_lengths
    
    def get_all_sample_numbers(self):
        """return numpy array that contains the sample numbers of each antenna. Divide this by the sample frequency to get time
        since the timestame of the first data point. Note that since these are, in general, different, they do NOT refer to sample
        0 of "get_data". Value is -1 if antenna does not exist """
        ##return self.SampleNumbers

        return self.sample_numbers
    
    
    def get_nominal_sample_number(self):
        """return the sample number of the 0th data sample returned by get_data.
        Divide by sample_frequency to get time from timestamp of the 0th data sample"""
        return self.nominal_sample_number
        
    def get_nominal_data_lengths(self):
        """return the number of data samples that are usable for each antenna, accounting for different starting sample numbers.
        returns array of ints. Value is -1 for antennas that do not exist."""
        #return self.nominal_DataLengths


        length_after_SN = self.data_lengths - ( self.sample_numbers - self.nominal_sample_number )
        return length_after_SN

    
    def get_ITRF_antenna_positions(self, out=None):
        """returns the ITRF positions of the antennas. Returns a 2D numpy array. 
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated.
        Does not account for polarization flips, but shouldn't need too.
        Is zero for antennas that don't exist"""



        if out is None:
            out = np.empty( (len(self.antenna_names), 3) )
        
        for ant_i in range(len(self.antenna_names)):

            metadata = self.antenna_metadata[ant_i]
        
            if not metadata is  None:

                out[ant_i] = metadata['ANTENNA_POSITION']

            else:

                out[ant_i] = 0

        return out
        
    def get_LOFAR_centered_positions(self, out=None):
        """returns the positions (as a 2D numpy array) of the antennas with respect to CS002. 
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated.
        Does not account for polarization flips, but shouldn't need too."""
        if out is None:
            out = np.empty( (len(self.antenna_names), 3) )
        
        md.convertITRFToLocal( self.get_ITRF_antenna_positions(), out=out )
            
        return out
    
    
    def get_timing_callibration_delays(self, out=None):
        """return the timing callibration of the anntennas, as a 1D np array.
        For now, this only extracts the antenna-level delays from the frequency dependent phase that is in the TBuf file.
        This also includes all known station delays. 
        Returns a 0 for non-existent antennas."""


        if self.antennaTimings is None:

            num_ants = len(self.antenna_names)

            self.antennaTimings = np.zeros( num_ants, dtype=float )
            self.zeroFreqPhases = np.zeros( num_ants, dtype=float )

            for ant_i in range(num_ants):

                metadata = self.antenna_metadata[ant_i]
                if metadata is None:
                    continue

                cal_curve = metadata["DIPOLE_CALIBRATION_GAIN_CURVE"]

                zeroFreqPhase, timing = md.convertPhase_to_Timing_LinFit(cal_curve)

                self.antennaTimings[ant_i] = timing
                self.zeroFreqPhases[ant_i] = zeroFreqPhase


                ## now add in totalcal
                antName = self.antenna_names[ ant_i ]
                if antName in self.total_cal.ant_delays:
                    self.antennaTimings[ant_i] += self.total_cal.ant_delays[antName]


                ## add in field clock cal  WARNING: I don't know the sign of this!
                self.antennaTimings[ant_i] -= metadata['CLOCK_OFFSET_DELAY']
                if metadata['CLOCK_OFFSET_DELAY_UNIT'] != 's':
                    print('ERROR:', 'CLOCK_OFFSET_DELAY_UNIT is not s. in get_timing_callibration_delays')
                    quit()

            ## finally, station delay in total cal
            if self.station_name in self.total_cal.station_delays:
                self.antennaTimings += self.total_cal.station_delays[self.station_name]


        if out is None:
            out = np.array(self.antennaTimings)
        else:
            out[:] = self.antennaTimings

        return out


    def get_zeroFreq_phases(self, out=None, addStationPhase=False):
        """return the calibrated phase per antenna at frequnecy 0. Is 0 if antenna doesn't exist. 
        if addStationPhase is True (currently not implemented), the phase of the entire station is added"""


        if self.antennaTimings is None:
            self.get_timing_callibration_delays( out )

        if out is None:
            out = np.array(self.zeroFreqPhases)
        else:
            out[:] = self.zeroFreqPhases

        if addStationPhase:
            print('WARNING: station phase not tested. at get_zeroFreq_phases')

            for ant_i in range(num_ants):

                metadata = self.antenna_metadata[ant_i]
                if metadata is None:
                    continue

                out[ant_i] += metadata['CLOCK_OFFSET_PHASE_ZERO']

        return out

    
    def get_total_delays(self, out=None):
        """Return the total delay for each antenna, accounting for all known delays, and nominal sample number. This function should be prefered over 'get_timing_callibration_delays', 
        but the offsets can have a large average. It is recomended to pick one antenna (on your referance station)
        and use it as a referance antenna so that it has zero timing delay. Note: this creates two defintions of T=0. I will call 'uncorrected time' is when the result of this function is
        used as-is, and a referance antenna is not choosen. (IE, the referance station can have a large total_delay offset), 'corrected time' will be otherwise.
        This function is literally the negative of get_time_from_second."""
        
        delays = self.get_timing_callibration_delays(out)
        delays -= self.get_nominal_sample_number()*(1/self.get_sample_frequency())
        
        return delays
    
    def get_time_from_second(self, out=None):
        """ return the time (in units of seconds) since the second of each antenna (which should be get_timestamp). accounting for delays. This is literally just the negative of get_total_delays"""
        out = self.get_total_delays(out)
        out *= -1
        return out
    
    def get_geometric_delays(self, source_location, out=None, antenna_locations=None, atmosphere_override=None):
        """
        Calculate travel time from a XYZ location to each antenna. out can be an array of length equal to number of antennas.
        antenna_locations is the table of antenna locations, given by get_LOFAR_centered_positions(). If None, it is calculated.
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

        if (self.geo_delay_temp is None) or (len(self.geo_delay_temp)<len(antenna_locations)):
            self.geo_delay_temp = np.array(antenna_locations)
            tmp = self.geo_delay_temp
        else:
            tmp = self.geo_delay_temp[:len(antenna_locations)]
            tmp[:] = antenna_locations


        atmo_to_use = atmosphere_override
        if atmo_to_use is None:
            atmo_to_use = self.total_cal.atmosphere
           
                
        v_airs = atmo_to_use.get_effective_lightSpeed(source_location, antenna_locations)

        tmp -= source_location
        tmp *= tmp
        np.sum(tmp, axis=1, out=out)
        np.sqrt(out, out=out)
        out /= v_airs
        return out


            

    def get_data(self, start_index, num_points, antenna_index=None, antenna_name=None, out=None):
        """return the raw data for a specific antenna, as an 1D int16 numpy array, of length num_points. First point returned is 
        start_index past get_nominal_sample_number(). Specify the antenna by giving the antenna_name (which is a string, same
        as output from get_antenna_names(), or as an integer antenna_index. An antenna_index of 0 is the first antenna in 
        get_antenna_names()."""
        
        if not (antenna_name is None):
            antenna_name = antenna_name_to_L2_name( antenna_name )

        if antenna_index is None:
            if antenna_name is None:
                raise LookupError("need either antenna_name or antenna_index")
            antenna_index = self.antenna_names.index(antenna_name)

        initial_point = (self.nominal_sample_number-self.sample_numbers[ antenna_index ]) + start_index
        final_point = initial_point+num_points


        dataObj = self.antenna_dataGroups[antenna_index]
        if dataObj is None:
            raise LookupError("do not have data for this antenna")

        antenna_name = self.antenna_names[ antenna_index ]
        
        if final_point >= self.data_lengths[antenna_index]:
            print("WARNING! data point", final_point, "is off end of file", self.data_lengths[antenna_index] )

        try:
            if out is None:
                RET = dataObj[initial_point:final_point]
                
            else:
                dataObj.read_direct(out, np.s_[initial_point:final_point], np.s_[0:num_points])
                RET = out


            if  (antenna_name in self.total_cal.sign_flips):
                RET *= -1

        except BaseException as error:

            print('error reading HDF5 TBB file.')
            print('   station', self.get_station_name(), 'internal ant name', antenna_name)
            print('   init n final point', initial_point, final_point)
            print('   known length:', self.get_full_data_lengths()[antenna_index], 'actual len', len(dataObj)  )
            print('   error msg:', repr(error) )
            print('RERAISING')
            raise error

        return RET