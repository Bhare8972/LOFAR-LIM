#!/usr/bin/env python3

from os import listdir
from os.path import isfile, join

import numpy as np

from matplotlib import pyplot as plt
import numpy as np

from LoLIM.utilities import RTD, Sname_to_SId_dict
from LoLIM.NumLib.FFT import complex_fft_obj
from LoLIM.atmosphere import default_atmosphere

from scipy.integrate import simpson


from scipy import constants
C  = constants.c
u0 = constants.mu_0
e0 = constants.epsilon_0
Z0 = u0*C

## TODO: add function to make a point-cloud instead of just one pulse
## and add function to inject noise
class pulse_generator:

    def __init__(self, antenna_function, trace_length, sample_time=5.0e-9, additional_freq_filter=None, atmosphere=None):
        """
        antenna_function should be a function with arguments: (frequencies, zenith, azimuth, invert=False, freq_fill=0.0, out), with angles in degrees
        trace_length should be length of desired number of samples
        sample_time is time between samples default 5e-9
        additional_freq_filter, if given, should be a function that takes frequency array in Hz and returns a 1D array to multiply by data.
        """
        
        self.antenna_function = antenna_function
        self.trace_length = trace_length
        self.sample_time = sample_time

      ## MAKE SOME MEMORY ##
        self.stat_X_dipole = np.empty(trace_length, dtype=np.cdouble)  ## the waveform in frequency space
        self.stat_Y_dipole = np.empty(trace_length, dtype=np.cdouble)
        self.stat_azimuthal_field = np.empty(trace_length, dtype=np.cdouble)
        self.stat_zenithal_field = np.empty(trace_length, dtype=np.cdouble)
        self.tmp_memory = np.empty(trace_length, dtype=np.cdouble)

        self.Jones_tmp = np.empty( (trace_length, 2,2), dtype=np.cdouble )

      ## some setup
        self.FFT_OBJ = complex_fft_obj( trace_length )
        self.frequencies = self.FFT_OBJ.freqs( self.sample_time )

        if additional_freq_filter is None:
            self.additional_filter = np.ones(trace_length)
        else:
            self.additional_filter = additional_freq_filter(self.frequencies)

        self.previous_station_data = []

        if atmosphere is None:
            self.atmosphere = default_atmosphere
        else:
            self.atmosphere = atmosphere


        self.initial_buffer_samples = 8 ## shift pulse right by this much so that pulse doesn't wrap over to end

    def sampleNumber_only(self, antenna_XYZs, XYZT_location, out_b=None, centering_mode=None):
        """
        Returns the sample_number per antenna only, with no simulation. Same as second output of __call__.
        NOT IMPLEMTNED YET
        """
        print("NOT YET IMPLEMENTED")
        quit()

    def energy(self, XYZ_moment, freq_min, freq_max):
        """ give the amount of energy (folded with flat freq. filter between freq_min and freq_max (in Hz)) emitted by a moment of units  Cm/s^2."""

        ## make filter
        filter = np.zeros( len(self.frequencies) )
        filter[ np.logical_and(self.frequencies>=freq_min, self.frequencies<=freq_max) ] = 1


        ## for each component, filter, and sum squares
        self.stat_zenithal_field[:] = 0
        for i in range(3):
            self.stat_azimuthal_field[:] = 0
            self.stat_azimuthal_field[0] = XYZ_moment[i]

            self.FFT_OBJ.fft( self.stat_azimuthal_field )
            self.stat_azimuthal_field *= filter
            self.FFT_OBJ.ifft( self.stat_azimuthal_field )

            self.stat_azimuthal_field[:] = np.abs(self.stat_azimuthal_field)
            self.stat_azimuthal_field *= self.stat_azimuthal_field 
            self.stat_zenithal_field += self.stat_azimuthal_field


        ## convert to power
        self.stat_zenithal_field *= Z0/(12*np.pi*C*C)

        ## integrate to energy
        E = simpson(self.stat_zenithal_field, dx=5.0e-9)
        return E



    def __call__(self, antenna_XYZs, antenna_polarization, XYZT_location, XYZ_moment, out_a=None, out_b=None, station_integer=None, centering_mode=None):
        """
        Generate a pulse as observed by a station. Assumes that every antenna in this station is roughly same angle from source (but not distance)
        antenna_XYZs should be np.array( (num_antennas, 3), dtype=double ) of antenan positions
        antenna_polarization should be np.array( num_antennas ). Where 0 means antenna is X, and 1 means Y
        XYZT_location should be numpy.array( 4, dtype=double )
        XYZ_moment numpy.array( 3, dtype=cdouble ) is what I'm calling the ``radiation moment''. Is units of Cm/s^2. Looks like first derivative of current... but is a vector.
        If given, out_a should be np.array( (num_antennas, trace_length), dtype=cdouble ), and out_b=np.array( num_antennas, dtype=int ).
        station_integer, if given, is index of station. Used to store and skip calculstions if this funciton is in a loop (does generate new memory on first use). Assumes angle to source doesn't change.
        centering_mode, defines how pulse is positioned in traces. Options:
            'float_beginning' : default option. Puts all pulses at beginning of trace with slight buffer. each antenna has different start index
            'float_middle'    : same, but pulses are in middle of trace
            integer           : hard code start index of all antennas to this number
            'relative_beginning' : NOT IMPLEMENTED move pulses to beginning, but all antennas have same start index
            'relative_middle;  :  all antnenas have same start index, but arrival time is roughly in middle of trace on average

        Retrurns two arrays, the first array is of type np.array( (num_antennas, trace_length), dtype=cdouble ), which gives the simulation as function of antenna.
            the second return is np.array( num_antennas, dtype=int ), which is the sample_number of the first sample in the first trace
        """

        if out_a is None:
            out_a = np.empty( (len(antenna_XYZs), self.trace_length), dtype=np.cdouble )
        if out_b is None:
            out_b = np.empty( len(antenna_XYZs), dtype=int )

    ### setup angles and whatnot
        if (station_integer is None) or (station_integer >= len(self.previous_station_data) ) or ( self.previous_station_data[ station_integer ] is None):
            station_XYZ = np.average( antenna_XYZs, axis=0 )

            station_delta_XYZ = XYZT_location[0:3] - station_XYZ

            stat_R = np.linalg.norm( station_delta_XYZ )  ## soley used to project  X Y Z polarization onto zenthal and azimuthal components

            stat_zenith = np.arccos(station_delta_XYZ[2] / stat_R)
            stat_azimuth = np.arctan2(station_delta_XYZ[1], station_delta_XYZ[0])

            self.antenna_function(frequencies=self.frequencies, zenith=stat_zenith * RTD, azimuth=stat_azimuth * RTD, invert=False, freq_fill=0.0, out=self.Jones_tmp)

            Jones_tmp = self.Jones_tmp

            ## STORE IF USEFUL
            if station_integer is not None:
                if station_integer >= len(self.previous_station_data):
                    self.previous_station_data += [ None ]*( station_integer + 1 - len(self.previous_station_data) )

                self.previous_station_data[ station_integer ] = (stat_zenith, stat_azimuth, np.array(self.Jones_tmp), station_XYZ )

        else:
            stat_zenith, stat_azimuth, Jones_tmp, station_XYZ = self.previous_station_data[ station_integer ]

        sin_stat_azimuth = np.sin(stat_azimuth)
        cos_stat_azimuth = np.cos(stat_azimuth)
        sin_stat_zenith = np.sin(stat_zenith)
        cos_stat_zenith = np.cos(stat_zenith)

    ### calc e-fields
        azimuthal_field = 0.0
        zenithal_field = 0.0

        ## X-orriented current
        zenithal_field  +=  cos_stat_zenith * cos_stat_azimuth * XYZ_moment[0]
        azimuthal_field += -sin_stat_azimuth * XYZ_moment[0]

        ## Y-orriented current
        zenithal_field  += cos_stat_zenith * sin_stat_azimuth * XYZ_moment[1]
        azimuthal_field += cos_stat_azimuth * XYZ_moment[1]

        ## Z-orriented current
        zenithal_field  += -sin_stat_zenith * XYZ_moment[2]

        azimuthal_field *= Z0/(4*np.pi*C)
        zenithal_field  *= Z0/(4*np.pi*C)


    ### apply jones matrix
        self.stat_azimuthal_field[:] = 0
        self.stat_azimuthal_field[0] = azimuthal_field

        self.stat_zenithal_field[:] = 0.0
        self.stat_zenithal_field[0] = zenithal_field

        self.FFT_OBJ.fft( self.stat_zenithal_field )
        self.FFT_OBJ.fft( self.stat_azimuthal_field )

        # R = np.linalg.norm( antenna_XYZs[ 0 ] - XYZT_location[0:3])
        # aveFFT_E_ze = np.average( np.abs(self.stat_zenithal_field) )/R
        # aveFFT_E_az = np.average( np.abs(self.stat_azimuthal_field) )/R

        self.stat_X_dipole[:] = self.stat_zenithal_field
        self.stat_X_dipole *= Jones_tmp[:,0,0]
        self.tmp_memory[:] = self.stat_azimuthal_field
        self.tmp_memory *=  Jones_tmp[:,0,1]
        self.stat_X_dipole += self.tmp_memory

        self.stat_Y_dipole[:] = self.stat_zenithal_field
        self.stat_Y_dipole *= Jones_tmp[:,1,0]
        self.tmp_memory[:] = self.stat_azimuthal_field
        self.tmp_memory *= Jones_tmp[:,1,1]
        self.stat_Y_dipole += self.tmp_memory
        # self.stat_Y_dipole = self.tmp_memory ### ONLY AZIMUTHAL

        self.stat_X_dipole *= self.additional_filter
        self.stat_Y_dipole *= self.additional_filter


        light_speed = self.atmosphere.get_effective_lightSpeed(XYZT_location[0:3], station_XYZ )

        min_freq_index = np.searchsorted(self.frequencies[:int(len(self.frequencies) / 2)], 30e6) - 1
        max_freq_index = np.searchsorted(self.frequencies[:int(len(self.frequencies) / 2)], 80e6)


    ### figure out how to center pulses ###
        # basic type checking
        if centering_mode is None:
            centering_mode = 'float_beginning'

        if isinstance(centering_mode, str):
            if not centering_mode in ['float_beginning', 'relative_middle', 'float_middle']:
                print('ERROR centering mode is:', centering_mode)
                print('needs to be one of:', ['float_beginning', 'relative_middle', 'float_middle'])
                quit()
        elif not isinstance(centering_mode, int):
            print('parameter centering_mode needs to be int or str')
            quit()

        # handle relative positioning
        if centering_mode == 'relative_middle':
            start_index = 0 ## will take average, or minimum
            N = 0
            for ant_i in range( len(antenna_XYZs) ):
                ant_loc = antenna_XYZs[ ant_i ]
                R = np.linalg.norm( ant_loc - XYZT_location[0:3])
                time_shift = XYZT_location[3] + (R/light_speed)
                shift_samples = int(time_shift / self.sample_time)

                shift_samples -= int( self.trace_length/2 )

                start_index += shift_samples
                N += 1

            centering_mode = int( start_index/N )


    ### apply per antenna
        for ant_i in range( len(antenna_XYZs) ):
            ant_loc = antenna_XYZs[ant_i]
            R = np.linalg.norm(ant_loc - XYZT_location[0:3])
            time_shift = XYZT_location[3] + (R / light_speed)
            thisAntenna_startSamples = int(time_shift / self.sample_time) - self.initial_buffer_samples


            if centering_mode == 'float_beginning':
                shift_samples = thisAntenna_startSamples
            elif centering_mode == 'float_middle': 
                shift_samples = int(time_shift / self.sample_time) - int( self.trace_length/2 )
            elif isinstance(centering_mode, int):
                if centering_mode > thisAntenna_startSamples:
                    print('WARNING! sample start is too late, and an antenna is wrapping around!')
                shift_samples = centering_mode


            subsample_shift = time_shift - (shift_samples * self.sample_time )


            self.tmp_memory[:] = self.frequencies
            self.tmp_memory *= -2j * np.pi * subsample_shift
            np.exp(self.tmp_memory, out=self.tmp_memory)

            if antenna_polarization[ ant_i ] == 0:  ## X-dipole
                self.tmp_memory *= self.stat_X_dipole
            else:  ## Y-dipole
                self.tmp_memory *= self.stat_Y_dipole

            self.tmp_memory *= 1.0 / R

            # D = np.abs(self.tmp_memory[min_freq_index:max_freq_index])
            # D *= D
            # plt.plot( self.frequencies[min_freq_index:max_freq_index]/(1e6), D )

            self.FFT_OBJ.ifft( self.tmp_memory )

            out_a[ ant_i, : ] = self.tmp_memory
            out_b[ ant_i ] = shift_samples

        return out_a, out_b



### GOAL of following code is to wrap above class in anouther flass that looks like the TBB file interfase

import LoLIM.IO.metadata as md
import LoLIM.utilities as util
import datetime

class TBB_simulation:
    def __init__(self, StationName, antenna_function, atmosphere, source_XYZT, source_XYZ_moment, noise=2, antenna_mode=None, simulation_length=10000, output_dtype=np.double):
        self.antenna_function = antenna_function
        self.atmosphere = atmosphere
        self.StationName = StationName
        self.StationID = util.Sname_to_SId_dict[ StationName ]
        self.antennaSet = 'LBA_OUTER'

        self.simulation_length = simulation_length
        self.output_dtype = output_dtype

        if self.output_dtype is not np.double:
            print( "output type ust be np.double for now" )
            quit()

        self.source_XYZT = np.array( source_XYZT )
        self.source_XYZ_moment = np.array( source_XYZ_moment )
        self.noise = noise



        
        self.ALL_ITRF_dipole_positions = md.getItrfAntennaPosition(self.StationName, self.antennaSet)
        self.ALL_local_dipole_positions = md.convertITRFToLocal(self.ALL_ITRF_dipole_positions)
        self.ALL_antenna_names = [  ]
        sid = str(self.StationID).zfill(3)
        for i in range( len(self.ALL_ITRF_dipole_positions) ):
            RCU = str( int(i/8) ).zfill(3)
            ant_num = str(i).zfill(3)
            self.ALL_antenna_names.append( sid + RCU + ant_num )


## antenna mode can be None, in which we use all antennas
        if antenna_mode is None:
            self.ITRF_dipole_positions = self.ALL_ITRF_dipole_positions
            self.dipoleNames = self.ALL_antenna_names
            self.local_dipole_positions = md.convertITRFToLocal(self.ITRF_dipole_positions)

## antenna mode can be integer, in which case we use that number of outermost LBA
        elif isinstance( antenna_mode, int ):
            N = antenna_mode*2

            self.ITRF_dipole_positions = self.ALL_ITRF_dipole_positions[-N:]
            self.dipoleNames = self.ALL_antenna_names[-N:]
            self.local_dipole_positions = md.convertITRFToLocal(self.ITRF_dipole_positions)

## antenna mode can be list of atnenan names
        elif isinstance( antenna_mode, list ):
            self.dipoleNames = []
            for an in antenna_mode:
                even_name = util.antName_to_even( antName_to_even )
                if even_name not in self.dipoleNames:
                    self.dipoleNames.append( even_name )
                    self.dipoleNames.append( util.even_antName_to_odd( even_name ) )

            self.ITRF_dipole_positions = self.ALL_ITRF_dipole_positions[ md.make_antennaID_filter( self.dipoleNames ) ]
            self.local_dipole_positions = md.convertITRFToLocal(self.ITRF_dipole_positions)

## antenna mode can be 2D array of antenna XYZ locations. to simulate a unqiuq array geometry (note, is assumed antennas are close together)
        elif isinstance( antenna_mode, np.array ) and len(antenna_mode.shape)==2:
            self.ITRF_dipole_positions = None ## GOOD LUCK!
            self.dipoleNames = []
            self.local_dipole_positions = np.empty( ( len(antenna_mode)*2, 3), dtype=np.double )
            for i,XYZ in enumerate(antenna_mode): 
                self.local_dipole_positions[ 2*i, : ] = XYZ
                self.local_dipole_positions[ 2*i+1, : ] = XYZ

                AN_even = sid + "111" + str(2*i).zfill(3)
                AN_odd = sid + "111" + str(2*i+1).zfill(3)

                self.dipoleNames.append( AN_even )
                self.dipoleNames.append( AN_odd )


        ## PULSE gen

        pulse_gen = pulse_generator(antenna_function, trace_length=simulation_length, sample_time=5.0e-9, additional_freq_filter=None, atmosphere=self.atmosphere)

        polarizations = np.array( [ ((i+1)%2) for i in range(len(self.dipoleNames)) ] )
        self.simulated_data, self.simulated_data_starts = pulse_gen(self.local_dipole_positions, polarizations, self.source_XYZT, self.source_XYZ_moment, out_a=None, out_b=None, station_integer=None, centering_mode='float_middle')
        self.nominal_sample_number = np.min(self.simulated_data_starts)

### extra methods not in original class.

    def get_all_antennaNames(self):
        return self.ALL_antenna_names

    def get_all_antennaPositions(self):
        return self.ALL_local_dipole_positions


### methods that should error if called
    def set_polarization_flips(self, even_antenna_names):
        print('set_polarization_flips should not be called in sim!')
        quit()

    def set_odd_polarization_delay(self, new_delay):
        print('set_odd_polarization_delay should not be called in sim!')
        quit()

    def set_station_delay(self, station_delay):
        print('set_station_delay should not be called in sim!')
        quit()
                
    def find_and_set_polarization_delay(self, verbose=False, tolerance=1e-9):
        print('find_and_set_polarization_delay should not be called in sim!')
        quit()

    def has_antenna(self, antenna_name=None, antenna_ID=None):
        print('has_antenna should not be called in sim!')
        quit()

### easy methods to overload

    def needs_metadata(self):
        return False

    def get_station_name(self):
        return self.StationName

    def get_station_delay(self):
        return 0.0

    def get_station_ID(self):
        return self.StationID

    def get_antenna_names(self):
        return self.dipoleNames
    
    def get_XYdipole_indeces(self):
        """Return 2D [i,j] array of integers. i is the index of the "full" dual-polarizated antenna (half of length of get_antenna_names). j is 0 for X-dipoles (odd for LBA_OUTER), 1 for Y-dipoles.
        Value is antenna_id (index of get_antenna_names). Will be -1 if antenna does not exist"""

        n_pairs = int(len(self.dipoleNames)/2)
        ret = np.full( (n_pairs,2), -1, dtype=np.int)
        for pair_i in range(n_pairs):
            # simulate LBA_outer only
            X_index = 2*pair_i + 1
            Y_index = 2*pair_i

            ret[pair_i, 0] = X_index
            ret[pair_i, 1] = Y_index

        return ret

    def get_antenna_set(self):
        return "LBA_OUTER"    

    def get_sample_frequency(self):
        """gets samples per second. Typically 200 MHz."""
        return 200.0e6

    def get_filter_selection(self):
        """return a string that represents the frequency filter used. Typically "LBA_10_90"""
        return "LBA_10_90"

    def get_timestamp(self):
        """return the POSIX timestamp of the first data point"""
        return 0 ## ??

    def get_timestamp_as_datetime(self):
        """return the POSIX timestampe of the first data point as a python datetime localized to UTC"""
        return datetime.datetime.fromtimestamp( self.get_timestamp(), tz=datetime.timezone.utc )


## methods related to SIM details
    def get_full_data_lengths(self):
        """get the number of samples stored for each antenna. Note that due to the fact that the antennas do not start recording
        at the exact same instant (in general), this full data length is not all usable
        returns array of ints. Value is -1 if antenna does not exist."""

        return np.array([ len(d) for d in  self.simulated_data ], dtype=int)

    def get_all_sample_numbers(self):
        return self.simulated_data_starts

    def get_nominal_sample_number(self):
        """return the sample number of the 0th data sample returned by get_data.
        Divide by sample_frequency to get time from timestamp of the 0th data sample"""
        return self.nominal_sample_number

    def get_nominal_data_lengths(self):
        return self.get_full_data_lengths()

    def get_ITRF_antenna_positions(self, out=None):
        """returns the ITRF positions of the antennas. Returns a 2D numpy array. 
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated.
        Does not account for polarization flips, but shouldn't need too."""

        if self.ITRF_dipole_positions is None:
            print( "ERROR in get_ITRF_antenna_positions" )
            quit()

        if out is None:
            out = np.empty( (len(self.dipoleNames), 3) )
        
        for ant_i, XYZ in enumerate(self.ITRF_dipole_positions):
            out[ant_i] = XYZ
            
        return out


    def get_LOFAR_centered_positions(self, out=None):
        """returns the positions (as a 2D numpy array) of the antennas with respect to CS002. 
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated.
        Does not account for polarization flips, but shouldn't need too."""
        
        if out is None:
            out = np.empty( (len(self.dipoleNames), 3) )
        
        for ant_i, XYZ in enumerate(self.local_dipole_positions):
            out[ant_i] = XYZ
            
        return out

    def get_timing_callibration_delays(self, out=None, force_file_delays=False):

        if out is None:
            out = np.empty( len(self.dipoleNames) )

        out[:] = -self.nominal_sample_number*5.0e-9

        return out

    def get_total_delays(self, out=None, force_file_delays=False):

        return self.get_timing_callibration_delays( out )


    def get_time_from_second(self, out=None, force_file_delays=False):

        out = self.get_timing_callibration_delays(out)
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
            atmo_to_use = self.atmosphere
                
        v_airs = atmo_to_use.get_effective_lightSpeed(source_location, antenna_locations)

        antenna_locations -= source_location
        antenna_locations *= antenna_locations
        np.sum(antenna_locations, axis=1, out=out)
        np.sqrt(out, out=out)
        out /= v_airs
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
            
        initial_point = start_index + self.nominal_sample_number - self.simulated_data_starts[antenna_index]
        final_point = initial_point+num_points

        ret = np.random.normal(scale=self.noise, size=num_points )

        if final_point <= 0 : ## before start
            return ret
        elif initial_point > self.simulation_length: ## after end
            return ret
        else: 
            if initial_point < 0:
                sim_initial_point = 0
                ret_initial_point = -initial_point
            else:
                sim_initial_point = initial_point
                ret_initial_point = 0

            if final_point >=  self.simulation_length:
                sim_final_point = self.simulation_length
                ret_final_point = ret_initial_point + self.simulation_length - sim_initial_point
            else:
                sim_final_point = final_point
                ret_final_point = num_points

            if ( sim_final_point-sim_initial_point ) != ( ret_final_point-ret_initial_point ):
                print('ERROR A in sim get_data')
                quit()

            ret[ ret_initial_point:ret_final_point ] += self.simulated_data[ antenna_index, sim_initial_point:sim_final_point ].real
                ## must be addition to preserve noise

            return ret

### everything below is for writing Olaf Format simulations

    def wroteOlafFormat(self, sim_name):
        """file name will be sim_name + '_{sname}.dat' """

        fname = sim_name + '_' + self.StationName + '.dat'
        with open(fname, 'w') as fout:

            ## header line
            fout.write( 'StartTime[ms]=  ')
            fout.write( str(self.nominal_sample_number*((5e-9)*(1e3))) )
            fout.write('  N_samples=  ')
            total_length = ( np.max(self.simulated_data_starts) - self.nominal_sample_number ) + self.simulation_length
            fout.write( str(total_length) )
            fout.write( " N_ant=    " )
            n_pairs = int(len(self.dipoleNames)/2)
            fout.write( str(n_pairs) )
            fout.write( "   P_noise=1\n")

            ## antenna positions
            names = self.get_antenna_names()
            positions = self.get_LOFAR_centered_positions()
            for ant_pair_i in range(n_pairs):
                even_i = ant_pair_i

                fout.write( str(int(names[even_i][-3:])) )
                fout.write(' NEh= ')
                fout.write( str(positions[even_i,1]) )  
                fout.write(' ')
                fout.write( str(positions[even_i,0]) )  
                fout.write(' ')
                fout.write( str(positions[even_i,2]) )  
                fout.write('\n')

            ## data
            for sample_i in range(total_length):
                for ant_i in range( len(self.dipoleNames) ):
                    ant_sample_i =  sample_i - ( self.simulated_data_starts[ant_i] - self.nominal_sample_number )

                    if 0 <= ant_sample_i < self.simulation_length:
                        d = self.simulated_data[ ant_i, ant_sample_i ].real
                    else:
                        d = 0

                    fout.write( str(d) )

                    if ant_i == len(self.dipoleNames)-1:
                        fout.write('\n')
                    else:
                        fout.write('    ')



def writeOlafStructureFile(sim_name, stationNames):
    """when writing sim files in Olaf's format, need structue file. This writes said structure file. file name will be sim_name+'_Structure.dat' """

    fname = sim_name + '_Structure.dat'

    with open(fname, 'w') as fout:
        for sname in stationNames:
            id = Sname_to_SId_dict[sname]
            fout.write( str(id) )
            fout.write('   ')
            fout.write( sname )
            fout.write('\n')





##### this is now all for reading Olaf Formatted simulation data
def read_file(fname):
    with open(fname, 'r') as fin:
        header = fin.readline().split()
        startime =  float(header[1])/1000.0
        Nsamples =  int( header[3] )
        NantPairs = int( header[5] )

        Antenna_XYZs = np.empty( (NantPairs,3) )
        for pairI in range(NantPairs):
            lineWords = fin.readline().split()

            Antenna_XYZs[ pairI,0 ] = float( lineWords[3] )
            Antenna_XYZs[ pairI,1 ] = float( lineWords[2] )
            Antenna_XYZs[ pairI,2 ] = float( lineWords[4] )

        datablock = np.empty( (NantPairs*2, Nsamples) )

        for sampI in range(Nsamples):
            dataline = fin.readline().split()
            for antI in range(NantPairs*2):
                datablock[ antI, sampI ] = float( dataline[antI] )


    return startime, Antenna_XYZs, datablock


def getFileNames(folder, SimName):
    """given a folder and simulation name, returns a dictionary. Keys are station names, values are file names prepended with the folder"""

    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    filteredFiles = [ f for f in onlyfiles if f.startswith(SimName) and f.endswith('.dat')]
    stationNames = [ f.split('.')[0].split('_')[-1] for f in filteredFiles ]

    FileDict = { sname:join(folder,f) for f,sname in zip(filteredFiles,stationNames) if sname in util.Sname_to_SId_dict }

    return FileDict



class TBB_ReadOlafSimulation:
    def __init__(self, StationName, fname, padBack=2**16, noise_STD=None):
        """padBack is number of samples to subrtract off start time. I.E. time of first sample = startTime - padBack*5e-9"""

        self.StationName = StationName
        self.StationID = util.Sname_to_SId_dict[self.StationName] 
        self.fname = fname
        self.padBack = padBack
        self.noise_STD = noise_STD

        self.StartTime, self.antennaPair_locations, self.data = read_file(fname)

        ## antennaNames
        prefix = str(self.StationID).zfill(3) + '000'
        self.dipoleNames = []
        self.antenna_locations = np.empty( (len(self.antennaPair_locations)*2, 3), dtype=float )
        for pair_i, loc in enumerate(self.antennaPair_locations):
            self.dipoleNames.append( prefix + str(pair_i*2).zfill(3) )
            self.dipoleNames.append( prefix + str(pair_i*2+1).zfill(3) )

            self.antenna_locations[ pair_i*2 ] = self.antennaPair_locations[ pair_i ]
            self.antenna_locations[ pair_i*2+1 ] = self.antennaPair_locations[ pair_i ]


### methods that should error if called
    def set_polarization_flips(self, even_antenna_names):
        print('set_polarization_flips should not be called in sim!')
        quit()

    def set_odd_polarization_delay(self, new_delay):
        print('set_odd_polarization_delay should not be called in sim!')
        quit()

    def set_station_delay(self, station_delay):
        print('set_station_delay should not be called in sim!')
        quit()
                
    def find_and_set_polarization_delay(self, verbose=False, tolerance=1e-9):
        print('find_and_set_polarization_delay should not be called in sim!')
        quit()

    def has_antenna(self, antenna_name=None, antenna_ID=None):
        print('has_antenna should not be called in sim!')
        quit()

### easy methods to overload

    def needs_metadata(self):
        return False

    def get_station_name(self):
        return self.StationName

    def get_station_delay(self):
        return 0.0

    def get_station_ID(self):
        return self.StationID 

    def get_antenna_names(self):
        return self.dipoleNames
    
    def get_XYdipole_indeces(self):
        """Return 2D [i,j] array of integers. i is the index of the "full" dual-polarizated antenna (half of length of get_antenna_names). j is 0 for X-dipoles (odd for LBA_OUTER), 1 for Y-dipoles.
        Value is antenna_id (index of get_antenna_names). Will be -1 if antenna does not exist"""

        n_pairs = int(len(self.dipoleNames)/2)
        ret = np.full( (n_pairs,2), -1, dtype=np.int)
        for pair_i in range(n_pairs):
            # simulate LBA_outer only
            X_index = 2*pair_i + 1
            Y_index = 2*pair_i

            ret[pair_i, 0] = X_index
            ret[pair_i, 1] = Y_index

        return ret

    def get_antenna_set(self):
        return "LBA_OUTER"    

    def get_sample_frequency(self):
        """gets samples per second. Typically 200 MHz."""
        return 200.0e6

    def get_filter_selection(self):
        """return a string that represents the frequency filter used. Typically "LBA_10_90"""
        return "LBA_10_90"

    def get_timestamp(self):
        """return the POSIX timestamp of the first data point"""
        return 0 ## ??

    def get_timestamp_as_datetime(self):
        """return the POSIX timestampe of the first data point as a python datetime localized to UTC"""
        return datetime.datetime.fromtimestamp( self.get_timestamp(), tz=datetime.timezone.utc )


## methods related to SIM details
    #def get_full_data_lengths(self):
        """get the number of samples stored for each antenna. Note that due to the fact that the antennas do not start recording
        at the exact same instant (in general), this full data length is not all usable
        returns array of ints. Value is -1 if antenna does not exist."""

        #return np.array([ len(d) for d in  self.simulated_data ], dtype=int)

    #def get_all_sample_numbers(self):
      #  return self.simulated_data_starts

    #def get_nominal_sample_number(self):
        """return the sample number of the 0th data sample returned by get_data.
        Divide by sample_frequency to get time from timestamp of the 0th data sample"""
    #    return 0 ## not even sure!

    #def get_nominal_data_lengths(self):
        #return self.get_full_data_lengths()

    def get_LOFAR_centered_positions(self, out=None):
        """returns the positions (as a 2D numpy array) of the antennas with respect to CS002. 
        if out is a numpy array, it is used to store the antenna positions, otherwise a new array is allocated.
        Does not account for polarization flips, but shouldn't need too."""
        
        if out is None:
            out = np.empty( (len(self.dipoleNames), 3) )
        
        for ant_i, XYZ in enumerate(self.antenna_locations):
            out[ant_i] = XYZ
            
        return out

    def get_timing_callibration_delays(self, out=None, force_file_delays=False):

        if out is None:
            out = np.empty( len(self.dipoleNames) )

        out[:] = -(self.StartTime - self.padBack*(5e-9))

        return out

    def get_total_delays(self, out=None, force_file_delays=False):

        return self.get_timing_callibration_delays( out )


    def get_time_from_second(self, out=None, force_file_delays=False):

        out = self.get_timing_callibration_delays(out)
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
            atmo_to_use = LOLIM_atmo.default_atmosphere
                
        v_airs = atmo_to_use.get_effective_lightSpeed(source_location, antenna_locations)

        antenna_locations -= source_location
        antenna_locations *= antenna_locations
        np.sum(antenna_locations, axis=1, out=out)
        np.sqrt(out, out=out)
        out /= v_airs
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


        if self.noise_STD is None:
            ret = np.zeros( num_points, dtype=float )
        else:
            ret = np.random.normal( scale=self.noise_STD, size=num_points )

        naive_initial_sample = start_index - self.padBack

        if naive_initial_sample < 0:
            ret_start = -naive_initial_sample
            data_start = 0
        elif naive_initial_sample >= self.data.shape[1]:
            print('')
            return ret
        else:
            ret_start = 0
            data_start = naive_initial_sample

        naive_end_sample = naive_initial_sample + num_points
        if naive_end_sample < 1:
            return ret
        elif naive_end_sample > self.data.shape[1]:
            ret_end = num_points - ( naive_end_sample - self.data.shape[1] )
            data_end = self.data.shape[1]
        else:
            ret_end = num_points
            data_end = naive_end_sample


        if (ret_end-ret_start) != (data_end-data_start):
            print('Sim Error should not be reached!')
            quit()


        ret[ret_start:ret_end] = self.data[antenna_index, data_start:data_end]

        return ret


