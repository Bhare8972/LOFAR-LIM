#!/usr/bin/env python3

import numpy as np

from matplotlib import pyplot as plt

from LoLIM.utilities import RTD
from LoLIM.FFT import complex_fft_obj
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
            'float_beginning' : default option. Puts all antennas at beginning. each antenna has different start index
            'float_middle'    : NOT IMPLEMENTED same, but antennas are in middle
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

            station_delta_XYZ = station_XYZ - XYZT_location[0:3]

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
        self.tmp_memory *= Jones_tmp[:,0,1]
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
            if not centering_mode in ['float_beginning', 'relative_middle']:
                print('ERROR centering mode is:', centering_mode)
                print('needs to be one of:', centering_mode)
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
