#!/usr/bin/env python3

""" Given a location and an antenna (and other supporting information), this utility will find and return the correct trace """

import numpy as np
from matplotlib import pyplot as plt

from LoLIM.utilities import processed_data_dir, v_air, antName_to_station
from LoLIM.signal_processing import remove_saturation, num_double_zeros, half_hann_window, simple_bandpass

from LoLIM.NumLib.FFT import complex_fft_obj

class getTrace_fromLoc:
    def __init__(self, data_file_dict, data_filter_dict, station_timing_calibration=None, return_dbl_zeros=False):
        """data_file_dict is dictionary where keys are station names and values are TBB files. data_filter_dict is same, but
        values are window_and_filter objects. station_timing_calibration is same, but values are the timing callibration of the stations"""


        print("WARNING! getTrace_fromLoc is being depreciated! (use dataLoaderHelper)")
        
        self.data_file_dict = data_file_dict
        self.data_filter_dict = data_filter_dict
#        self.station_timing_calibration = station_timing_calibration
        self.return_dbl_zeros = return_dbl_zeros
        self.atmosphere = None #atmosphere # I don't think this is needed. Atmo can be set by setting calibration when opening data file
        
        if station_timing_calibration is not None:
            for sname, TBB_file in data_file_dict.items():
                TBB_file.set_station_delay( station_timing_calibration[sname] )

    def get_TBBfile_dict(self):
        return self.data_file_dict

    def set_return_dbl_zeros(self, return_dbl_zeros):
        self.return_dbl_zeros = return_dbl_zeros
                
    def source_recieved_index(self, XYZT, ant_name):
        
        station_name = SId_to_Sname[ int(ant_name[:3]) ]
        station_data = self.data_file_dict[ station_name ]
        file_antenna_index = station_data.get_antenna_names().index( ant_name )
        
        total_time_offset = station_data.get_total_delays()[ file_antenna_index ]
        antenna_locations = station_data.get_LOFAR_centered_positions()
        predicted_arrival_time = station_data.get_geometric_delays(XYZT[:3], antenna_locations=antenna_locations[file_antenna_index:file_antenna_index+1]) + XYZT[3]
        
        data_arrival_index = int( predicted_arrival_time/5.0E-9 + total_time_offset/5.0E-9 )
        
        return data_arrival_index
        
    def get_trace_fromLoc(self, XYZT, ant_name, width, do_remove_RFI=True, do_remove_saturation=True, positive_saturation=2046, negative_saturation=-2047, removal_length=50, half_hann_length=50):
        """given the location of the source in XYZT, name of the antenna, and width (in num data samples) of the desired pulse, 
        return starting_index of the returned trace, total time calibration delay of that antenna (from TBB.get_total_delays), predicted arrival time of pulse at that antenna,
        and the time trace centered on the arrival time."""
        
        station_name = SId_to_Sname[ int(ant_name[:3]) ]
        station_data = self.data_file_dict[ station_name ]
        data_filter = self.data_filter_dict[ station_name ]
        file_antenna_index = station_data.get_antenna_names().index( ant_name )
        
#        ant_loc = station_data.get_LOFAR_centered_positions()[ file_antenna_index ]
#        ant_time_offset = station_data.get_timing_callibration_delays()[file_antenna_index] - station_data.get_nominal_sample_number()*5.0E-9
#        total_time_offset = ant_time_offset + self.station_timing_calibration[station_name]
#        predicted_arrival_time = np.linalg.norm( XYZT[:3]-ant_loc )/v_air + XYZT[3]
        
        total_time_offset = station_data.get_total_delays()[ file_antenna_index ]
        antenna_locations = station_data.get_LOFAR_centered_positions()
        predicted_arrival_time = station_data.get_geometric_delays(XYZT[:3], antenna_locations=antenna_locations[file_antenna_index:file_antenna_index+1], atmosphere_override=self.atmosphere)[0] + XYZT[3]
        
        data_arrival_index = int( predicted_arrival_time/5.0E-9 + total_time_offset/5.0E-9 )
        
        local_data_index = int( data_filter.blocksize*0.5 )
        data_start_sample = data_arrival_index - local_data_index      
        
        input_data = station_data.get_data(data_start_sample, data_filter.blocksize, antenna_index=file_antenna_index  )
        input_data = np.array( input_data, dtype=np.double )
        if do_remove_saturation:
            remove_saturation( input_data, positive_saturation, negative_saturation, removal_length, half_hann_length )

        width_before = int(width*0.5)
        width_after = int(width-width_before)

        if self.return_dbl_zeros :
            dbl_zeros = num_double_zeros( input_data[ local_data_index-width_before : local_data_index+width_after ] )

        if do_remove_RFI:
            data = data_filter.filter( input_data )[ local_data_index-width_before : local_data_index+width_after ]
        else:
            data = input_data[ local_data_index-width_before : local_data_index+width_after ]

        if self.return_dbl_zeros:
            return data_start_sample+local_data_index-width_before, total_time_offset, predicted_arrival_time, data, dbl_zeros
        else:
            return data_start_sample+local_data_index-width_before, total_time_offset, predicted_arrival_time, data
    
    def get_trace_fromIndex(self, starting_index, ant_name, width, do_remove_RFI=True, do_remove_saturation=True, positive_saturation=2046, negative_saturation=-2047, removal_length=50, saturation_half_hann_length=50):
        """similar to previous, but now retrieve trace based on location in file. For repeatability.
        Has same returns. predicted arrival time is just the time in middle of trace"""
        
        station_name = SId_to_Sname[ int(ant_name[:3]) ]
        station_data = self.data_file_dict[ station_name ]
        data_filter = self.data_filter_dict[ station_name ]
        file_antenna_index = station_data.get_antenna_names().index( ant_name )
        
        total_time_offset = station_data.get_total_delays()[ file_antenna_index ]
       
        local_data_index = int( data_filter.blocksize*0.5 )
        
        input_data = station_data.get_data(starting_index-local_data_index , data_filter.blocksize, antenna_index=file_antenna_index  )
        input_data = np.array( input_data, dtype=np.double )
        if do_remove_saturation:
            remove_saturation( input_data, positive_saturation, negative_saturation, removal_length, saturation_half_hann_length )

        if self.return_dbl_zeros :
            dbl_zeros = num_double_zeros( input_data[ local_data_index : local_data_index+width ] )
        
        
        if do_remove_RFI:
            data = data_filter.filter( input_data )[ local_data_index : local_data_index+width ]
        else:
            data = input_data[ local_data_index : local_data_index+width ]
        
        predicted_arrival_time = starting_index*5.0E-9 - total_time_offset + 0.5*width*5.0E-9

        if self.return_dbl_zeros:
            return starting_index, total_time_offset, predicted_arrival_time, data, dbl_zeros
        else:
            return starting_index, total_time_offset, predicted_arrival_time, data
    

### NOTE: this class has problem that I'm not sure how to handle multiple antenna modes per station. Assume class only manages one antenna mode. 
## In future we need to either extend this class, or add a new wrapper, that removes antenna function and does amplitude calibrations. 
class dataLoaderHelper:
    """ class designed to assist with loading data and doing level-0 signal analysis. Designed to replace getTrace_fromLoc. Assumes all input station data files are all in same antenna mode.
    Applies two frequency filters to the data. a "simple" band-pass filter, it has some default settings defined in method "set_simple_freqFilter". This settings
    can be changed by calling "set_simple_freqFilter". In addition, a RFI line filter can be applied. The two methods are "a" and "b", depeinding if using old or "adv" findRFI
    It needs to be extended to work in higher nyquist zones, but currently this is not implmented."""

    def __init__(self, data_file_dict):
        """data_file_dict is dictionary where keys are station names and values are TBB files. Assumes each TBB file is in same antenna mode, and each TBB file has calibration file applied."""
        
        self.data_file_dict = data_file_dict
        #self.atmosphere = None  # I don't think this is needed. Atmo can be set by setting calibration when opening data file

        #self.do_zero_saturation = False
        self.station_RFI_data = None

        self.set_simple_freqFilter()

        self.known_timeCalibration_information = {}
        ## key is station name, value is output of MultiFile_Dal1.get_time_from_second.
        ## reason for this is that MultiFile_Dal1.get_time_from_second re-computes the delays every time, thus is slow for regular use

        self.known_antenna_locs = {}
        ## key is station name, value is output of MultiFile_Dal1.get_LOFAR_centered_positions.
        ## reason for this is that MultiFile_Dal1.get_LOFAR_centered_positions re-computes the delays every time, thus is slow for regular use


        
## basic settings and such ##

    def get_TBBfile_dict(self):
        return self.data_file_dict

    #def set_zeroSaturation( self, do_zero_saturation, positive_saturation=2046, negative_saturation=-2047, removal_length=50, half_hann_length=50 ):
    #    self.do_zero_saturation = do_zero_saturation
    #    self.positive_saturation = positive_saturation
    #    self.negative_saturation = negative_saturation
    #    self.removal_length = removal_length
    #    self.half_hann_length = half_hann_length

    def set_simple_freqFilter(self, blocksize=2**16, lower_filter=30.0E6, upper_filter=80.0E6, half_window_percent=0.1, filter_roll_width = 2.5E6):
        ## save variables
        self.blocksize = blocksize
        self.lower_filter = lower_filter
        self.upper_filter = upper_filter
        self.half_window_percent = half_window_percent
        self.filter_roll_width = filter_roll_width

        ## make tukey windowing function
        self.half_hann_window, self.window_edge_size = half_hann_window(self.blocksize, self.half_window_percent, return_windowSize=True)

        ## get sample frequency, assuming all files the same
        singleFileObj = self.data_file_dict[ list(self.data_file_dict)[0] ] ## just selects one file object
        self.sample_frequency = singleFileObj.get_sample_frequency()

        ## generate band-pass filter
        self.FFT_frequencies = np.fft.fftfreq(self.blocksize, d=(1/self.sample_frequency))

        self.bandpass_filter = simple_bandpass(self.FFT_frequencies, lower_freq=lower_filter, upper_freq=upper_filter, roll_width = filter_roll_width)

        ## completly reject low-frequency bits
        self.bandpass_filter[0] = 0.0
        self.bandpass_filter[1] = 0.0


        ## set temporary data objects
        self.tmp_data_block = None
        self.single_data_block = np.empty( blocksize, dtype=complex )
        self.temp_timeShifter = np.empty( blocksize, dtype=complex )

        self.FFT_obj = complex_fft_obj( blocksize )
        
        
    def get_outputSampleFrequency(self):
        return self.sample_frequency

    def set_RFIFilter(self, station_RFI_data, is_adv=True):
        """Set RFI line filters. To be implmented better"""
        self.RFI_lineData_isAdv = is_adv
        self.station_RFI_data = station_RFI_data

        ## filer out un-needed antenna modes
        if self.RFI_lineData_isAdv:
            ## first get known antenna set and filter. assume all TBB files are the same
            exemplar_file = self.data_file_dict[ list(self.data_file_dict.keys())[0] ]
            antenna_set = exemplar_file.get_antenna_set()
            filter_selection = exemplar_file.get_filter_selection()

            ## now search for correct RFI line data
            old_RFI_data = self.station_RFI_data
            self.station_RFI_data = {}
            for sname, all_datums in old_RFI_data.items():
                usable_datums = []
                have_X = False
                have_Y = False

                for d in all_datums:

                    if (d["antenna_set"]==antenna_set) and (d["filter_selection"]==filter_selection):
                        usable_datums.append( d )
                        if d['antenna_polarization'] == 'Y':
                            have_Y = True
                        elif d['antenna_polarization'] == 'X':
                            have_X = True

                if (len(usable_datums)!=2) or (not have_X) or (not have_Y):
                    raise Exception("RFI line data is not usable")

                self.station_RFI_data[sname] = usable_datums



    def get_window_edge_size(self):
        return self.window_edge_size + 1

    def get_maxTraceLength(self):
        """return maximum tracelenth that can be returned by this class. Is blocksize - 2*window_edge_size"""
        return self.blocksize - 2*self.window_edge_size - 2

## some data-processing helpers ##

    def get_timeCalibration(self, sname):
        """return same result as MultiFile_Dal1.get_time_from_second. Difference is this method caches result so that multiple calls are fast"""
        if sname in self.known_timeCalibration_information:
            return self.known_timeCalibration_information[sname]

        else:
            cals = self.data_file_dict[sname].get_time_from_second()
            self.known_timeCalibration_information[sname] = cals 
            return cals

    def get_antennaLocs(self, sname):
        """return same result as MultiFile_Dal1.get_LOFAR_centered_positions. Difference is this method caches result so that multiple calls are faster"""

        if sname in self.known_antenna_locs:
            return self.known_antenna_locs[sname]

        else:
            locs = self.data_file_dict[sname].get_LOFAR_centered_positions()
            self.known_antenna_locs[sname] = locs
            return locs
        



    def XYZT_to_arrivalTime(self, XYZT, ant_name):
        """given a source at location and time XYZT, return the arrival time at an antenna. (e.g. only geometric delay)""" 

        station_name = SId_to_Sname[ int(ant_name[:3]) ]
        station_data = self.data_file_dict[ station_name ]
        file_antenna_index = station_data.get_antenna_names().index( ant_name )

        antenna_locations = self.get_antennaLocs(station_name)
        geo_delay = station_data.get_geometric_delays(XYZT[:3], antenna_locations=antenna_locations[file_antenna_index:file_antenna_index+1])[0]

        return geo_delay


## NOTE: could add anouther mode that gives sample number, but allows for sub-sample shift
    def get_SingleProcessed_block(self, initial_sample, ant_name, initial_sample_mode, perform_ifft=False):
        """return one processed block of data.  initial_sample indicates the start of the block, depending on value of initial_sample_mode.
        if initial_sample_mode is 's', than initial_sample is expected to be an integer which number of samples past nominal_sample_number (timing calibration is ignored) that is the first sample of returned data.
        if initial_sample_mode is 't', than intial_sample is expected to be seconds past the POSIX timestamp of the first sample (all time calibrations are accounted for, and sub-sample shift is accounted for).
        If perform_ifft is True, than return result in time-domain, else return result in frequency domain.
        Note, returned array will have edges of size window_edge_size (+1 sample if initial_sample_mode is 't')
        Frequencies are filtered according to settings.
        Return complex-valued numpy array. This array points to an internal buffer that can be modifed again by this class. Also return number of double-zeros in raw data"""


        ## book-keeping
        station_name = antName_to_station( ant_name )
        station_data = self.data_file_dict[ station_name ]


        if initial_sample_mode == 's':

            if not isinstance(initial_sample, int):
                raise Exception("For initial_sample_mode=='s', expected initial_sample to be an int")

            initial_sample_index = initial_sample
            sub_sample = None

        elif initial_sample_mode == 't':
            station_time_calibration = self.get_timeCalibration( station_name )

            file_antenna_index = station_data.get_antenna_names().index( ant_name )
            antenna_time_calibration = station_time_calibration[file_antenna_index]  ## this is time of sample 0, in seconds, past the time stamp

            initial_sample_index_dec = (initial_sample - antenna_time_calibration)*self.sample_frequency
            initial_sample_index = int(initial_sample_index_dec)
            sub_sample = initial_sample_index_dec - initial_sample_index



        #intSampleIndex = int( initial_sample_index )
        #residual_sample_index = initial_sample_index - intSampleIndex

        #if sample_is_from_second:
        #    intSampleIndex -= station_data.get_nominal_sample_number()

        ## get data
        self.tmp_data_block = station_data.get_data(initial_sample_index, self.blocksize, antenna_name=ant_name)#, out=self.tmp_data_block)

        dbl_zeros = num_double_zeros( self.tmp_data_block )

        self.single_data_block[:] = self.tmp_data_block

        #if self.do_zero_saturation:
        #    remove_saturation( input_data, self.positive_saturation, self.negative_saturation, self.removal_length, self.half_hann_length )


        ## signal-processing
        self.single_data_block *= self.half_hann_window

        self.FFT_obj.fft( self.single_data_block )

        self.single_data_block *= self.bandpass_filter

        if (self.station_RFI_data is not None):
            station_RFI_filter = self.station_RFI_data[station_name] ## assume this is already filtered for antenna type

            if self.RFI_lineData_isAdv: ## need to find correct polrization 
                Y_polarized_dipoles = station_data.get_antenna_names()[::2]
                pol = 'Y' if ant_name in Y_polarized_dipoles else 'X'

                for f in station_RFI_filter:
                    if f['antenna_polarization'] == pol:
                        station_RFI_filter = f 
                        break

            if station_RFI_filter['blocksize'] != self.blocksize:
                raise Exception("ERROR! blocksize in RFI data is different than loading blocksize")

            RFI_lines = station_RFI_filter['dirty_channels']
            self.single_data_block[RFI_lines] = 0.0



        if (not sub_sample is None) and (sub_sample != 0):
            self.temp_timeShifter[:] = self.FFT_frequencies
            self.temp_timeShifter *= 2j*np.pi*sub_sample/self.sample_frequency ## this shifts the signal to earlier times by residual_sample_index number of samples
            np.exp( self.temp_timeShifter, out=self.temp_timeShifter )

            self.single_data_block *= self.temp_timeShifter


        if perform_ifft:
            self.FFT_obj.ifft( self.single_data_block )

        return self.single_data_block, dbl_zeros


    def FFTdatablock_from_loc(self, XYZT, antenna_name):
        """similar to get_SingleProcessed_block, except first sample is defined by location in sky, and always returns in fourier space"""

        #first_index = self.source_recieved_index(XYZT, antenna_name)

        arrival_time = self.XYZT_to_arrivalTime( XYZT, antenna_name)
        return self.get_SingleProcessed_block(arrival_time, antenna_name, initial_sample_mode='t', perform_ifft=False)

    def trace_from_loc(self, leading_num_samples, XYZT, trailing_num_samples, antenna_name):
        """similar to get_SingleProcessed_block. Except always returns a data trace in time-domain. Where:
        return_data[leading_num_samples] corresponds to an emitter at XYZT, and len(return_data) = leading_num_samples+trailing_num_samples.
        Note, this data has all edge-effects removed. Thus maximum trace length to be returned is get_maxTraceLength
        Like other functions also returns number double zeros, however this is for entire block NOT just returned data section.
        Like other functions, data returned shares memory, and thus will be corruped by future function calls"""


        True_leading_samples = leading_num_samples + window_edge_size + 1
        arrival_time = self.XYZT_to_arrivalTime( XYZT, antenna_name)

        data_block, dblzeros = self.get_SingleProcessed_block( arrival_time-(True_leading_samples/self.sample_frequency), antenna_name, initial_sample_mode='t', perform_ifft=True)

        return data_block[ window_edge_size + 1 : window_edge_size + 1 + leading_num_samples+trailing_num_samples], dblzeros

    def longTrace_from_loc(self, leading_num_samples, XYZT, trailing_num_samples, antenna_name):
        """simlar to trace_from_loc. However, it stiches many blocks together, allowing for long traces to be returned.
        As a result, it does not return number double zeros (only data), and new data block is allocated (NOT shared with future function calls)"""

        data_to_return = np.empty( leading_num_samples+trailing_num_samples, dtype=complex )

        arrival_time = self.XYZT_to_arrivalTime( XYZT, antenna_name)
        current_time = arrival_index-(leading_num_samples+window_edge_size+1)/self.sample_frequency

        return_sample_we_at = 0
        num_samples_needed = leading_num_samples+trailing_num_samples
        maxSamples_per_block = self.get_maxTraceLength()
        while num_samples_needed>0:
            data_block, dblzeros = self.get_SingleProcessed_block(current_time, antenna_name, initial_sample_mode='t', perform_ifft=True)

            if num_samples_needed > maxSamples_per_block:
                L = maxSamples_per_block
            else:
                L = num_samples_needed


            data_to_return[ return_sample_we_at:return_sample_we_at+L] = data_block[ window_edge_size+1 : window_edge_size+1+L ]

            return_sample_we_at += L
            current_time += L/self.sample_frequency
            num_samples_needed -= L

        return data_to_return




    
    
    
# if __name__ == '__main__':
    
#     from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1
#     from LoLIM.findRFI import window_and_filter
#     from LoLIM.read_pulse_data import  read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
#     from LoLIM.interferometry import read_interferometric_PSE as R_IPSE
    
#     from matplotlib import pyplot as plt
    
    
#     ##### NOTE: general structure of this code is good, but somethings are outdated ####
    
    
#     timeID = "D20170929T202255.000Z"
#     input_folder = "interferometry_out4"
#     IPSE_block = 20
#     IPSE_unique_ID = 2083 
#     antenna_num = 150
#     station_delay_file = "station_delays.txt"
    
#     pulse_length = 1000
    
# #    timeID = "D20160712T173455.100Z"
# #    input_folder = "interferometry_out3_lowAmp_goodDelays"
# #    IPSE_block = 505
# #    IPSE_unique_ID = 50500
# #    antenna_num = 0
# #    station_delay_file = "station_delays_5.txt"
    
#     polarization_flips = "polarization_flips.txt"
#     bad_antennas = "bad_antennas.txt"  ##TODO NOTE: this isn't working
#     additional_antenna_delays = "ant_delays.txt"
    
    
#     processed_data_folder = processed_data_dir(timeID)
    
#     polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
#     bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
#     additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
#     station_timing_offsets = read_station_delays( processed_data_folder+'/'+station_delay_file )
    
#     raw_fpaths = filePaths_by_stationName(timeID)
#     raw_data_files = {sname:MultiFile_Dal1(fpaths, force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays) \
#                       for sname,fpaths in raw_fpaths.items() if sname in station_timing_offsets}
    
#     data_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in station_timing_offsets}
    
    
#     trace_locator = getTrace_fromLoc( raw_data_files, data_filters, station_timing_offsets )
    
    
#     interferometry_header, IPSE_list = R_IPSE.load_interferometric_PSE( processed_data_folder + "/" + input_folder, blocks_to_open=[IPSE_block] )
#     IPSE = [IPSE for IPSE in IPSE_list if IPSE.unique_index==IPSE_unique_ID][0]
    
#     print( 'station:', interferometry_header.antenna_data[antenna_num].station )
#     print( 'antenna:', interferometry_header.antenna_data[antenna_num].name )
    
#     if pulse_length is None:
#         pulse_length = interferometry_header.pulse_length
    
    
#     print("saved trace")
#     T = np.array(IPSE.file_dataset[antenna_num])
#     plt.plot( np.abs(T) )    
#     plt.plot( np.real(T) )   
#     plt.show()
    
#     print(IPSE.XYZT, interferometry_header.antenna_data[antenna_num].name)
    
#     start_sample, total_time_offset, arrival_time, extracted_trace = trace_locator.get_trace_fromLoc(IPSE.XYZT, interferometry_header.antenna_data[antenna_num].name, pulse_length, do_remove_RFI=True, do_remove_saturation=True)
    
    
    
#     print( start_sample )
#     print("extracted trace")
#     plt.plot( np.abs(extracted_trace) )
#     plt.plot( np.real(extracted_trace) )
#     plt.show()
    
    
    
    
    
    
    
    
    