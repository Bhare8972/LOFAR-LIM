#!/usr/bin/env python3

""" Given a location and an antenna (and other supporting information), this utility will find and return the correct trace """

import numpy as np
from matplotlib import pyplot as plt

from LoLIM.utilities import processed_data_dir, v_air, SId_to_Sname
from LoLIM.signal_processing import remove_saturation, num_double_zeros

class getTrace_fromLoc:
    def __init__(self, data_file_dict, data_filter_dict, station_timing_calibration=None, return_dbl_zeros=False):
        """data_file_dict is dictionary where keys are station names and values are TBB files. data_filter_dict is same, but
        values are window_and_filter objects. station_timing_calibration is same, but values are the timing callibration of the stations"""
        
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
    
    
    
    
if __name__ == '__main__':
    
    from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1
    from LoLIM.findRFI import window_and_filter
    from LoLIM.read_pulse_data import  read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
    from LoLIM.interferometry import read_interferometric_PSE as R_IPSE
    
    from matplotlib import pyplot as plt
    
    
    ##### NOTE: general structure of this code is good, but somethings are outdated ####
    
    
    timeID = "D20170929T202255.000Z"
    input_folder = "interferometry_out4"
    IPSE_block = 20
    IPSE_unique_ID = 2083 
    antenna_num = 150
    station_delay_file = "station_delays.txt"
    
    pulse_length = 1000
    
#    timeID = "D20160712T173455.100Z"
#    input_folder = "interferometry_out3_lowAmp_goodDelays"
#    IPSE_block = 505
#    IPSE_unique_ID = 50500
#    antenna_num = 0
#    station_delay_file = "station_delays_5.txt"
    
    polarization_flips = "polarization_flips.txt"
    bad_antennas = "bad_antennas.txt"  ##TODO NOTE: this isn't working
    additional_antenna_delays = "ant_delays.txt"
    
    
    processed_data_folder = processed_data_dir(timeID)
    
    polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
    additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
    station_timing_offsets = read_station_delays( processed_data_folder+'/'+station_delay_file )
    
    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(fpaths, force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays) \
                      for sname,fpaths in raw_fpaths.items() if sname in station_timing_offsets}
    
    data_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in station_timing_offsets}
    
    
    trace_locator = getTrace_fromLoc( raw_data_files, data_filters, station_timing_offsets )
    
    
    interferometry_header, IPSE_list = R_IPSE.load_interferometric_PSE( processed_data_folder + "/" + input_folder, blocks_to_open=[IPSE_block] )
    IPSE = [IPSE for IPSE in IPSE_list if IPSE.unique_index==IPSE_unique_ID][0]
    
    print( 'station:', interferometry_header.antenna_data[antenna_num].station )
    print( 'antenna:', interferometry_header.antenna_data[antenna_num].name )
    
    if pulse_length is None:
        pulse_length = interferometry_header.pulse_length
    
    
    print("saved trace")
    T = np.array(IPSE.file_dataset[antenna_num])
    plt.plot( np.abs(T) )    
    plt.plot( np.real(T) )   
    plt.show()
    
    print(IPSE.XYZT, interferometry_header.antenna_data[antenna_num].name)
    
    start_sample, total_time_offset, arrival_time, extracted_trace = trace_locator.get_trace_fromLoc(IPSE.XYZT, interferometry_header.antenna_data[antenna_num].name, pulse_length, do_remove_RFI=True, do_remove_saturation=True)
    
    
    
    print( start_sample )
    print("extracted trace")
    plt.plot( np.abs(extracted_trace) )
    plt.plot( np.real(extracted_trace) )
    plt.show()
    
    
    
    
    
    
    
    
    