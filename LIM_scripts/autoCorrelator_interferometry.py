#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


from LoLIM.utilities import log, processed_data_dir, v_air, SId_to_Sname, Sname_to_SId_dict, RTD, even_antName_to_odd
from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1
from LoLIM.findRFI import window_and_filter
from LoLIM.signal_processing import parabolic_fit, remove_saturation, FFT_time_shift, half_hann_window, upsample_and_correlate

class source_object():
    def __init__(self, XYZT):
        self.XYZT = XYZT
        
        ant_names = reference_data_file.get_antenna_names()
        ant_locations = reference_data_file.get_LOFAR_centered_positions()
        ant_timing_calibrations = reference_data_file.get_timing_callibration_delays()
        
        known_station_delay = -reference_data_file.get_nominal_sample_number()*5.0E-9
        
        hann_window = half_hann_window( trace_length,  half_percent=0.1)
        
        self.ant_locs = []
        self.ant_names = []
        self.ant_start_time = []
        self.even_data_list = []
        self.odd_data_list = []
        
        
        half_TL = int(trace_length/2)
        
        for even_ant_i in range(0,len(ant_names),2):
            even_arrival_time = np.linalg.norm( self.XYZT[:3]-ant_locations[even_ant_i])/v_air + self.XYZT[3]
            even_recieved_time = even_arrival_time + ant_timing_calibrations[even_ant_i] + known_station_delay
            even_receieved_index = int(even_recieved_time/5.0E-9)
            even_subsample_remainder = even_recieved_time - even_receieved_index*5.0E-9
            
            local_received_index = int(blocksize*0.5)
            even_data_start_index = even_receieved_index - local_received_index
            even_data = reference_data_file.get_data( even_data_start_index, num_points=blocksize, antenna_index=even_ant_i )
            
            even_data = np.array(even_data, dtype=np.float)
            remove_saturation(even_data, positive_saturation_amplitude, negative_saturation_amplitude)
            
            
            even_FFT_data = referance_filter.filter_FFT( even_data )
            
            FFT_time_shift( referance_filter.get_frequencies(), even_FFT_data, -even_subsample_remainder )
            even_data = np.fft.ifft(even_FFT_data, axis=-1)
            
            even_data = np.real( even_data )[ local_received_index-half_TL : local_received_index+half_TL  ]
            even_data *= hann_window
            
            
            
            odd_recieved_time = even_arrival_time + ant_timing_calibrations[even_ant_i+1] + known_station_delay
            odd_receieved_index = int(odd_recieved_time/5.0E-9)
            odd_subsample_remainder = odd_recieved_time - odd_receieved_index*5.0E-9
            
            odd_data_start_index = odd_receieved_index - local_received_index
            odd_data = reference_data_file.get_data( odd_data_start_index, num_points=blocksize, antenna_index=even_ant_i+1 )
            
            odd_data = np.array(odd_data, dtype=np.float)
            remove_saturation(odd_data, positive_saturation_amplitude, negative_saturation_amplitude)
            
            odd_FFT_data = referance_filter.filter_FFT( odd_data )
            
            FFT_time_shift( referance_filter.get_frequencies(), odd_FFT_data,  -odd_subsample_remainder+ant_timing_calibrations[even_ant_i]-ant_timing_calibrations[even_ant_i+1])
            odd_data = np.fft.ifft(odd_FFT_data, axis=-1)## now the odd data has same time base as even data
            
            odd_data = np.real(odd_data)[ local_received_index-half_TL : local_received_index+half_TL ] 
            odd_data *= hann_window

            even_max = np.max(even_data)
            odd_max = np.max(odd_data)
            
            if even_max>min_amp and odd_max>min_amp:

#                A = max(np.max(even_data), np.max(odd_data))
    
#                data_time = np.arange(trace_length)*5.0E-9 + even_arrival_time -half_TL*5.0E-9
#                plt.plot(data_time, even_data/(2*A)+even_ant_i, 'g')
#                plt.plot(data_time, odd_data/(2*A)+even_ant_i, 'm')
                
#                plt.plot( [even_arrival_time,even_arrival_time], [even_ant_i-0.5,even_ant_i+0.5] )
                
                self.ant_locs.append( ant_locations[even_ant_i] )
                self.ant_names.append( ant_names[even_ant_i] )
                self.ant_start_time.append( even_arrival_time - half_TL*5.0E-9 )
                self.even_data_list.append( even_data/even_max )
                self.odd_data_list.append( odd_data/odd_max )
            
#        plt.show()
                
    def correlateEvE(self, data_file, data_filter, guess_time_delay, timing_error):
        
        ant_names = data_file.get_antenna_names()
        ant_locations = data_file.get_LOFAR_centered_positions()
        ant_timing_calibrations = data_file.get_timing_callibration_delays()
        
        known_station_delay = guess_time_delay-data_file.get_nominal_sample_number()*5.0E-9
        
        
        half_TL = int(timing_error/5.0E-9)
        station_trace_length = half_TL*2
        hann_window = half_hann_window( station_trace_length,  half_percent=0.1)
     
        
        cross_correlator = upsample_and_correlate( station_trace_length, upsample_factor )
        out = np.zeros( cross_correlator.out_length )
        
        
        even_tst_data = []
        offset = int( (station_trace_length - len(self.even_data_list[0]))/2 )
        for ED in self.even_data_list:
            tst_data = np.zeros( station_trace_length )
            
            tst_data[offset:offset+trace_length] = ED
            
            even_tst_data.append( tst_data )
        
        
        
        for even_ant_i in range(0,len(ant_names),2):
            even_arrival_time = np.linalg.norm( self.XYZT[:3]-ant_locations[even_ant_i])/v_air + self.XYZT[3]
            even_recieved_time = even_arrival_time + ant_timing_calibrations[even_ant_i] + known_station_delay
            receieved_index = int(even_recieved_time/5.0E-9)
            subsample_remainder = even_recieved_time - receieved_index*5.0E-9
            
            local_received_index = int(blocksize*0.5)
            even_data_start_index = receieved_index - local_received_index
            even_data = data_file.get_data( even_data_start_index, num_points=blocksize, antenna_index=even_ant_i )
            
            even_data = np.array(even_data, dtype=np.float)
            remove_saturation(even_data, positive_saturation_amplitude, negative_saturation_amplitude)
            
            even_FFT_data = data_filter.filter_FFT( even_data )
            
            FFT_time_shift( data_filter.get_frequencies(), even_FFT_data, -subsample_remainder )
            even_data = np.fft.ifft(even_FFT_data, axis=-1)
            
            even_data = np.real( even_data )[ local_received_index-half_TL : local_received_index+half_TL  ]
            even_data *= hann_window
            
            for TD in even_tst_data:
                out += np.real( cross_correlator.run( TD, even_data/np.max(even_data) ) )
            
        return out
    
    def correlateEvO(self, data_file, data_filter, guess_time_delay, timing_error):
        
        ant_names = data_file.get_antenna_names()
        ant_locations = data_file.get_LOFAR_centered_positions()
        ant_timing_calibrations = data_file.get_timing_callibration_delays()
        
        known_station_delay = guess_time_delay-data_file.get_nominal_sample_number()*5.0E-9
        
        
        half_TL = int(timing_error/5.0E-9)
        station_trace_length = half_TL*2
        hann_window = half_hann_window( station_trace_length,  half_percent=0.1)
     
        
        cross_correlator = upsample_and_correlate( station_trace_length, upsample_factor )
        out = np.zeros( cross_correlator.out_length )
        
        
        even_tst_data = []
        offset = int( (station_trace_length - len(self.even_data_list[0]))/2 )
        for ED in self.even_data_list:
            tst_data = np.zeros( station_trace_length )
            
            tst_data[offset:offset+trace_length] = ED
            
            even_tst_data.append( tst_data )
        
        
        
        for even_ant_i in range(0,len(ant_names),2):
            odd_arrival_time = np.linalg.norm( self.XYZT[:3]-ant_locations[even_ant_i+1])/v_air + self.XYZT[3]
            odd_recieved_time = odd_arrival_time + ant_timing_calibrations[even_ant_i+1] + known_station_delay
            receieved_index = int(odd_recieved_time/5.0E-9)
            subsample_remainder = odd_recieved_time - receieved_index*5.0E-9
            
            local_received_index = int(blocksize*0.5)
            odd_data_start_index = receieved_index - local_received_index
            odd_data = data_file.get_data( odd_data_start_index, num_points=blocksize, antenna_index=even_ant_i+1 )
            
            odd_data = np.array(odd_data, dtype=np.float)
            remove_saturation(odd_data, positive_saturation_amplitude, negative_saturation_amplitude)
            
            odd_FFT_data = data_filter.filter_FFT( odd_data )
            
            FFT_time_shift( data_filter.get_frequencies(), odd_FFT_data, -subsample_remainder + ant_timing_calibrations[even_ant_i]-ant_timing_calibrations[even_ant_i+1] )
            odd_data = np.fft.ifft(odd_FFT_data, axis=-1)
            
            odd_data = np.real( odd_data )[ local_received_index-half_TL : local_received_index+half_TL  ]
            odd_data *= hann_window
            
            for TD in even_tst_data:
                out += np.real( cross_correlator.run( TD, odd_data/np.max(odd_data) ) )
            
        return out
    
    def correlateSvS(self, data_file, data_filter, guess_time_delay, timing_error):
        
        ant_names = data_file.get_antenna_names()
        ant_locations = data_file.get_LOFAR_centered_positions()
        ant_timing_calibrations = data_file.get_timing_callibration_delays()
        
        known_station_delay = guess_time_delay-data_file.get_nominal_sample_number()*5.0E-9
        
        
        half_TL = int(timing_error/5.0E-9)
        station_trace_length = half_TL*2
        hann_window = half_hann_window( station_trace_length,  half_percent=0.1)
     
        
        cross_correlator = upsample_and_correlate( station_trace_length, upsample_factor )
        out = np.zeros( cross_correlator.out_length )
        
        
        even_tst_data = []
        offset = int( (station_trace_length - len(self.even_data_list[0]))/2 )
        for ED,OD in zip(self.even_data_list,self.odd_data_list):
            tst_data = np.zeros( station_trace_length )
            
            tst_data[offset:offset+trace_length] = ED
            tst_data[offset:offset+trace_length] += OD
            
            even_tst_data.append( tst_data )
        
        
        
        for even_ant_i in range(0,len(ant_names),2):
            arrival_time = np.linalg.norm( self.XYZT[:3]-ant_locations[even_ant_i])/v_air + self.XYZT[3]
            
            even_recieved_time = arrival_time + ant_timing_calibrations[even_ant_i] + known_station_delay
            receieved_index = int(even_recieved_time/5.0E-9)
            subsample_remainder = even_recieved_time - receieved_index*5.0E-9
            
            local_received_index = int(blocksize*0.5)
            even_data_start_index = receieved_index - local_received_index
            even_data = data_file.get_data( even_data_start_index, num_points=blocksize, antenna_index=even_ant_i )
            
            even_data = np.array(even_data, dtype=np.float)
            remove_saturation(even_data, positive_saturation_amplitude, negative_saturation_amplitude)
            
            even_FFT_data = data_filter.filter_FFT( even_data )
            
            FFT_time_shift( data_filter.get_frequencies(), even_FFT_data, -subsample_remainder )
            even_data = np.fft.ifft(even_FFT_data, axis=-1)
            
            even_data = np.real( even_data )[ local_received_index-half_TL : local_received_index+half_TL  ]
            even_data *= hann_window
            
            
            
            odd_recieved_time = arrival_time + ant_timing_calibrations[even_ant_i+1] + known_station_delay
            receieved_index = int(odd_recieved_time/5.0E-9)
            subsample_remainder = odd_recieved_time - receieved_index*5.0E-9
            
            local_received_index = int(blocksize*0.5)
            odd_data_start_index = receieved_index - local_received_index
            odd_data = data_file.get_data( odd_data_start_index, num_points=blocksize, antenna_index=even_ant_i+1 )
            
            odd_data = np.array(odd_data, dtype=np.float)
            remove_saturation(odd_data, positive_saturation_amplitude, negative_saturation_amplitude)
            
            odd_FFT_data = data_filter.filter_FFT( odd_data )
            
            FFT_time_shift( data_filter.get_frequencies(), odd_FFT_data, -subsample_remainder + ant_timing_calibrations[even_ant_i]-ant_timing_calibrations[even_ant_i+1] )
            odd_data = np.fft.ifft(odd_FFT_data, axis=-1)
            
            odd_data = np.real( odd_data )[ local_received_index-half_TL : local_received_index+half_TL  ]
            odd_data *= hann_window
            
            
            even_max = np.max(even_data)
            odd_max = np.max(odd_data)
            
            if even_max>min_amp and odd_max>min_amp:
                for TD in even_tst_data:
                    out += np.real( cross_correlator.run( TD, odd_data/odd_max+even_data/even_max ) )
            
        return out
            
        
    
    
if __name__=="__main__":
    timeID = "D20170929T202255.000Z"
    output_folder = "autoCorrelator_interferometry"
    
    ###TODO: include known antenna corrections
    
    sources = [ 
            [ -17458.7832805 , 9147.33910803 , 2561.96347286 , 1.17337126996 ],
            [ -15810.2625335 , 8242.84768561 , 1634.49102349 , 1.20861145958 ],
            [ -15974.1108256 , 10934.8998241 , 4882.71756089 , 1.22938898396 ],
            [ -16361.1824681 , 9944.65047674 , 3078.29020013 , 1.16010393052 ],
            [ -15904.9794194 , 9869.99678101 , 3496.57295038 , 1.16635726386 ],
            [ -16230.4501499 , 10307.5643518 , 3818.81997483 , 1.1797433305 ],
            [ -16068.3951488 , 9189.89050515 , 3592.9892939 , 1.15317625519 ],
            [ -16062.8217345 , 9181.0520059 , 3623.02612132 , 1.15321317577 ],
    ]
     
    trace_length = 50 ##num data points on reference station
    
    reference_station = "CS002"
    station_to_correlate= "RS208"
    
    timing_error = 1.0E-4
    
    upsample_factor = 16
    
    guess_timing_delays = {
'CS003' :  1.40560400665e-06 , ## diff to guess: 4.5596689259e-07
'CS004' :  4.32241812582e-07 , ## diff to guess: 7.29762519972e-07
'CS005' :  -2.19141774625e-07 , ## diff to guess: -1.49126567272e-07
'CS006' :  4.32824190102e-07 , ## diff to guess: 4.48181793014e-08
'CS007' :  3.99184781356e-07 , ## diff to guess: 7.35832780287e-08
'CS011' :  -5.87066323359e-07 , ## diff to guess: -1.17500496441e-06
'CS013' :  -1.81294775392e-06 , ## diff to guess: 1.25812313897e-06
'CS017' :  -8.44316849912e-06 , ## diff to guess: -3.01243822616e-06
'CS021' :  9.27494082077e-07 , ## diff to guess: 2.58621974831e-06
'CS030' :  -2.73919980817e-06 , ## diff to guess: 3.18378210222e-06
'CS032' :  -1.56687585636e-06 , ## diff to guess: 4.21394323749e-06
'CS101' :  -8.17787145972e-06 , ## diff to guess: -4.70975066499e-06
'CS103' :  -2.85317351407e-05 , ## diff to guess: -1.12140033722e-05
'RS208' :  7.11009676471e-06 , ## diff to guess: -1.03482774903e-05
'CS301' :  -7.11667989706e-07 , ## diff to guess: 6.23299212564e-07
'CS302' :  -5.33236917528e-06 , ## diff to guess: 7.55885340113e-06
'RS306' :  7.20907366205e-06 , ## diff to guess: 4.35870853386e-05
'RS307' :  7.33044668701e-06 , ## diff to guess: 4.70103471884e-05
'RS310' :  8.12572903877e-06 , ## diff to guess: 9.46208518041e-05
'CS401' :  -9.42131948959e-07 , ## diff to guess: 4.83673218568e-06
'RS406' :  -4.37233503223e-05 , ## diff to guess: 0.0
'RS409' :  8.43305056992e-06 , ## diff to guess: 0.000106761401308
'CS501' :  -9.61121569986e-06 , ## diff to guess: 1.29069318714e-06
'RS503' :  6.94990149833e-06 , ## diff to guess: 7.70036115781e-06
'RS508' :  7.45483955743e-06 , ## diff to guess: -2.1748842352e-05
'RS509' :  7.81716469529e-06 , ## diff to guess: 1.01272143067e-05
            }
    
    
#    guess_timing_delays = {
#        "CS002":0.0,
#        "CS003":1.0E-6,
#        "CS004":0.0,
#        "CS005":0.0,
#        "CS006":0.0,
#        "CS007":0.0,
#        "CS011":0.0,
#        "CS013":-0.000003,
#        "CS017":-7.0E-6,
#        "CS021":-8E-7,
#        "CS026":-7E-6,
#        "CS030":-5.5E-6,
#        "CS032":-5.5E-6 + 2E-6 + 1E-7,
#        "CS101":-7E-6,
#        "CS103":-22.5E-6-20E-8,
#        "RS106":35E-6 +30E-8 +12E-7,
#        "CS201":-7.5E-6,
#        "RS205":25E-6,
#        "RS208":8E-5+3E-6,
#        "CS301":6.5E-7,
#        "CS302":-3.0E-6-35E-7,
#        "RS305":-6E-6,
#        "RS306":-7E-6,
#        "RS307":175E-7+8E-7,
#        "RS310":8E-5+6E-6,
#        "CS401":-3E-6,
#        "RS406":-25E-6,
#        "RS407":5E-6 -8E-6 -15E-7,
#        "RS409":8E-6,
#        "CS501":-12E-6,
#        "RS503":-30.0E-8-10E-7,
#        "RS508":6E-5+5E-7,
#        "RS509":10E-5+15E-7,
#        }
    location_correct = False
    
    positive_saturation_amplitude = 2046
    negative_saturation_amplitude = -2047
    
    min_amp = 5
    
    
    raw_fpaths = filePaths_by_stationName(timeID)
    
    reference_data_file = MultiFile_Dal1(raw_fpaths[reference_station], force_metadata_ant_pos=True)
    station_data_file = MultiFile_Dal1(raw_fpaths[station_to_correlate], force_metadata_ant_pos=True)
    
    referance_filter = window_and_filter(timeID=timeID,sname=reference_station)
    station_filter = window_and_filter(timeID=timeID,sname=station_to_correlate)
    
    blocksize = referance_filter.blocksize
    
    
    guess_delay = guess_timing_delays[station_to_correlate]
    if location_correct:
        ave_source_location = np.zeros(3)
        for s_loc in sources:
            ave_source_location += np.array(s_loc)[:3]
            
        ave_source_location /= len(sources)
        
        ref_loc = reference_data_file.get_LOFAR_centered_positions()
        ref_loc = np.average(ref_loc, axis=0)
        
        stat_loc = station_data_file.get_LOFAR_centered_positions()
        stat_loc = np.average(stat_loc, axis=0)
        
        guess_delay -= np.linalg.norm( ave_source_location-stat_loc )/v_air - np.linalg.norm( ave_source_location-ref_loc )/v_air
    
    correlation = None
    i=0
    print(station_to_correlate)
    for source_loc in sources:
        source = source_object( np.array(source_loc) )
        ret = source.correlateSvS( station_data_file, station_filter, guess_delay, timing_error )
        
        if correlation is None:
            correlation = ret
        else:
            correlation += ret
    
        ret /= 2*np.max(ret)
        
        plt.plot(ret+i)
        
        i+=1
        
    plt.show()
    
    plt.plot(correlation)
    plt.show()
    
    
    
    
    
    
    
    
    
    