#!/usr/bin/env python3

from os.path import isdir

import numpy as np
import h5py
from scipy.signal import resample

from LoLIM.getTrace_fromLoc import getTrace_fromLoc 
from LoLIM.signal_processing import parabolic_fit


def save_EventByLoc(event_index, event_XYZT, output_folder, trace_locator, pulse_width=50, upsample_factor=2, min_ant_amp=2):

    

    if not isdir(output_folder):
        print('no folder:', output_folder)
        print('  quiting')
        quit()



    TBBfile_dict = trace_locator.get_TBBfile_dict()
    trace_locator.set_return_dbl_zeros( False )


    out_fname = output_folder + "/potSource_"+str(event_index)+".h5"
    out_file = h5py.File(out_fname, "w")

    print('saving event', event_index)
    for sname, TBB_file in TBBfile_dict.items():

        antenna_names = TBB_file.get_antenna_names()
        station_timing_delay = TBB_file.get_station_delay()
        
        
        h5_statGroup = out_file.create_group( sname )
        print()
        print(sname)
        
        for even_ant_i in range(0,len(antenna_names),2):

            even_ant_name = antenna_names[ even_ant_i ]
            odd_ant_name = antenna_names[ even_ant_i+1 ]
            
            starting_index, PolE_offset, throw, PolE_trace = trace_locator.get_trace_fromLoc(event_XYZT, even_ant_name, pulse_width, do_remove_RFI=True, do_remove_saturation=True)
            throw, PolO_offset, throw, PolO_trace = trace_locator.get_trace_fromIndex(starting_index, odd_ant_name, pulse_width, do_remove_RFI=True, do_remove_saturation=True)
            
            PolE_HE = np.abs(PolE_trace)
            PolO_HE = np.abs(PolO_trace)
            
            
            h5_Ant_dataset = h5_statGroup.create_dataset(even_ant_name, (4, pulse_width ), dtype=np.double)
            
            h5_Ant_dataset[0] = np.real(PolE_trace)
            h5_Ant_dataset[1] = PolE_HE
            h5_Ant_dataset[2] = np.real(PolO_trace)
            h5_Ant_dataset[3] = PolO_HE
            
            ### note that peak time should NOT account for station timing offsets!
            
            h5_Ant_dataset.attrs['starting_index'] = starting_index
            h5_Ant_dataset.attrs['PolE_timeOffset'] = -(PolE_offset - station_timing_delay)
            h5_Ant_dataset.attrs['PolO_timeOffset'] = -(PolO_offset - station_timing_delay)
            h5_Ant_dataset.attrs['PolE_timeOffset_CS'] = (PolE_offset - station_timing_delay)
            h5_Ant_dataset.attrs['PolO_timeOffset_CS'] = (PolO_offset - station_timing_delay)
            
            
            
            
            sample_time = 5.0E-9
            if upsample_factor > 1:
                PolE_HE = resample(PolE_HE, len(PolE_HE)*upsample_factor )
                PolO_HE = resample(PolO_HE, len(PolO_HE)*upsample_factor )
                
                sample_time /= upsample_factor
            
            
            if np.max(PolE_HE)> min_ant_amp:
                
                
                PolE_peak_finder = parabolic_fit( PolE_HE )
                h5_Ant_dataset.attrs['PolE_peakTime'] =  starting_index*5.0E-9 - (PolE_offset-station_timing_delay) + PolE_peak_finder.peak_index*sample_time
            else:
                h5_Ant_dataset.attrs['PolE_peakTime'] = np.nan
            
            if np.max(PolO_HE)> min_ant_amp:
                
                
                PolO_peak_finder = parabolic_fit( PolO_HE )
                h5_Ant_dataset.attrs['PolO_peakTime'] =  starting_index*5.0E-9 - (PolO_offset-station_timing_delay) + PolO_peak_finder.peak_index*sample_time
            else:
                h5_Ant_dataset.attrs['PolO_peakTime'] =  np.nan
            
            print("  ", even_ant_name, h5_Ant_dataset.attrs['PolE_peakTime'], h5_Ant_dataset.attrs['PolO_peakTime'])
                
            
    out_file.close()