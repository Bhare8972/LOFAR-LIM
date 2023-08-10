#!/usr/bin/env python3


from os import mkdir
from os.path import isdir

# from pickle import load

from matplotlib import pyplot as plt
import numpy as np

from LoLIM.make_planewave_fits import planewave_fitter, MultiFile_Dal1
from LoLIM.IO.raw_tbb_IO import  filePaths_by_stationName
from LoLIM.utilities import v_air, processed_data_dir

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"



if __name__ == "__main__":
    
    timeID = 'D20200814T124803.228Z'
    first_block = 3500
    num_blocks  = 100
    
    cal_file = 'calibration_v2.txt'
    
    out_folder = 'olaf_planewave_tests_2'
    
    
    timeID_dir = processed_data_dir(timeID)
    
    cal_file = timeID_dir + '/' + cal_file
    out_folder = timeID_dir + '/' + out_folder
    
    if not isdir(out_folder):
        mkdir(out_folder)
    
    raw_fpaths = filePaths_by_stationName(timeID)
    
    for sname, fpaths in raw_fpaths.items():
        
        print("processing station", sname)
        
        TBB_data = MultiFile_Dal1( fpaths, force_metadata_ant_pos = True, total_cal = cal_file)
        ant_names = TBB_data.get_antenna_names()
        
        if len(ant_names)==0:
            continue
        
        fitter_even = planewave_fitter(TBB_data, polarization=0, initial_block=first_block, number_of_blocks=num_blocks, 
                                  pulses_per_block=2, pulse_length=50 + int(100/(v_air*5.0E-9)), min_amplitude=50, 
                                  upsample_factor=4, min_num_antennas=4,
                        max_num_planewaves=1000, timeID=timeID, blocksize = 2**16,
                        positive_saturation = 2046, negative_saturation = -2047,
                        saturation_post_removal_length = 50, saturation_half_hann_length = 50, verbose=True)
        
        if fitter_even.num_found_planewaves == 0:
            print(' no even planewaves')
            continue
        
        even_RMSs, even_Zeniths, even_Azimuths, even_ant_fits = fitter_even.go_fit()
        
        
        
        fitter_odd = planewave_fitter(TBB_data, polarization=1, initial_block=first_block, number_of_blocks=num_blocks, 
                          pulses_per_block=2, pulse_length=50 + int(100/(v_air*5.0E-9)), min_amplitude=50, 
                          upsample_factor=4, min_num_antennas=4,
                max_num_planewaves=1000, timeID=timeID, blocksize = 2**16,
                positive_saturation = 2046, negative_saturation = -2047,
                saturation_post_removal_length = 50, saturation_half_hann_length = 50, verbose=True)
        
        if fitter_odd.num_found_planewaves == 0:
            print(' no odd planewaves')
            continue
        
        odd_RMSs, odd_Zeniths, odd_Azimuths, odd_ant_fits = fitter_odd.go_fit()
        
        even_ant_name = ant_names[::2]
        odd_ant_names = ant_names[1::2]
        


        
    
        plt.hist(even_RMSs, bins=50, range=[0,10e-9], color='b')
        # plt.hist(even_RMSs, bins=50, color='b')
        plt.axvline(x=1.5e-9, c='r')
        plt.savefig(out_folder+'/'+sname+'_even.png')
        plt.xlim(0.0, 10e-9)
        ### TODO: fit a distribution and print the max and width
#        plt.show()
        plt.close()
        
        plt.hist(odd_RMSs, bins=50, range=[0,10e-9], color='b')
        # plt.hist(odd_RMSs, bins=50, color='b')
        plt.axvline(x=1.5e-9, c='r')
        plt.savefig(out_folder+'/'+sname+'_odd.png')
        plt.xlim(0.0, 10e-9)
#        plt.show()
        plt.close()
        
        ## TODO: record number of fits per antenna?
        
        ## to file 
        out_file = open( out_folder + '/' + sname+ "_LOG.txt", 'w' )
        print("station", sname, file = out_file)
        print(' ', len(even_RMSs), 'even fits.', len(odd_RMSs), 'odd fits', file = out_file)
        print(' even', file = out_file)
        for ant_name, RMS in zip(even_ant_name,even_ant_fits):
            print(' ', ant_name, RMS, file = out_file)
        print(' odd', file = out_file)
        for ant_name, RMS in zip(odd_ant_names,odd_ant_fits):
            print(' ', ant_name, RMS, file = out_file)
            
            
            ## to screen 
        print(' ', len(even_RMSs), 'even fits.', len(odd_RMSs), 'odd fits')
        print(' even')
        for ant_name, RMS in zip(even_ant_name,even_ant_fits):
            print(' ', ant_name, RMS)
        print(' odd')
        for ant_name, RMS in zip(odd_ant_names,odd_ant_fits):
            print(' ', ant_name, RMS)
        print()
    
#    timeID = "D20180809T145549.143Z"
#     timeID = "D20170929T202255.000Z"
#     output_folder = "/planewave_RMS_histograms"
    

#     stations = filePaths_by_stationName(timeID)
    
#     processed_data_dir = processed_data_dir(timeID)
#     output_fpath = processed_data_dir + output_folder
#     if not isdir(output_fpath):
#         mkdir(output_fpath)
        
#     with open( processed_data_dir + "/findRFI/findRFI_results", 'rb' ) as fin:
#         find_RFI = load(fin)
    
#     for sname in stations.keys():
# #        if (sname in ['CS002', 'CS003', 'CS004', 'CS005', 'CS006', 'CS007',
# #                      'CS011', 'CS013', 'CS017', 'CS021', 'CS026']) or (sname not in find_RFI):
# #            continue
    
#         sname= 'CS002'
        
#         print("processing", sname)
        
#         RMSs, zeniths, azimuths = planewave_fits(timeID = timeID, 
#                        station = sname, 
#                        polarization  = 0, 
#                        initial_block = 3500,
#                        number_of_blocks = 500, 
#                        pulses_per_block = 2, 
#                        pulse_length = 50 + int(100/(v_air*5.0E-9)), 
#                        min_amplitude = 50, 
#                        upsample_factor = 4, 
#                        min_num_antennas = 4,
#                        max_num_planewaves = 1000,
#                        verbose = False, ## doesn't do anything anyway
#                        polarization_flips="polarization_flips.txt", bad_antennas="bad_antennas.txt", additional_antenna_delays = "ant_delays.txt",  
#                        positive_saturation = 2046, negative_saturation = -2047, saturation_post_removal_length = 50, saturation_half_hann_length = 50)
        
#         print(np.average(zeniths),np.average(azimuths) )
        
#         print(len(RMSs), "found planewaves")
#         plt.hist(RMSs, bins=50, range=[0,10e-9])
#         plt.savefig(output_fpath+'/'+sname+'.png')
#         ### TODO: fit a distribution and print the max and width
# #        plt.show()
#         plt.close()
        
#         break
    
    