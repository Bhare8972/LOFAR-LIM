#!/usr/bin/env python3

""" this script replaces both "histogram_planewave_fits", and "get_antenna_delays".  It looks through the SVN repository for all calibrations that were uploaded within some dt of 
the lightnign flash (typically choosen to be 1 year). Then it systematically downloads each callibration and fits planewaves. The best callibration is remembered and kept, 
histogram plots of the planewaves are made, and logs of the RMS values per antenna are also made. This script is very slow."""

from os import mkdir
from os.path import isdir

from pickle import load
from datetime import timedelta, datetime

from matplotlib import pyplot as plt
import numpy as np

from LoLIM.make_planewave_fits import planewave_fitter
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.utilities import v_air, processed_data_dir, logger, raw_data_dir
from LoLIM.get_phase_callibration import get_station_history, get_ordered_revisions, get_phase_callibration


## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"


if __name__ == "__main__":
    
    timeID = "D20190424T194432.504Z"
    output_folder = "/find_calibration_out"
    
    actually_find_callibration = True ## if true, loops over every astron callibration in time, and finds best
    ## if false, only uses present calibration information, useful to get planewave fit info
    
    history_folder = "./svn_phase_cal_history"
    get_all_timings = True ## if false, only get stations that need it
    mode = 'LBA_OUTER' ## set to None to get ALL files. ##TODO: check mode in file! may need 30-90 filter (Does this setting even work?)
    
    min_num_planewaves = 100
    RMS_cut = 1.0E-8
    
    max_dt = timedelta(days = 2*365)
    
    initial_block = 3000
    number_of_blocks = 2000
    pulses_per_block = 2
    max_num_planewaves = 200 ## per polarization, so multiply by two
    
    stations = filePaths_by_stationName(timeID)
    
    processed_data_folder = processed_data_dir(timeID)
    output_fpath = processed_data_folder + output_folder
    if not isdir(output_fpath):
        mkdir(output_fpath)
        
    if not isdir(history_folder):
        mkdir(history_folder)
        
    antenna_output_folder = output_fpath + '/antenna_fits'
    if not isdir(antenna_output_folder):
        mkdir(antenna_output_folder)
        
        
        
    with open( processed_data_folder + "/findRFI/findRFI_results", 'rb' ) as fin:
        find_RFI = load(fin)
    
    
    
    fpaths = filePaths_by_stationName(timeID)
    polarization_flips = processed_data_folder + '/' + "polarization_flips.txt"
    bad_antennas = processed_data_folder + '/' + "bad_antennas.txt"
   # additional_antenna_delays = processed_data_folder + '/' + "ant_delays.txt"
    additional_antenna_delays = None
    
    
    station_log = logger()
    station_log.take_stdout()
    for sname in stations.keys():
        
        #### open the station
        
        station_log.set(output_fpath+'/'+sname+'_log.txt')
        print("processing", sname)
        
        TBB_data = MultiFile_Dal1( fpaths[sname], polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays )
    
        
        get_station_history( sname, history_folder )
        revisions = get_ordered_revisions(sname, history_folder, datetime.fromtimestamp( TBB_data.get_timestamp() ), max_dt)
        print(len(revisions), 'calibration revisions')
        
        if sname in find_RFI:
            do_find_RFI = timeID
        else:
            do_find_RFI = None
            print("WARNING!: this station doesn't have RFI data. Just doing 30-90 filter.")
        
        #### find the planewaves
        
        fitter_even = planewave_fitter( TBB_data = TBB_data,
                   timeID = do_find_RFI, 
                   polarization  = 0,
                   initial_block = initial_block,
                   number_of_blocks = number_of_blocks, 
                   pulses_per_block = pulses_per_block, 
                   pulse_length = 50 + int(100/(v_air*5.0E-9)), 
                   min_amplitude = 50, 
                   upsample_factor = 4, 
                   min_num_antennas = 4,
                   max_num_planewaves = max_num_planewaves,
                   verbose = False, ## doesn't do anything anyway
                   positive_saturation = 2046, negative_saturation = -2047, saturation_post_removal_length = 50, saturation_half_hann_length = 50)

        fitter_odd = planewave_fitter( TBB_data = TBB_data,
                   timeID = do_find_RFI, 
                   polarization  = 1,
                   initial_block = initial_block,
                   number_of_blocks = number_of_blocks, 
                   pulses_per_block = pulses_per_block, 
                   pulse_length = 50 + int(100/(v_air*5.0E-9)), 
                   min_amplitude = 50, 
                   upsample_factor = 4, 
                   min_num_antennas = 4,
                   max_num_planewaves = max_num_planewaves,
                   verbose = False, ## doesn't do anything anyway
                   positive_saturation = 2046, negative_saturation = -2047, saturation_post_removal_length = 50, saturation_half_hann_length = 50)

        print( fitter_even.num_found_planewaves, "planewaves found on even antennas" )
        print( fitter_odd.num_found_planewaves, "planewaves found on odd antennas" )
        
        if fitter_even.num_found_planewaves+fitter_odd.num_found_planewaves  <  min_num_planewaves:
            print("too few planewaves. Skipping station")
            continue
        
        print()
        print()
        print()
        
        ### now loop over available calibrations
            
        prev = None
        min_delta = 1.0E-10 ## if difference in calibration is smaller, then skip
        best_revision = None
        best_RMS = np.inf
        best_combined_RMSs = None
        even_antenna_RMSs = None
        odd_antenna_RMSs = None
        even_antenna_SecDer = None
        odd_antenna_SecDer = None
        for rev_i, revision in enumerate(revisions):
            ## load new calibrations
            if actually_find_callibration:
                print(sname, revision, '(', rev_i+1, '/', len(revisions), ')')
                get_phase_callibration( sname, revision,  raw_data_dir(timeID), mode, force=True )
            
            new = TBB_data.get_timing_callibration_delays()
            if prev is not None: ## check is really necisary to find planewaves
                new -= prev
                np.abs(new,out=new)
                if not np.any( new>min_delta ):
                    print('skipping!')
                    print()
                    print()
                    continue
            
            prev = new
            
            ## fit planewaves with new callibrations
            even_RMSs, throw, throw, even_ant_fits = fitter_even.go_fit( max_RMS = RMS_cut)
            odd_RMSs, throw, throw, odd_ant_fits = fitter_odd.go_fit( max_RMS = RMS_cut)
            
            combined_RMSs = np.append( even_RMSs[even_RMSs < RMS_cut], odd_RMSs[odd_RMSs < RMS_cut] )
            aveRMS = np.average( combined_RMSs )
            
            print('ave RMS:', aveRMS)
            print()
            print()
            print()
            
            if aveRMS < best_RMS:
                best_revision = revision
                best_RMS = aveRMS
                even_antenna_RMSs = even_ant_fits
                odd_antenna_RMSs = odd_ant_fits
                best_combined_RMSs = combined_RMSs
#                even_antenna_SecDer = fitter_even.get_second_derivatives( even_RMSs, max_RMS = RMS_cut )
#                odd_antenna_SecDer = fitter_odd.get_second_derivatives( odd_RMSs, max_RMS = RMS_cut )
                
                
            ## break loop if not actually searching for callibration
            if not actually_find_callibration:
                break
                
                
        if actually_find_callibration:
            print('using revision:', best_revision, 'with RMS:', best_RMS)
            get_phase_callibration( sname, best_revision,  raw_data_dir(timeID), mode, force=True )
        
        print('even antenna fits:')
        print( even_antenna_RMSs )
        print('odd antenna fits:')
        print( odd_antenna_RMSs )
        print()
        print()
        
        plt.hist(best_combined_RMSs, bins=50, range=[0,RMS_cut])
        plt.savefig(output_fpath+'/'+sname+'.png')
        ### TODO: fit a distribution and print the max and width
#        plt.show()
        plt.axvline(x=1.5E-9)
        plt.close() 
        
        with open(antenna_output_folder + '/' + sname + ".txt", 'w+') as fout:
            even_ant_names = TBB_data.get_antenna_names()[::2]
            odd_ant_names = TBB_data.get_antenna_names()[1::2]
            
            for ant_name, RMS in zip(even_ant_names, even_antenna_RMSs):
                fout.write(ant_name)
                fout.write(' ')
                fout.write(str(RMS))
                fout.write('\n')
            
            
            for ant_name, RMS in zip(odd_ant_names, odd_antenna_RMSs):
                fout.write(ant_name)
                fout.write(' ')
                fout.write(str(RMS))
                fout.write('\n')
        
#        with open(antenna_output_folder + '/' + sname + "_secDer.txt", 'w+') as fout:
#            even_ant_names = TBB_data.get_antenna_names()[::2]
#            odd_ant_names = TBB_data.get_antenna_names()[1::2]
#            
#            for ant_name, SecDer in zip(even_ant_names, even_antenna_SecDer):
#                fout.write(ant_name)
#                fout.write(' ')
#                fout.write(str(SecDer))
#                fout.write('\n')
#            
#            
#            for ant_name, SecDer in zip(odd_ant_names, odd_antenna_SecDer):
#                fout.write(ant_name)
#                fout.write(' ')
#                fout.write(str(SecDer))
#                fout.write('\n')
            
            
    