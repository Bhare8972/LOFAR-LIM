#!/usr/bin/env python3

"""This script is for when "find_best_calibrations fails. Since that script relies upon planewave fitting, this can occur when planewaves cannot be found for a station
for whatever reason. In this case, callibration files are still needed. There are two reasonable options. First is to use the callibrations in the raw data file, second
is to use the callibrations closest in time to the measurment. This file is used under the assumption the first option is better since ASTRON has a better idea which 
tables are valid and which are not. The function of this script is to pull the timing callibration from the raw data file and compare it (via RMS) to the callibration 
from a tablea, and pick the table with lowest RMS."""

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
    output_log_folder = "/mostSimilar_antenna_callibrations"
    
    history_folder = "./svn_phase_cal_history"
    mode = 'LBA_OUTER' ## set to None to get ALL files. ##TODO: check mode in file! may need 30-90 filter (Does this setting even work?)
   
    station = 'CS401'
    
    max_dt = timedelta(days = 2*365) # timespan over which to search callibrations


    
    processed_data_folder = processed_data_dir(timeID)
    output_fpath = processed_data_folder + output_log_folder
    if not isdir(output_fpath):
        mkdir(output_fpath)
        
    if not isdir(history_folder):
        mkdir(history_folder)
        
    
    fpaths = filePaths_by_stationName(timeID)
    polarization_flips = processed_data_folder + '/' + "polarization_flips.txt"
    bad_antennas = processed_data_folder + '/' + "bad_antennas.txt"
   # additional_antenna_delays = processed_data_folder + '/' + "ant_delays.txt"
    additional_antenna_delays = None
    
    
    station_log = logger()
    station_log.take_stdout()
    
    #### open the station
    
    print("processing", station)
    station_log.set(output_fpath+'/'+station+'_log.txt')
    
    TBB_data = MultiFile_Dal1( fpaths[station], polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays )
    
    if TBB_data.needs_metadata():
        print(station, "does not have delay metadata. This will not work.")
    
    ant_names = TBB_data.get_antenna_names()
    file_delays = TBB_data.get_timing_callibration_delays(force_file_delays=True)
    print()
    print("file delays")
    for name, delay in zip(ant_names, file_delays):
        print(name,  delay)
    
    get_station_history( station, history_folder )
    revisions = get_ordered_revisions(station, history_folder, datetime.fromtimestamp( TBB_data.get_timestamp() ), max_dt)
    print()
    print(len(revisions), 'calibration revisions')
    
    if len(revisions)==0:
        quit()
        
        
    best_revision = None
    best_RMS = np.inf
    for rev_i, revision in enumerate(revisions):
        ## load new calibrations
        print()
        print( revision, '(', rev_i+1, '/', len(revisions), ')')
        get_phase_callibration( station, revision,  raw_data_dir(timeID), mode, force=True )
        
        new_delays = TBB_data.get_timing_callibration_delays()
        diff = new_delays - file_delays
        diff *= diff
        RMS = np.average( diff )
        print("  RMS:", RMS)
        
        if RMS < best_RMS:
            best_revision = revision
            best_RMS = RMS
            
            
    print()
    print("setting best revision:", best_revision)
    get_phase_callibration( station, best_revision,  raw_data_dir(timeID), mode, force=True )
    delays = TBB_data.get_timing_callibration_delays()
    for name, delay in zip(ant_names, delays):
        print(name, delay)
    
        
        
            
    