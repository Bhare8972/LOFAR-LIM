#!/usr/bin/env python3

""" Often, LOFAR does not add the appropriate meta-data to the TBB data files. This is particularly problamatic for the phase
callibration. This module has a series of utilities designed to download the correct phase callibration data.

The phase callibration data is stored in an SVN repository. 

This module has a funtion "download_phase_callibrations", which does exactly what it says. The main script at the bottom serves as an example how to open 
data and download the phase callibrations. When run, the script will download the calibration data for all stations that need it, for a given event."""

import os
from datetime import datetime

def get_station_history(station_name, history_folder):
    """given a station name (as string) and a folder to store it in, save the history of the SVN repository"""
    cmd = "svn log https://svn.astron.nl/Station/trunk/CalTables/"+station_name+" > "+history_folder+"/"+station_name+"_hist.txt"
    os.system(cmd)
    
def get_latest_previous_revision(station_name, history_folder, timestamp):
    """for a history_folder that has a svn log for station name, find the last revision before timestamp, which should be 
    a python datetime object"""
    fname = history_folder+'/'+station_name+"_hist.txt"
    revision = None
    date = None
    with open(fname, 'r') as fin:
        for line in fin:
            if line[0]=='r':
                #### we found a line that specifies a revision!, now get the revision and date ####
                words = line.split()
                line_revision = words[0]
                line_date =  datetime.strptime( words[4], "%Y-%m-%d" )
                #### decide if this is one we could want ###
                if line_date < timestamp:
                    if (date is None) or line_date>date:
                        date = line_date
                        revision = line_revision
    return revision
                    
def get_phase_callibration(station, revision, folder):
    """download the phase callibration for a particular revision and station to folder. Folder should be default raw data folder"""
    cmd = "svn checkout -r "+revision+" https://svn.astron.nl/Station/trunk/CalTables/"+station+" "+folder+'/'+station
    os.system(cmd)

def download_phase_callibrations(station, history_folder, timestamp, folder):
    """This function is a wrapper that first downloads the history for this station, finds the revision closest to the timestamp
    then downloads that revision to a folder"""
    get_station_history( station, history_folder )
    revision = get_latest_previous_revision( station, history_folder, timestamp )
    get_phase_callibration( station, revision, folder )
    
if __name__ == "__main__":
    #### a full working example of opening a file, checking if it needs metadata, and downloading if necisary
    import LoLIM.utilities as utils
    from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
    
    from os import mkdir
    from os.path import isdir
    
    timeID = "D20180813T153001.413Z"
    history_folder = "./svn_phase_cal_history"
    
    skip = []#['RS407'] ##stations to skip
    
    
    if not isdir(history_folder):
        mkdir(history_folder)
        
    fpaths = filePaths_by_stationName(timeID)
    stations = fpaths.keys()
#    stations = ["RS407"]
    
    for station in stations:
        if station in skip:
            continue
        
        TBB_data = MultiFile_Dal1( fpaths[station] )
        timestamp = datetime.fromtimestamp( TBB_data.get_timestamp() )
        
        if TBB_data.needs_metadata():
            print("downloading for station:", station)
            download_phase_callibrations(station, history_folder, timestamp, utils.raw_data_dir(timeID) )
            
        print( station)
        print( TBB_data.get_timing_callibration_delays() )
    
    
    
    
    
    
    
    
    
    
    
    