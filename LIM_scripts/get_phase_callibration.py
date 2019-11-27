#!/usr/bin/env python3

""" Often, LOFAR does not add the appropriate meta-data to the TBB data files. This is particularly problamatic for the phase
callibration. This module has a series of utilities designed to download the correct phase callibration data.

The phase callibration data is stored in an SVN repository. 

This module has a funtion "download_phase_callibrations", which does exactly what it says. The main script at the bottom serves as an example how to open 
data and download the phase callibrations. When run, the script will download the calibration data for all stations that need it, for a given event."""

import os
from datetime import datetime
from os import mkdir

from os.path import isdir

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

def get_ordered_revisions(station_name, history_folder, timestamp, max_timediff):
    """ like get_latest_previous_revision. Except it returns a list of revisions that are all within max_timediff (a timedelta object) of timestamp. List is ordered by time relative to timestamp"""
    
    info = []
    
    fname = history_folder+'/'+station_name+"_hist.txt"
    with open(fname, 'r') as fin:
        for line in fin:
            if line[0]=='r':
                #### we found a line that specifies a revision!, now get the revision and date ####
                words = line.split()
                line_revision = words[0]
                line_date =  datetime.strptime( words[4], "%Y-%m-%d" )
                
                dt = abs( line_date-timestamp )
                if dt < max_timediff:
                    info.append( [line_revision, dt] )
                    
    info.sort(key=lambda X: X[1])
    return [X[0] for X in info]
                    
def get_phase_callibration(station, revision, folder, mode=None, force=False):
    """download the phase callibration for a particular revision and station to folder. Folder should be default raw data folder"""
    
    if force:
        force_CMD = ' --force '
    else:
        force_CMD = ' '
        
    if not isdir(folder+'/'+station):
        mkdir(folder+'/'+station)

    if mode is not None:
        mode_name = {"LBA_OUTER": "LBA_OUTER-10_90",
                     "LBA_INNER": "LBA_INNER-10_90",
                           "HBA": "HBA-110_190"}[ mode ] ## this seems like a wierd way to do this.....
        stationNr = station[2:] ## oddly not the stationID
        
        
        cmd = "svn export"+force_CMD+"-r "+revision+" https://svn.astron.nl/Station/trunk/CalTables/"+station+ '/CalTable-' + stationNr + '-' + mode_name + '.dat'+" "+folder+'/'+station + '/CalTable-' + stationNr + '-' + mode_name + '.dat'
    else:
        cmd = "svn export"+force_CMD+"-r "+revision+" https://svn.astron.nl/Station/trunk/CalTables/"+station+" "+folder+'/'+station
    os.system(cmd)

def download_phase_callibrations(station, history_folder, timestamp, folder, mode=None):
    """This function is a wrapper that first downloads the history for this station, finds the revision closest to the timestamp
    then downloads that revision to a folder"""
    get_station_history( station, history_folder )
    revision = get_latest_previous_revision( station, history_folder, timestamp )
    get_phase_callibration( station, revision, folder, mode )
    

    
#### two random, but useful, utility functions
def write_timing_noise(folder, sname, antenna_names, antenna_noise):
    with open(folder + '/' + sname + ".txt", 'w+') as fout:
        for ant_name, noise in zip(antenna_names, antenna_noise):
            fout.write(ant_name)
            fout.write(' ')
            fout.write(str(noise))
            fout.write('\n')
            
def read_timing_noise(folder, sname=None):
    output_dictionary = {}
    nan_antennas_list = []
    
    def read_data(fname):
        with open(folder + '/' + sname + ".txt", 'r') as fin:
            for line in fin:
                data = fin.split()
                ant_name = data[0]
                noise = float(data[1])
                output_dictionary[ant_name] = noise
                if noise != noise: ## is nan
                    pass
    
    if (sname is None) and (fname is not None):
        fnames = os.listdir
    elif (sname is not None) and (fname is None):
        fname = folder + '/' + sname + ".txt"
    
    
    
    
    
    
    