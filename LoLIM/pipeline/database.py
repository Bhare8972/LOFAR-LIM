#!/usr/bin/env python3

"""this provides some helper functions for creating and reading the JSON database. Hopefully avoiding data races"""


from os.path import isfile, dirname, abspath, join
import time
import os
import datetime
from shutil import copyfile


import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from LoLIM import LOFARFlashData
from LoLIM import utilities as utils

from LoLIM.pipeline.readSettings import readSettings


flashState_categories = CategoricalDtype(["ready_download_data", "download_data", "initial_statistics", "find_RFI", "impulsive_imager", "LTA_upload", "local_cleanup"], ordered=False)
flashSubState_categories = CategoricalDtype(["ready", "running", "complete", "error"], ordered=False)
pipelineMode_categories = CategoricalDtype(["off", "all", 
                        "ready_download_data","download_data", "initial_statistics", "find_RFI", "impulsive_imager"], ordered=False)


def makeEmptyDatabase(flashDatabase_fname, stormDatabase_fname):
    """make a new database at location database_path only if database_path does not exist. returns True if succsesful, False otherwise. 
    Populates database with info in LoLIM.LOFARflashData"""

    if isfile(flashDatabase_fname):
        print("ERROR: cannot make new database, file already exists at:", flashDatabase_fname)
        return False

    if isfile(stormDatabase_fname):
        print("ERROR: cannot make new storm database, file already exists at:", stormDatabase_fname)
        return False

    DF =  pd.DataFrame( { 
        'TimeID':pd.Series([],dtype=str),
        'year':pd.Series([],dtype=str),
        'month':pd.Series([],dtype=str),
        'day':pd.Series([],dtype=str),
        'state':pd.Series([],dtype=flashState_categories),
        'sub_state':pd.Series([],dtype=flashSubState_categories),
        'rawDataDownloaded':pd.Series([],dtype=bool),
        'calibration_file':pd.Series([],dtype=str),
        'pipeline_priority':pd.Series([],dtype=int),                  ## larger is higher priority.    5 is normal.
        'pipeline_mode':pd.Series([],dtype=pipelineMode_categories),  ## what should the pipeline do (default: all)
        'deleteData_on_complete':pd.Series([],dtype=bool), ## default True
        'slurm_processes':[],
        'storm':pd.Series([],dtype=str),
        }, 
        index=pd.Series([],dtype=str) ## flash name
        )


    SDF =  pd.DataFrame( { 
        'LTAloc_file':pd.Series([],dtype=str),
        'MD5Sum_file':pd.Series([],dtype=str),
        'year':pd.Series([],dtype=str),
        'month':pd.Series([],dtype=str),
        'day':pd.Series([],dtype=str),
        'hour':pd.Series([],dtype=str),
        "ID_prefix":pd.Series([],dtype=str),     ## eg:  L2025065
        "numberFlashes":pd.Series([],dtype=int),
        }, 
        index=pd.Series([],dtype=str)            ## eg:23G
        )



    ## I know this is not quite correct way to inject new data, 

    storm_dict = {} 
    for flashname, flash_timeID in LOFARFlashData.__FlashNameToTimeIDDict__.items():
        storm = flashname.split('-')[0]

        year = utils.year_from_timeID(flash_timeID)
        month = utils.month_from_timeID(flash_timeID)
        day = utils.day_from_timeID(flash_timeID)
        hour = utils.hour_from_timeID(flash_timeID)

        data = {"TimeID":flash_timeID,  "year":year, 'month':month, "day":day, 'state':"ready_download_data", "sub_state":"ready", 'rawDataDownloaded':False, 'calibration_file':'', 'pipeline_priority':5,
            'pipeline_mode':"off", 'deleteData_on_complete':True, 'slurm_processes':[],  'storm':storm }
        DF.loc[flashname] = data

        if storm not in storm_dict:
            storm_info = { 'LTAloc_file':'', 'MD5Sum_file':'', "year":year, 'month':month, "day":day, "hour":hour, "ID_prefix":'', "numberFlashes":1}
        else:
            storm_info = storm_dict[storm]

            if int(month) < int(storm_info['month']):
                storm_info['month'] = month
                storm_info['day'] = day
                storm_info['hour'] = hour

            elif int(day) < int(storm_info['day']):
                storm_info['day'] = day
                storm_info['hour'] = hour

            elif int(hour) < int(storm_info['hour']):
                storm_info['hour'] = hour

            storm_info['numberFlashes'] += 1

        storm_dict[storm] = storm_info


    for stormname, storminfo in storm_dict.items():
        SDF.loc[stormname] = storminfo



    with open(flashDatabase_fname, 'w') as fout:
        fout.write( DF.to_json(orient='table') )

    with open(stormDatabase_fname, 'w') as fout:
        fout.write( SDF.to_json(orient='table') )

    return True





class database_manager:
    """ attempts to manage reading and writing to the database, so that only one process does so at a time.
    Works by making a temp file to track if file is open or not. Thus, it has a fail mode if two processes try to accses in close time intervals.
    If it sees the database is open, it will wait up to timeout [minutes] to try and reopen it. and throw an error otherwise"""

    def __init__(self, database_fname, timeout_mins=10):
        self.database_fname = database_fname
        self.database_path =  dirname(abspath( self.database_fname ))
        self.database_file = os.path.basename( self.database_fname )
        self.timeout_mins = timeout_mins

        self.lockFname_start = self.database_file + "_lock"
        self.lockFname =  os.path.join( self.database_path,  self.lockFname_start + str(np.random.randint(0,99999)) )

        self.is_open = False

    def __enter__(self):

        start_time = time.time()

        while True:
            ## check timeout
            if (time.time()-start_time)/60.0 > self.timeout_mins:
                raise TimeoutError("Could not lock accses to database: "+self.database_fname)

            ## make a lock file so other managers know we are trying to accses file
            with open( self.lockFname,'w') as lock_file:
                lock_file.write('Hi!')

            ## now we check if other lockfiles exist
            all_files = ( f for f in os.listdir(self.database_path) if os.path.isfile( os.path.join(self.database_path, f) ) )
            lock_files = [f for f in all_files if f.startswith(self.lockFname_start ) ]


            if len(lock_files) == 0:
                ## there are NO lock files, which is wierd given we just made one
                raise TimeoutError("lock file does not exist, this should not happen")
            elif len(lock_files) > 1:
                ## there are other lock files!!
                ## remove ours
                os.remove( self.lockFname )
                ## wait random time
                time.sleep( 0.5 + np.random.random()  )

            else:
                ## only our lock file exists!
                break
            
        

        return self


    def __exit__(self, exception_type, exception_value, exception_traceback):
        #Exception handling here ??
    
        if isfile( self.lockFname ) :
            os.remove( self.lockFname )
        else:
            print('WARNING: somehow lock file was removed:', self.lockFname)


    def read_database(self):
        return pd.read_json(self.database_fname, orient='table')

    def save_database(self, db):    
        with open(self.database_fname, 'w') as fout:
            fout.write( db.to_json(orient='table') )




 ## USSAGE:
# with database_manager(loc, timeout) as man:
#       db = man.get_database()
#       man.save_database( db )


def backupFile(fname):
    """ given a filename (ostensilby a databse), copy the file to a new location that is fname+'%'+current date and time"""

    newfname = fname+'%'+datetime.datetime.now().strftime('D%Y%m%dT%H%M%S') 
    copyfile( fname, newfname )


def cleanupBackups(database_fname, min_num_backups=1, maxBackupDays=30):
    backupFolder, fileName = os.path.split(database_fname)

    allBackups_fnames = [f for f in os.listdir(backupFolder) if os.path.isfile( os.path.join(backupFolder, f)) and ('%' in f) and f.startswith(fileName)]
    backupDatetimes = [ datetime.datetime.strptime(f.split('%')[-1], 'D%Y%m%dT%H%M%S')  for f in allBackups_fnames ]

    ## need to sort these two arrays together
    comboData = list( zip(backupDatetimes, allBackups_fnames) )
    comboData.sort( key = lambda x: x[0], reverse=True )

    #decide what to keep
    backupDate = datetime.datetime.now() - datetime.timedelta(days=maxBackupDays)
    backups_fnames_to_keep = []
    lastFname_index = None
    for index, (backup_datetime, fname) in enumerate(comboData):
        if backup_datetime < backupDate:
            lastFname_index = index
            break
        else:
            backups_fnames_to_keep.append( fname )



    ## keep more if not enough
    if lastFname_index is not None:
        num_needed = min_num_backups - len(backups_fnames_to_keep)
        if num_needed >0:
            num_left = len(comboData) - lastFname_index
            if num_left > 0:
                for i in range(lastFname_index, lastFname_index+min(num_left,num_needed) ):
                    backups_fnames_to_keep.append( comboData[i][1] )


    ## now remove files that are not kept
    for fname in allBackups_fnames:
        if fname not in backups_fnames_to_keep:
            os.remove( os.path.join( backupFolder, fname ) )






def flashDatabase_printActiveFlashes(flashDatabase=None, settingsFname=None):
    if flashDatabase is None:
        databaseSettings = readSettings( settingsFname )
        flashDatabase_location = databaseSettings['flashDatabase_location']
        with database_manager(flashDatabase_location) as flashdatabase_manager:
            flashDatabase = flashdatabase_manager.read_database()

    activeFlashDatbase = flashDatabase[ flashDatabase["pipeline_mode"]!="off" ] 

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(len(activeFlashDatbase), "active flashes")
    print(  activeFlashDatbase  )

    #for flashName, dataRow in activeFlashDatbase.iterrows():
    #    print(flashName, dataRow)
    #    print()

