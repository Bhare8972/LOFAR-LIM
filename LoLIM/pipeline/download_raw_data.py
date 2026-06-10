#!/usr/bin/env python3

"""this file contains utilties for the first steps in the pipeline, that is downloading data using wget from the LTA"""

import sys
import os
import hashlib
import time
import json

import pandas as pd

from LoLIM import utilities as utils
from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.database import database_manager

from LoLIM.pipeline.jobInterface import standard_jobInterface

class downloadData_jobInterface(standard_jobInterface):
    def stage_name(self):
        """return the name of this stage"""
        raise "download_data"

    def command(self):
        """return string of command to run"""
        raise "python -m LoLIM.pipeline.download_raw_data"


    def postProcessing(self):
        flashName = self.flashName
        databaseSettings = self.settings_data

        stageName = "download_data"
        #dataRow = self.flashDatabase_row

        #flashOutputFolder = os.path.join(  databaseSettings["processed_data_loc"], dataRow['year'], flashName )
        flashOutputFolder = self.flashOutputFolder
        tableOutputFname = os.path.join(flashOutputFolder, databaseSettings['ready_download_data']['output_folderName'], databaseSettings['ready_download_data']['dataFileTableName'])

        with database_manager(tableOutputFname) as dataFileTableManager:
            dataFileTable = dataFileTableManager.read_database()  ## label is an integer index starts at 0

        allDownloaded = True
        for i,row in dataFileTable.iterrows():
            if not row['file_downloaded']:
                allDownloaded = False
                #print( row['fileName'], 'NOT downloaded' )
           # else:
                #print( row['fileName'], 'downloaded' )

        if allDownloaded:
            self.flashDatabase_row['rawDataDownloaded'] = True
            return True
        else:
            return False

        

def findAll_LTAloc_Files(folder, folder_organized_by_year=False):
    """given a folder, return a list of tuples. Each tuple has two strings. First string is absolute path and filename of the LTA locations file. Second is same, but for the md5 sum file. 
    if folder_organized_by_year is false, this function will only search the given folder. If True, this function will search subfolders that have integer names (and NOT this folder)"""

    folder_contents = os.listdir(folder)
    files_to_check = folder_contents

    ## compensate if organized by year
    if folder_organized_by_year:
        files_to_check = []

        for FC in folder_contents:
            if FC.isdigit() and os.path.isdir( join(folder, FC) ):
                files_to_check += [join(FC,f) for f in os.listdir( join(folder, FC) ) ]


    ## check files are good
    files_to_check = [join(folder,f) for f in files_to_check]
    good_files = [ f for f in files_to_check if os.path.isfile(f) and f.endswith('LTAfiles') ]

    ## check for md5 sum
    MDF5sums_files = []
    for f in good_files:
        MD5Sum_fname = f.rsplit( ".", 1 )[ 0 ] + ".md5"

        n = None
        if os.path.isfile(MD5Sum_fname):
            n = MD5Sum_fname

    return [ (f,m) for f,m in zip(good_files, MDF5sums_files) ]


def LTALocFname_to_ObservationID( LTALocFname ):
    f = LTALocFname.rsplit('/',1)[0]
    return f.split('.')[0]



class LTAlocations:
    def __init__(self, LTA_locations_file, md5Sums_file=None, year=None, flash_timeID=None):
        """read all file locations from a file and MD5 sums. If year is specified (as string), only include files from that year. If flash_timeID is specified, only include files for that flash"""

        self.LTA_locations_file = LTA_locations_file
        self.md5Sums_file = md5Sums_file


        timeID_has_subseconds = (flash_timeID is not None) and ('.' in flash_timeID)

        self.file_locs = {}
        with open(self.LTA_locations_file, 'r') as fin:
            for line in fin:
                line_data = line.split()
                
                if len(line_data) == 0:
                    continue
                
                floc = line_data[-1]
                fname = floc.split('/')[-1]
                TimeID = utils.get_timeID(fname)

                if not timeID_has_subseconds:
                    TimeID_check = TimeID.split('.')[0]
                else:
                    TimeID_check = TimeID

                if (flash_timeID is not None) and (flash_timeID != TimeID_check):
                    continue

                self.year = utils.year_from_timeID(TimeID)
                if (year is not None) and (year != self.year ):
                    continue

                if TimeID not in self.file_locs:
                    self.file_locs[TimeID] = []
                    
                self.file_locs[TimeID].append( floc )


        self.MD5_data = {}
        if self.md5Sums_file is not None:
            with open(self.md5Sums_file) as md5_fin:
                for line in md5_fin:
                    line_data = line.split()
                    if len(line_data) == 0:
                        continue
                    
                    inputmd5sum, file = line_data
                    file = file.split('/')[-1]
                    self.MD5_data[file] = inputmd5sum

    def getTimeIDs(self):
        """return list of TimeIDS"""

        return list( self.file_locs.keys() )

    def locs_by_TimeID(self, TimeID):
        return self.file_locs[TimeID]

    def loc_to_fname(self, file_loc):
        return file_loc.split('/')[-1]

    def getMD5_per_fLoc(self, file_loc):
        fname = file_loc.split('/')[-1]

        if fname in self.MD5_data :
            return self.MD5_data[fname]
        else:
            return None



#def default_wget_prefix():
#    return "https://lofar-download.grid.surfsara.nl/lofigrid/SRMFifoGet.py?surl="

def md5sum(fname, blocksize=65536):

        hash = hashlib.md5()
        with open(fname, "rb") as f:
            for block in iter(lambda: f.read(blocksize), b""):
                hash.update(block)
        return hash.hexdigest()

def download_file_wget(lta_loc, download_folder, wgetRC=None, prefix=None, returnMD5=True):
    """attempt to download a file with wget. if returnMD5 is false return only file size as integer bytes. If returnMD5 is true, return size and MD5 sum. If size is 0, then return empty string as MD5sum  """


    if prefix is None:
        prefix = ''


    partial_fname = lta_loc.split('/')[-1]
    download_loc = os.path.join( download_folder, partial_fname)

    cmd = "wget --no-check-certificate -nv " + prefix + lta_loc+" -O "+download_loc
    if wgetRC!=None:
        cmd = "WGETRC=" + wgetRC + " " + cmd

    os.system(cmd)

    if os.path.isfile( download_loc ):
        fs = os.path.getsize( download_loc )
    else:
        fs = 0

    if not returnMD5:
        return fs

    if (fs == 0):
        return 0, ""

    foundMD5 = md5sum(download_loc)

    return fs, foundMD5




### HIGHERLEVEL PIPELINE FUNCTIONS ###


def RunStage_downloadData(flashName, outputFolder, logFolderLocation, flashDatabaseData, jobNumber, maxJobs, databaseSettings, stageSettings):

    jobNumber = int(jobNumber)
    maxJobs = int(maxJobs)

    timeID = flashDatabaseData["TimeID"]
    if '.' not in timeID:
        print("timeID for flash", flashName, "is not complete (", timeID,")")
        return False

    dataFolder = databaseSettings['raw_data_loc']
    year = flashDatabaseData['year']
    yearDataFolder = os.path.join( dataFolder, year )
    flashDataFolder = os.path.join( yearDataFolder, timeID )

    if not os.path.isdir( yearDataFolder ):
        os.mkdir( yearDataFolder )
    if not os.path.isdir( flashDataFolder ):
        os.mkdir( flashDataFolder )

    wgetRC_fname = databaseSettings['download_data']['wgetRC_file']


    flashOutputFolder = os.path.join(  databaseSettings["processed_data_loc"], year, flashName )
    tableOutputFname = os.path.join(flashOutputFolder, databaseSettings['ready_download_data']['output_folderName'], databaseSettings['ready_download_data']['dataFileTableName'])

    with database_manager(tableOutputFname) as dataFileTableManager:
        dataFileTable = dataFileTableManager.read_database()  ## label is an integer index starts at 0

    Num = len(dataFileTable)
    Num_todo = int(Num/maxJobs)
    #initial_index = Num_todo*jobNumber
    #final_index = Num if (jobNumber==(maxJobs-1)) else ( initial_index+ Num_todo)
    workIndex = [ i*Num_todo for i in range(maxJobs) ]
    workIndex.append( Num )
    initial_index = int( workIndex[jobNumber] )
    final_index = int( workIndex[jobNumber+1] )

    all_downloaded = True
    update_datums = []
    LTA_loc_prefix = stageSettings['LTA_loc_prefix']
    for i in range(initial_index, final_index):
        LTA_location = dataFileTable.loc[i,'LTA_location']
        LTA_MD5Sum   = dataFileTable.loc[i,'LTA_MD5Sum']
        fileName     = dataFileTable.loc[i,'fileName']
        local_MD5Sum = dataFileTable.loc[i,'local_MD5Sum']
        file_downloaded = dataFileTable.loc[i,'file_downloaded']

        fullFileName = os.path.join( flashDataFolder,fileName )

        print('fileName:', fileName)


        if os.path.isfile( fullFileName ):
            if file_downloaded:
                print('already downloaded: skipping')
                continue

            else:
                if local_MD5Sum == "":
                    ## the database says is not downloaded, but we don't have it. 
                    ## If MD5 exists, than could be previous download error and we redownload
                    ## if MD5 does NOT exist, than could be a database reset, and thus we check
                    foundMD5 = md5sum( fullFileName )
                    if foundMD5==LTA_MD5Sum:  ## file is downloaded, so we update the database

                        print('previously downloaded with good md5: skipping')
                        update_datums.append( [i, True, foundMD5] )
                        continue

                # else: we trust the database that file is no good


        #if LTA_loc_prefix == '%default':
        #    LTA_loc_prefix = default_wget_prefix()

        print('downloading')
        A = time.time()
        fileSize, MD5_sum = download_file_wget(LTA_location, flashDataFolder, wgetRC=wgetRC_fname, prefix=LTA_loc_prefix, returnMD5=True)
        print(' done. T elapsed:', (time.time()-A)/60.0, "[mins]")

        if fileSize == 0:
            all_downloaded = False
            update_datums.append( [i, False, '0'] )
            print('  FAIL: final size 0')

        elif (LTA_MD5Sum!='') and (MD5_sum != LTA_MD5Sum):
            update_datums.append( [i, False, LTA_MD5Sum] )
            all_downloaded = False
            print('  FAIL: MD5 bad')

        else:
            update_datums.append( [i, True, LTA_MD5Sum] )
            print('  SUCCSESS!')

    with database_manager(tableOutputFname) as dataFileTableManager:
        dataFileTable = dataFileTableManager.read_database()

        for i, isdown, md5 in update_datums:
            dataFileTable.loc[i, 'file_downloaded'] = isdown
            dataFileTable.loc[i, 'LTA_MD5Sum'] = md5

        dataFileTableManager.save_database( dataFileTable )

    return all_downloaded







if __name__ == "__main__":
    settingsFname  = sys.argv[1]
    flashName      = sys.argv[2]
    jobNumber      = sys.argv[3]
    maxJobs        = sys.argv[4]
    numCPUs        = sys.argv[5]


## open database and pipeline settings
    stageName = "download_data"

    databaseSettings = readSettings( settingsFname )

    with database_manager(databaseSettings['flashDatabase_location']) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()

    dataRow = flashDatabase.loc[flashName]
    flashYear = dataRow['year']

    flashOutputFolder = os.path.join(  databaseSettings["processed_data_loc"], flashYear, flashName)

    outputFolder = os.path.join(  flashOutputFolder, databaseSettings[stageName]['output_folderName'])
    logFolderLocation = os.path.join( outputFolder, databaseSettings[stageName]['log_folderName'])

    logFileName = os.path.join( logFolderLocation, databaseSettings[stageName]['log_fileName'])
    logFileName = logFileName + '_job'+str(jobNumber)+'.txt'

    log = utils.logger( logFileName, False )
    log.take_stdout()
    log.take_stderr()

    print("stage:", stageName)
    print("  flash:", flashName, "settings:", settingsFname, "jobNumber:",jobNumber, "maxJobs:",maxJobs, "numCPUs:",numCPUs)


## now check for stage settings
    stage_settings_fname = os.path.join( outputFolder, databaseSettings[stageName]['settingsFileName'] ) 
    if os.path.isfile(stage_settings_fname):
        stageSettings = json.load( open(stage_settings_fname, 'r') )
    else:
        stageSettings = databaseSettings[stageName]['settings']


## now run!
    succsses = RunStage_downloadData(flashName, outputFolder, logFolderLocation, dataRow, jobNumber, maxJobs, databaseSettings, stageSettings)



## check if good!
    if succsses:
        print('is succsesful!')

        completionFile = os.path.join( outputFolder, databaseSettings[stageName]['log_folderName'], databaseSettings[stageName]['succesfulCompletion_FileName']+"_"+jobNumber)
        with open(completionFile, 'w') as fout:
            fout.write("1")
    else:
        print('not succsesful')


    del log



