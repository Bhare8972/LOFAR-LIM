#!/usr/bin/env python3

""" a set of utilities to inject new flashes into the pipeline """

import os
import shutil

import numpy as np

from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.database import database_manager, backupFile, pipelineMode_categories, flashState_categories
from LoLIM.pipeline.download_raw_data import LTAlocations
import LoLIM.utilities as utils


## storm name utitilities

def incrementLetterGroup( lettergroup ):
    """ if a letter group is a string of capitol letters, return the next one. e.g. if 'AQZ' return 'ARZ' """

    sqnc = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    letterList = [l for l in lettergroup][::-1]

    currentIndex = 0
    while True:
        if currentIndex == len(letterList):
            letterList.append( 'A' )
            break

        elif letterList[currentIndex]=='Z':
            letterList[currentIndex] = 'A'
            currentIndex += 1

        elif letterList[currentIndex] not in sqnc:
            print("ERROR in incrementing letter group:", lettergroup)
            quit()

        else:
            i = sqnc.index( letterList[currentIndex] )
            letterList[currentIndex] = sqnc[i+1]
            break

    return ''.join( letterList[::-1] )


def lttrGrp_is_gtrThanOrEql(A, B):
    """return true if lettergroup A is greater than or equal to B. False otherwise
    That is, if incrementLetterGroup is applied to B 0 or more times will it become A"""


    if len(A) > len(B):
        return True

    elif len(A) < len(B):
        return False

    elif A==B:
        return True

    else:
        ### they are of equal length but not trivially equal
        return A >= B


def inject_storm(observationID, year, databaseSettings, newFlash_pipelineMode="off"):

    """
    inject storm into storm database. Will return True/False if succsseful.
    Also inject flashes into flash database.
    Will create database backups before changing database


    Assumes that storm is LTA_locations_folder, and locations file is observationID+".LTAfiles", and md5 sum file (optional) is observationID+".md5"
    Will check if storm is already in database. If so, will check consistancy.

    observationID should be a string. e.g. "L2028003"
    year should be a string with four digits. e.g. "2023"
    databaseSettings should be a dictionary of the database settings, or a sting of its location

    newFlash_pipelineMode should be a string, out of database.pipelineMode_categories, to set for new flashes. Will not change flashes already in pipeline (default "off")
    """

    if isinstance(databaseSettings, str):
        databaseSettings = readSettings( databaseSettings )

    if newFlash_pipelineMode not in pipelineMode_categories.categories:
        print("ERROR: pipeline mode: "+newFlash_pipelineMode+" not in list")
        return False


    stormDatabase_location = databaseSettings['stormDatabase_location']
    flashDatabase_location = databaseSettings['flashDatabase_location']

    LTA_locs_folder = databaseSettings['download_data']['LTA_locations_folder']
    LTA_locs_subfolder = os.path.join(LTA_locs_folder, year) if databaseSettings['download_data']['LTA_locations_folder_organized_byYear'] else LTA_locs_folder

    LTAlocFile = os.path.join( LTA_locs_subfolder, observationID+".LTAfiles" )
    MD5SumFile = os.path.join( LTA_locs_subfolder, observationID+".md5" )

    if not os.path.isfile(LTAlocFile):
        print("ERROR: cannot find LTAloc file at:", LTAlocFile )
        return False

    haveMD5_file = True
    if not os.path.isfile(MD5SumFile):
        MD5SumFile = None
        haveMD5_file = False
        print('WARNING: cannot find MD5sum file:', MD5SumFile)

    if not os.path.isfile(stormDatabase_location):
        print("ERROR: cannot find storm database at:", stormDatabase_location )
        return False

    if not os.path.isfile(flashDatabase_location):
        print("ERROR: cannot find flash database at:", flashDatabase_location )
        return False


    backupFile(stormDatabase_location)
    backupFile(flashDatabase_location)

    ### read databases
    with database_manager(stormDatabase_location) as stormdatbase_manager:
        stormDatabase = stormdatbase_manager.read_database()

    with database_manager(flashDatabase_location) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()


    ## get starting month and day
    minMonth = "99"
    minDay = '99'
    minHour = '99'

    locData = LTAlocations( LTAlocFile, MD5SumFile, year )
    flashTimeIDs = locData.getTimeIDs()
    flashTimeIDs.sort()

    truncated_flashTimeIDS = [] ## same as flashTimeIDs, but with .sssZ bit removed
    for ID in flashTimeIDs:
        FlashYear = utils.year_from_timeID( ID )
        truncated_flashTimeIDS.append( ID.split('.')[0] )

        if FlashYear != year:
            print('ERROR: a flash is from year '+year+" but given year is "+year)
            return False

        Flashmonth = utils.month_from_timeID( ID )
        Flashday = utils.day_from_timeID( ID )
        Flashhour = utils.hour_from_timeID( ID )

        if Flashmonth < minMonth:
            minMonth = Flashmonth 
            minDay = Flashday 
            minHour = Flashhour 

        elif Flashday < minDay:
            minDay = Flashday 
            minHour = Flashhour 

        elif Flashhour < minHour:
            minHour = Flashhour



    ## getStormName
    nextAvail_ltrGrp = 'A'
    stormsOfTheYear = stormDatabase[ stormDatabase['year']==year ]

    storm_is_new = True
    stormName = None
    for stormName_i, storm_info in stormsOfTheYear.iterrows():
        strm_ltrGrp = stormName_i[2:]

        if lttrGrp_is_gtrThanOrEql(strm_ltrGrp, nextAvail_ltrGrp):
            nextAvail_ltrGrp = incrementLetterGroup(strm_ltrGrp)


        if  storm_info['ID_prefix']  == '':
            if storm_info['month']==minMonth and storm_info['day']==minDay and storm_info['hour']==minHour:
                storm_is_new = False
                stormName = stormName_i
                break

        elif storm_info['ID_prefix'] == observationID:
            storm_is_new = False
            stormName = stormName_i
            break



    ### update database!!
    MD5_fileName = observationID+".md5"  if haveMD5_file else ""

    if storm_is_new:
        stormName = year[2:]+nextAvail_ltrGrp
        stormDatabase.loc[stormName] = { 'LTAloc_file': observationID+".LTAfiles" , 'MD5Sum_file':MD5_fileName, "year":year, 'month':minMonth, "day":minDay, "hour":minHour, \
"ID_prefix":observationID, "numberFlashes":len(flashTimeIDs)}


        i=1
        for flash_timeID in flashTimeIDs:
            flashMonth = utils.month_from_timeID(flash_timeID)
            flashDay = utils.month_from_timeID(flash_timeID)
            data = {"TimeID":flash_timeID,  "year":year, 'month':flashMonth, "day":flashDay, 'state':"ready_download_data", "sub_state":"ready", 'rawDataDownloaded':False, 'calibration_file':None, 'pipeline_priority':5,
            'pipeline_mode':newFlash_pipelineMode, 'deleteData_on_complete':True, 'slurm_processes':[],  'storm':stormName }
            flashName = stormName+'-'+str(i)
            flashDatabase.loc[ flashName ] = data

            i += 1



    else:
    ### check what needs to be updated

        stormData = stormDatabase.loc[stormName]

        ### first check easy things (not number of flashes)
        if stormData['LTAloc_file'] == '':
            stormDatabase.at[stormName,'LTAloc_file'] = observationID+".LTAfiles"

        if stormData['MD5Sum_file'] == '':
            stormDatabase.at[stormName,'MD5Sum_file'] = MD5_fileName

        if  stormData['year'] != year:
            stormDatabase.at[stormName,'year'] = year

        if  stormData['month'] != minMonth:
            stormDatabase.at[stormName,'month'] = minMonth


        if  stormData['day'] != minDay:
            stormDatabase.at[stormName,'day'] = minDay


        if  stormData['hour'] != minHour:
            stormDatabase.at[stormName,'hour'] = minHour


        if  stormData['ID_prefix'] == '':
            stormDatabase.at[stormName,'ID_prefix'] = observationID



        flashData = flashDatabase[ flashDatabase['storm']==stormName ]
        available_FlashIndex = 1  
        flashesAreStored = np.zeros(shape=len(flashTimeIDs), dtype=bool)  ## is True if this flash is in database, false otherwise. Correspons to flashTimeIDs
        flashNames_not_in_Observation = []
        for flashName, flashData in flashData.iterrows():
            index = int(flashName.split('-')[-1])
            if available_FlashIndex <= index:
                available_FlashIndex = index+1

            if flashData["TimeID"] in flashTimeIDs:
                flashesAreStored[ flashTimeIDs.index(flashData["TimeID"]) ] = True

            elif flashData["TimeID"] in truncated_flashTimeIDS:
                index = truncated_flashTimeIDS.index(flashData["TimeID"])
                flashesAreStored[ index ] = True
                flashDatabase.at[flashName, "TimeID"] = flashTimeIDs[ index ]

            else:
                flashNames_not_in_Observation.append(  flashName)

        ## add flashes if needed
        for flash_timeID, inPipeline in zip(flashTimeIDs, flashesAreStored):
            if inPipeline:
                continue

            flashMonth = utils.month_from_timeID(flash_timeID)
            flashDay = utils.day_from_timeID(flash_timeID)
            data = {"TimeID":flash_timeID,  "year":year, 'month':flashMonth, "day":flashDay, 'state':"ready_download_data", "sub_state":"ready", 'rawDataDownloaded':False, 'calibration_file':'', 'pipeline_priority':5,
            'pipeline_mode':newFlash_pipelineMode, 'deleteData_on_complete':True, 'slurm_processes':[],  'storm':stormName }
            flashName = stormName+'-'+str(available_FlashIndex)
            flashDatabase.loc[ flashName ] = data

            available_FlashIndex += 1

        stormDatabase.at[stormName,'numberFlashes'] = len(flashNames_not_in_Observation) + len(flashesAreStored)



    ### write databases
    with database_manager(stormDatabase_location) as stormdatbase_manager:
        stormdatbase_manager.save_database( stormDatabase )

    with database_manager(flashDatabase_location) as flashdatabase_manager:
        flashdatabase_manager.save_database( flashDatabase )


    return True

def inject_observations(databaseSettings, newFlash_pipelineMode="off"):
    """wrapper over inject_storm. Searches all files in inject_LTA_locations folder, if ends in LTAfiles then inject those flashes into the pipeline with the associated mode. 
    Copy all such LTAfiles and md5 files to the approprate folder, and then delete all contents inject_LTA_locations"""

    inject_LTA_locations = databaseSettings["inject_LTA_locations"]
    all_inject_files = [ f for f in os.listdir(inject_LTA_locations) if os.path.isfile(os.path.join(inject_LTA_locations,f)) ]

    LTAfiles = [f for f in all_inject_files if f.endswith('LTAfiles')]

    LTA_locs_folder = databaseSettings['download_data']['LTA_locations_folder']

    for ltaf in LTAfiles:
        obsID = ltaf.split('.')[0]

        ## get year
        locs = LTAlocations( os.path.join(inject_LTA_locations,ltaf) )
        LTA_locs_subfolder = os.path.join(LTA_locs_folder, locs.year) if databaseSettings['download_data']['LTA_locations_folder_organized_byYear'] else LTA_locs_folder

        if not os.path.isdir( LTA_locs_subfolder ):
            os.mkdir( LTA_locs_subfolder )

        ## copy LTAfiles
        shutil.copyfile(os.path.join(inject_LTA_locations,ltaf), os.path.join(LTA_locs_subfolder,ltaf))
        ## copy md5sum file if exist
        MD5file = obsID+'.md5'
        if os.path.isfile( os.path.join(inject_LTA_locations,MD5file) ):
            shutil.copyfile(os.path.join(inject_LTA_locations,MD5file), os.path.join(LTA_locs_subfolder,MD5file))

        ## inject!
        inject_storm(observationID=obsID, year=locs.year, databaseSettings=databaseSettings, newFlash_pipelineMode=newFlash_pipelineMode)



    ## clear the directory
    for fname in all_inject_files:
        try:
            os.remove( os.path.join(inject_LTA_locations,fname) )

        except Exception as e:
            print('exception', e)



def changeFlashStatus(flashName, databaseSettings, new_pipelineMode, new_Status=None):


    if new_pipelineMode not in pipelineMode_categories.categories:
        print("ERROR: pipeline mode: "+new_pipelineMode+" not in list")
        return False

    if new_Status is not None:
        if new_Status not in flashState_categories.categories:
            print("ERROR: pipeline mode: "+new_Status+" not in list")
            return False


    if isinstance(databaseSettings, str):
        databaseSettings = readSettings( databaseSettings )


    flashDatabase_location = databaseSettings['flashDatabase_location']

    backupFile(flashDatabase_location)

    with database_manager(flashDatabase_location) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()

        if not flashName in flashDatabase.index:
            print("ERROR flash", flashName, "not in database")
            return False

        flashDatabase.at[flashName, 'pipeline_mode'] = new_pipelineMode

        if new_Status is not None:
            flashDatabase.at[flashName, 'state'] = new_Status
            flashDatabase.at[flashName, 'sub_state'] = 'ready'

        flashdatabase_manager.save_database( flashDatabase )