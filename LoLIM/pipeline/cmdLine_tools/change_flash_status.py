#!/usr/bin/env python3


import argparse

from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.database import database_manager, backupFile

if __name__ == "__main__":
    print('program change flash status')

    flash_state_categories = ["ready_download_data", "download_data", "initial_statistics", "find_RFI", "impulsive_imager", "LTA_upload", "local_cleanup"]

    parser = argparse.ArgumentParser(description='change the state of one flash in database. If mode or state are changed, then sub_state is set to ready')
    parser.add_argument("settings", help="location of the settings JSON file")
    parser.add_argument("flash", help="name of the flash. E.G 20B-1")

    parser.add_argument("--pipeline_mode", help="new pipeline mode", required=False,
        choices=['off', 'all']+flash_state_categories)
    parser.add_argument("--state", help="new state", required=False,
        choices=flash_state_categories)


    parser.add_argument("--calibration_file", help="name and location of calibration file", required=False)

    parser.add_argument("--pipeline_priority", help="priority of this flash in the pipeline. Larger is higher priority", required=False, type=int)

    parser.add_argument("--deleteData_on_complete", help="True/False if delete all data on complete", required=False, choices=['True','False'])

    
    args = parser.parse_args()

    print('read settings')
    databaseSettings = readSettings( args.settings )

    stormDatabase_location = databaseSettings['stormDatabase_location']
    flashDatabase_location = databaseSettings['flashDatabase_location']

    print('backup database!')
    backupFile(stormDatabase_location)
    backupFile(flashDatabase_location)

    with database_manager(flashDatabase_location) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()




        if not args.flash in flashDatabase.index:
            print("ERROR flash", args.flash, "not in database")
            quit()

        print('current flash status:')
        print(flashDatabase.loc[args.flash])


        print()
        database_changed = False
        madeSubState_ready = False

        if args.pipeline_mode != None:
            flashDatabase.at[args.flash, 'pipeline_mode'] = args.pipeline_mode
            database_changed = True
            madeSubState_ready = True

        if args.state != None:
            flashDatabase.at[args.flash, 'state'] = args.state
            database_changed = True
            madeSubState_ready = True

        if args.calibration_file != None:
            flashDatabase.at[args.flash, 'calibration_file'] = args.calibration_file
            database_changed = True

        if args.pipeline_priority != None:
            flashDatabase.at[args.flash, 'pipeline_priority'] = int(args.pipeline_priority)
            database_changed = True

        if args.deleteData_on_complete != None:
            if  args.deleteData_on_complete == 'True':
                flashDatabase.at[args.flash, 'deleteData_on_complete'] = True

            elif  args.deleteData_on_complete == 'False':
                flashDatabase.at[args.flash, 'deleteData_on_complete'] = False

            database_changed = True

        if madeSubState_ready:
            flashDatabase.at[args.flash, 'sub_state'] = 'ready'
            database_changed = True ## just in case



        if database_changed:
            print('database entry changed:')
            print()
            print(flashDatabase.loc[args.flash])

            flashdatabase_manager.save_database( flashDatabase )

        else:
            print('database not changed')

