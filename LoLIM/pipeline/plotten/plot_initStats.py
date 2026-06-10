#!/usr/bin/env python3

import os
import argparse

import numpy as np
from matplotlib import pyplot as plt

from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.database import database_manager
from LoLIM.pipeline.initial_statistics import readInitialStatisticsFile

from LoLIM import prettytable

if __name__ == "__main__":
    print('program plot initial_statistics result')

    parser = argparse.ArgumentParser()
    parser.add_argument("settings", help="location of the settings JSON file")
    parser.add_argument("flash", help="name of the flash. E.G 20B-1")

    parser.add_argument("--stations", action="extend", nargs="*", help="list of stations to plot. If not given then all stations are plotted.")

    parser.add_argument("--outfolder", help="folder where plots are put. Default is log folder of initial_statistics. Give a '0' if saved plots are not desired.")

    parser.add_argument("--show", action="store_true", help="if given, show plot with interactive plotter. Else only save to file. If this is not given, and --outfolder is 0 than this script will do nothing interesting")

    args = parser.parse_args()



    print('read settings')
    databaseSettings = readSettings( args.settings )

    print('read database')
    with database_manager(databaseSettings['flashDatabase_location']) as flashdatabase_manager:
        flashDatabase = flashdatabase_manager.read_database()

    dataRow = flashDatabase.loc[args.flash]
    flashYear = dataRow['year']

## some file and folder organization
    flashOutputFolder = os.path.join(  databaseSettings["processed_data_loc"], flashYear, args.flash)
    data_outputFolder = os.path.join(  flashOutputFolder, databaseSettings["initial_statistics"]['output_folderName'])
    logFolderLocation = os.path.join( data_outputFolder, databaseSettings["initial_statistics"]['log_folderName'])
    #output_fname = os.path.join( outputFolder, sname+'_stats.json' )


    output_folder = logFolderLocation
    if not (args.outfolder is None):
        output_folder = args.outfolder


    if not os.path.isdir( data_outputFolder ):
        print('ERROR! path:', data_outputFolder, 'does not exist')
        quit()


    available_datafiles = { f[0:5]:f for f in os.listdir(data_outputFolder) if f.endswith('_stats.json')}

    if args.stations is None:
        todo_dict = available_datafiles
    else:
        todo_dict = {}
        for s in args.stations:
            if s in available_datafiles:
                todo_dict[s] = available_datafiles[s]
            else:
                print('WARNING: station', s, 'does not have an output in', data_outputFolder)


    for sname, datafile in todo_dict.items():

        print('station:', sname)

        statsData = readInitialStatisticsFile( databaseSettings, flashOutputFolder, sname )
        
        #{'sname':sname, 'ant_names':ant_names, 'saturation_max':saturation_max, 'saturation_min':saturation_min, 
        #'blockSize':blockSize, 'num_blocks':num_blocks, 'maxOverBlocks':maxOverBlocks, 
        #'fracSaturation_perAntenna':fracSaturation_perAntenna, 'fracDblZero_perAntenna':fracDblZero_perAntenna }

        print('  sat-max:', statsData['saturation_max'], 'sat-min:', statsData['saturation_min'])
        print('  blockSize:', statsData['blockSize'], 'num_blocks:', statsData['num_blocks'])

        antenna_names = statsData['ant_names']
        fracSat = statsData['fracSaturation_perAntenna']
        fracDblZero = statsData['fracDblZero_perAntenna']

        print(fracSat)
        

        x = prettytable.PrettyTable(["ant. name", "% saturation", "% dbl zero"])
        x.float_format = "3.0f"
        x.align["ant. name"] = "l" # Left align city names

        for ant in antenna_names:
            x.add_row([ant, fracSat[ant]*100, fracDblZero[ant]*100])
        print(x)


        max_over_blocks = statsData['maxOverBlocks']
        sat = max_over_blocks>2046
        print('fraction blocks with saturation:', np.sum(sat)/len(sat))


        fig,ax = plt.subplots(1,1)
        ax.plot( np.arange(statsData['num_blocks']), statsData['maxOverBlocks'] )
        ax.set_ylabel('block max')
        ax.set_xlabel('block number')

        if output_folder != '0':
            outfile = os.path.join( output_folder, sname+'_blockMax.pdf')
            fig.savefig(fname=outfile)

        if args.show:
            plt.show()



        print()
        print()
        print()

