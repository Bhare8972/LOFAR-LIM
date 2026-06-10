#!/usr/bin/env python3

import os
import argparse

import numpy as np
from matplotlib import pyplot as plt

from LoLIM.pipeline.readSettings import readSettings
from LoLIM.pipeline.database import database_manager
from LoLIM.pipeline.run_find_RFI import readFindRFI_results # readFindRFI_results( databaseSettings, flashOutputFolder, sname_list )

from LoLIM import prettytable

if __name__ == "__main__":
    print('program plot findRFI result')

    parser = argparse.ArgumentParser()
    parser.add_argument("settings", help="location of the settings JSON file")
    parser.add_argument("flash", help="name of the flash. E.G 20B-1")

    parser.add_argument("--stations", action="extend", nargs="*", help="list of stations to plot. If not given then all stations are plotted.")

    parser.add_argument("--outfolder", help="folder where plots are put. Default is log folder of findRFI. Give a '0' if saved plots are not desired.")

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
    data_outputFolder = os.path.join(  flashOutputFolder, databaseSettings["find_RFI"]['output_folderName'])
    logFolderLocation = os.path.join( data_outputFolder, databaseSettings["find_RFI"]['log_folderName'])
    #output_fname = os.path.join( outputFolder, sname+'_stats.json' )


    output_folder = logFolderLocation
    if not( args.outfolder is None):
        output_folder = args.outfolder


    if not os.path.isdir( data_outputFolder ):
        print('ERROR! path:', data_outputFolder, 'does not exist')
        quit()


    available_stations = [ f[0:5] for f in os.listdir(data_outputFolder) if f.endswith('_advFindRFI.json')]

    if args.stations is None:
        stations_todo = available_stations
    else:
        stations_todo = []
        for s in args.stations:
            if s in available_stations:
                stations_todo.append( s )
            else:
                print('WARNING: station', s, 'does not have an output in', data_outputFolder)


    RFI_data = readFindRFI_results( databaseSettings, flashOutputFolder, stations_todo )


    for sname, RFIdata in RFI_data.items():

        print('station:', sname)
        for antSet in RFIdata:

        # RFI_output_dict["blocksize"] = block_size
        # RFI_output_dict["initial_block"] = initial_block
        # RFI_output_dict["num_blocks"] = num_blocks
        # RFI_output_dict["max_blocks"] = max_blocks

        # RFI_output_dict["dirty_channels"] = list(dirty_channels + lower_frequency_index)

        # cleaned_spectrum = np.array( spectrum_mean )
        # cleaned_spectrum[:, dirty_channels] = 0.0
        # RFI_output_dict["cleaned_power"] = list( 2*np.sum( cleaned_spectrum, axis=1 ) )

        # RFI_output_dict["antenna_names"] = list(pol_antenna_names)
        # RFI_output_dict["antennas_good"] = list( antenna_is_good )

        #RFI_output_dict["median_phase_spread_byChannel"] = median_phase_spread_byChannel
        #RFI_output_dict["reference_spectrum"] = spectrum_mean[ref_antenna_pairi]
        

        # RFI_output_dict["antenna_set"] = TBB_in_file.get_antenna_set()
        # RFI_output_dict["filter_selection"] = TBB_in_file.get_filter_selection()
        # RFI_output_dict["antenna_polarization"] = pol_name

        # RFI_output_dict["timestamp"] = TBB_in_file.get_timestamp()
        # RFI_output_dict["lower_frequency_index"] = lower_frequency_index
        # RFI_output_dict["upper_frequency_index"] = upper_frequency_index
            print('  ant. set:', antSet["antenna_set"], "pol", antSet["antenna_polarization"] )


            lower_frequency_index =  antSet["lower_frequency_index"]
            upper_frequency_index =  antSet["upper_frequency_index"]
            block_size = antSet["blocksize"]
            sample_frequency = 1/(5.0e-9)

            median_phase_spread_byChannel = antSet["median_phase_spread_byChannel"]


            frequencies = np.fft.fftfreq(block_size, 1.0/sample_frequency)
            cut_frequencies = frequencies[lower_frequency_index:upper_frequency_index]
            cut_frequencies_MHZ = cut_frequencies*(1e-6)

            median_spread = np.median( median_phase_spread_byChannel )
            sorted_phase_spreads = np.sort( median_phase_spread_byChannel )
            N = len(median_phase_spread_byChannel)
            noise = sorted_phase_spreads[int(N*0.95)] - sorted_phase_spreads[int(N/2)]


            plt.figure()
            plt.plot(cut_frequencies_MHZ, median_phase_spread_byChannel)
            plt.axhline( median_spread-3*noise, color='r')
            plt.title("Phase spread vs frequency. ant set:"+antSet["antenna_set"]+"\nPolarization "+antSet["antenna_polarization"])
            plt.ylabel("Spread value")
            plt.xlabel("Frequency [MHz]")
            
            if output_folder != '0':
                print(output_folder, sname, '_antSet', antSet["antenna_set"], "_pol:", antSet["antenna_polarization"], '_RFIPhase.pdf')
                outfile = os.path.join( output_folder, sname+'_antSet'+antSet["antenna_set"]+"_pol:"+antSet["antenna_polarization"]+'_RFIPhase.pdf')
                plt.savefig(fname=outfile)

            if args.show:
                plt.show()

            plt.close()


            ref_spectrum = antSet["reference_spectrum"]
            dirty_channels_adj = np.array(antSet["dirty_channels"])-lower_frequency_index

            plt.figure()
            plt.title("Amplitude vs frequency. ant set:"+antSet["antenna_set"]+"\nPolarization "+antSet["antenna_polarization"])
            plt.plot(cut_frequencies_MHZ,                 ref_spectrum)
            plt.plot(cut_frequencies_MHZ[dirty_channels_adj], ref_spectrum[dirty_channels_adj], 'ro')
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("magnitude")
            plt.yscale('log')#, nonposy='clip')

            if output_folder != '0':
                outfile = os.path.join( output_folder, sname+'_antSet'+antSet["antenna_set"]+"_pol:"+antSet["antenna_polarization"]+'_RFIamp.pdf')
                plt.savefig(fname=outfile)

            if args.show:
                plt.show()

            plt.close()


            print()
            print()
            print()

