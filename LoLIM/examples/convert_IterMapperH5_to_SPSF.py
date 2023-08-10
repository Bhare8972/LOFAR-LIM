#!/usr/bin/env python3

from LoLIM.IO.SPSF_readwrite import make_SPSF_from_data
from LoLIM.iterativeMapper.mapper_header import read_header

import numpy as np
from matplotlib import pyplot as plt


from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = "D20190424T194432.504Z"
    itermapper_folder = 'iterMapper_9Aug2023Cal_evenAnts'
    output_fname = 'itermapper_9Aug2023.txt'


    RMS_units = 1e-9

    ## no
    max_RMS = np.inf    
    min_num_RS = 0
    cut_txt = 'no cuts applied'

    ## loose
    # max_RMS = 10    
    # min_num_RS = 3
    # cut_txt = 'light cuts applied:  RMS < 10 ns,  number remote stations >= 3'

    ## tight
    # max_RMS = 3   
    # min_num_RS = 10
    # cut_txt = 'tight cuts applied:  RMS < 3 ns,  number remote stations >= 10'


    header = read_header(itermapper_folder, timeID)
    station_names = header.get_orderedStations() 
    data = header.load_data_as_sources()

    data_collumns = ['unique_id', 'distance_east[m]', 'distance_north[m]', 'distance_up[m]', 'time_from_second[s]', 'RMS[ns]', 'number_RS', 'ref_amplitude', 'max_sqrt_eig[m]']#, 'stations_used']
    data_types =    ['i',          'd',               'd',                  'd',               'd',                    'd',       'i',         'd',               'd']#,                's'+str(len(station_names)) ]


    num_station_used = np.zeros(len(station_names), dtype=float)
    ave_RMS_withStation = np.zeros(len(station_names), dtype=float)
    TMP = np.zeros(len(station_names), dtype=float)

    # TODO:  need spsf data types adn strings!!

    def data_reader():
        global TMP
        global num_station_used
        global ave_RMS_withStation
        for d in data:
            RMS = d.RMS/RMS_units

            if RMS > max_RMS:
                continue
            if d.numRS < min_num_RS:
                continue

            # station_filter = d.get_stationFilter()
            # station_filter_string = ''.join( ['1' if B else '0' for B in station_filter ] )


            # TMP[:] = station_filter
            # num_station_used += TMP
            # TMP *= RMS
            # ave_RMS_withStation += TMP

            out = [d.uniqueID, d.XYZT[0], d.XYZT[1], d.XYZT[2], d.XYZT[3],
                   RMS, d.numRS, d.refAmp, float(d.max_sqrtEig().real)*2]#, station_filter_string ]  ##multiply error by two, becouse i think it is underestimated
            yield out


    notes = {'station_order':station_names}
    comments = ["data imaged by LOFAR impulsive imager", cut_txt, "number_RS : number of remote stations included in fit. Higher is better", 
    "ref_amplitude : amplitude in digitizer volts on a reference antenna", "max_sqrt_eig : location error estimate",
     "stations_used : boolean array showing which stations were used in fit. Order given by station_order"]
    new_SPSF = make_SPSF_from_data(timeID[:-5], data_collumns, data_reader(), data_format=data_types, extra_notes=notes, comments=comments )

    processed_data_folder = utilities.processed_data_dir(timeID)
    out_loc = processed_data_folder +'/' + output_fname
    new_SPSF.write_to_file( out_loc )


    # ave_RMS_withStation /= num_station_used
    # print( num_station_used )
    # print( ave_RMS_withStation )

    # print()
    # print('stations not used', header.stations_to_exclude)

    # plt.bar( station_names, num_station_used )
    # plt.ylabel( 'times station used' )
    # plt.show()

    # plt.bar( station_names, ave_RMS_withStation )
    # plt.ylabel( 'ave RMS' )
    # plt.show()