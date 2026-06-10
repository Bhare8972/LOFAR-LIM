#!/usr/bin/env python3

from os import listdir
from os.path import isfile, join

import datetime

import numpy as np

from LoLIM.IO.SPSF_readwrite import make_SPSF_from_data
from LoLIM.version import __version__
from LoLIM.IO import metadata as md

from LoLIM.iterativeMapper.iterative_mapper import read_header

def read_results(read_header, print_progress=True):

    fnames = (join(read_header.tmp_out_data,f) for f in listdir(read_header.tmp_out_data))
    fnames = [f for f in fnames if isfile(f) ]


## time data
    planewave_fit_times = []
    loadData_times = []
    pointsource_fit_times = []
    total_times = []


## pointsource data
    easting = []
    northing = []
    height = []
    time = []
    RMS = []
    RC2 = []
    ref_amp = []
    loc_err = []
    station_mask = []
    num_RS = []

## data reload Dist
    data_reloads = []

    for fi,fname in enumerate(fnames):
        if print_progress:
            print(fi, '/', len(fnames), end='\r')

        with open(fname,'r') as fin:
            fin.readline()

        ### read time data
            dataLine = fin.readline()
            datums = dataLine.split()
            planewave_fit_times.append( int(datums[1]) )
            loadData_times.append( int(datums[3]) )
            pointsource_fit_times.append( int(datums[5]) )
            total_times.append( int(datums[7]) )

        ## read pointsource data
            dataLine = fin.readline()
            datums = dataLine.split()
            num_pointsources = int(datums[1])

            fin.readline()


            tmp_easting = np.empty(num_pointsources, dtype=float)
            tmp_northing = np.empty(num_pointsources, dtype=float)
            tmp_height = np.empty(num_pointsources, dtype=float)
            tmp_time = np.empty(num_pointsources, dtype=float)
            tmp_RMS = np.empty(num_pointsources, dtype=float)
            tmp_RC2 = np.empty(num_pointsources, dtype=float)
            tmp_refAmp = np.empty(num_pointsources, dtype=float)
            tmp_loc_error = np.empty(num_pointsources, dtype=float)
            tmp_mask = np.empty(num_pointsources, dtype=int)
            tmp_numRS = np.empty(num_pointsources, dtype=int)

            for i in range(num_pointsources):
                dataLine = fin.readline()
                datums = dataLine.split()

                tmp_easting[i] = float( datums[1] )
                tmp_northing[i] = float( datums[2] )
                tmp_height[i] = float( datums[3] )
                tmp_time[i] = float( datums[4] )
                tmp_RMS[i] = float( datums[5] )
                tmp_RC2[i] = float( datums[6] )
                tmp_refAmp[i] = float( datums[7] )
                tmp_loc_error[i] = float( datums[8] )
                tmp_mask[i] = int( datums[9] )
                tmp_numRS[i] = int( datums[10] )


            easting = np.append( easting,          tmp_easting )
            northing = np.append( northing,         tmp_northing )
            height = np.append( height,             tmp_height )
            time = np.append( time,                 tmp_time )
            RMS = np.append( RMS,                   tmp_RMS )
            RC2 = np.append( RC2,                   tmp_RC2 )
            ref_amp = np.append( ref_amp,           tmp_refAmp )
            loc_err = np.append( loc_err,           tmp_loc_error )
            station_mask = np.append( station_mask, tmp_mask )
            num_RS = np.append( num_RS,             tmp_numRS )

        ## read loading distribution

            dataLine = fin.readline()
            datums = dataLine.split()

            if datums[0] != 'reloadDt': ## sanity check
                print("ERROR in reading!")
                quit()

            dataLine = fin.readline()
            datums = dataLine.split()
            tmp_reloadDist = [float(d) for d in datums]
            data_reloads = np.append( data_reloads, tmp_reloadDist )


    if print_progress:
        print()
    ret_CPUTimeData = {'planewaveFitTime':planewave_fit_times, 'loadDataTime':loadData_times, 'pointsourceFitTime':pointsource_fit_times, 'totalTime':total_times}

    ret_pointsourceData = {'easting':easting, 'northing':northing, 'height':height, 'time':time, 
    'RMS':RMS, 'RC2':RC2, 'ref_amp':ref_amp, 'loc_err':loc_err, 'station_mask':station_mask, 'num_RS':num_RS}

    return ret_CPUTimeData, ret_pointsourceData, data_reloads






def results_to_SPSF(header_folder, out_location, maxRMS=4e-9, minNumRS=5, print_progress=True):


    header = read_header(header_folder)


    if print_progress:
        print('read data from file')

    _, dataDict, _ = read_results(header, print_progress=print_progress)


    if print_progress:
        print('generate SPSF object')

    easting = dataDict['easting']
    northing = dataDict['northing']
    height = dataDict['height']
    time = dataDict['time']
    RMS = dataDict['RMS']
    ref_amp = dataDict['ref_amp']
    loc_err = dataDict['loc_err']
    num_RS = dataDict['num_RS']


    tranformation_matrix = np.linalg.inv(  md.get_ITRFToLocal_matrix() )
    phase_center = md.ITRFCS002


    def zip_data():
        for i in range(len(easting)):

            if (RMS[i]>maxRMS) or (num_RS[i]<minNumRS):
                continue


            ITRF_XYZ = np.dot( tranformation_matrix, np.array([easting[i], northing[i], height[i]]) )
            ITRF_XYZ += phase_center

            latLonAlt = md.ITRF_to_geoditic(ITRF_XYZ)

            if print_progress and ((i%1000)==0):
                print('source', int(i/1000) , 'k /', int(len(easting)/1000), 'k                      ', end='\r' )

            yield (i, easting[i], northing[i], height[i],  time[i], latLonAlt[0], latLonAlt[1], latLonAlt[2], RMS[i]/(1.0e-9), int(num_RS[i]), ref_amp[i], loc_err[i])


    collumn_names = ( 'unique_ID', 'easting[m]', 'northing[m]', 'up[m]', 'time_from_second[s]', 'latitude', 'longitude', 'altitude[m]', 'RMS[ns]', 'num_RS', 'reference_amplitude', 'location_error_estimate[m]' )
    data_format = (      'i',         'd',         'd',          'd',        'd',                  'd',        'd',        'd',            'd',       'i',       'd',                  'd')
    format_strings = [None,         '.2f',        '.2f',          '.2f',     '.10f',                '.9f',    '.9f',        '.2f',         '.2f',     None,      '.2f',               '.2f'      ]

    comments = [
    'data pre-filtered: RMS<'+str(maxRMS)+" and number participating RS>="+str(minNumRS),
    'NOTE: this is a very loose cut and the data must be cut further!',
    ]

    notes = {
    'Software':['LoLIM', __version__],
    'start_processing_time':[header.headerCreation_time],
    'finish_processing_time':[str( datetime.datetime.now( datetime.timezone.utc ) )],
    }

    SPSF_object = make_SPSF_from_data(timeID=header.timeID, collumn_names=collumn_names, data_iterator=zip_data(), data_format=data_format, extra_notes=notes, comments=comments)
    SPSF_object.max_num_data = len(easting)

    if print_progress:
        print()

    if print_progress:
        print('finally, write to file')

    SPSF_object.write_to_file(out_location, format_strings=format_strings)

    if print_progress:
        print('writing data to SPSF file complete')
            