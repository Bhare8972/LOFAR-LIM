#!/usr/bin/env python3

from LoLIM.IO.SPSF_readwrite import make_SPSF_from_data
from LoLIM.iterativeMapper.mapper_header import read_header


from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = "D20190424T194432.504Z"
    itermapper_folder = 'iterMapper_TotalCal_oddAnts'
    output_fname = 'all_sources.txt'


    max_RMS = 10
    RMS_units = 1e-9
    min_num_RS = 3


    header = read_header(itermapper_folder, timeID)
    data = header.load_data_as_sources()

    data_collumns = ['unique_id', 'distance_east', 'distance_north', 'distance_up', 'time_from_second', 'RMS', 'number_RS', 'ref_amplitude', 'number_AntsThrown', 'max_sqrt_eig']


    def data_reader():
        for d in data:
            RMS = d.RMS/RMS_units

            if RMS > max_RMS:
                continue
            if d.numRS < min_num_RS:
                continue

            out = [d.uniqueID, d.XYZT[0], d.XYZT[1], d.XYZT[2], d.XYZT[3],
                   RMS, d.numRS, d.refAmp, d.numThrows, float(d.max_sqrtEig()) ]
            yield out

    new_SPSF = make_SPSF_from_data(timeID, data_collumns, data_reader() )

    processed_data_folder = utilities.processed_data_dir(timeID)
    out_loc = processed_data_folder +'/' + output_fname
    new_SPSF.write_to_file( out_loc )