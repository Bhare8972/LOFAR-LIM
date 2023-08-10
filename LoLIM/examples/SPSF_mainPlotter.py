#!/usr/bin/env python3

import numpy as np

from LoLIM.main_plotter import *
#from LoLIM.interferometry import read_interferometric_PSE as R_IPSE

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__=="__main__":
    timeID = "D20190424T194432.504Z"


    #### make color maps and coordinate systems ####
#    cmap = gen_cmap('plasma', 0.0, 0.8)
    cmap = gen_olaf_cmap()
    coordinate_systemA = typical_transform([0.0,0.0,0.0], 1.153)
    # coordinate_systemB = AzEl_transform([0.0,0.0,0.0], 1.153)
    
    
    #### make the widget ####
    plotter = Active3DPlotter( [coordinate_systemA] )
    plotter.show()
    
    
    ## LOAD the data
    from LoLIM.IO.SPSF_readwrite import pointSource_data
    data_folder = utilities.processed_data_dir(timeID)     ##only need for next line. Uncomment if not needed
    input_fname =  data_folder + '/itermapper_oddAnts.txt' ## this defines where the data file is
    SPSF_file = pointSource_data(input_fname)
    
    name = '19A1_flash'
    
    new_dataset = SPSF_to_DataSet(SPSF_file, name, cmap, marker='s', marker_size=5, color_mode='time')
    plotter.add_dataset( new_dataset )




    ## RUN!
    plotter.qApp.exec_()
