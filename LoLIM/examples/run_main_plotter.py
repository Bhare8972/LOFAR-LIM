#!/usr/bin/env python3

import numpy as np

from LoLIM.main_plotter import *
from LoLIM.interferometry import read_interferometric_PSE as R_IPSE
from LoLIM import read_LMA as LMA

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/home/brian/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__=="__main__":
    
    #### make color maps and coordinate systems ####
#    cmap = gen_cmap('plasma', 0.0, 0.8)
    cmap = gen_olaf_cmap()
    coordinate_systemA = typical_transform([0.0,0.0,0.0], 1.153)
#    coordinate_system = camera_transform([0.0,0.0,0.0], 1.153)
    coordinate_systemB = AzEl_transform([0.0,0.0,0.0], 1.153)
    
    
    
#    timeID = "D20160712T173455.100Z"
#    input_folder = "interferometry_out3_lowAmp_goodDelays"#_complexSum"
    
    
#    input_folder = "interferometry_out3"
#    input_folder = "interferometry_out4_tstNORMAL"
#    input_folder = "interferometry_out4_tstS2abs"
#    input_folder = "interferometry_out_S2abs"
#    input_folder = "interferometry_out4_tstS2normabsBefore"
#    input_folder = "interferometry_out4_tstS2normabsBefore_noCore2"
#    input_folder = "interferometry_out4_sumLog"
#    input_folder = "interferometry_out4_noRemSaturation"
#    input_folder = "interferometry_out4_no_erase"
#    input_folder = "interferometry_out4_PrefStatRS306"
    
    
    
    timeID = "D20170929T202255.000Z"
    input_folder = "interferometry_out4" ## use this one
    
#    timeID = "D20180813T153001.413Z"
#    input_folder = "interferometry_out_fastTest2"
#    input_folder = "interferometry_out_WO41"
#    input_folder = "interferometry_out2"
    
    
    
    
#    timeID = "D20180921T194259.023Z"
#    input_folder = "interferometry_out"
    
    processed_data_folder = processed_data_dir(timeID)
    data_dir = processed_data_folder + "/" + input_folder
    
    
    #### make the widget ####
    plotter = Active3DPlotter( [coordinate_systemA, coordinate_systemB] )
#    plotter.setWindowTitle("LOFAR-LIM data viewer")
    plotter.show()
    
    if True:
        interferometry_header, IPSE = R_IPSE.load_interferometric_PSE( data_dir )#, blocks_to_open=[338,339,340,341] )
#        IPSE = R_IPSE.filter_IPSE(IPSE, [[-15340,-15240], [10330,10350], [4800,4925], [1.229,1.245] ])
        
        IPSE_dataset = IPSE_to_DataSet(IPSE, "IPSE", cmap)
#        IPSE_dataset = DataSet_interferometricPointSources(IPSE, marker='s', marker_size=5, color_mode="time", name="PSE", cmap=cmap)
        plotter.add_dataset( IPSE_dataset )
    
    
    
    from LoLIM.iterativeMapper.iterative_mapper import read_header
    
#    header = read_header( 'iterMapper_20_CS002',  'D20170929T202255.000Z')
#    data = header.load_data_as_sources()
#    new_dataset = iterPSE_to_DataSet(data, '20_CS002', cmap)
#    new_dataset.X_offset = -27.0
#    new_dataset.Y_offset = 11.0
#    new_dataset.Z_offset = 10.0
#    plotter.add_dataset( new_dataset )
    
    header = read_header( 'iterMapper_50_CS002',  'D20170929T202255.000Z')
    data = header.load_data_as_sources()
    new_dataset = iterPSE_to_DataSet(data, '50_CS002', cmap)
    #new_dataset.X_offset = -27.0
    #new_dataset.Y_offset = 11.0
    #new_dataset.Z_offset = 10.0
    plotter.add_dataset( new_dataset )
    
    
#    header = read_header( 'iterMapper_100_CS002',  'D20170929T202255.000Z')
#    data = header.load_data_as_sources()
#    new_dataset = iterPSE_to_DataSet(data, '100_CS002', cmap)
#    new_dataset.X_offset = -27.0
#    new_dataset.Y_offset = 11.0
#    new_dataset.Z_offset = 10.0
#    plotter.add_dataset( new_dataset )
#    
#    
#    header = read_header( 'iterMapper_200_CS002',  'D20170929T202255.000Z')
#    data = header.load_data_as_sources()
#    new_dataset = iterPSE_to_DataSet(data, '200_CS002', cmap)
#    new_dataset.X_offset = -27.0
#    new_dataset.Y_offset = 11.0
#    new_dataset.Z_offset = 10.0
#    plotter.add_dataset( new_dataset )

    
    
    
    plotter.qApp.exec_()
