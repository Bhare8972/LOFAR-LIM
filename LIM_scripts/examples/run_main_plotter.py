#!/usr/bin/env python3

from LoLIM.main_plotter import *
from LoLIM.interferometry import read_interferometric_PSE as R_IPSE

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__=="__main__":
    
    #### make color maps and coordinate systems ####
    cmap = gen_cmap('plasma', 0.0, 0.8)
    coordinate_system = typical_transform([0.0,0.0,0.0], 0.0)
    
    
    
#    timeID = "D20160712T173455.100Z"
#    input_folder = "interferometry_out3_lowAmp_goodDelays"#_complexSum"
    timeID = "D20170929T202255.000Z"
#    input_folder = "interferometry_out3"
#    input_folder = "interferometry_out4_tstNORMAL"
#    input_folder = "interferometry_out4_tstS2abs"
#    input_folder = "interferometry_out_S2abs"
#    input_folder = "interferometry_out4_tstS2normabsBefore"
#    input_folder = "interferometry_out4_tstS2normabsBefore_noCore2"
#    input_folder = "interferometry_out4_sumLog"
    input_folder = "interferometry_out4"
#    input_folder = "interferometry_out4_noRemSaturation"
#    input_folder = "interferometry_out4_no_erase"
#    input_folder = "interferometry_out4_PrefStatRS306"
    
    processed_data_folder = processed_data_dir(timeID)
    data_dir = processed_data_folder + "/" + input_folder
    
    
    #### make the widget ####
    qApp = QtWidgets.QApplication(sys.argv)
    plotter = Active3DPlotter( coordinate_system )
    plotter.setWindowTitle("LOFAR-LIM data viewer")
    plotter.show()
    
    interferometry_header, IPSE = R_IPSE.load_interferometric_PSE( data_dir )#, blocks_to_open=[338,339,340,341] )
#    IPSE = R_IPSE.filter_IPSE(IPSE, [[-15340,-15240], [10330,10350], [4800,4925], [1.229,1.245] ])
    
    IPSE_dataset = DataSet_interferometricPointSources(IPSE, marker='s', marker_size=5, color_mode="time", name="PSE", cmap=cmap)
    plotter.add_dataset( IPSE_dataset )
    
    
    
    qApp.exec_()