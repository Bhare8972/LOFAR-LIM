#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt


from LoLIM.utilities import processed_data_dir, v_air
from LoLIM.interferometry import read_interferometric_PSE as R_IPSE
from LoLIM.interferometry import impulsive_imager_tools as inter_tools


## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/public/raw_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
#    timeID = "D20160712T173455.100Z"
#    input_folder = "interferometry_out3_lowAmp_goodDelays"
    timeID = "D20170929T202255.000Z"
#    input_folder = "interferometry_out3"
#    input_folder = "interferometry_out4_tstNORMAL"
#    input_folder = "interferometry_out4_tstS2abs"
    input_folder = "interferometry_out4"
    
    unique_ID = 55760
    block_help = [557]
    
#    image_widths = [20,10,40]
    image_widths = [30,30,120]
    max_pixel_size = 0.2
    
    num_threads = 4
    
    ### open data ###
    processed_data_folder = processed_data_dir(timeID)
    data_dir = processed_data_folder + "/" + input_folder
    interferometry_header, IPSE_list = R_IPSE.load_interferometric_PSE( data_dir, blocks_to_open=block_help )
    
    IPSE_to_image = [IPSE for IPSE in IPSE_list if IPSE.unique_index==unique_ID][0]
#    prefered_antenna_index = IPSE_to_image.prefered_antenna_index
    
    ## prep imager ###
#    antennas = interferometry_header.antenna_data ## all
    antennas = [ant for ant in interferometry_header.antenna_data if ant.station[:2]=='RS'] ## only remote stations
    
    num_antennas = len( antennas )
    antenna_delays = np.zeros(num_antennas)
    antenna_locs = np.zeros( (num_antennas,3) )
    for new_index, ant_info in enumerate(antennas):
#        antenna_delays[ ant_info.antenna_index ] = ant_info.timing_delay
#        antenna_locs[ ant_info.antenna_index ] = ant_info.location
        antenna_delays[ new_index ] = ant_info.timing_delay
        antenna_locs[ new_index ] = ant_info.location
        
        if ant_info.antenna_index == IPSE_to_image.prefered_antenna_index:
            prefered_antenna_index = new_index
#        
    imager = inter_tools.image_data_stage2_absBefore( antenna_locs, antenna_delays, interferometry_header.trace_length_stage2, interferometry_header.upsample_factor )
#    imager = inter_tools.image_data_stage2ABSafter( antenna_locs, antenna_delays, interferometry_header.trace_length_stage2, interferometry_header.upsample_factor )
    imaging_function = imager.intensity_multiprocessed_ABSbefore
            
#    imager =  inter_tools.image_data_sumLog(antenna_locs, antenna_delays, interferometry_header.trace_length_stage2, interferometry_header.upsample_factor )
#    imaging_function = imager.intensity_multiprocessed_sumLog
            
    file_dataset = IPSE_to_image.h5_dataset_opener.get_object()

    for ant_i in range(num_antennas):
        old_ant_index = antennas[ant_i].antenna_index
        
        modeled_dt = -(  np.linalg.norm( antenna_locs[ prefered_antenna_index ]-IPSE_to_image.loc ) - 
                       np.linalg.norm( antenna_locs[ant_i]-IPSE_to_image.loc )   )/v_air
        modeled_dt -= antenna_delays[ prefered_antenna_index ] -  antenna_delays[ant_i]
        
        modeled_dt = int(modeled_dt/5.0E-9)*5.0E-9 ## modulas to 5 ns
        
        read_data = np.array( file_dataset[old_ant_index], dtype=np.complex )
        imager.set_data( read_data, ant_i, -modeled_dt )
    imager.prepare_image( )
    
    
    
    ### prep variables ###
    nX_pixels = int(2*image_widths[0]/max_pixel_size) + 1
    nY_pixels = int(2*image_widths[1]/max_pixel_size) + 1
    nZ_pixels = int(2*image_widths[2]/max_pixel_size) + 1
    
    X_array = np.linspace( IPSE_to_image.loc[0]-image_widths[0], IPSE_to_image.loc[0]+image_widths[0], nX_pixels )
    Y_array = np.linspace( IPSE_to_image.loc[1]-image_widths[1], IPSE_to_image.loc[1]+image_widths[1], nY_pixels )
    Z_array = np.linspace( IPSE_to_image.loc[2]-image_widths[2], IPSE_to_image.loc[2]+image_widths[2], nZ_pixels )
    
    ## image X, Y ##
    print("image XY")
    XYZs = np.zeros( (nX_pixels*nY_pixels, 3) )
    for xi in range(nX_pixels):
        for yj in range(nY_pixels):
            i = nY_pixels*xi + yj
            XYZs[i,0] = X_array[xi]
            XYZs[i,1] = Y_array[yj]
            XYZs[i,2] = IPSE_to_image.loc[2]
            
    image = np.zeros( nX_pixels*nY_pixels )
    imaging_function( XYZs, image, num_threads )
    
    print("plotting XY")
    
    image *= -1
    image = np.swapaxes(image.reshape( nX_pixels, nY_pixels ), 0,1)
    
    plt.pcolormesh(X_array-IPSE_to_image.loc[0], Y_array-IPSE_to_image.loc[1], image, vmin=0.3, vmax=1.0)
    plt.colorbar()
    circle1 = plt.Circle((0.0,0.0), radius=0.25, alpha=.3, color='k')
    plt.gca().add_patch( circle1 )
    plt.show()
    
    
    ## image X, Z ##
    print("image XZ")
    XYZs = np.zeros( (nX_pixels*nZ_pixels, 3) )
    for xi in range(nX_pixels):
        for zk in range(nZ_pixels):
            i = nZ_pixels*xi + zk
            XYZs[i,0] = X_array[xi]
            XYZs[i,1] = IPSE_to_image.loc[1]
            XYZs[i,2] = Z_array[zk]
            
    image = np.zeros( nX_pixels*nZ_pixels )
    imaging_function( XYZs, image, num_threads )
    
    print("plotting XZ")
    
    image *= -1
    image = np.swapaxes(image.reshape( nX_pixels, nZ_pixels ), 0,1)
    
    plt.pcolormesh(X_array-IPSE_to_image.loc[0], Z_array-IPSE_to_image.loc[2], image, vmin=0.3, vmax=1.0)
    plt.colorbar()
    circle1 = plt.Circle((0.0,0.0), radius=0.25, alpha=.3, color='k')
    plt.gca().add_patch( circle1 )
    plt.show()
    
    
    ## image X, Z ##
#    print("image YZ")
#    XYZs = np.zeros( (nY_pixels*nZ_pixels, 3) )
#    for yj in range(nY_pixels):
#        for zk in range(nZ_pixels):
#            i = nZ_pixels*yj + zk
#            XYZs[i,0] = IPSE_to_image.loc[0]
#            XYZs[i,1] = Y_array[yj]
#            XYZs[i,2] = Z_array[zk]
#            
#    image = np.zeros( nY_pixels*nZ_pixels )
#    imaging_function( XYZs, image, num_threads )
#    
#    print("plotting YZ")
#    
#    image *= -1
#    image = np.swapaxes( image.reshape( nY_pixels, nZ_pixels ), 0,1)
#    
#    plt.pcolormesh(Y_array, Z_array, image)
#    plt.colorbar()
#    circle1 = plt.Circle((IPSE_to_image.loc[1],IPSE_to_image.loc[2]), radius=0.25, alpha=.3, color='k')
#    plt.gca().add_patch( circle1 )
#    plt.show()
    
        
    
    