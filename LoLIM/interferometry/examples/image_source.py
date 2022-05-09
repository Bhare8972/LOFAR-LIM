#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt


from LoLIM.utilities import processed_data_dir, v_air
from LoLIM.interferometry import read_interferometric_PSE as R_IPSE
from LoLIM.interferometry import impulsive_imager_tools as inter_tools
from LoLIM.signal_processing import half_hann_window

from scipy.optimize import minimize

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

if __name__ == "__main__":
    timeID = "D20180813T153001.413Z"
    
#    input_folder = "interferometry_out2_TEST"
#    unique_ID = 24070
#    N = 10
    
    
    input_folder = "interferometry_out2"
#    unique_ID = 240700
    unique_ID = 240783
    N = 100
    
    block_help = [int(unique_ID/N)]
    
    image_widths = [100,100,100]
#    image_widths = [30,30,120]
    max_pixel_size = 0.2
    
    num_threads = 4
    
    ### open data ###
    processed_data_folder = processed_data_dir(timeID)
    data_dir = processed_data_folder + "/" + input_folder
    interferometry_header, IPSE_list = R_IPSE.load_interferometric_PSE( data_dir, blocks_to_open=block_help )
    
    IPSE_to_image = [IPSE for IPSE in IPSE_list if IPSE.unique_index==unique_ID][0]
    print("intensity:", IPSE_to_image.intensity)
    print("  loc:", IPSE_to_image.loc)

#    prefered_antenna_index = IPSE_to_image.prefered_antenna_index
    
    ## prep imager ###
#    antennas = interferometry_header.antenna_data ## all
    antennas = [ant for ant in interferometry_header.antenna_data if interferometry_header.use_core_stations_S2 or np.isfinite( ant.no_core_ant_i ) ] 
    
    num_antennas = len( antennas )
    antenna_delays = np.zeros(num_antennas)
    antenna_locs = np.zeros( (num_antennas,3) )
    for ant_info in antennas:
        
        index = ant_info.no_core_ant_i
        if interferometry_header.use_core_stations_S2:
            index = ant_info.with_core_ant_i
        
        antenna_delays[ index ] = ant_info.timing_delay
        antenna_locs[ index ] = ant_info.location
        
#        if ant_info.antenna_index == IPSE_to_image.prefered_antenna_index:
#            prefered_antenna_index = index
        
    pref_ant_loc = None
    pref_ant_delay = None
    for ant in interferometry_header.antenna_data:
        if ant.with_core_ant_i == IPSE_to_image.prefered_antenna_index:
            pref_ant_loc = np.array( ant.location )
            pref_ant_delay = ant.timing_delay
        
        
    imager = inter_tools.image_data_stage2_absBefore( antenna_locs, antenna_delays, interferometry_header.trace_length_stage2, interferometry_header.upsample_factor )
#    imager = inter_tools.image_data_stage2ABSafter( antenna_locs, antenna_delays, interferometry_header.trace_length_stage2, interferometry_header.upsample_factor )
    imaging_function = imager.intensity_multiprocessed_ABSbefore
            
#    imager =  inter_tools.image_data_sumLog(antenna_locs, antenna_delays, interferometry_header.trace_length_stage2, interferometry_header.upsample_factor )
#    imaging_function = imager.intensity_multiprocessed_sumLog
    
    
    file_dataset = IPSE_to_image.h5_dataset_opener.get_object()
    stage_2_window = half_hann_window(interferometry_header.pulse_length, interferometry_header.hann_window_fraction)
    for ant_i in range(num_antennas):
        old_ant_index = antennas[ant_i].antenna_index
        
        modeled_dt = -(  np.linalg.norm( pref_ant_loc - IPSE_to_image.loc ) - 
                       np.linalg.norm( antenna_locs[ant_i]-IPSE_to_image.loc )   )/v_air
        A =   modeled_dt 
        modeled_dt -= pref_ant_delay -  antenna_delays[ant_i]
        modeled_dt /= 5.0E-9
        modeled_dt += IPSE_to_image.peak_index
        modeled_dt = -int(modeled_dt)*5.0E-9
        
        read_data = np.array( file_dataset[old_ant_index], dtype=np.complex )
        read_data *= stage_2_window
        
        imager.set_data( read_data, ant_i, modeled_dt )
    imager.prepare_image( )
    
    
    ### prep variables ###
    nX_pixels = int(2*image_widths[0]/max_pixel_size) + 1
    nY_pixels = int(2*image_widths[1]/max_pixel_size) + 1
    nZ_pixels = int(2*image_widths[2]/max_pixel_size) + 1
    
    X_array = np.linspace( IPSE_to_image.loc[0]-image_widths[0], IPSE_to_image.loc[0]+image_widths[0], nX_pixels )
    Y_array = np.linspace( IPSE_to_image.loc[1]-image_widths[1], IPSE_to_image.loc[1]+image_widths[1], nY_pixels )
    Z_array = np.linspace( IPSE_to_image.loc[2]-image_widths[2], IPSE_to_image.loc[2]+image_widths[2], nZ_pixels )
    
    ## image just X
    print("image X")
    XYZs = np.zeros( (nX_pixels, 3) )
    for xi in range(nX_pixels):
        XYZs[xi,0] = X_array[xi]
        XYZs[xi,1] = IPSE_to_image.loc[1]
        XYZs[xi,2] = IPSE_to_image.loc[2]
            
    image = np.zeros( nX_pixels )
    imaging_function( XYZs, image, num_threads )
        
    print("plotting X")
    XYZs[:,0] -= IPSE_to_image.loc[0]
    image *= -1
    plt.plot(XYZs[:,0], image)
    
    ## calculate the variance
    step = 5
    values = []
    for X in [-step, 0.0, step]:
        I = -imager.intensity_ABSbefore( np.array([IPSE_to_image.loc[0]+X,IPSE_to_image.loc[1],IPSE_to_image.loc[2]] ) )
        values.append( -np.log( I ) )
    
    D = ( values[2]-2*values[1]+values[0] ) /(step*step )
    var = 1.0/D
    intensity = -imager.intensity_ABSbefore( np.array([IPSE_to_image.loc[0],IPSE_to_image.loc[1],IPSE_to_image.loc[2]] ) )
    print( var,intensity )
    def G(X):
        return intensity*np.exp(-X*X*D*0.5)
    plt.plot(XYZs[:,0], G(XYZs[:,0]), 'k')
    
    
    plt.show()
    
    
    ## image just Y
    print("image Y")
    XYZs = np.zeros( (nY_pixels, 3) )
    for i in range(nY_pixels):
        XYZs[i,0] = IPSE_to_image.loc[0]
        XYZs[i,1] = Y_array[i]
        XYZs[i,2] = IPSE_to_image.loc[2]
            
    image = np.zeros( nY_pixels )
    imaging_function( XYZs, image, num_threads )
    
    print("plotting Y")
    XYZs[:,1] -= IPSE_to_image.loc[1]
    image *= -1
    
#    for XYZ, I in zip(XYZs, image):
#        print(XYZ[1], I)
    plt.plot(XYZs[:,1], image)
    
    ## calculate the variance
    step = 5
    values = []
    for X in [-step, 0.0, step]:
        I = -imager.intensity_ABSbefore( np.array([IPSE_to_image.loc[0],IPSE_to_image.loc[1]+X,IPSE_to_image.loc[2]] ) )
        values.append( -np.log( I ) )
    
    D = ( values[2]-2*values[1]+values[0] ) /(step*step )
    var = 1.0/D
    intensity = -imager.intensity_ABSbefore( np.array([IPSE_to_image.loc[0],IPSE_to_image.loc[1],IPSE_to_image.loc[2]] ) )
    print( var,intensity )
    def G(X):
        return intensity*np.exp(-X*X*D*0.5)
    plt.plot(XYZs[:,1], G(XYZs[:,1]), 'k')
    
    plt.show()
    
    ## image just Z
    print("image Z")
    XYZs = np.zeros( (nZ_pixels, 3) )
    for i in range(nZ_pixels):
        XYZs[i,0] = IPSE_to_image.loc[0]
        XYZs[i,1] = IPSE_to_image.loc[1]
        XYZs[i,2] = Z_array[i]
            
    image = np.zeros( nZ_pixels )
    imaging_function( XYZs, image, num_threads )
    
    print("plotting Z")
    XYZs[:,2] -= IPSE_to_image.loc[2]
    image *= -1
    plt.plot(XYZs[:,2], image)
    
    ## calculate the variance
    step = 5
    values = []
    for X in [-step, 0.0, step]:
        I = -imager.intensity_ABSbefore( np.array([IPSE_to_image.loc[0],IPSE_to_image.loc[1],IPSE_to_image.loc[2]+X] ) )
        values.append( -np.log( I ) )
    
    D = ( values[2]-2*values[1]+values[0] ) /(step*step )
    print("ARG", values)
    var = 1.0/D
    intensity = -imager.intensity_ABSbefore( np.array([IPSE_to_image.loc[0],IPSE_to_image.loc[1],IPSE_to_image.loc[2]] ) )
    print( var,intensity )
    def G(X):
        return intensity*np.exp(-X*X*D*0.5)
    plt.plot(XYZs[:,2], G(XYZs[:,2]), 'k')
    
    plt.show()
    
    quit()
    
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
    
        
    
    