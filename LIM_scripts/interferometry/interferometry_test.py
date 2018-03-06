#!/usr/bin/env python3

import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy import fftpack

from scipy.optimize import least_squares, minimize

from LoLIM.utilities import v_air, RTD, processed_data_dir
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName
from LoLIM.findRFI import FindRFI, window_and_filter
from LoLIM.IO.metadata import getClockCorrections
from LoLIM.read_pulse_data import read_antenna_delays, read_station_delays
from LoLIM.signal_processing import half_hann_window, correlateFFT, parabolic_fit

#### some algorithms for choosing pairs of antennas ####
def pairData_NumAntPerStat(input_files, num_ant_per_stat=1):
    """return the antenna-pair data for choosing one antenna per station"""
    num_stations = len(input_files)
    
    antennas = []
    for file_index in range(num_stations):
        for x in range(num_ant_per_stat):
            antennas.append( [file_index,x*2] )
    
    num_antennas = len( antennas )
    num_pairs = int( num_antennas*(num_antennas-1)*0.5 )
    pairs = np.zeros((num_pairs,2), dtype=np.int64)
    
    pair_i = 0
    for i in range(num_antennas):
        for j in range(num_antennas):
            if j<=i:
                continue
            
            pairs[pair_i,0] = i
            pairs[pair_i,1] = j
            
            pair_i += 1
            
    return np.array(antennas, dtype=int), pairs

def find_max_duration(antenna_locations, bounding_box ):
    class max_duration_finder:
        def __init__(self, loc1, loc2, center):
            self.loc1 = loc1
            self.loc2 = loc2
            self.center = center
            self.time_detect_center = np.linalg.norm(self.center-self.loc2) - np.linalg.norm(self.center-self.loc1)
            
        def __call__(self, source_position):
            time_detect_source = np.linalg.norm(source_position-self.loc2) - np.linalg.norm(source_position-self.loc1)
            
            return -np.abs(self.time_detect_center - time_detect_source)/v_air
        
        
        
    center = np.average(bounding_box, axis=-1)
    
    best = 0
    best_i = None
    best_j = None
    best_position = None
    for ant_i in range(len(antenna_locations)):
        for ant_j in range(ant_i+1,len(antenna_locations)):
            
            minimizer = max_duration_finder(antenna_locations[ant_i], antenna_locations[ant_j], center)
        
            RES = minimize(minimizer, center, bounds=bounding_box, method='TNC')
            
            if np.abs(RES.fun) > best:
                best = np.abs(RES.fun)
                best_i = ant_i
                best_j = ant_j
                best_position = RES.x
                
    return best, (best_i, best_j, best_position, center)


class Inertf_TOA_minimizer:
    def __init__(self, antenna_locations, ant_dt):
        self.ant_locs = antenna_locations
        self.ant_dx = ant_dt
        
        self.num_antennas = len( self.ant_locs )
        
        self.residual_tmp = np.empty( len(self.ant_dx), np.double )
        self.jac_tmp = np.empty( (len(self.ant_dx),3) , np.double )
        
    def residuals(self, source_loc):
        pair_i = 0
        for ant_a in range( self.num_antennas ):
            for ant_b in range(ant_a+1, self.num_antennas ):
                
                loc_A = self.ant_locs[ ant_a ]
                loc_B = self.ant_locs[ ant_b ]
                
                model_dx = np.linalg.norm(loc_A-source_loc) - np.linalg.norm(loc_B-source_loc)
                self.residual_tmp[ pair_i ] = model_dx*(1.0/v_air) - self.ant_dx[ pair_i ]
                
                pair_i += 1
                
        return np.array( self.residual_tmp )
    
    def jacobian(self, source_loc):
        pair_i = 0
        for ant_a in range( self.num_antennas ):
            for ant_b in range(ant_a+1, self.num_antennas ):
                
                loc_A = self.ant_locs[ ant_a ]
                loc_B = self.ant_locs[ ant_b ]
                
                dA = source_loc-loc_A
                dA /= np.linalg.norm( dA )
                self.jac_tmp[pair_i, :] = dA
                
                dB = source_loc-loc_B
                dB /= np.linalg.norm( dB )
                self.jac_tmp[pair_i, :] -= dB
                
                pair_i += 1
                
        return np.array( self.jac_tmp )


class TOA_minimizer:
    def __init__(self, antenna_locations, ant_dt):
        self.ant_locs = antenna_locations
        self.ant_dt = ant_dt
        
        self.num_antennas = len( self.ant_locs )
        
        self.residual_tmp = np.empty( len(self.ant_dt), np.double )
        self.jacobian_workspace = np.empty( (len(self.ant_dt),4) , np.double )
        
    def residuals(self, source_loc):
        source_loc[2] = np.abs(source_loc[2])
        for ant_a in range( self.num_antennas ):
                
            loc_A = self.ant_locs[ ant_a ]
            
            model_dt = source_loc[3] + np.linalg.norm(loc_A-source_loc[0:3])/v_air 
            self.residual_tmp[ ant_a ] = model_dt - self.ant_dt[ ant_a ]
                
        return np.array( self.residual_tmp )
        
#        self.residual_tmp[:] = self.ant_dt 
#        self.residual_tmp[:] -= source_loc[3]
#        self.residual_tmp[:] *= v_air
#        self.residual_tmp[:] *= self.residual_tmp[:]
#        
#        R2 = self.ant_locs - source_loc[0:3]
#        R2 *= R2
#        self.residual_tmp[:] -= R2[:,0]
#        self.residual_tmp[:] -= R2[:,1]
#        self.residual_tmp[:] -= R2[:,2]
#        
##        self.residual_tmp[ np.logical_not(np.isfinite(self.residual_tmp)) ] = 0.0
#        
#        return np.array( self.residual_tmp )
        
    
    def jacobian(self, source_loc):
        source_loc[2] = np.abs(source_loc[2])
        for ant_a in range( self.num_antennas ):
                
            loc_A = self.ant_locs[ ant_a ]
            
            dA = source_loc[0:3] - loc_A
            dA /= (np.linalg.norm( dA )*v_air)
            self.jacobian_workspace[ant_a, 0:3] = dA
            self.jacobian_workspace[ant_a, 3] = 1
            
            
                
##        return np.array( self.jac_tmp )
#        self.jacobian_workspace[:, 0:3] = source_loc[0:3]
#        self.jacobian_workspace[:, 0:3] -= self.ant_locs
#        self.jacobian_workspace[:, 0:3] *= -2.0
#        
#        self.jacobian_workspace[:, 3] = self.ant_dt
#        self.jacobian_workspace[:, 3] -= source_loc[3]
#        self.jacobian_workspace[:, 3] *= -v_air*v_air*2
        
#        mask = np.logical_not(np.isfinite(self.jacobian_workspace[:, 3])) 
#        self.jacobian_workspace[mask, :] = 0
        
        return np.array( self.jacobian_workspace )
    
    def SSqE(self, XYZT):
        R2 = self.ant_locs - XYZT[0:3]
        R2 *= R2
        
        theory = np.sum(R2, axis=1)
        np.sqrt(theory, out=theory)
        
        theory *= 1.0/v_air
        theory += XYZT[3] - self.ant_dt
        
        print("ave err:", np.average(theory) )
        
        theory *= theory
        
        
#        theory[ np.logical_not(np.isfinite(theory) ) ] = 0.0
        
        return np.sum(theory)
        




class Tfree_TOA_minimizer:
    def __init__(self, antenna_locations, ant_dt):
        self.ant_locs = antenna_locations
        self.ant_dt = ant_dt
        
        self.num_antennas = len( self.ant_locs )
        
        self.residual_tmp = np.empty( len(self.ant_dt), np.double )
        self.jacobian_workspace = np.empty( (len(self.ant_dt),4) , np.double )
        
    def residuals(self, XYZ):
        XYZ[2] = np.abs(XYZ[2])
        for ant_a in range( self.num_antennas ):
                
            loc_A = self.ant_locs[ ant_a ]
            
            model_dt = np.linalg.norm(loc_A-XYZ[0:3])/v_air 
            self.residual_tmp[ ant_a ] = model_dt - self.ant_dt[ ant_a ]
            
        self.residual_tmp -= np.average(self.residual_tmp)
                
        return np.array( self.residual_tmp )
        
#        self.residual_tmp[:] = self.ant_dt 
#        self.residual_tmp[:] -= source_loc[3]
#        self.residual_tmp[:] *= v_air
#        self.residual_tmp[:] *= self.residual_tmp[:]
#        
#        R2 = self.ant_locs - source_loc[0:3]
#        R2 *= R2
#        self.residual_tmp[:] -= R2[:,0]
#        self.residual_tmp[:] -= R2[:,1]
#        self.residual_tmp[:] -= R2[:,2]
#        
##        self.residual_tmp[ np.logical_not(np.isfinite(self.residual_tmp)) ] = 0.0
#        
#        return np.array( self.residual_tmp )
        
    
#    def jacobian(self, source_loc):
#        source_loc[2] = np.abs(source_loc[2])
#        for ant_a in range( self.num_antennas ):
#                
#            loc_A = self.ant_locs[ ant_a ]
#            
#            dA = source_loc[0:3] - loc_A
#            dA /= (np.linalg.norm( dA )*v_air)
#            self.jacobian_workspace[ant_a, 0:3] = dA
#            self.jacobian_workspace[ant_a, 3] = 1
            
            
                
##        return np.array( self.jac_tmp )
#        self.jacobian_workspace[:, 0:3] = source_loc[0:3]
#        self.jacobian_workspace[:, 0:3] -= self.ant_locs
#        self.jacobian_workspace[:, 0:3] *= -2.0
#        
#        self.jacobian_workspace[:, 3] = self.ant_dt
#        self.jacobian_workspace[:, 3] -= source_loc[3]
#        self.jacobian_workspace[:, 3] *= -v_air*v_air*2
        
#        mask = np.logical_not(np.isfinite(self.jacobian_workspace[:, 3])) 
#        self.jacobian_workspace[mask, :] = 0
        
        return np.array( self.jacobian_workspace )
    
    def SSqE(self, XYZ):
        R2 = self.ant_locs - XYZ[0:3]
        R2 *= R2
        
        theory = np.sum(R2, axis=1)
        np.sqrt(theory, out=theory)
        
        theory *= 1.0/v_air
        theory -= self.ant_dt
        
        theory -= np.average(theory)
        
        theory *= theory
        
        
#        theory[ np.logical_not(np.isfinite(theory) ) ] = 0.0
        
        return np.sum(theory)


if __name__ == "__main__":
    
    timeID = "D20160712T173455.100Z"
    stations_to_exclude = []
    
    
    station_delays = "station_delays_4.txt"
    additional_antenna_delays = "ant_delays.txt"
    
    do_RFI_filtering = False
    
    block_size = 2**16
    
    block_index = int( (3.819/5.0E-9) )
    data_index = int( (2**16)*0.2 ) + 700 + 1000
    
    findRFI_initial_block = 5
    findRFI_num_blocks = 20
    findRFI_max_blocks = 100
    
    
    bounding_box = np.array([[28000.0,36000.0], [23000.0,30000.0], [2000.0,7000.0]], dtype=np.double)
    
    
    hann_window_fraction = 0.1 ## the window length will be increased by this fraction
    pulse_length = 300.0E-9 ## this is multiplied by 2 and added to the wing-time needed
    
    upsample_factor = 8
    
    
    
    #### open data files and find RFI ####
    raw_fpaths = filePaths_by_stationName(timeID)
    station_names = []
    input_files = []
    RFI_filters = []
    CS002_index = None
    for station, fpaths  in raw_fpaths.items():
        if station not in stations_to_exclude:
            print("opening", station)
            station_names.append( station )
            
            if station=='CS002':
                CS002_index = len(station_names)-1
            
            input_files.append( MultiFile_Dal1( fpaths, force_metadata_ant_pos=True ) )
            
            RFI_result = None
            if do_RFI_filtering:
                RFI_result = FindRFI(input_files[-1], block_size, findRFI_initial_block, findRFI_num_blocks, findRFI_max_blocks, verbose=False, figure_location=None)
            RFI_filters.append( window_and_filter(block_size, find_RFI=RFI_result) )
            
    print("Data opened and FindRFI completed.")
            
            
            
            
    
    #### choose antenna pairs ####
    boundingBox_center = np.average(bounding_box, axis=-1)
    antennas_to_use, antenna_pairs = pairData_NumAntPerStat(input_files, 1 )
    num_antennas = len(antennas_to_use)
    
    
    #### get antenna locations and delays ####
    antenna_locations = np.zeros((num_antennas, 3), dtype=np.double)
    antenna_delays = np.zeros(num_antennas, dtype=np.double)
    antenna_data_offsets = np.zeros(num_antennas, dtype=np.long)
    
    clock_corrections = getClockCorrections()
    CS002_correction = -clock_corrections["CS002"] - input_files[CS002_index].get_nominal_sample_number()*5.0E-9 ## wierd sign. But is just to correct for previous definiitions
    
    processed_data_dir = processed_data_dir(timeID)
    station_timing_offsets = read_station_delays( processed_data_dir+'/'+station_delays )
    
    
    extra_ant_delays = read_antenna_delays( processed_data_dir+'/'+additional_antenna_delays )
    
    for ant_i, (station_i, station_ant_i) in enumerate(antennas_to_use):
        data_file = input_files[station_i]
        station = station_names[station_i]
        
        ant_name = data_file.get_antenna_names()[ station_ant_i ]
        antenna_locations[ant_i] = data_file.get_LOFAR_centered_positions()[ station_ant_i ]
        antenna_delays[ant_i] = data_file.get_timing_callibration_delays()[ station_ant_i ]
        
        ## account for station timing offsets
        antenna_delays[ant_i]  += station_timing_offsets[station] +  (-clock_corrections[station] - data_file.get_nominal_sample_number()*5.0E-9) - CS002_correction 
        
        ## add additional timing delays
        ##note this only works for even antennas!
        if ant_name in extra_ant_delays:
            antenna_delays[ant_i] += extra_ant_delays[ ant_name ][0]
            
            
        ## now we accound for distance to the source
        travel_time = np.linalg.norm( antenna_locations[ant_i] - boundingBox_center )/v_air
        antenna_data_offsets[ant_i] = int(travel_time/5.0E-9)
            
        
        
        #### now adjust the data offsets and antenna delays so they are consistent
            
        ## first we amount to adjust the offset by time delays mod the sampling time
        offset_adjust = int( antenna_delays[ant_i]/5.0E-9 ) ##this needs to be added to offsets and subtracted from delays
        
        ## then we can adjust the delays accounting for the data offset
        antenna_delays[ant_i] -= antenna_data_offsets[ant_i]*5.0E-9 
        
        ##now we finally account for large time delays
        antenna_data_offsets[ant_i] += offset_adjust
        antenna_delays[ant_i] -= offset_adjust*5.0E-9
            
            
        
    #### calculate trace length needed  ####
    data_WingTime_size, info = find_max_duration(antenna_locations, bounding_box)
    print()
    print("wing-time size (us):", data_WingTime_size)
    print("        antennas:", antenna_locations[info[0]], antenna_locations[info[1]])
    print("        points:", info[2], info[3])
    print()
    
    ##minimum length needed
    trace_length =  int( (pulse_length+data_WingTime_size)*2*(1.0+hann_window_fraction)/5.0E-9 )
    ## bump up to next power of 2
    trace_length = 2**( int(np.log2( trace_length )) + 1 )
    
    print("using traces of length:", trace_length, "points.", trace_length*5.0E-3, "us")
    
    
    antenna_times = np.empty(num_antennas, dtype=np.double)
    
    
    #### open and filter data###
    data = np.empty((num_antennas,trace_length), dtype=np.complex)
    tmp = np.empty( block_size, dtype=np.double )
    current_height = 0
    
#    synth_data = np.array([ 26785.69474169,  24299.92250268,     39.56884082])
    
    for ant_i, (station_i, station_ant_i) in enumerate(antennas_to_use):
        data_file = input_files[station_i]
        RFI_filter = RFI_filters[station_i]
        
        offset = antenna_data_offsets[ant_i]
        tmp[:] = data_file.get_data(block_index+offset, block_size, antenna_index=station_ant_i) ## get the data. accounting for the offsets calculated earlier
            
        filtered = RFI_filter.filter( tmp, whiten=False )
        data[ant_i] = filtered[data_index:data_index+trace_length] ## filter out RFI, then select the bit we want
        HE = np.abs( data[ant_i] )
        HE_max = np.max( HE )
        
        T = np.argmax( HE )
        parabola_peak = parabolic_fit(HE, T, 5)
        T = float( parabola_peak.peak_index )
        antenna_times[ant_i] = T*5.0E-9 - antenna_delays[ant_i]
        
        
        plt.plot(np.real(data[ant_i])+current_height)
        plt.plot(np.abs(data[ant_i])+current_height)
        
        plt.plot( [T,T],[current_height, current_height+HE_max], 'r-o')
        
        current_height += HE_max
        
    
    
        
    plt.show()
    print('Data loaded and filtered')
    
    guess = np.random.rand(3)*(bounding_box[:,1]-bounding_box[:,0]) + bounding_box[:,0]
    
    fitter = Tfree_TOA_minimizer(antenna_locations, antenna_times)
    res = minimize( fitter.SSqE, boundingBox_center, method="Nelder-Mead", options={"xatol":1.0E-20, 'maxiter':10000} )
    
    print(res)
    
    print( "RMS", np.sqrt( fitter.SSqE(res.x)/len(antenna_times) ) )
    
    print(guess)
    quit()
    
    
    guess = np.append(boundingBox_center, np.average(antenna_times))
    
    fitter = TOA_minimizer(antenna_locations, antenna_times)
    res = least_squares( fitter.residuals, guess, jac=fitter.jacobian, method='lm', ftol=1.0E-15, xtol=1.0E-15, gtol=1.0E-15, x_scale='jac' )
#    res = least_squares( fitter.residuals, guess, jac="3-point", method='trf', ftol=1.0E-15, xtol=1.0E-15, gtol=1.0E-15, x_scale='jac'  )
    
    print(res)
    
    print( "RMS", np.sqrt( fitter.SSqE(res.x)/len(antenna_times) ) )
    
    quit()
    
    
    #### apply signal processing ####
    data *= half_hann_window(trace_length, hann_window_fraction)
    
#    plt.plot(np.abs(data[2]))
#    plt.plot(np.abs(data[3]), 'r')
#    plt.show()
    
    data = fftpack.fft(data, n=trace_length*2, axis=-1)



    ## use cross correlation to get time of peaks
    ant_dt = np.empty( int(num_antennas*(num_antennas-1)/2), dtype=np.double  )
    Xcorrelation = np.empty(upsample_factor*trace_length*2, dtype=np.complex)
    pair_i = 0
    for ant_a in range( num_antennas ):
        for ant_b in range(ant_a+1, num_antennas ):
            
            correlateFFT(data[ ant_a ], data[ ant_b ], Xcorrelation  )
    
            P = np.argmax(Xcorrelation)
                
            
            parabola_peak = parabolic_fit(np.real(Xcorrelation), P, 5)
            P = float( parabola_peak.peak_index )
    
            if P > len(Xcorrelation)/2:
                P -= len(Xcorrelation)
                
            P *= 5.0E-9/upsample_factor
       
            ant_dt[ pair_i ] = P - (antenna_delays[ ant_a ] - antenna_delays[ ant_b ])
            
            pair_i += 1
    
    fitter = Inertf_TOA_minimizer(antenna_locations, ant_dt)
    
    res = least_squares( fitter.residuals, boundingBox_center, jac='3-point', bounds=np.swapaxes(bounding_box, 0 ,1), ftol=1.0E-15, xtol=1.0E-15, gtol=1.0E-15, x_scale='jac'  )
    
    print( res )
    
    
    
    
    
    
    ##positive means A comes AFTER B
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            