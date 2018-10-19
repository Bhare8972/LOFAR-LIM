#!/usr/bin/env python3

#python
import time
from os import mkdir
from os.path import isdir, isfile

#external
import numpy as np
from scipy.optimize import least_squares, minimize
from prettytable import PrettyTable
import matplotlib.pyplot as plt

#mine
from utilities import log, processed_data_dir, v_air, SNum_to_SName_dict
from binary_IO import read_long, write_long, write_double_array, write_string, write_double
#from porta_code import code_logger
#from RunningStat import RunningStat

from read_pulse_data import writeTXT_station_delays,read_station_info, curtain_plot_CodeLog

from read_PSE import read_PSE_timeID
from planewave_functions import read_SSPW_timeID_multiDir


class fitting_PSE:
    def __init__(self, PSE):
        self.PSE = PSE
        PSE.load_antenna_data( True )
        
        if PSE.PolE_RMS<PSE.PolO_RMS:
            self.polarity = 0
        else:
            self.polarity = 1
        
        self.initial_loc = PSE.PolE_loc if self.polarity==0 else PSE.PolO_loc
        
        self.SSPW_list = []
        
    def set_loc(self, location):
        self.XYZ_location = location
        
    def set_loc_offset(self, offset):
        self.XYZ_cur_loc = offset+self.XYZ_location

    def add_SSPW(self, new_SSPW):
        self.SSPW_list.append(new_SSPW)

    def fitting_prep(self, station_delay_input_order, stations_to_keep, PSE_ant_locs, SSPW_ant_locs):
        print("PSE", self.PSE.unique_index)
        self.station_delay_input_order = station_delay_input_order
        
        ant_X = []
        ant_Y = []
        ant_Z = []
        pulse_times = []
#        self.station_index_range = {}
        antenna_names = []
        
        for ant_name, ant_info in self.PSE.antenna_data.items():
            station_number = ant_name[0:3]
            sname = SNum_to_SName_dict[ station_number ]
            if sname not in stations_to_keep:
                continue
            
            pt = ant_info.PolE_peak_time if self.polarity==0 else ant_info.PolO_peak_time 
            if not np.isfinite(pt):
                continue
            
            ant_loc = PSE_ant_locs[ ant_name ]
            ant_X.append( ant_loc[0] )
            ant_Y.append( ant_loc[1] )
            ant_Z.append( ant_loc[2] )
            
            pulse_times.append( pt )
            antenna_names.append( ant_name )
            
            
        ### add SSPW
        for SSPW in self.SSPW_list:
            sname = SSPW.sname
            print("  SSPW", SSPW.unique_index, SSPW.sname)
            
            n_ant = 0
            for ant_name, ant_info in SSPW.ant_data.items():
                pt = ant_info.PolE_peak_time if self.polarity==0 else ant_info.PolO_peak_time 
                if not np.isfinite(pt):
                    continue
                
                ant_loc = SSPW_ant_locs[ ant_name ]
                ant_X.append( ant_loc[0] )
                ant_Y.append( ant_loc[1] )
                ant_Z.append( ant_loc[2] )
            
                pulse_times.append( pt )
                antenna_names.append( ant_name )
                n_ant += 1
            print("    num ant:", n_ant)
            
        self.antenna_X = np.array(ant_X)
        self.antenna_Y = np.array(ant_Y)
        self.antenna_Z = np.array(ant_Z)
        self.pulse_times = np.array(pulse_times)
        self.antenna_names = np.array(antenna_names)
        
        sorted_indeces = np.argsort( self.antenna_names )
        self.antenna_X = self.antenna_X[sorted_indeces]
        self.antenna_Y = self.antenna_Y[sorted_indeces]
        self.antenna_Z = self.antenna_Z[sorted_indeces]
        self.pulse_times = self.pulse_times[sorted_indeces]
        self.antenna_names = self.antenna_names[sorted_indeces]
        
        
        self.station_index_range = {}
        current_station = None
        start_index = None
        for idx, ant_name in enumerate(self.antenna_names):
            sname = SNum_to_SName_dict[ ant_name[0:3] ]
            
            if start_index is None:
                start_index = idx
                current_station = sname
            elif sname != current_station:
                self.station_index_range[ current_station ] = [start_index, idx]
                start_index = idx
                current_station = sname
        self.station_index_range[ current_station ] = [start_index, idx+1]
        
        
        self.ordered_station_index_range = [ (self.station_index_range[sname_] if (sname_ in self.station_index_range) else [None,None]) for sname_ in  self.station_delay_input_order]
        
    def add_new_ant_offsets(self, ant_offset_dict):
        for ant_name, offset in ant_offset_dict.items():
            if ant_name in self.antenna_names:
                i = self.antenna_names.index(ant_name)
                self.pulse_times[i] -= offset
            
    def try_location_LS(self, delays, T, out):
        X,Y,Z = self.XYZ_cur_loc
        Z = np.abs(Z)
        
        delta_X_sq = self.antenna_X-X
        delta_Y_sq = self.antenna_Y-Y
        delta_Z_sq = self.antenna_Z-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
            
        out[:] = T - self.pulse_times
        
        ##now account for delays
        for index_range, delay in zip(self.ordered_station_index_range,  delays):
            first,last = index_range
            if first is not None:
                out[first:last] += delay ##note the wierd sign
                
                
        out *= v_air
        out *= out ##this is now delta_t^2 *C^2
        
        out -= delta_X_sq
        out -= delta_Y_sq
        out -= delta_Z_sq
    
    def try_location_JAC(self, delays, T, out_T, out_delays):
        X,Y,Z = self.XYZ_cur_loc
        Z = np.abs(Z)
        
        out_T[:] = T - self.pulse_times
        out_T[:] *= 2*v_air*v_air
        
        delay_i = 0
        for index_range, delay in zip(self.ordered_station_index_range,  delays):
            first,last = index_range
            if first is not None:
                out_T[first:last] += delay*2*v_air*v_air
                out_delays[first:last,delay_i] = out_T[first:last]
            delay_i += 1
            
    def SSqE_fit(self, delays, T):
        X,Y,Z = self.XYZ_cur_loc
        Z = np.abs(Z)
        
        delta_X_sq = self.antenna_X-X
        delta_Y_sq = self.antenna_Y-Y
        delta_Z_sq = self.antenna_Z-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
        distance = delta_X_sq
        distance += delta_Y_sq
        distance += delta_Z_sq
        
        np.sqrt(distance, out=distance)
        distance *= 1.0/v_air
        
        distance += T
        distance -= self.pulse_times
        
        ##now account for delays
        for index_range, delay in zip(self.ordered_station_index_range,  delays):
            first,last = index_range
            if first is not None:
                distance[first:last] += delay ##note the wierd sign
                
        distance *= distance
        return np.sum(distance)
    
    
            
#    def RMS_fit_byStation(self, delays, XYZT_location):
#        X,Y,Z,T = XYZT_location
#        Z = np.abs(Z)
#        
#        delta_X_sq = self.antenna_X-X
#        delta_Y_sq = self.antenna_Y-Y
#        delta_Z_sq = self.antenna_Z-Z
#        
#        delta_X_sq *= delta_X_sq
#        delta_Y_sq *= delta_Y_sq
#        delta_Z_sq *= delta_Z_sq
#        
#        distance = delta_X_sq
#        distance += delta_Y_sq
#        distance += delta_Z_sq
#        
#        np.sqrt(distance, out=distance)
#        distance *= 1.0/v_air
#        
#        distance += T
#        distance -= self.pulse_times
#        
#        ##now account for delays
#        for index_range, delay in zip(self.ordered_station_index_range,  delays):
#            first,last = index_range
#            if first is not None:
#                distance[first:last] += delay ##note the wierd sign
#                
#        distance *= distance
#        
#        ret = {}
#        for sname, index_range in self.station_index_range.items():
#            first,last = index_range
#            
#            ret[sname] = np.sqrt( np.sum(distance[first:last])/float(last-first) )
#        
#        return ret
#    
#    def get_DataTime(self, sname, sorted_antenna_names):
#        out = []
#        
#        index_range = self.station_index_range[sname]
#        if index_range[0] is None:
#            return [np.inf]*len(sorted_antenna_names)
#        first,last = index_range
#        
#        ant_names = self.antenna_names[first:last]
#        data_times = self.pulse_times[first:last]
#            
#        for ant_name in sorted_antenna_names:
#            if ant_name in ant_names:
#                ant_index = np.where(ant_names==ant_name)[0][0]
#                out.append( data_times[ ant_index ] )
#            else:
#                out.append( np.inf )
#        
#        return out
        
#    def get_ModelTime(self, sname, sorted_antenna_names, sdelay):
#        out = []
#        for ant_name in sorted_antenna_names:
#            
#            if ant_name in self.antenna_names:
#                ant_idx = np.where( self.antenna_names==ant_name )[0][0]
#                ant_loc = np.array([ self.antenna_X[ant_idx], self.antenna_Y[ant_idx], self.antenna_Z[ant_idx]])
#                
#                ant_loc -= self.source_location[0:3]
#                model_time = np.linalg.norm( ant_loc )/v_air + self.source_location[3] + sdelay
#                
#                out.append(model_time)
#            else:
#                out.append( np.inf )
#        return out
    
#    
#    def estimate_T(self, delays, XYZT_location, workspace):
#        X,Y,Z,T = XYZT_location
#        Z = np.abs(Z)
#        
#        delta_X_sq = self.antenna_X-X
#        delta_Y_sq = self.antenna_Y-Y
#        delta_Z_sq = self.antenna_Z-Z
#        
#        delta_X_sq *= delta_X_sq
#        delta_Y_sq *= delta_Y_sq
#        delta_Z_sq *= delta_Z_sq
#        
#            
#        workspace[:] = delta_X_sq
#        workspace[:] += delta_Y_sq
#        workspace[:] += delta_Z_sq
#        
#        np.sqrt(workspace, out=workspace)
#        
#        workspace[:] -= self.pulse_times*v_air ## this is now source time
#        
#        ##now account for delays
#        for index_range, delay in zip(self.ordered_station_index_range,  delays):
#            first,last = index_range
#            if first is not None:
#                workspace[first:last] += delay*v_air ##note the wierd sign
#                
#                
#        ave_error = np.average(workspace)
#        return -ave_error/v_air

if __name__=="__main__":
    ##opening data
    timeID = "D20160712T173455.100Z"
    output_folder = "plot_ChiSq_surface"
    
    SSPW_folders = ['SSPW_data']
    PSE_folder = "correlate_SSPW_excludeRS"
    
    
    flash_XY = [25000.0, 30000.0]
    num_Z_points = 30
    num_rho_points = 30
    alt_delta = 1500
    rho_delta = 1000
    
    referance_station = "CS002"
    stations_to_keep = ['CS001', 'CS002', 'CS013', 'CS006', 'CS031', 'CS028', 'CS011', 'CS030', 'CS026', 'CS032', 'CS021', 'CS004', 'CS302']
    stations_to_correlate = ['RS106', 'RS205', 'RS208', 'RS305', 'RS306', 'RS307', 'RS406', 'RS407', 'RS503', 'RS508', 'RS509']   ## to correlate from the SSPW
    
    ## to find delays
    stations_to_find_offsets = stations_to_correlate + stations_to_keep
    
    ## to find delays
    correlation_table =  {
            2:[80334, None  , None  , None , None  , None , None , 205312, 108726, None, None],
            3:[None , 187384, None  , None , 179554, None , 44449, None  , 108728, 17  , None],
            6:[80376, 187416, None  , None , None  , None , 44481, None  , 108764, 63  , 62553],
            7:[80441, 187480, 136608, None , None  , 38255, None , 205419, 108825, 150 , 62622],
#            11:[None , None  , None  , None , None  , None , None , None  , None  , None, None ],
            12:[80479, None  , None  , 29882, 179658, None , 44589, None  , 108865, None, None],
            
            26:[84563, 191298, None  , 33588, 183144, None , None , None  , 112675, None, None ],
#            32:[85073, 191774, 139657, None , None  , None , None , None  , None  , None, None ],
#            34:[85466, 192118, None  , None , 183903, None , None , 210375, 113539, None, None ],
#            36:[85894, None  , None  , None , None  , None , 49557, 210803, 113942, None, None ],
#            37:[85974, 192582, None  , None , None  , None , 49637, None  , 114025, None, None ],
            }

    correlation_table_SSPW_group = { ### since we are loading multiple SSPW files, the SSPW unique IDs are not actually unique...
            2:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            3:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            6:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            7:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            12:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            
            26:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            32:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            34:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            36:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            37:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            }
    
   
    initial_guess = np.array([  2.27641128e-07,   9.64467134e-08,   4.55404079e-08,
        -3.50793896e-08,   3.42436977e-09,  -1.03138733e-07,
        -6.37671269e-08,  -2.41010390e-07,   2.09883079e-08,
        -6.11864971e-07,  -1.34991028e-06,   3.56163761e-09,
        -4.73826308e-09,   2.07081197e-09,  -1.61643521e-08,
         5.21639888e-09,  -7.76217407e-09,  -3.08118519e-09,
         1.21113529e-08,  -4.31124913e-09,   4.71161759e-10,
        -6.40984610e-10,  -1.84263794e-08,   3.12762867e+04,
         2.77384050e+04,   4.00852854e+03,   3.81932687e+00,
         3.13307814e+04,   2.77798365e+04,   3.91782297e+03,
         3.81951468e+00,   3.12829207e+04,   2.80147549e+04,
         3.46423008e+03,   3.82087309e+00,   3.10766065e+04,
         2.81211521e+04,   3.40019650e+03,   3.82266925e+00,
         3.10883662e+04,   2.81595324e+04,   3.28586857e+03,
         3.82359654e+00,   3.11659726e+04,   2.75321778e+04,
         5.11146282e+03,   3.88040054e+00])


    
    if referance_station in stations_to_find_offsets:
        stations_to_find_offsets.remove( referance_station )
    
    #### setup directory variables ####
    processed_data_dir = processed_data_dir(timeID)
    
    data_dir = processed_data_dir + "/" + output_folder
    if not isdir(data_dir):
        mkdir(data_dir)
        
    logging_folder = data_dir + '/logs_and_plots'
    if not isdir(logging_folder):
        mkdir(logging_folder)

    #Setup logger and open initial data set
    log.set(logging_folder + "/log_out.txt") ## TODo: save all output to a specific output folder
    log.take_stderr()
    log.take_stdout()
    
    
    
    
    
    log("Time ID:", timeID)
    log("output folder name:", output_folder)
    log("date and time run:", time.strftime("%c") )
    log("SSPW folder:", SSPW_folders)
    log("PSE folder:", PSE_folder)
    log("referance station:", referance_station)
    log("stations to keep:", stations_to_keep)
    log("stations to correlate:", stations_to_correlate)
    log("stations to find delays:", stations_to_find_offsets)
    log("flash XY", flash_XY)
    log("num. z points:", num_Z_points)
    log("num. rho points:", num_rho_points)
    log("altitude delta:", alt_delta)
    log("radial delta:", rho_delta)
    log("correlation matrix:", correlation_table)
    log("SSPW group:", correlation_table_SSPW_group)
    log("iniital guess:", initial_guess)
    
    

            
    #### open known PSE ####
    print("reading PSE")
    PSE_data = read_PSE_timeID(timeID, PSE_folder, data_loc="/home/brian/processed_files")
    PSE_ant_locs = PSE_data["ant_locations"]
    PSE_list = PSE_data["PSE_list"]
    old_delays = PSE_data["stat_delays"]
    PSE_list.sort(key=lambda PSE: PSE.unique_index)
    
    ##open known SSPW##
    print("reading SSPW")
    SSPW_data_dict = read_SSPW_timeID_multiDir(timeID, SSPW_folders, data_loc="/home/brian/processed_files", stations=stations_to_correlate) 
    SSPW_multiple_dict = SSPW_data_dict["SSPW_multi_dicts"]
    SSPW_locs_dict = SSPW_data_dict["ant_locations"]
    

    ## setup initial guess
    N_delays = len(stations_to_find_offsets)
    initial_delays = initial_guess[:N_delays]
    initial_locations = initial_guess[N_delays:]
    initial_times = []
    
    ##### correlate SSPW to PSE according to matrix
    PSE_to_correlate = []
    i=0
    for PSE_index, SSPW_indeces in correlation_table.items():
        
        ## find PSE
        found_PSE = None
        for PSE in PSE_list:
            if PSE.unique_index == PSE_index:
                found_PSE = PSE
                break
        
        if found_PSE is None:
            print("error! cannot find PSE")
            break
        
        new_PSE_to_fit = fitting_PSE( found_PSE )
        new_PSE_to_fit.set_loc( initial_locations[i:i+3] )
        PSE_to_correlate.append(new_PSE_to_fit)
        
        initial_times.append( initial_locations[i+3] )
        i += 4
        
            
        ## correlate SSPW
        group_indeces = correlation_table_SSPW_group[PSE_index]
        
        for sname, SSPW_index, SSPW_group_index in zip(stations_to_correlate, SSPW_indeces, group_indeces):
            if SSPW_index is None:
                continue
            
            SSPW_dict = SSPW_multiple_dict[SSPW_group_index]
            for SSPW in SSPW_dict[sname]:
                if SSPW.unique_index==SSPW_index:
                    new_PSE_to_fit.add_SSPW(SSPW)
                    break
                    
        ## prep the PSE
        new_PSE_to_fit.fitting_prep(stations_to_find_offsets, stations_to_keep, PSE_ant_locs, SSPW_locs_dict)
        print()
        
            
            
    #### prep for fitting! ####
    
    initial_guess = np.append(initial_delays, initial_times)

    N_ant = 0
    for PSE in PSE_to_correlate:
        N_ant += len(PSE.antenna_X)
        
        
        
    workspace_sol = np.zeros(N_ant, dtype=np.double)
    workspace_jac = np.zeros((N_ant, len(initial_guess)), dtype=np.double)
    
    
    
    def objective_fun(sol):
        global workspace_sol
        delays = sol[:N_delays]
        ant_i = 0
        param_i = N_delays
        for PSE in PSE_to_correlate:
            N_stat_ant = len(PSE.antenna_X)
            
            PSE.try_location_LS(delays, sol[param_i], workspace_sol[ant_i:ant_i+N_stat_ant])
            ant_i += N_stat_ant
            param_i += 1
            
        return workspace_sol
        
#        workspace_sol *= workspace_sol
#        return np.sum(workspace_sol)
        
    
    def objective_jac(sol):
        global workspace_jac
        delays = sol[:N_delays]
        ant_i = 0
        param_i = N_delays
        for PSE in PSE_to_correlate:
            N_stat_ant = len(PSE.antenna_X)
            
            PSE.try_location_JAC(delays, sol[param_i],  workspace_jac[ant_i:ant_i+N_stat_ant, param_i],  workspace_jac[ant_i:ant_i+N_stat_ant, 0:N_delays])
            ant_i += N_stat_ant
            param_i += 1
            
        return workspace_jac
            
    
    print()
    print()
    
    
    ## check the best fit:
    total_AveSSqE = 0.0
    new_station_delays = initial_guess[:N_delays]
    param_i = N_delays
    for PSE in PSE_to_correlate:
        
        PSE.set_loc_offset( np.zeros(3) )
        total_AveSSqE += PSE.SSqE_fit( new_station_delays,  initial_guess[param_i] )
        param_i += 1
        
    RMS_fit = np.sqrt(total_AveSSqE/N_ant)
    print("initial fit:", RMS_fit)
    print()
    print()
    
    
    #### now we loop over our grid and fit delays and time for each location ####
    Z_grid, rho_grid = np.meshgrid( np.linspace(-alt_delta/2.0, alt_delta/2.0, num_Z_points),  np.linspace(-rho_delta/2.0, rho_delta/2.0, num_rho_points)  )
    fit_value_grid = np.zeros_like(Z_grid)
    radial_norm = np.array(flash_XY)/np.linalg.norm(flash_XY)
    
    for z_i in range(num_Z_points):
        for rho_i in range(num_rho_points):
            
            offset = np.append( rho_grid[z_i,rho_i]*radial_norm, [ Z_grid[z_i,rho_i] ] )
            for PSE in PSE_to_correlate:
                PSE.set_loc_offset( offset )
                
            fit_res = least_squares(objective_fun, initial_guess, jac=objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
            
            
            total_AveSSqE = 0.0
            new_station_delays = fit_res.x[:N_delays] 
            param_i = N_delays
            for PSE in PSE_to_correlate:
                total_AveSSqE += PSE.SSqE_fit( new_station_delays,  fit_res.x[param_i] )
                param_i += 1
            
            RMS_fit = np.sqrt(total_AveSSqE/N_ant)
            fit_value_grid[z_i, rho_i] = RMS_fit
            
            print( z_i, rho_i,':', RMS_fit )
            print("  ", fit_res.message)
            
            
    Z_step = alt_delta/(num_Z_points-1)
    rho_step = rho_delta/(num_rho_points-1)
    
    Z_grid_edges, rho_grid_edges = np.meshgrid( np.linspace(-alt_delta/2.0-Z_step/2.0, alt_delta/2.0+Z_step/2.0, num_Z_points+1),  np.linspace(-rho_delta/2.0-rho_step/2.0, rho_delta/2.0+rho_step/2.0, num_rho_points+1)  )
    
    plt.pcolormesh( rho_grid_edges, Z_grid_edges, fit_value_grid)
    plt.colorbar()
    plt.show()
    
    
        
    
    
    
    
    