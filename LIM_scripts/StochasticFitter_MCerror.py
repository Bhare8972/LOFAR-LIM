#!/usr/bin/env python3

#python
import time
from os import mkdir
from os.path import isdir, isfile

#external
import numpy as np
from scipy.optimize import least_squares, minimize
from prettytable import PrettyTable

#mine
from LoLIM.utilities import log, processed_data_dir, v_air, SId_to_Sname#SNum_to_SName_dict
from LoLIM.IO.binary_IO import read_long, write_long, write_double_array, write_string, write_double
#from porta_code import code_logger
#from RunningStat import RunningStat

from LoLIM.read_pulse_data import writeTXT_station_delays,read_station_info, curtain_plot_CodeLog

from LoLIM.read_PSE import read_PSE_timeID
from LoLIM.planewave_functions import read_SSPW_timeID_multiDir


class fitting_PSE:
    def __init__(self, PSE, PSE_polarity):
        self.PSE = PSE
        PSE.load_antenna_data( True )
        
        if PSE.PolE_RMS<PSE.PolO_RMS:
            self.polarity = 0
        else:
            self.polarity = 1
            
        if PSE_polarity is not None:
            self.polarity = PSE_polarity
        
        self.initial_loc = PSE.PolE_loc if self.polarity==0 else PSE.PolO_loc
        
        self.SSPW_list = []

    def add_SSPW(self, new_SSPW):
        self.SSPW_list.append(new_SSPW)

    def fitting_prep(self, station_delay_input_order, stations_to_keep, stations_use_PSE, PSE_ant_locs, SSPW_ant_locs):
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
            sname = SId_to_Sname[ int(station_number) ]
            if not ( (sname in stations_to_keep) or (sname in stations_use_PSE) ) :
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
        self.pre_err_pulse_times = np.array(pulse_times)
        self.antenna_names = np.array(antenna_names)
        
        sorted_indeces = np.argsort( self.antenna_names )
        self.antenna_X = self.antenna_X[sorted_indeces]
        self.antenna_Y = self.antenna_Y[sorted_indeces]
        self.antenna_Z = self.antenna_Z[sorted_indeces]
        self.pre_err_pulse_times = self.pre_err_pulse_times[sorted_indeces]
        self.antenna_names = self.antenna_names[sorted_indeces]
        
        self.pulse_times = np.array( self.pre_err_pulse_times )
        
        self.station_index_range = {}
        current_station = None
        start_index = None
        for idx, ant_name in enumerate(self.antenna_names):
            sname = SId_to_Sname[ int(ant_name[0:3]) ]
            
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
                
    def set_error(self, new_ant_timing_error):
        self.pulse_times = self.pre_err_pulse_times + np.random.normal(scale=new_ant_timing_error, size=len(self.pre_err_pulse_times))
            
    def try_location_LS(self, delays, XYZT_location, out):
        X,Y,Z,T = XYZT_location
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
    
    def try_location_JAC(self, delays, XYZT_location, out_loc, out_delays):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        out_loc[:,0] = X
        out_loc[:,0] -= self.antenna_X
        out_loc[:,0] *= -2
        
        out_loc[:,1] = Y
        out_loc[:,1] -= self.antenna_Y
        out_loc[:,1] *= -2
        
        out_loc[:,2] = Z
        out_loc[:,2] -= self.antenna_Z
        out_loc[:,2] *= -2
        
        
        out_loc[:,3] = T - self.pulse_times
        out_loc[:,3] *= 2*v_air*v_air
        
        delay_i = 0
        for index_range, delay in zip(self.ordered_station_index_range,  delays):
            first,last = index_range
            if first is not None:
                out_loc[first:last,3] += delay*2*v_air*v_air
                out_delays[first:last,delay_i] = out_loc[first:last,3]
            delay_i += 1
            
    def SSqE_fit(self, delays, XYZT_location):
        X,Y,Z,T = XYZT_location
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
    
    
            
    def RMS_fit_byStation(self, delays, XYZT_location):
        X,Y,Z,T = XYZT_location
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
        
        ret = {}
        for sname, index_range in self.station_index_range.items():
            first,last = index_range
            
            ret[sname] = np.sqrt( np.sum(distance[first:last])/float(last-first) )
        
        return ret
    
    def get_DataTime(self, sname, sorted_antenna_names):
        out = []
        
        index_range = self.station_index_range[sname]
        if index_range[0] is None:
            return [np.inf]*len(sorted_antenna_names)
        first,last = index_range
        
        ant_names = self.antenna_names[first:last]
        data_times = self.pulse_times[first:last]
            
        for ant_name in sorted_antenna_names:
            if ant_name in ant_names:
                ant_index = np.where(ant_names==ant_name)[0][0]
                out.append( data_times[ ant_index ] )
            else:
                out.append( np.inf )
        
        return out
        
    def get_ModelTime(self, sname, sorted_antenna_names, sdelay):
        out = []
        for ant_name in sorted_antenna_names:
            
            if ant_name in self.antenna_names:
                ant_idx = np.where( self.antenna_names==ant_name )[0][0]
                ant_loc = np.array([ self.antenna_X[ant_idx], self.antenna_Y[ant_idx], self.antenna_Z[ant_idx]])
                
                ant_loc -= self.source_location[0:3]
                model_time = np.linalg.norm( ant_loc )/v_air + self.source_location[3] + sdelay
                
                out.append(model_time)
            else:
                out.append( np.inf )
        return out
    
    
    def estimate_T(self, delays, XYZT_location, workspace):
        X,Y,Z,T = XYZT_location
        Z = np.abs(Z)
        
        delta_X_sq = self.antenna_X-X
        delta_Y_sq = self.antenna_Y-Y
        delta_Z_sq = self.antenna_Z-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
            
        workspace[:] = delta_X_sq
        workspace[:] += delta_Y_sq
        workspace[:] += delta_Z_sq
        
        np.sqrt(workspace, out=workspace)
        
        workspace[:] -= self.pulse_times*v_air ## this is now source time
        
        ##now account for delays
        for index_range, delay in zip(self.ordered_station_index_range,  delays):
            first,last = index_range
            if first is not None:
                workspace[first:last] += delay*v_air ##note the wierd sign
                
                
        ave_error = np.average(workspace)
        return -ave_error/v_air

if __name__=="__main__":
    ##opening data
    timeID = "D20170929T202255.000Z"
    output_folder = "stocastic_fitter_runA_err2ns"
    
    SSPW_folders = ['SSPW2_tmp']
    PSE_folder = "handcorrelate_SSPW"
    
    num_itter = 1000
    timing_error = 2.0E-9
    
    referance_station = "CS002"
    stations_to_keep = []## to keep from the PSE
    stations_to_correlate = ["CS002", "CS003", "CS004", "CS005", "CS006", "CS007", "CS011", "CS013", "CS017", "CS021", "CS030", "CS032", "CS101", "CS103",
                             "RS208", "CS301", "CS302", "RS306", "RS307", "RS310", "CS401", "RS406", "RS409", "CS501", "RS503", "RS508", "RS509"]  
                                ## to correlate from the SSPW
    
    ## to find delays
    stations_to_find_offsets = stations_to_correlate + stations_to_keep

   
## -1 means use data from PSE None means no data
    correlation_table =  {
#  "CS002", "CS003", "CS004", "CS005", "CS006", "CS007", "CS011", "CS013", "CS017", "CS021", "CS030", "CS032", "CS101", "CS103", 
#  "RS208", "CS301", "CS302", "RS306", "RS307", "RS310", "CS401", "RS406", "RS409", "CS501", "RS503", "RS508", "RS509"
   
0:[  -1   ,    1808,      -1,      -1,      -1,     -1 ,      -1,      -1,      -1,      -1,      -1,     None,    -1 ,    -1  ,
        -1,   -1   ,    -1  ,    -1  ,      -1,     -1 ,    -1  ,      -1,      -1,      -1,      -1,     None,  None],

1:[  -1   ,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,      -1,       -1,   None,    -1  ,
      -1  ,   -1   ,    -1  ,    -1  ,    None,     -1 ,    -1  ,    None,      -1,      -1,      -1,       -1,    -1 ],
  
2:[  -1   ,      -1,      -1,      -1,      -1,  19428 ,      -1,      -1,      -1,      -1,      -1,       -1,     -1,    -1  ,
      -1  ,   -1   ,    -1  ,    None,      -1,     -1 ,    -1  ,      -1,      -1,      -1,      -1,       -1,    None],
   
3:[  -1   ,    1825,      -1,    None,    6191,      -1,      -1,      -1,      -1,   None ,      -1,       -1,     -1,   24648,
      -1  ,      -1,    -1  ,    -1  ,      -1,     -1 ,    -1  ,      -1,      -1,      -1,      -1,       -1,    -1 ],
   
4:[  -1   ,      -1,      -1,      -1,    6200,      -1,  40265 ,      -1,      -1,      -1,      -1,       -1,     -1,    -1  ,
      -1  ,   -1   ,    -1  ,    -1  ,    None,  18376 ,    -1  ,      -1,      -1,    None,      -1,     None,    -1 ],
   
            }
    
    ### since we are loading multiple SSPW files, the SSPW unique IDs are not actually unique. Either have a list of indeces, where each index
    ## refers to the file in "SSPW_folders", or a None, which implies all indeces are from the first "SSPW_folders"
    correlation_table_SSPW_group = { 
            0:None,
            1:None,
            2:None,
            3:None,
            4:None,
            }
    
    ##set polarization to fit. Do not need to have PSE. if 0, use even annennas, if 1 use odd antennas. If None, or not-extent in this list,
    ## then use default polarization
    PSE_polarization = {
            0:1,
            1:1,
            2:1,
            3:0,
            4:0,
            }

    
    initial_guess = np.array(
            [  1.40427809e-06,   4.31025200e-07,  -2.20496710e-07,
         4.31343516e-07,   3.98578573e-07,  -5.88519822e-07,
        -1.81445018e-06,  -8.44276344e-06,   9.25953807e-07,
        -2.73962044e-06,  -1.57047793e-06,  -8.17613911e-06,
        -2.85284318e-05,   6.68084426e-06,  -7.19812324e-07,
        -5.35571542e-06,   7.10156300e-06,   6.39487995e-06,
         6.51643900e-06,   1.49666497e-06,   2.43805669e-05,
         4.36558182e-06,  -9.61191607e-06,   6.93244635e-06,
         6.71737582e-06,   7.27924311e-06,  -1.54706398e+04,
         8.87008324e+03,   3.42588066e+03,   1.15317861e+00,
        -1.57684642e+04,   9.06902196e+03,   3.49821834e+03,
         1.15321428e+00,  -1.54479311e+04,   8.85861563e+03,
         3.43834812e+03,   1.15324252e+00,  -1.59066697e+04,
         9.15824077e+03,   3.42515884e+03,   1.15344813e+00,
        -1.57110000e+04,   8.98502233e+03,   3.26820689e+03,
         1.15356325e+00])
    

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
    log("num trials:", num_itter)
    log("timing error:", timing_error)
    log("correlation matrix:", correlation_table)
    log("SSPW group", correlation_table_SSPW_group)
    log("iniital guess:", repr(initial_guess))
    
    

            
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
    


    ##### correlate SSPW to PSE according to matrix
    PSE_to_correlate = []
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
        
        PSE_polarity = None
        if PSE_index in PSE_polarization:
            PSE_polarity = PSE_polarization[PSE_index]
        
        new_PSE_to_fit = fitting_PSE( found_PSE, PSE_polarity )
        
        PSE_to_correlate.append(new_PSE_to_fit)
            
        ## correlate SSPW
        group_indeces = correlation_table_SSPW_group[PSE_index]
        if group_indeces is None:
            group_indeces = [0]*len(stations_to_correlate)
        
        stations_to_PSE_use = []
        for sname, SSPW_index, SSPW_group_index in zip(stations_to_correlate, SSPW_indeces, group_indeces):
            
            if SSPW_index is None:
                continue
            if SSPW_index == -1:
                stations_to_PSE_use.append(sname)
                continue
            
            SSPW_dict = SSPW_multiple_dict[SSPW_group_index]
            
            for SSPW in SSPW_dict[sname]:
                if SSPW.unique_index==SSPW_index:
                    new_PSE_to_fit.add_SSPW(SSPW)
                    break
                    
        ## prep the PSE
        new_PSE_to_fit.fitting_prep(stations_to_find_offsets, stations_to_keep, stations_to_PSE_use, PSE_ant_locs, SSPW_locs_dict)
        print()
        
            
            
    #### prep for fitting! ####
    N_delays = len(stations_to_find_offsets)

    i = N_delays
    N_ant = 0
    for PSE in PSE_to_correlate:
        N_ant += len(PSE.antenna_X)
        
    workspace_sol = np.zeros(N_ant, dtype=np.double)
#    workspace_jac = np.zeros((N_ant, N_delays+4*len(PSE_to_correlate)), dtype=np.double)
    
    
    def objective_fun(sol):
#        global workspace_sol
        workspace_sol = np.zeros(N_ant, dtype=np.double)
        delays = sol[:N_delays]
        ant_i = 0
        param_i = N_delays
        for PSE in PSE_to_correlate:
            N_stat_ant = len(PSE.antenna_X)
            
            PSE.try_location_LS(delays, sol[param_i:param_i+4], workspace_sol[ant_i:ant_i+N_stat_ant])
            ant_i += N_stat_ant
            param_i += 4
            
        
            
        return workspace_sol
        
#        workspace_sol *= workspace_sol
#        return np.sum(workspace_sol)
        
    
    def objective_jac(sol):
#        global workspace_jac
        workspace_jac = np.zeros((N_ant, N_delays+4*len(PSE_to_correlate)), dtype=np.double)
        
        if np.isnan(workspace_jac).any():
            print("JAC NAN A")
            
        delays = sol[:N_delays]
        ant_i = 0
        param_i = N_delays
        for PSE in PSE_to_correlate:
            N_stat_ant = len(PSE.antenna_X)
            
            PSE.try_location_JAC(delays, sol[param_i:param_i+4],  workspace_jac[ant_i:ant_i+N_stat_ant, param_i:param_i+4],  workspace_jac[ant_i:ant_i+N_stat_ant, 0:N_delays])
            ant_i += N_stat_ant
            param_i += 4
        
        if np.isnan(workspace_jac).any():
            print("JAC NAN B")
            
        return workspace_jac
            
    
    print()
    print()
    
    
    
    ### initial fit, with no additional noise ###
    fit_res = least_squares(objective_fun, initial_guess, jac=objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
    initial_guess = fit_res.x 
    
    initial_RMS = 0.0
    new_station_delays = initial_guess[:N_delays]
    param_i = N_delays
    for PSE in PSE_to_correlate:
        initial_RMS += PSE.SSqE_fit( new_station_delays,  initial_guess[param_i:param_i+4] )
        param_i += 4
    initial_RMS = np.sqrt(initial_RMS/N_ant)
    
    print("initial RMS:", initial_RMS)
    print()
    
    
    
    
    #### run fit N times ####
    ave_X_byItter = np.zeros( num_itter )
    ave_Y_byItter = np.zeros( num_itter )
    ave_Z_byItter = np.zeros( num_itter )
    ave_T_byItter = np.zeros( num_itter )
    delays_byItter = np.zeros( (num_itter,N_delays) )
    
    num_PSE = len(PSE_to_correlate)
    rel_X_byItter = np.zeros( (num_itter, num_PSE) )
    rel_Y_byItter = np.zeros( (num_itter, num_PSE) )
    rel_Z_byItter = np.zeros( (num_itter, num_PSE) )
    rel_T_byItter = np.zeros( (num_itter, num_PSE) )
    
    rel_phi_ByItter = np.zeros( (num_itter, num_PSE) )
    rel_rho_ByItter = np.zeros( (num_itter, num_PSE) )
    
    for run_i in range(num_itter):
        
        for PSE in PSE_to_correlate:
            PSE.set_error( timing_error )
        
        fit_res = least_squares(objective_fun, initial_guess, jac=objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')

        total_RMS = 0.0
        new_station_delays = fit_res.x[:N_delays] 
        param_i = N_delays
        for PSE in PSE_to_correlate:
            total_RMS += PSE.SSqE_fit( new_station_delays,  fit_res.x[param_i:param_i+4] )
            param_i += 4
            
        total_RMS = np.sqrt(total_RMS/N_ant)
        
        ave_X = np.average( fit_res.x[N_delays:][0::4] )
        ave_Y = np.average( fit_res.x[N_delays:][1::4] )
        ave_Z = np.average( fit_res.x[N_delays:][2::4] )
        ave_T = np.average( fit_res.x[N_delays:][3::4] )
            
        print("run", run_i, "fit:", total_RMS)
        print("  ave X:", ave_X, "ave Y:", ave_Y,"ave Z:", ave_Z, "ave T:", ave_T)
        print("  ", fit_res.message)
        
        ave_X_byItter[ run_i ] = ave_X
        ave_Y_byItter[ run_i ] = ave_Y
        ave_Z_byItter[ run_i ] = ave_Z
        ave_T_byItter[ run_i ] = ave_T
        delays_byItter[ run_i ] = new_station_delays
        
        param_i = N_delays
        PSE_i = 0
        for PSE in PSE_to_correlate:
            X,Y,Z,T = fit_res.x[param_i:param_i+4]
            
            rel_X_byItter[run_i, PSE_i] = X-ave_X
            rel_Y_byItter[run_i, PSE_i] = Y-ave_Y
            rel_Z_byItter[run_i, PSE_i] = Z-ave_Z
            rel_T_byItter[run_i, PSE_i] = T-ave_T
            
            azimuth = np.arctan2(Y, X)
            sin_phi = np.sin(azimuth)
            cos_phi = np.cos(azimuth)
            
            rel_rho_ByItter[run_i, PSE_i] = cos_phi*(X-ave_X) + sin_phi*(Y-ave_Y)
            rel_phi_ByItter[run_i, PSE_i] = -sin_phi*(X-ave_X) + cos_phi*(Y-ave_Y)
            
            param_i += 4
            PSE_i += 1
    
    print()
    print()
    
    print("ave X std:", np.std(ave_X_byItter) )
    print("ave Y std:", np.std(ave_Y_byItter) )
    print("ave Z std:", np.std(ave_Z_byItter) )
    print("ave T std:", np.std(ave_T_byItter) )
    
    print()
    print()
    for sname, i in zip(stations_to_find_offsets, range(N_delays) ):
        print("station", sname, "std:", np.std(delays_byItter[:,i]) )
        
        
    print()
    print()
    PSE_i = 0
    rel_X = []
    rel_Y = []
    rel_Z = []
    rel_T = []
    rel_phi = []
    rel_rho = []
    for PSE in PSE_to_correlate:
        rel_X_error = np.std( rel_X_byItter[:, PSE_i] )
        rel_Y_error = np.std( rel_Y_byItter[:, PSE_i] )
        rel_Z_error = np.std( rel_Z_byItter[:, PSE_i] )
        rel_T_error = np.std( rel_T_byItter[:, PSE_i] )
        
        rel_phi_error = np.std( rel_phi_ByItter[:, PSE_i] )
        rel_rho_error = np.std( rel_rho_ByItter[:, PSE_i] )
        
        PSE_i += 1
        
        print(PSE.PSE.unique_index, ':', rel_X_error, rel_Y_error, rel_Z_error, rel_T_error)
        print("   ", rel_phi_error, rel_rho_error)
        
        rel_X.append( rel_X_error )
        rel_Y.append( rel_Y_error )
        rel_Z.append( rel_Z_error )
        rel_T.append( rel_T_error )
        rel_phi.append( rel_phi_error )
        rel_rho.append( rel_rho_error )
        
    print("ave, std, min, max")
    print("X:", np.average(rel_X), np.std(rel_X), np.min(rel_X), np.max(rel_X) ) 
    print("Y:", np.average(rel_Y), np.std(rel_Y), np.min(rel_Y), np.max(rel_Y) ) 
    print("Z:", np.average(rel_Z), np.std(rel_Z), np.min(rel_Z), np.max(rel_Z) )
    print("T:", np.average(rel_T), np.std(rel_T), np.min(rel_T), np.max(rel_T) )  
    print("phi:", np.average(rel_phi), np.std(rel_phi), np.min(rel_phi), np.max(rel_phi) ) 
    print("rho:", np.average(rel_rho), np.std(rel_rho), np.min(rel_rho), np.max(rel_rho) ) 
    
        
#    plt.hist(ave_X_byItter, bins=30)
#    plt.show()
        
    
    
    
    
    