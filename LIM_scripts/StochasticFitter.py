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
    timeID = "D20160712T173455.100Z"
    output_folder = "stocastic_fitter_runA"
    
    SSPW_folders = ['SSPW_data']
    PSE_folder = "correlate_SSPW_excludeRS"
    
    max_num_itter = 2000
    itters_till_convergence =  100
    delay_width = 100.0E-9 
    position_width = delay_width/3.0E-9
    
    
    referance_station = "CS002"
    stations_to_keep = ['CS001', 'CS002', 'CS013', 'CS006', 'CS031', 'CS028', 'CS011', 'CS030', 'CS026', 'CS032', 'CS021', 'CS004', 'CS302']#, 'RS106', 'RS205', 'RS208', 'RS305', 'RS306', 'RS307', 'RS406', 'RS407', 'RS503', 'RS508', 'RS509']  ## to keep from the PSE
    stations_to_correlate = ['RS106', 'RS205', 'RS208', 'RS305', 'RS306', 'RS307', 'RS406', 'RS407', 'RS503', 'RS508', 'RS509']  ## to correlate from the SSPW
    
    ## to find delays
    stations_to_find_offsets = stations_to_correlate + stations_to_keep

    correlation_table =  {
            2:[80334, None  , None  , None , None  , None , None , 205312, 108726, None, None],
            3:[None , 187384, None  , None , 179554, None , 44449, None  , 108728, 17  , None],
            6:[80376, 187416, None  , None , None  , None , 44481, None  , 108764, 63  , 62553],
            7:[80441, 187480, 136608, None , None  , 38255, None , 205419, 108825, 150 , 62622],
#            11:[None , None  , None  , None , None  , None , None , None  , None  , None, None ],
            12:[80479, None  , None  , 29882, 179658, None , 44589, None  , 108865, None, None],
            
            26:[84563, 191298, None  , 33588, 183144, None , None , None  , 112675, None, None ],
            32:[85073, 191774, 139657, None , None  , None , None , None  , None  , None, None ],
            34:[85466, 192118, None  , None , 183903, None , None , 210375, 113539, None, None ],
            36:[85894, None  , None  , None , None  , None , 49557, 210803, 113942, None, None ],
            37:[85974, 192582, None  , None , None  , None , 49637, None  , 114025, None, None ],
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
    
    
    

    initial_guess = np.array([ -5.14212874e-07,  -4.09300285e-07,  -1.37456406e-06,
         1.40592032e-07,   1.92089724e-07,  -2.63776140e-07,
         1.05842464e-06,   1.52628872e-06,   3.22253192e-07,
         2.64585736e-06,   1.97801644e-06,  -1.77618755e-08,
         2.71348447e-08,  -6.95719329e-09,   3.77398845e-08,
         7.22358436e-08,  -3.10145556e-08,   8.01027360e-08,
        -2.28380742e-08,   6.74290715e-09,   4.18816133e-08,
        -1.68437422e-09,  -5.70384818e-08,   3.05421187e+04,
         2.85010492e+04,   2.34086165e+03,   3.81932739e+00,
         3.06064132e+04,   2.85403282e+04,   1.93110645e+03,
         3.81951522e+00,   3.05307373e+04,   2.87568257e+04,
         4.53107578e+02,   3.82087373e+00,   3.03125196e+04,
         2.88557720e+04,   1.09510010e+02,   3.82266991e+00,
         3.03088501e+04,   2.88663274e+04,   9.69604871e+02,
         3.82359723e+00,   3.05631025e+04,   2.84423446e+04,
         3.88948534e+03,   3.88040043e+00,   3.05053832e+04,
         2.85821856e+04,   3.95497377e+03,   3.88727216e+00,
         3.13676325e+04,   2.86899006e+04,   3.55635443e+03,
         3.89168231e+00,   3.06872444e+04,   2.89099678e+04,
         6.20709424e+02,   3.89796547e+00,   3.12144844e+04,
         2.84792220e+04,   2.85726972e+03,   3.89923199e+00])

 
    print_fit_table = True
    stations_to_plot = {'RS106':0, 'RS205':0, 'RS208':0, 'RS305':0, 'RS306':0, 'RS307':0, 'RS406':0, 'RS407':0, 'RS503':0, 'RS508':0, 'RS509':0} ## index is SSPW folder to use
    
    
    
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
    log("max num. iters:", max_num_itter)
    log("num convergence itters:", itters_till_convergence)
    log("position width:", position_width)
    log("delay width:", delay_width)
    log("correlation matrix:", correlation_table)
    log("SSPW group", correlation_table_SSPW_group)
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
    SSPW_data_dict = read_SSPW_timeID_multiDir(timeID, SSPW_folders, data_loc="/home/brian/processed_files", stations=stations_to_correlate, max_block=93000) 
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
        
        new_PSE_to_fit = fitting_PSE( found_PSE )
        
        PSE_to_correlate.append(new_PSE_to_fit)
            
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
    N_delays = len(stations_to_find_offsets)
    
    _initial_guess_ = initial_guess ## becouse I'm lazy
    
    initial_guess = np.zeros(N_delays + 4*len(PSE_to_correlate), dtype=np.double)

    i = N_delays
    N_ant = 0
    for PSE in PSE_to_correlate:
        initial_guess[i:i+4] = PSE.initial_loc
        initial_guess[i+2] = np.abs(initial_guess[i+2]) ##make sure Z guess is positive
        i += 4
        N_ant += len(PSE.antenna_X)
        
    if _initial_guess_ is not None:
        initial_guess[:len(_initial_guess_)] = _initial_guess_
        
        
        
        
    workspace_sol = np.zeros(N_ant, dtype=np.double)
    workspace_jac = np.zeros((N_ant, N_delays+4*len(PSE_to_correlate)), dtype=np.double)
    
    
    
    def objective_fun(sol):
        global workspace_sol
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
        global workspace_jac
        delays = sol[:N_delays]
        ant_i = 0
        param_i = N_delays
        for PSE in PSE_to_correlate:
            N_stat_ant = len(PSE.antenna_X)
            
            PSE.try_location_JAC(delays, sol[param_i:param_i+4],  workspace_jac[ant_i:ant_i+N_stat_ant, param_i:param_i+4],  workspace_jac[ant_i:ant_i+N_stat_ant, 0:N_delays])
            ant_i += N_stat_ant
            param_i += 4
            
        return workspace_jac
            
    
    print()
    print()
    
    
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
    best_solution = initial_guess
    best_RMS = initial_RMS
    new_guess = np.array( initial_guess )
    itters_since_change = 0
    for run_i in range(max_num_itter):
        itters_since_change += 1
        
        new_guess[:N_delays] = np.random.normal(scale=delay_width, size=N_delays) + best_solution[:N_delays] ## note use of best_solution, allows for translation. Faster convergence?
        
        param_i = N_delays
        ant_i = 0
        for PSE in PSE_to_correlate:
            new_guess[param_i:param_i+3] = np.random.normal(scale=position_width, size=3) + best_solution[param_i:param_i+3]
            
            N_stat_ant = len(PSE.antenna_X)
            new_guess[param_i+3] = PSE.estimate_T(new_guess[:N_delays], new_guess[param_i:param_i+4], workspace_sol[ant_i:ant_i+N_stat_ant])
            ant_i += N_stat_ant
            param_i += 4
        
        
        fit_res = least_squares(objective_fun, new_guess, jac=objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')

        total_RMS = 0.0
        new_station_delays = fit_res.x[:N_delays] 
        param_i = N_delays
        for PSE in PSE_to_correlate:
            total_RMS += PSE.SSqE_fit( new_station_delays,  fit_res.x[param_i:param_i+4] )
            param_i += 4
            
        total_RMS = np.sqrt(total_RMS/N_ant)
            
        print("run", run_i, "fit:", total_RMS)
        print("  ", fit_res.message)
        
        if total_RMS < best_RMS:
            print("  best fit so far")
            best_RMS = total_RMS
            best_solution = fit_res.x
            itters_since_change = 0
        print()
        
        if itters_since_change == itters_till_convergence:
            break
        
    if itters_since_change != itters_till_convergence:
        print("solution not converged")
    else:
        print("solution converged")
        
        
    
    ####  print out results ####
            
    if print_fit_table:
        fit_table = PrettyTable()
        fit_table.field_names = ['id'] + stations_to_correlate + ['total']
        fit_table.float_format = '.2E'
        
        fit_table_OLD = PrettyTable()
        fit_table_OLD .field_names = ['id'] + stations_to_keep + ['total']
        fit_table_OLD .float_format = '.2E'
    
    i = N_delays
    total_RMS = 0.0
    new_station_delays = best_solution[:N_delays] 
    for PSE in PSE_to_correlate:
        
        loc = best_solution[i:i+4]
        loc[2] = np.abs(loc[2])
        SSqE = PSE.SSqE_fit( new_station_delays,  loc )
        PSE.source_location = loc
        PSE.RMS_fit = np.sqrt(SSqE/len(PSE.antenna_X))
        
        print("PSE", PSE.PSE.unique_index)
        print("  RMS:", PSE.RMS_fit)
        print("  loc:", loc)
        
        if print_fit_table:
            fit_table_row = [PSE.PSE.unique_index]
            station_fits = PSE.RMS_fit_byStation( new_station_delays,  loc )
            
            for sname in stations_to_correlate:
                if sname in station_fits:
                    fit_table_row.append( station_fits[sname] )
                else:
                    fit_table_row.append( '' )
            fit_table_row.append( np.sqrt(SSqE/len(PSE.antenna_X)) )
            fit_table.add_row( fit_table_row )
            
            fit_table_row_OLD = [PSE.PSE.unique_index]
            for sname in stations_to_keep:
                if sname in station_fits:
                    fit_table_row_OLD.append( station_fits[sname] )
                else:
                    fit_table_row_OLD.append( '' )
            fit_table_row_OLD.append( np.sqrt(SSqE/len(PSE.antenna_X)) )
            fit_table_OLD.add_row( fit_table_row_OLD )
        
        
        i += 4
        total_RMS += SSqE
        
        
    if print_fit_table:
        print()
        print()
        print( fit_table_OLD )
        print()
        print()
        print( fit_table )
        
    print()
    print("total RMS:", np.sqrt(total_RMS/N_ant))
    
    #### open station info ####
    StationInfo_dict = read_station_info(timeID)
    
    ##### make plots of fits ####
    for sname, SSPW_group_index in stations_to_plot.items():
        sdata = StationInfo_dict[sname]
        SSPW_dict = SSPW_multiple_dict[ SSPW_group_index ]
        
        if sname in SSPW_dict:
            planewave_events = SSPW_dict[sname]
        else:
            planewave_events = []
        
        sdelay = 0.0
        if sname in stations_to_find_offsets:
            sdelay += new_station_delays[ stations_to_find_offsets.index(sname) ]
#        if sname in station_delay_dict:
#            sdelay += station_delay_dict[sname]
        
        
#        SSPW_amp_X = []
#        SSPW_even_amp_Y = []
#        SSPW_odd_amp_Y = []
#        for SSPW in planewave_events:
#            T = SSPW.ZAT[2]
#            even_amp = SSPW.get_ave_even_amp()
#            odd_amp = SSPW.get_ave_odd_amp()
            
#            SSPW_amp_X.append(T)
#            SSPW_even_amp_Y.append(even_amp)
#            SSPW_odd_amp_Y.append(odd_amp)
            
        data_SSPW  = [e.get_DataTime(sdata.sorted_antenna_names) for e in planewave_events]
        annotations  = [e.unique_index for e in planewave_events]
        
        CP = curtain_plot_CodeLog(StationInfo_dict, logging_folder + "/SSPWvsPSE_"+sname)
        CP.addEventList(sname, data_SSPW, 'b', marker='o', size=50, annotation_list=annotations, annotation_size=20)
        

        data_PSE = [PSE.get_DataTime(sname, sdata.sorted_antenna_names) for PSE in PSE_to_correlate if sname in PSE.station_index_range]                
        model_PSE = [PSE.get_ModelTime(sname, sdata.sorted_antenna_names, sdelay) for PSE in PSE_to_correlate if sname in PSE.station_index_range]
        CP.addEventList(sname, data_PSE, 'r', marker='+', size=100)
        CP.addEventList(sname, model_PSE, 'r', marker='o', size=50)
        
        
#        PSE_amp_X = []
#        PSE_even_amp_Y = []
#        PSE_odd_amp_Y = []
        for PSE in PSE_to_correlate:
            distance = np.linalg.norm( PSE.source_location[0:3] - sdata.get_station_location() )
            arrival_time = PSE.source_location[3] + distance/v_air + sdelay
            
            CP.CL.add_function( "plt.axvline", x=arrival_time, c='r' )
            CP.CL.add_function( "plt.annotate", str(PSE.PSE.unique_index), xy=(arrival_time, np.max(CP.station_offsets[sname]) ), size=20)
            
#            PSE_amp_X.append(arrival_time)
#            PSE_even_amp_Y.append(PSE.orig_even_amp)
#            PSE_odd_amp_Y.append(PSE.orig_odd_amp)
#            
            
        
#        low_point = np.min(CP.station_offsets[sname])
#        top_point = np.max(CP.station_offsets[sname]) - low_point
#        MAX_AMP = max( np.max(PSE_even_amp_Y),  np.max(PSE_odd_amp_Y),  np.max(SSPW_even_amp_Y),  np.max(SSPW_odd_amp_Y))
#        
#        SSPW_even_amp_Y = np.array(SSPW_even_amp_Y)*top_point/MAX_AMP + low_point
#        SSPW_odd_amp_Y = np.array(SSPW_odd_amp_Y)*top_point/MAX_AMP + low_point
#        PSE_even_amp_Y = np.array(PSE_even_amp_Y)*top_point/MAX_AMP + low_point
#        PSE_odd_amp_Y = np.array(PSE_odd_amp_Y)*top_point/MAX_AMP + low_point
#        
#        CP.CL.add_function("plt.plot", PSE_amp_X, PSE_even_amp_Y, 'g+')
#        CP.CL.add_function("plt.plot", PSE_amp_X, PSE_odd_amp_Y, 'm+')
#        CP.CL.add_function("plt.plot", SSPW_amp_X, SSPW_even_amp_Y, 'go')
#        CP.CL.add_function("plt.plot", SSPW_amp_X, SSPW_odd_amp_Y, 'mo')
            
        CP.save()
            
    #### save point sources to binary ####
#    with open(data_dir + "/PSE_data", 'wb') as fout:
#        write_long(fout, len(PSE_to_correlate))
#        for PSE in PSE_to_correlate:
#            PSE.save_binary(fout)
            
    #### save station delays ####
    print()
    print("new delays:")
    for sname, new_delay in zip(stations_to_find_offsets, new_station_delays):
        if sname not in old_delays:
            old_delays[sname] = 0.0
        old_delays[sname] += new_delay
        print(sname, ":", new_delay)
    writeTXT_station_delays(data_dir + "/station_delays.txt", old_delays)
    
    print("new solution:")
    print( repr(best_solution) )
        
    
    
    
    
    