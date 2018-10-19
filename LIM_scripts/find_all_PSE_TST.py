#!/usr/bin/env python3

##internal
import time
from os import mkdir
from os.path import isdir

##import external packages
import numpy as np
from scipy.optimize import least_squares, minimize

##my packages
from LoLIM.utilities import log, processed_data_dir, v_air
from LoLIM.read_pulse_data import read_station_info, refilter_pulses, curtain_plot_CodeLog, AntennaPulse_dict__TO__AntennaTime_dict, getNwriteBin_modData
from LoLIM.porta_code import code_logger
from LoLIM.IO.binary_IO import write_long, write_double_array, write_string, write_double

PSE_next_unique_index = 0
class PointSourceEvent:
    def __init__(self, pulse_dict, guess_location, station_info_dict):
        global PSE_next_unique_index
        self.unique_index = PSE_next_unique_index
        PSE_next_unique_index += 1
        
        self.pulse_dict = pulse_dict
        self.station_info_dict = station_info_dict
        
        ant_locations = []
        PolE_times = []
        PolO_times = []
        self.antennas_included = []
        
        self.num_even_ant = 0
        self.num_odd_ant = 0
        
        for pulse in self.pulse_dict.values():
            loc = pulse.antenna_info.location
            ant_locations.append(loc)
            self.antennas_included.append( pulse.even_antenna_name )
            
            PolE_times.append( pulse.PolE_peak_time ) ###NOTE that peak times are infinity if something is wrong with antenna
            PolO_times.append( pulse.PolO_peak_time ) ###NOTE that peak times are infinity if something is wrong with antenna
            
            if np.isfinite( pulse.PolE_peak_time ):
                self.num_even_ant += 1
            if np.isfinite( pulse.PolO_peak_time ):
                self.num_odd_ant += 1
            
        self.antenna_locations = np.array(ant_locations)
        PolE_times = np.array(PolE_times)
        PolO_times = np.array(PolO_times)
        
        self.residual_workspace = np.zeros(len(PolE_times))
        self.jacobian_workspace = np.zeros( (len(PolE_times), 4) )
         
        ### fit even polarisation ###
        ## TODO: fit simultaniously!
        guess_location[2] = np.abs( guess_location[2] )
        
        self.ant_times = PolE_times
        EvenPol_min = least_squares(self.objective_RES, guess_location, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', max_nfev=1000)
        self.PolE_loc = EvenPol_min.x
        self.PolE_loc[2] = np.abs(self.PolE_loc[2])
        self.PolE_RMS = np.sqrt( self.SSqE(self.PolE_loc)/float(self.num_even_ant) )
        
        ### fit odd polarisation ###
        self.ant_times = PolO_times
        OddPol_min = least_squares(self.objective_RES, guess_location, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', max_nfev=1000)
        self.PolO_loc = OddPol_min.x
        self.PolO_loc[2] = np.abs(self.PolO_loc[2])
        self.PolO_RMS = np.sqrt( self.SSqE(self.PolO_loc)/float(self.num_odd_ant) )
        
        
        print()
        print("PSE", self.unique_index)
        print("  even red chi sq:", self.PolE_RMS, ',', self.num_even_ant, "antennas")
        print("  even_loc:", self.PolE_loc)
        print("  ", EvenPol_min.message)
        print("  odd red chi sq:", self.PolO_RMS, ',', self.num_odd_ant, "antennas")
        print("  odd loc:", self.PolO_loc)
        print("  ", OddPol_min.message)
        print()
        
        ## free resources
        del self.residual_workspace
        del self.jacobian_workspace
        del self.ant_times
        del self.antenna_locations
        
    def objective_RES(self, XYZT):
        XYZT[2] = np.abs(XYZT[2])
        
        self.residual_workspace[:] = self.ant_times 
        self.residual_workspace[:] -= XYZT[3]
        self.residual_workspace[:] *= v_air
        self.residual_workspace[:] *= self.residual_workspace[:]
        
        R2 = self.antenna_locations - XYZT[0:3]
        R2 *= R2
        self.residual_workspace[:] -= R2[:,0]
        self.residual_workspace[:] -= R2[:,1]
        self.residual_workspace[:] -= R2[:,2]
        
        self.residual_workspace[ np.logical_not(np.isfinite(self.residual_workspace)) ] = 0.0
        
        return self.residual_workspace
    
    def objective_JAC(self, XYZT):
        XYZT[2] = np.abs(XYZT[2])
        
        self.jacobian_workspace[:, 0:3] = XYZT[0:3]
        self.jacobian_workspace[:, 0:3] -= self.antenna_locations
        self.jacobian_workspace[:, 0:3] *= -2.0
        
        self.jacobian_workspace[:, 3] = self.ant_times
        self.jacobian_workspace[:, 3] -= XYZT[3]
        self.jacobian_workspace[:, 3] *= -v_air*v_air*2
        
        mask = np.logical_not(np.isfinite(self.jacobian_workspace[:, 3])) 
        self.jacobian_workspace[mask, :] = 0
        
        return self.jacobian_workspace
    
    def SSqE(self, XYZT):
        R2 = self.antenna_locations - XYZT[0:3]
        R2 *= R2
        
        theory = np.sum(R2, axis=1)
        np.sqrt(theory, out=theory)
        
        theory *= 1.0/v_air
        theory += XYZT[3] - self.ant_times
        
        theory *= theory
        
        
        theory[ np.logical_not(np.isfinite(theory) ) ] = 0.0
        
        return np.sum(theory)
    
    def get_ModelTime(self, station_info, polarity,inclusion=0):
        ret = []
        for ant_name in station_info.sorted_antenna_names:
            
            if inclusion != 1:
                included = ant_name in self.antennas_included
                if inclusion==1 and not included:
                    ret.append(np.inf)
                    continue
                if inclusion==2 and included:
                    ret.append(np.inf)
                    continue
                    
            ant_loc = station_info.AntennaInfo_dict[ant_name].location
            
            if polarity == 0:
                dx = np.linalg.norm( self.PolE_loc[0:3]-ant_loc)/v_air
                ret.append( dx+self.PolE_loc[3] )
            elif polarity == 1:
                dx = np.linalg.norm( self.PolO_loc[0:3]-ant_loc)/v_air
                ret.append( dx+self.PolO_loc[3] )
                
        return ret
        
    
    def save_as_binary(self, fout):
        
        write_long(fout, self.unique_index)
        write_double(fout, self.PolE_RMS)
        write_double(fout, 0.0) ## reduced chi-squared
        write_double(fout, 0.0) ## power
        write_double_array(fout, self.PolE_loc)
        write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0])) ## standerd errors
        
        write_double(fout, self.PolO_RMS)
        write_double(fout, 0.0) ## reduced chi-squared
        write_double(fout, 0.0) ## power
        write_double_array(fout, self.PolO_loc)
        write_double_array(fout, np.array([0.0, 0.0, 0.0, 0.0]))## standerd errors
        
        write_double(fout, 5.0E-9) ##seconds per sample
        write_long(fout, self.num_even_ant)
        write_long(fout, self.num_odd_ant)
        
        write_long(fout, 1) ## 1 means that we do save trace data
        
        write_long(fout, len(self.pulse_dict))
        for ant_name, pulse in self.pulse_dict.items():
            write_string(fout, ant_name)
            write_long(fout, pulse.section_number)
            write_long(fout, pulse.unique_index)
            write_long(fout, pulse.starting_index)
            write_long(fout, pulse.antenna_status)
            
            write_double(fout, pulse.PolE_peak_time)
            write_double(fout, pulse.PolE_estimated_timing_error)
            write_double(fout, pulse.PolE_time_offset)
            write_double(fout, np.max(pulse.even_antenna_hilbert_envelope))
            write_double(fout, pulse.even_data_STD)
            
            write_double(fout, pulse.PolO_peak_time)
            write_double(fout, pulse.PolO_estimated_timing_error)
            write_double(fout, pulse.PolO_time_offset)
            write_double(fout, np.max(pulse.odd_antenna_hilbert_envelope))
            write_double(fout, pulse.odd_data_STD)
            
            write_double_array(fout, pulse.even_antenna_hilbert_envelope)
            write_double_array(fout, pulse.even_antenna_data)
            write_double_array(fout, pulse.odd_antenna_hilbert_envelope)
            write_double_array(fout, pulse.odd_antenna_data)
            
        
        

def find_PSE_itteration(ant_locs, ant_pulse_times, search_time, search_indeces):
    
    ## assume globals search_location, bin_half_width
    
    num_antennas_needed = 5
    num_antennas_total = len( ant_locs )
    
    starter_K_vector = np.sum( ant_locs*ant_locs, axis=1 )
    starter_K_vector -= starter_K_vector[0]
    starter_K_vector *= 0.5
    
    starter_matrix = np.zeros( (len(ant_locs), 4) )
    starter_matrix[:,:-1] = ant_locs
    starter_matrix -= starter_matrix[0]
    
    def recursive_iterator(ant_index, antenna_mask, pulse_times, used_pulse_indeces):
        
        if np.sum( antenna_mask ) == num_antennas_needed:
            
            ## we have a solution to test
            matrix = starter_matrix[ antenna_mask ] ## note that this copies the matrix
            k_vector = starter_K_vector[ antenna_mask ]
            
            pulse_times = np.array(pulse_times)
            
            pulse_times_sq =  pulse_times*pulse_times
            pulse_times_sq -= pulse_times_sq[0]
            pulse_times_sq *= v_air*v_air
            
            k_vector -= 0.5*pulse_times_sq
            
            pulse_times -= pulse_times[0]
            pulse_times *= -v_air*v_air
            
            matrix[:,3] = pulse_times
            
            XYZT, residuals, rank, s = np.linalg.lstsq(matrix, k_vector)
            
            if len(residuals) == 1:
                return residuals[0], XYZT, used_pulse_indeces
            else:
                return np.inf, np.zeros(4), used_pulse_indeces
                
            
        elif (num_antennas_total-ant_index) <  (num_antennas_needed-np.sum( antenna_mask )) :
            
            ### the number of antennas available is less than number of antennas needed.
            ## no solutions possible
            return np.inf, np.zeros(4), used_pulse_indeces
        
        else:
            
            ## we do not have enough antennas, and are not at the end
            
            ## the solutions without this antenna
            current_solution = recursive_iterator(ant_index+1, antenna_mask, pulse_times, used_pulse_indeces)
            
            if len( ant_pulse_times[ant_index] ) - search_indeces[ant_index] == 0:
                return current_solution
                
            #### solutions WITH this antenna
            T_recieved = np.linalg.norm( search_location - ant_locs[ant_index] )/v_air
            min_t = search_time - T_recieved
            max_t = search_time + T_recieved
            
            found_start = None
            
            for pulse_index in range(search_indeces[ant_index], len( ant_pulse_times[ant_index] ) ):
                pulse_time = ant_pulse_times[ant_index][pulse_index]
                
                if not np.isfinite(pulse_time):
                    continue
                
                if found_start is None:
                    if pulse_time < min_t:
                        found_start = False
                    else:
                        found_start = True
                        
                elif not found_start:
                    if found_start >= min_t:
                        found_start = True
                        search_indeces[ant_index] = pulse_index
                        
                if found_start:
                    if pulse_time>max_t:
                        break
                    
                    else:
                        ## a solution to try
                        new_mask = np.array( antenna_mask )
                        new_mask[ant_index] = 1
                        new_pulse_times = pulse_times + [ pulse_time ] ## note that this will copy the pulse_times list
                        new_pulse_Ids = np.array( used_pulse_indeces )
                        new_pulse_Ids[ant_index] = pulse_index
                        
                        new_solution = recursive_iterator(ant_index+1, new_mask, new_pulse_times, new_pulse_Ids)
                        
                        if new_solution[0] < current_solution[0]:
                            current_solution = new_solution
                            
            return current_solution
        
    return recursive_iterator(0, np.zeros(num_antennas_total, dtype=bool), [], -1*np.ones(num_antennas_total, dtype=int) )



##### routines for initial fitting #####


def get_pulse_indecesNres(XYZT, antPos_dicts, antTime_dicts, T_window):
    XYZT[2] = np.abs(XYZT[2])
    
    pulse_indeces = []
    pulse_residuals = []
    for sname, pulse_dict in antTime_dicts.items():
        for ant_name, pulse_times in pulse_dict.items():
            if len(pulse_times) == 0:
                continue
            
            ant_loc = antPos_dicts[sname][ant_name]
            DT = np.linalg.norm( XYZT[0:3]-ant_loc )/v_air
            
            residuals = np.abs( pulse_times - (XYZT[3] + DT) )
            idx = np.argmin( residuals )
            
            if residuals[idx]<T_window and np.isfinite(residuals[idx]):
                pulse_residuals.append( residuals[idx] )
                pulse_indeces.append( [idx, sname, ant_name] )
            
    return pulse_indeces, np.array(pulse_residuals)

## objective functions
def individual_obj_fun_RES(XYZT, antPos_dicts, antTime_dicts, pulse_indeces):
    global residual_workspace
    XYZT[2] = np.abs(XYZT[2])
    
    ant_i = -1
    for pulse_i, sname, ant_name in pulse_indeces:
        ant_i += 1
        
        if pulse_i<0:
            residual_workspace[ant_i] = 0.0
            continue
            
        pulse_time = antTime_dicts[sname][ant_name][ pulse_i ]
         
        ant_location = antPos_dicts[sname][ant_name]
        
        R2 = ant_location - XYZT[0:3]
        R2 *= R2
        R2 = np.sum(R2)
        
        ant_T_diff = pulse_time - XYZT[3]
        ant_T_diff *= v_air
        ant_T_diff *= ant_T_diff
        
        ant_T_diff -= R2
        
        residual_workspace[ant_i] = ant_T_diff
                
                
    return residual_workspace

def individual_obj_fun_JAC(XYZT, antPos_dicts, antTime_dicts, pulse_indeces):
    global jacobian_workspace
    XYZT[2] = np.abs(XYZT[2])
    
    ant_i = -1
    for pulse_i, sname, ant_name in pulse_indeces:
        ant_i += 1
            
        if pulse_i<0:
            jacobian_workspace[ant_i] = 0.0
            continue
        
        pulse_time = antTime_dicts[sname][ant_name][ pulse_i ]
        ant_location = antPos_dicts[sname][ant_name]
            
        jacobian_workspace[ant_i, 0:3] = 2.0*( ant_location - XYZT[0:3] )
        jacobian_workspace[ant_i, 3] = 2.0*(v_air*v_air)* ( XYZT[3] - pulse_time )
                
    return jacobian_workspace


def initial_obj_fun_RES(XYZT, ant_locs, pulse_times):
    global residual_workspace
    XYZT[2] = np.abs(XYZT[2])
    
    residual_workspace[:] = pulse_times
    residual_workspace -= XYZT[3]
    residual_workspace *= v_air
    residual_workspace *= residual_workspace

    R2 = ant_locs - XYZT[0:3]
    R2 *= R2
    residual_workspace -= R2[:, 0]
    residual_workspace -= R2[:, 1]
    residual_workspace -= R2[:, 2]
    
    return residual_workspace

def initial_obj_fun_JAC(XYZT, ant_locs, pulse_times):
    global jacobian_workspace
    XYZT[2] = np.abs(XYZT[2])

    jacobian_workspace[:, 0:3] = 2.0*( ant_locs - XYZT[0:3] )
    jacobian_workspace[:, 3] = 2.0*(v_air*v_air)* ( XYZT[3] - pulse_times )
                
    return jacobian_workspace

def get_initial_RMS(XYZT, ant_locs, pulse_times):
    
    XYZT[2] = np.abs(XYZT[2])
    
    res = np.linalg.norm( ant_locs - XYZT[0:3], axis=1)/v_air
    res += XYZT[3]
    res -= pulse_times
    
    return np.sqrt( np.average(res**2) )


if __name__ == "__main__":
    
    timeID = "D20170929T202255.000Z"
    output_folder = "allPSE_TST_BREAKS"
    
    plot_station_map = True
    
    stations_to_exclude =  ["CS028","RS106", "RS305", "RS205", "CS201", "RS407", "RS406"]
    
    num_blocks_per_step = 100
    initial_block = 3500
    num_steps = 100
    
    ant_timing_calibrations = "cal_tables/TxtAntDelay"
    polarization_flips = "polarization_flips.txt"
    bad_antennas = "bad_antennas.txt"
    station_delays = "station_delays.txt"
    additional_antenna_delays = "ant_delays.txt"
    
    min_signal_SNR = 20
    
    search_location = np.array( [-1.58423240e+04,   9.08114847e+03,   3.34469757e+03] )
    bin_half_width = 5000/v_air
    
    max_ant_time_residual = 1000.0E-9 #### if the residual of an antenna time is greater than this, the antenna is thrown out
    num_ant_threshold = 50
    
    
    
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
    
    
    #### print details
    print("Time ID:", timeID)
    print("output folder name:", output_folder)
    print("date and time run:", time.strftime("%c") )
    print("stations to exclude:", stations_to_exclude)
    print("initial block", initial_block, "num blocks per step:", initial_block, "num steps", num_steps)
    print("min signal SNR", min_signal_SNR)
    print("antenna timing calibrations:", ant_timing_calibrations)
    print("antenna polarization flips file:", polarization_flips)
    print("bad antennas file:", bad_antennas)
    print("station delays file:", station_delays)
    print("ant delays file:", additional_antenna_delays)
    print()
    print("search location:", search_location)
    print("half width (m):", bin_half_width*v_air)
    print("maximum residual:", max_ant_time_residual)
    print("num ant threshold:", num_ant_threshold)
    
    ##station data
    print()
    print("opening station data")
    
    polarization_flips = processed_data_dir + '/' + polarization_flips
    bad_antennas = processed_data_dir + '/' + bad_antennas
    station_delays = processed_data_dir + '/' + station_delays
    ant_timing_calibrations = processed_data_dir + '/' + ant_timing_calibrations
    additional_antenna_delays = processed_data_dir + '/' + additional_antenna_delays
    
    StationInfo_dict = read_station_info(timeID, stations_to_exclude=stations_to_exclude, bad_antennas_fname=bad_antennas, ant_delays_fname=additional_antenna_delays, 
                                         pol_flips_fname=polarization_flips, station_delays_fname=station_delays, txt_cal_table_folder=ant_timing_calibrations)
        

    if plot_station_map:
        CL = code_logger(logging_folder+ "/station_map")
        CL.add_statement("import numpy as np")
        CL.add_statement("import matplotlib.pyplot as plt")
        
        for sname,sdata in StationInfo_dict.items():
            ant_X = np.array([ ant.location[0] for ant in sdata.AntennaInfo_dict.values() ])
            ant_Y = np.array([ ant.location[1] for ant in sdata.AntennaInfo_dict.values() ])
            station_X = np.average(ant_X)
            station_Y = np.average(ant_Y)
            
            CL.add_function("plt.scatter", ant_X, ant_Y)
            CL.add_function("plt.annotate", sname, xy=(station_X, station_Y), size=30)
            
        CL.add_statement( "plt.tick_params(axis='both', which='major', labelsize=30)" )
        CL.add_statement( "plt.show()" )
        CL.save()
        
        
    prefered_antennas = [ sdata.sorted_antenna_names[0] for sdata in StationInfo_dict.values() ] ## select one prefered antenna from every station
    

    for iter_i in range(num_steps):
        print( "Opening Data. Itter:", iter_i )
        
        current_block = initial_block + iter_i*num_blocks_per_step
        
        AntennaPulse_dicts = {sname:sdata.read_pulse_data(approx_min_block=current_block, approx_max_block=current_block+num_blocks_per_step) for sname,sdata in StationInfo_dict.items()}

        ##refilter pulses
        #### NOTE: we are only keeping pulses if they have good even polarization, due to way pulse finding works
        AntennaPulse_dicts_refiltered = {sname:refilter_pulses(AntPulse_dict, min_signal_SNR, pol_requirement=1) for sname,AntPulse_dict in AntennaPulse_dicts.items()}
        
        
        ## get antenna positions and pulse times
        PolE_pulseTime_dicts = {}
#        PolO_pulseTime_dicts = {}
        antPos_dicts = {}
        
        prefered_ant_pulsetimes = []
        prefered_ant_locs = []
        
        initial_search_time = np.inf
        final_search_time = 0.0
        
        for sname, antPulse_dict in AntennaPulse_dicts_refiltered.items():
            
            PolE_pulseTime_dicts[sname] = {}
#            PolO_pulseTime_dicts[sname] = {}
            antPos_dicts[sname] = {}
            
            for ant_name, pulse_list in antPulse_dict.items():
                antPos_dicts[sname][ant_name] = StationInfo_dict[sname].AntennaInfo_dict[ant_name].location
                
                PolE_pulseTime_dicts[sname][ant_name] = np.array( [pulse.PolE_peak_time for pulse in pulse_list] )
#                PolO_pulseTime_dicts[sname][ant_name] = np.array( [pulse.PolO_peak_time for pulse in pulse_list] )
                
                if len(PolE_pulseTime_dicts[sname][ant_name]) == 0:
                    continue

                if ant_name in prefered_antennas:
                    prefered_ant_pulsetimes.append( PolE_pulseTime_dicts[sname][ant_name] )
                    prefered_ant_locs.append( antPos_dicts[sname][ant_name] )
                
#                min_T = min( np.min(PolE_pulseTime_dicts[sname][ant_name]), np.min(PolO_pulseTime_dicts[sname][ant_name]) )
#                max_T = max( np.max(PolE_pulseTime_dicts[sname][ant_name]), np.max(PolO_pulseTime_dicts[sname][ant_name]) )
                min_T = np.min(PolE_pulseTime_dicts[sname][ant_name])
                max_T = np.max(PolE_pulseTime_dicts[sname][ant_name])
                DT = np.linalg.norm( search_location-antPos_dicts[sname][ant_name] )/v_air
                min_T -= DT
                max_T += DT
                
                if min_T < initial_search_time:
                    initial_search_time = min_T
                if max_T > final_search_time and np.isfinite(max_T):
                    final_search_time = max_T
                
        prefered_ant_locs = np.array( prefered_ant_locs )
        prefered_ant_nextIndex = np.zeros( len(prefered_ant_locs), dtype=int )
                
        ### now we loop over time, finding sources ###
        PSE_list = []
        current_time = initial_search_time
        while current_time<final_search_time:
            print("search time:", current_time, "out of:", final_search_time)
            
            while len(prefered_ant_locs) >= 5:
                
                 ## first we try various combinations of pulses of 5 antennas (stations)
                res, XYZT_guess, used_pulse_indeces = find_PSE_itteration(prefered_ant_locs, prefered_ant_pulsetimes, current_time, prefered_ant_nextIndex)
                XYZT_guess[2] = np.abs(XYZT_guess[2])
                ### PROBLEM !: this does not effectively find point sources (probably not quite an issue)

                event_ant_locs = []
                event_pulse_times = []
                
                for ant_ind, pulse_index in enumerate(used_pulse_indeces):
                    if pulse_index!=-1:
                        event_ant_locs.append( prefered_ant_locs[ant_ind] )
                        event_pulse_times.append( prefered_ant_pulsetimes[ant_ind][pulse_index] )
                        
                        prefered_ant_pulsetimes[ant_ind][pulse_index] = np.inf
                        
                if not np.isfinite(res):
                    break
                
                print()
                print("best res:", res, XYZT_guess)
                
                ## now we use the best solution to locate the event on as many antennas as possible
                event_ant_locs = np.array( event_ant_locs )
                event_pulse_times = np.array( event_pulse_times )
                
                residual_workspace = np.zeros(len(event_ant_locs))
                jacobian_workspace = np.zeros( (len(event_ant_locs), 4) )
                
                EvenPol_min = least_squares(initial_obj_fun_RES, XYZT_guess, jac=initial_obj_fun_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', 
                        args=(event_ant_locs, event_pulse_times), max_nfev=1000)
                XYZT_guess = EvenPol_min.x
                XYZT_guess[2] = np.abs(XYZT_guess[2])
                
                RMS = get_initial_RMS(XYZT_guess, event_ant_locs, event_pulse_times)
                
                print("best RMS and loc:", RMS, XYZT_guess, EvenPol_min.message)
                print()
                
                if RMS>max_ant_time_residual:
#                    break
                    continue
                
                
                
                
                #### loop to maximize fit and number of antennas
                current_location = XYZT_guess
                current_location[2] = np.abs(current_location[2])
                pulse_indeces, residuals = get_pulse_indecesNres(current_location, antPos_dicts, PolE_pulseTime_dicts, max_ant_time_residual)
                
                num_ant = len(pulse_indeces)
                
                found_PSE = False
                ### PROBLEM 2: this looping algorithm can't converge on a point source. (initial times do not give a result that is sufficently constrained)
                while num_ant>=5:
                    current_location[2] = np.abs(current_location[2])
                    
                    residual_workspace = np.zeros(num_ant)
                    jacobian_workspace = np.zeros( (num_ant, 4) )

                    EvenPol_min = least_squares(individual_obj_fun_RES, current_location, jac=individual_obj_fun_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', 
                                            args=(antPos_dicts, PolE_pulseTime_dicts, pulse_indeces), max_nfev=1000)
                

                    pulse_indeces, residuals= get_pulse_indecesNres(EvenPol_min.x, antPos_dicts, PolE_pulseTime_dicts, max_ant_time_residual)
                    new_num_antennas = len(pulse_indeces)
                    
                    if new_num_antennas ==0:
                        print("WARINGING! zero ant")
                        break
                    
                    print("loop:")
                    print("  ", EvenPol_min.message)
                    print("   maximum residual:", np.max(np.abs(residuals)))
                    print("   loc:", EvenPol_min.x)
                    print("   num ant:", new_num_antennas, "RMS:", np.sqrt( np.sum(residuals**2)/new_num_antennas ) )
                    
                    if new_num_antennas<num_ant:
                        print("WARNING! number of antennas is reduced!")
                    
                    if new_num_antennas <5:
                        print("not enough ant")
                        break
                    elif new_num_antennas == num_ant:
                        ### we have converged!!!
                        if new_num_antennas < num_ant_threshold:
                            print("not enough ant")
                            break
                        
                        ## remove the point sources=
                        PSE_pulse_dict = {}
                        for pulse_i, sname, ant_name in pulse_indeces:
                            PSE_pulse_dict[ant_name] = AntennaPulse_dicts_refiltered[sname][ant_name][ pulse_i ]
                            PolE_pulseTime_dicts[sname][ant_name][pulse_i] = np.inf
                        
                        new_point_source = PointSourceEvent(PSE_pulse_dict, EvenPol_min.x, StationInfo_dict)
                        PSE_list.append( new_point_source )
                        found_PSE = True
                        
                        break
                    else:
                        num_ant = new_num_antennas
                        current_location = EvenPol_min.x
            
                if not found_PSE:
                    break ### PROBLEM 3: we break too early. It may be possible that find_PSE_itteration can return a bad solution BEFORE it returns a good one 
            
            current_time += bin_half_width
            
        ### need mechanism to recyle unused pulses!
            
        print("finished block. plotting and saving")
            
        #### plot left-over pulses
        CP = curtain_plot_CodeLog(StationInfo_dict, logging_folder + "/leftover_singleStation_events_"+str(current_block))
        
        ####plot the pulses
        for sname, pulses in AntennaPulse_dicts_refiltered.items():
            ATD = AntennaPulse_dict__TO__AntennaTime_dict(pulses, StationInfo_dict[sname])
            CP.add_AntennaTime_dict(sname, ATD, color='b', marker='o', size=50)
            
        ### plot the PSE
        for sname,station_info in StationInfo_dict.items():
            inclusion_model_times = [ PSE.get_ModelTime(station_info, 0, 1) for PSE in PSE_list ]
            exculsion_model_times = [ PSE.get_ModelTime(station_info, 0, 2) for PSE in PSE_list ]
            annotations = [ PSE.unique_index for PSE in PSE_list ]
                        
            CP.addEventList(sname, inclusion_model_times, 'g', marker='+', size=50, annotation_list=annotations, annotation_size=20)
            CP.addEventList(sname, exculsion_model_times, 'r', marker='+', size=50)
            
        CP.annotate_station_names(size=30, t_offset=-3.0E-6)
        CP.save()
        
        
        #### save the PSE!
        with open(data_dir + '/point_sources_'+str(current_block), 'wb') as fout:
            
            write_long(fout, 1) ## means mod data is next
            getNwriteBin_modData(fout, StationInfo_dict)
                    
            ant_pos_dict = {}
            for sinfo in StationInfo_dict.values():
                for ant_name, ant_data in sinfo.AntennaInfo_dict.items():
                    ant_pos_dict[ant_name] = ant_data.location
             
            write_long(fout, 2) ## means antenna location data is next
            write_long(fout, len(ant_pos_dict))
            for ant_name, ant_loc in ant_pos_dict.items():
                write_string(fout, ant_name)
                write_double_array(fout, ant_loc)
                
            
            write_long(fout,3) ## means PSE data is next
            write_long(fout, len(PSE_list))
            for PSE in PSE_list:
                PSE.save_as_binary(fout)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

