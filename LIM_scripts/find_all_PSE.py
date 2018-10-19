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
        self.ant_times = PolE_times
        EvenPol_min = least_squares(self.objective_RES, guess_location, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', max_nfev=1000)
        self.PolE_loc = EvenPol_min.x
        self.PolE_RMS = np.sqrt( self.SSqE(self.PolE_loc)/float(self.num_even_ant) )
        
        ### fit odd polarisation ###
        self.ant_times = PolO_times
        OddPol_min = least_squares(self.objective_RES, guess_location, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', max_nfev=1000)
        self.PolO_loc = OddPol_min.x
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
            
        
        

##### roaming hyper-bubble #####
def count_antennas(XYZT, half_R_t, antPos_dicts, antTime_dicts):
    max_diff = 0.0
    n_ant = 0
    for sname, antPulse_dict in antTime_dicts.items():
        for ant_name, pulse_list in antPulse_dict.items():
                    
            ant_loc = antPos_dicts[sname][ant_name]
            DT = np.linalg.norm( XYZT[0:3]-ant_loc )/v_air
                    
            min_box = XYZT[3] + DT - half_R_t
            max_box = XYZT[3] + DT + half_R_t

            minInd = np.searchsorted(antTime_dicts[sname][ant_name], min_box)
                    
            maxInd = np.searchsorted(antTime_dicts[sname][ant_name], max_box)
                    
            if (maxInd-minInd)>0:
                n_ant += 1
                delta_T = antTime_dicts[sname][ant_name][minInd:maxInd] - (XYZT[3] + DT)
                max_delta = np.max( np.abs(delta_T) )
                if max_delta>max_diff:
                    max_diff = max_delta
            
    return n_ant, max_diff
        
def AlgorithmOfTheRoamingHypersphere(min_half_R_t, initial_half_R_t, initial_XYZT, antPos_dicts, antTime_dicts):
    
    initial_XYZT = np.array(initial_XYZT)
    initial_XYZT[0:3] *= 1.0/v_air ##put everything in seconds, so that Nelder-Mead tollarances are consistent
    
    def obj_fun(XYZT):
        ### hope that this is safe to modify....=
        scaled_XYZT = np.array(XYZT)
        scaled_XYZT[0:3] *= v_air ##put correct things back into meters
        
        N, trash = count_antennas(scaled_XYZT, current_half_R_t, antPos_dicts, antTime_dicts)
        if N == 0:
            return 2
        else:
            return 1.0/N
    
    current_half_R_t = initial_half_R_t
    current_XYZT  = initial_XYZT
    i = 0
    while current_half_R_t > min_half_R_t:
        i += 1
        
        min_ret = minimize(obj_fun, current_XYZT, method='Nelder-Mead', options={'xatol':current_half_R_t/100.0})
        current_XYZT = min_ret.x
        
        
        rescaled_XYZT = np.array(current_XYZT)
        rescaled_XYZT[0:3] *= v_air ##put correct things back into meters
        current_N,  max_dif = count_antennas(rescaled_XYZT, current_half_R_t, antPos_dicts, antTime_dicts)
#        print("RHS itter:", i, current_half_R_t, current_N, " "*50, end='\r', flush=True)
        
#        print()
#        print( current_half_R_t )
#        print( current_N, i)
#        print( rescaled_XYZT, min_ret.message )
#        print( max_dif/current_half_R_t,  current_N)

        new_current_half_R_t = max_dif*0.99
        
        if not new_current_half_R_t<current_half_R_t:
            current_half_R_t *= 0.75
        else: 
            current_half_R_t = new_current_half_R_t
        
    i += 1
    current_half_R_t = min_half_R_t
    min_ret = minimize(obj_fun, current_XYZT, method='Nelder-Mead', options={'xatol':current_half_R_t/100.0})
    current_XYZT = min_ret.x
    
    rescaled_XYZT = np.array(current_XYZT)
    rescaled_XYZT[0:3] *= v_air ##put correct things back into meters
    current_N,  max_dif = count_antennas(rescaled_XYZT, current_half_R_t, antPos_dicts, antTime_dicts)
    print("RHS itter:", i, current_half_R_t, current_N, " "*50)
        
    
#    current_XYZT[0:3] *= v_air ##back into meters

    return rescaled_XYZT, current_N

#def AlgorithmOfTheRoamingHypersphere_2(min_half_R_t, initial_half_R_t, initial_XYZT, antPos_dicts, antTime_dicts):
#    
#    #### write algorithm that works simular to the minimization loop that runs after the hyperspher algorithm. Except reduce the time diff each time. 
#    
#    ## have two loops, an inner loop that minimizes location each run acn clauclates number of antennas in radius. Stops running once number of antennas is stable. 
#    
#    ## an outer loop that reduces radius each turn. Stops once radius is small enough. 
#    
#    
#    current_half_R_t = initial_half_R_t
#    
#    pulse_indeces, residuals= get_pulse_indecesNres(initial_XYZT, antPos_dicts, PolE_pulseTime_dicts, current_half_R_t)
#    current_num_antennas = len(pulse_indeces)
#    
#    current_location = np.array( initial_XYZT )
##    current_location *= 1.0/v_air ##convert units to seconds
#    
#    outer_i = 0
#    while current_half_R_t > min_half_R_t and current_num_antennas>=num_ant_threshold:
#        outer_i += 1
#        
#        pulse_indeces, residuals= get_pulse_indecesNres(current_location, antPos_dicts, PolE_pulseTime_dicts, current_half_R_t)
#        current_num_antennas = len(pulse_indeces)
#        print("outer loop", outer_i, current_half_R_t, current_num_antennas)
#        
#        inner_i = 0
#        while True:
#            inner_i += 1
#            current_location[2] = np.abs(current_location[2])
#
#            EvenPol_min = least_squares(individual_obj_fun_RES, current_location, jac=individual_obj_fun_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', 
#                                        max_nfev=1000, args=(antPos_dicts, PolE_pulseTime_dicts, pulse_indeces))
#            
#            rescaled_loc = np.array(EvenPol_min.x)
##            rescaled_loc[0:3] *= v_air
#
#            pulse_indeces, residuals= get_pulse_indecesNres(rescaled_loc, antPos_dicts, PolE_pulseTime_dicts, current_half_R_t)
#            new_num_antennas = len(pulse_indeces)
#                    
#            print("RHS loop", outer_i, inner_i)
#            print("  ", EvenPol_min.message)
#            print("   loc:", rescaled_loc)
#            print("   num ant:", new_num_antennas, "RMS:", np.sqrt( np.sum(residuals**2)/new_num_antennas ) )
#                    
#            if new_num_antennas ==0:
#                print("WARINGING! zero ant")
#                break
#            
#            max_residual = np.max(np.abs(residuals))
#            print("   maximum residual:", max_residual)
#                    
#            if new_num_antennas<current_num_antennas:
#                print("WARNING! number of antennas is reduced!")
#                    
#            if new_num_antennas <5:
#                print("not enough ant")
#                break
#            elif new_num_antennas == current_num_antennas:
#                current_num_antennas = new_num_antennas
#                current_location = EvenPol_min.x
#                break
#            else:
#                current_num_antennas = new_num_antennas
#                current_location = EvenPol_min.x
#                
#                
#        new_current_half_R_t = max_residual*0.99
#        if not new_current_half_R_t<current_half_R_t:
#            current_half_R_t *= 0.75
#        else: 
#            current_half_R_t = new_current_half_R_t
        
        
    

##### routines for initial fitting #####


def get_pulse_indecesNres(XYZT, antPos_dicts, antTime_dicts, T_window):
    XYZT[2] = np.abs(XYZT[2])
    
    pulse_indeces = []
    pulse_residuals = []
    for sname, pulse_dict in antTime_dicts.items():
        for ant_name, pulse_times in pulse_dict.items():
            
            ant_loc = antPos_dicts[sname][ant_name]
            DT = np.linalg.norm( XYZT[0:3]-ant_loc )/v_air
            
            residuals = np.abs( antTime_dicts[sname][ant_name] - (XYZT[3] + DT) )
            idx = np.argmin( residuals )
            
            if residuals[idx]<T_window:
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


if __name__ == "__main__":
    
    timeID = "D20170929T202255.000Z"
    output_folder = "allPSE_runA"
    
    plot_station_map = True
    
    stations_to_exclude =  ["CS028","RS106", "RS305", "RS205", "CS201", "RS407"]#, "RS406"]
    
    num_blocks_per_step = 100
    initial_block = 3500
    num_steps = 10
    
    ant_timing_calibrations = "cal_tables/TxtAntDelay"
    polarization_flips = "polarization_flips.txt"
    bad_antennas = "bad_antennas.txt"
    additional_antenna_delays = "ant_delays.txt"
    
    station_delays = "/station_delays.txt"
    
    min_signal_SNR = 0.0
    
    search_location = np.array( [-1.58423240e+04,   9.08114847e+03,   4.0e+03] )
    hyphersphere_initial_size = 20000.0
    hypersphere_min_size = 10.0E-9 ##1000
     
    max_ant_time_residual = 100.0E-9 #### if the residual of an antenna time is greater than this, the antenna is thrown out ## 1000E-9
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
    print("Hypersphere search location:", search_location)
    print("Hypersphere initial size (m):", hyphersphere_initial_size)
    print("Hypersphere min size (s):", hypersphere_min_size)
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
        
    print("using stations:", StationInfo_dict.keys())

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

    half_radius_time = hyphersphere_initial_size/(2.0*v_air)

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
        
        initial_sphere_search_time = np.inf
        final_sphere_search_time = 0.0
        
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
                
#                min_T = min( np.min(PolE_pulseTime_dicts[sname][ant_name]), np.min(PolO_pulseTime_dicts[sname][ant_name]) )
#                max_T = max( np.max(PolE_pulseTime_dicts[sname][ant_name]), np.max(PolO_pulseTime_dicts[sname][ant_name]) )
                min_T = np.min(PolE_pulseTime_dicts[sname][ant_name])
                max_T = np.max(PolE_pulseTime_dicts[sname][ant_name])
                DT = np.linalg.norm( search_location-antPos_dicts[sname][ant_name] )/v_air
                min_T -= DT
                max_T += DT
                
                if min_T < initial_sphere_search_time:
                    initial_sphere_search_time = min_T
                if max_T > final_sphere_search_time and np.isfinite(max_T):
                    final_sphere_search_time = max_T
                
                
        ### now we loop over time, finding sources ###
        PSE_list = []
        current_time = initial_sphere_search_time
        while current_time<final_sphere_search_time:
            print("search time:", current_time, "out of:", final_sphere_search_time)
            current_box_XYZT = np.append(search_location, [current_time])
            
            while True: ## loop untill all PSE are found
                num_ant,  trash = count_antennas(current_box_XYZT, half_radius_time, antPos_dicts, PolE_pulseTime_dicts)
                
                if num_ant < num_ant_threshold:
                    break
                
                #### prune the pulse dictionaries and count active antennas ###
                num_active_ant = 0
                for sname, antpulse_dict in AntennaPulse_dicts_refiltered.items():
                    #### remove antennas with no pulses
                    ants_to_remove = [antname for antname,pulse_list in antpulse_dict.items() if (len(pulse_list) == 0)]
                    for ant in ants_to_remove:
                        del antpulse_dict[ant]
                        
                        del PolE_pulseTime_dicts[sname][ant]
#                        del PolO_pulseTime_dicts[sname][ant]
                        del antPos_dicts[sname][ant]
                        
                    #### count number of antennas
                    num_active_ant += len(antpulse_dict)
                    
                residual_workspace = np.zeros(num_active_ant)
                jacobian_workspace = np.zeros( (num_active_ant, 4) )
                print(num_active_ant, "active antennas")
                
                
                
                
                print()
                print("initial found PSE:", current_time, num_ant)
                print("hypersphere:")
                guess_XYZT, num_ant = AlgorithmOfTheRoamingHypersphere(hypersphere_min_size, half_radius_time, current_box_XYZT, antPos_dicts, PolE_pulseTime_dicts)
#                guess_XYZT = current_box_XYZT
#                num_itter = 0
#                print("hypersphere results:", num_ant, "num itter:", num_itter)
                print("   ", guess_XYZT)
                
                if num_ant < 10:
                    print("not enough ant")
                    break
                    
                
                
                
                
                
                #### loop to maximize fit and number of antennas
                current_location = guess_XYZT
                current_location[2] = np.abs(current_location[2])
                pulse_indeces, residuals = get_pulse_indecesNres(current_location, antPos_dicts, PolE_pulseTime_dicts, max_ant_time_residual)
                
                new_num_antennas = len(pulse_indeces)
                
                
#                print(new_num_antennas, num_ant)
#                if new_num_antennas < num_ant:
#                    quit()
                    
                previous_num_antennas = None
                num_ant = new_num_antennas
                
                found_PSE = False
                
                while True:
                    current_location[2] = np.abs(current_location[2])

                    EvenPol_min = least_squares(individual_obj_fun_RES, current_location, jac=individual_obj_fun_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', 
                                            args=(antPos_dicts, PolE_pulseTime_dicts, pulse_indeces), max_nfev=1000)
                

                    pulse_indeces, residuals= get_pulse_indecesNres(EvenPol_min.x, antPos_dicts, PolE_pulseTime_dicts, max_ant_time_residual)
                    new_num_antennas = len(pulse_indeces)
                    
                    if new_num_antennas ==0:
                        print("WARNING! zero ant")
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
                    elif new_num_antennas == num_ant or (previous_num_antennas is not None and new_num_antennas==previous_num_antennas) :
                        ### we have converged!!!
                        if new_num_antennas < num_ant_threshold:
                            print("not enough ant")
                            break
                        
                        ## remove the point sources=
                        PSE_pulse_dict = {}
                        for pulse_i, sname, ant_name in pulse_indeces:
                            PSE_pulse_dict[ant_name] = AntennaPulse_dicts_refiltered[sname][ant_name][ pulse_i ]
                            del AntennaPulse_dicts_refiltered[sname][ant_name][ pulse_i ]
                            PolE_pulseTime_dicts[sname][ant_name] = np.delete(PolE_pulseTime_dicts[sname][ant_name], pulse_i )
#                            PolO_pulseTime_dicts[sname][ant_name] = np.delete(PolO_pulseTime_dicts[sname][ant_name], pulse_i )
                        
                        new_point_source = PointSourceEvent(PSE_pulse_dict, EvenPol_min.x, StationInfo_dict)
                        PSE_list.append( new_point_source )
                        found_PSE = True
                        
                        break
                    else:
                        previous_num_antennas = num_ant
                        num_ant = new_num_antennas
                        current_location = EvenPol_min.x
                        
                if not found_PSE:
                    break
            
            current_time += half_radius_time
            
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
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

