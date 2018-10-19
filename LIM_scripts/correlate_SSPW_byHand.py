#!/usr/bin/env python3
##python
from os import mkdir
from os.path import isdir
import time
import bisect

##external
import numpy as np
from scipy.optimize import brute, minimize, least_squares

#mine
from LoLIM.utilities import v_air, log, processed_data_dir, SId_to_Sname#SNum_to_SName_dict
from LoLIM.planewave_functions import read_SSPW_timeID, multi_planewave_intersections
from LoLIM.IO.binary_IO import write_long, write_double, write_double_array, write_string
from LoLIM.read_PSE import writeBin_modData_T2


class pointsource_event(object):
    def __init__(self, ant_loc_dict, SSPW_dict, referance_station, initial_guess, station_offsets_guess={}):
        self.ant_loc_dict = ant_loc_dict
        self.SSPW_dict = SSPW_dict
        self.referance_station = referance_station
        
        self.unique_index = None
        
        self.source_location_guess = initial_guess
        
#        self.source_location_guess_PWI, T, junk, junk = multi_planewave_intersections(SSPW_dict, referance_station, False)
#        self.source_location_guess_PWI = np.append(self.source_location_guess_PWI, [T])
        
            
        
        ## setup for fitting ##
        ant_locations = []
        PolE_times = []
        PolO_times = []
        self.ant_info = []
        self.station_index_range = {}
        
        self.num_even_ant = 0
        self.num_odd_ant = 0
        
        PolE_amps = []
        PolO_amps = []
        
        for station_name, SSPW in self.SSPW_dict.items():
            station_slice_start = len(ant_locations)
            
            station_delay_guess = 0.0
            if station_name in station_offsets_guess:
                station_delay_guess = station_offsets_guess[station_name]
                
        
            for ant_data in SSPW.ant_data.values():
                if ant_data.antenna_status == 3:
                    continue
                
                loc = ant_loc_dict[ant_data.ant_name]
                ant_locations.append( loc )
                self.ant_info.append( ant_data )
                
                PolE_times.append( ant_data.PolE_peak_time-station_delay_guess ) ##note that peak time is infinity if something is wrong with antenna
                PolO_times.append( ant_data.PolO_peak_time-station_delay_guess )
                 
                if np.isfinite( ant_data.PolE_peak_time ):
                    self.num_even_ant += 1
                    PolE_amps.append( np.max(ant_data.PolE_hilbert_envelope) )
                    
                if np.isfinite( ant_data.PolO_peak_time ):
                    self.num_odd_ant += 1
                    PolO_amps.append( np.max(ant_data.PolO_hilbert_envelope) )
                    
            self.station_index_range[ station_name ] = [station_slice_start, len(ant_locations)]
            
        self.antenna_locations = np.array(ant_locations)
        self.PolE_times = np.array(PolE_times)
        self.PolO_times = np.array(PolO_times)

        self.PolE_amps = np.array( PolE_amps )
        self.PolO_amps = np.array( PolO_amps )
        
        self.ave_PolE_amp = np.average( self.PolE_amps )
        self.ave_PolO_amp = np.average( self.PolO_amps )
        self.std_PolE_amp = np.std( self.PolE_amps )
        self.std_PolO_amp = np.std( self.PolO_amps )
        
        self.num_antennas = len(self.antenna_locations)
        self.residual_workspace = np.zeros(2*self.num_antennas)
        self.jacobian_workspace = np.zeros((2*self.num_antennas, 8))
        
        guess = np.append( self.source_location_guess, self.source_location_guess )
        IGuess_min = least_squares(self.objective_RES, guess, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', max_nfev=1000)
        fit_res = IGuess_min
        
#        guess = np.append( self.source_location_guess_PWI, self.source_location_guess_PWI )
#        PWIGuess_min = least_squares(self.objective_RES, guess, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', max_nfev=1000)
        
#        if IGuess_min.cost < PWIGuess_min.cost:
#            fit_res = IGuess_min
#        else:
#            fit_res = PWIGuess_min
            
        self.PolE_loc = fit_res.x[:4]
        self.PolO_loc = fit_res.x[4:]
        self.fit_message = fit_res.message
        
        PolE_SSqE, PolO_SSqE = self.SSqE(fit_res.x)
        self.PolE_RMS = np.sqrt( PolE_SSqE/self.num_even_ant )
        self.PolO_RMS = np.sqrt( PolO_SSqE/self.num_odd_ant )
        
        
        del self.residual_workspace
        del self.jacobian_workspace
        
        
    def objective_RES(self, XYZT_2):
        
        self.residual_workspace[:self.num_antennas] = self.PolE_times 
        self.residual_workspace[:self.num_antennas] -= XYZT_2[3]
        
        self.residual_workspace[self.num_antennas:] = self.PolO_times 
        self.residual_workspace[self.num_antennas:] -= XYZT_2[7]
        
        self.residual_workspace[:] *= v_air
        self.residual_workspace[:] *= self.residual_workspace[:]
        
        E_R2 = self.antenna_locations - XYZT_2[0:3]
        E_R2 *= E_R2
        self.residual_workspace[:self.num_antennas] -= np.sum( E_R2 )
        
        
        O_R2 = self.antenna_locations - XYZT_2[4:7]
        O_R2 *= O_R2
        self.residual_workspace[self.num_antennas:] -= np.sum( E_R2 )
        
        self.residual_workspace[ np.logical_not(np.isfinite(self.residual_workspace)) ] = 0.0
        
        return np.array( self.residual_workspace )
    
    def objective_JAC(self, XYZT_2):
        self.jacobian_workspace[:self.num_antennas, 0:3] = XYZT_2[0:3]
        self.jacobian_workspace[:self.num_antennas, 0:3] -= self.antenna_locations
        self.jacobian_workspace[:self.num_antennas, 0:3] *= -2.0
        
        
        self.jacobian_workspace[self.num_antennas:, 4:7] = XYZT_2[4:7]
        self.jacobian_workspace[self.num_antennas:, 4:7] -= self.antenna_locations
        self.jacobian_workspace[self.num_antennas:, 4:7] *= -2.0
        
        self.jacobian_workspace[:self.num_antennas, 3] = self.PolE_times
        self.jacobian_workspace[:self.num_antennas, 3] -= XYZT_2[3]
        self.jacobian_workspace[:self.num_antennas, 3] *= -v_air*v_air*2
        
        self.jacobian_workspace[self.num_antennas:, 7] = self.PolO_times
        self.jacobian_workspace[self.num_antennas:, 7] -= XYZT_2[7]
        self.jacobian_workspace[self.num_antennas:, 7] *= -v_air*v_air*2
        
        mask = np.logical_not(np.isfinite(self.jacobian_workspace[:, 3])) 
        self.jacobian_workspace[mask, :] = 0
        
        mask = np.logical_not(np.isfinite(self.jacobian_workspace[:, 7])) 
        self.jacobian_workspace[mask, :] = 0
        
        return np.array( self.jacobian_workspace )
    
    def SSqE(self, XYZT_2):
        E_R2 = self.antenna_locations - XYZT_2[0:3]
        E_R2 *= E_R2
        
        E_theory = np.sum(E_R2, axis=1)
        np.sqrt(E_theory, out=E_theory)
        
        E_theory *= 1.0/v_air
        E_theory += XYZT_2[3] - self.PolE_times
        
        E_theory *= E_theory
        E_theory[ np.logical_not(np.isfinite(E_theory) ) ] = 0.0
        
        
        O_R2 = self.antenna_locations - XYZT_2[4:7]
        O_R2 *= O_R2
        
        O_theory = np.sum(O_R2, axis=1)
        np.sqrt(O_theory, out=O_theory)
        
        O_theory *= 1.0/v_air
        O_theory += XYZT_2[7] - self.PolO_times
        
        O_theory *= O_theory
        O_theory[ np.logical_not(np.isfinite(O_theory) ) ] = 0.0
        
        return np.sum(E_theory), np.sum(O_theory)
    
    
    
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
        
        write_long(fout, len(self.ant_info))
        for ant_data in self.ant_info:
            write_string(fout, ant_data.ant_name)
            write_long(fout, ant_data.pulse_section_number)
            write_long(fout, ant_data.pulse_unique_index)
            write_long(fout, ant_data.pulse_starting_index)
            write_long(fout, ant_data.antenna_status)
            
            write_double(fout, ant_data.PolE_peak_time)
            write_double(fout, 0.0)
            write_double(fout, ant_data.PolE_time_offset)
            write_double(fout, np.max(ant_data.PolE_hilbert_envelope))
            write_double(fout, ant_data.PolE_std)
            
            write_double(fout, ant_data.PolO_peak_time)
            write_double(fout, 0.0)
            write_double(fout, ant_data.PolO_time_offset)
            write_double(fout, np.max(ant_data.PolO_hilbert_envelope))
            write_double(fout, ant_data.PolO_std)
            
            write_double_array(fout, ant_data.PolE_hilbert_envelope)
            write_double_array(fout, ant_data.PolE_antenna_data)
            write_double_array(fout, ant_data.PolO_hilbert_envelope)
            write_double_array(fout, ant_data.PolO_antenna_data)


    
    
    
if __name__ == "__main__":
        
    ##opening data
    timeID = "D20170929T202255.000Z"
    
    output_folder = "handcorrelate_SSPW"
    
    SSPW_folder = "/SSPW"
    first_block = 3450
    num_blocks = 200
    
    station_offsets_guess = {
        "CS002" : 0.0,
        "CS003" : 1.40604830769e-06,
        "CS004" : 4.30394544075e-07,
        "CS005" : -2.20261890579e-07,
        "CS006" : 4.33335851395e-07,
        "CS007" : 4.00830327457e-07,
        "CS011" : -5.90195835997e-07,
        "CS013" : -1.80837037148e-06,
        "CS017" : -8.44121045587e-06,
        "CS021" : 9.29644391506e-07,
        "CS030" : -2.7261703611e-06,
        "CS032" : -1.70410037907e-06,
        "CS101" : -8.16593767923e-06,
        "CS103" : -2.85284698181e-05,
        "RS106" : 6.9014917479e-06,
        "CS201" : -1.04924709196e-05,
        "RS205" : -9.64484475983e-09,
        "RS208" : 6.89820543045e-06,
        "CS301" : -7.28095713039e-07,
        "CS302" : -5.36414352515e-06,
        "RS305" : 6.26687518009e-06,
        "RS306" : 7.01257893086e-06,
        "RS307" : 6.79091242133e-06,
        "RS310" : 6.69851747251e-06,
        "CS401" : -1.16597207744e-06,
        "RS406" : 5.45597491921e-07,
        "RS409" : 8.22050319577e-06,
        "CS501" : -9.59499655097e-06,
        "RS503" : 6.991534839e-06,
        "RS508" : 6.99681743893e-06,
        "RS509" : 6.97295496941e-06,     
        }
    
    unique_index = 7
    guess_location = [-15000.0, 9000.0, 5000.0, 1.32211980e+00]
    SSPW_to_use = {
            "CS002":274381,
            "CS003":23093,
            "CS004":12711,
            "CS005":87180,
            "CS006":None,
            "CS007":144716,
            "CS011":318596,
            "CS013":240928,
            "CS017":107488,
            "CS021":155984,
            "CS030":252618,
            "CS032":211086,
            "CS101":35160,
            "CS103":175837,
            "RS208":98564,
            "CS301":77047,
            "CS302":45432,
            "RS306":None,
            "RS307":329859,
            "RS310":None,
            "RS401":234187,
            "RS406":None,
            "RS409":None,
            "CS501":119480,
            "RS503":263319,
            "RS508":None,
            "RS509":130016,
            }
    
    
    #### setup directory variables ####
    processed_data_dir = processed_data_dir(timeID)
    
    data_dir = processed_data_dir + "/" + output_folder
    if not isdir(data_dir):
        mkdir(data_dir)
        
    logging_folder = data_dir + '/logs_and_plots'
    if not isdir(logging_folder):
        mkdir(logging_folder)

    #Setup logger and open initial data set
    log.set(logging_folder + "/log_out_"+str(unique_index)+".txt") ## TODo: save all output to a specific output folder
    log.take_stderr()
    log.take_stdout()
    
    
    log("PSE ID", unique_index)
    log("station offset guess:", station_offsets_guess)
    log()
    log()
    log("SSPW:", SSPW_to_use)
    log()
    log()
    
    planewave_info_dict = read_SSPW_timeID(timeID, SSPW_folder, min_block=first_block, max_block=first_block+num_blocks)
#    allSSPW_dict = planewave_info_dict["SSPW_dict"]
    ant_locs = planewave_info_dict["ant_locations"]
    
    ## find SSPW ##
    station_locs = {}
    for ant, loc in ant_locs.items():
        sname = SId_to_Sname[ int(ant[:3]) ]
        
        if sname not in station_locs:
            station_locs[sname] = []
        station_locs[sname].append(loc)
        
    station_locs = {sname:np.average(all_locs, axis=0) for sname,all_locs in station_locs.items()}
    
    SSPW_dict = {}
    for sname, SSPW_list in planewave_info_dict["SSPW_dict"].items():
        if sname not in SSPW_to_use:
            continue
        
        SSPW_ID = SSPW_to_use[sname]
        if SSPW_ID is None:
            continue
        
        found=False
        for SSPW in SSPW_list:
            if SSPW.unique_index == SSPW_ID:
                print("found", SSPW_ID, "fit", SSPW.fit)
                SSPW.prep_fitting(station_locs[sname], ant_locs )
                SSPW_dict[sname] = SSPW 
                found=True
                break
            
        if not found:
            log("could not find SSPW", SSPW_ID, "for station", sname)
          
    
    ### fit the PSE!
    new_PSE = pointsource_event(ant_locs, SSPW_dict, "CS002", guess_location, station_offsets_guess)
    new_PSE.unique_index = unique_index
    
    print(new_PSE.PolE_RMS, new_PSE.PolE_loc)
    print(new_PSE.PolO_RMS, new_PSE.PolO_loc)
    
    with open(data_dir + '/point_sources_'+str(unique_index), 'wb') as fout:
        
        write_long(fout, 5)
        writeBin_modData_T2(fout, station_delay_dict=planewave_info_dict['stat_delays'], ant_delay_dict=planewave_info_dict["ant_delays"], bad_ant_list=planewave_info_dict['bad_ants'], flipped_pol_list=planewave_info_dict['pol_flips'])
    
        write_long(fout, 2) ## means antenna location data is next
        write_long(fout, len(ant_locs))
        for ant_name, ant_loc in ant_locs.items():
            write_string(fout, ant_name)
            write_double_array(fout, ant_loc)
                
            
        write_long(fout,3) ## means PSE data is next
        write_long(fout, 1)
        new_PSE.save_as_binary(fout)
    
    
    