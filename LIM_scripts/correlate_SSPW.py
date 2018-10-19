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


PSE_next_unique_i = 0
class pointsource_event(object):
    def __init__(self, ant_loc_dict, SSPW_dict, referance_station, initial_guess, station_offsets_guess={}):
        self.ant_loc_dict = ant_loc_dict
        self.SSPW_dict = SSPW_dict
        self.referance_station = referance_station
        
        global PSE_next_unique_i
        self.unique_index = PSE_next_unique_i
        PSE_next_unique_i += 1
        
        self.source_location_guess = initial_guess
        
        self.source_location_guess_PWI, T, junk, junk = multi_planewave_intersections(SSPW_dict, referance_station, False)
        self.source_location_guess_PWI = np.append(self.source_location_guess_PWI, [T])
        
            
        
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
    
        
        guess = np.append( self.source_location_guess_PWI, self.source_location_guess_PWI )
        PWIGuess_min = least_squares(self.objective_RES, guess, jac=self.objective_JAC, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac', max_nfev=1000)
        
        if IGuess_min.cost < PWIGuess_min.cost:
            fit_res = IGuess_min
        else:
            fit_res = PWIGuess_min
            
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


def correlate_stations(ant_loc_dict, SSPW_dict, referance_station, final_fit_val, min_time_between_planewaves, station_offsets_guess={}):
    #### algorithm:
        ## loop over each SSPW in ref. station, call it referance SSPW
        ##   for each ref. SSPW,  find  location on the SSPW ray that minimizes the spherical wave equations (choosing best fitting SSPW on all stations)
        ##      am assuming that timing error in each station is less than min_time_between_planewaves
        
    
    #### first we get the physical center of each station
    station_locs = {}
    for ant, loc in ant_loc_dict.items():
        sname = SId_to_Sname[ int(ant[:3]) ]
        
        if sname not in station_locs:
            station_locs[sname] = []
        station_locs[sname].append(loc)
        
    station_locs = {sname:np.average(all_locs, axis=0) for sname,all_locs in station_locs.items()}
    
    for sname,SSPW_list in SSPW_dict.items():
        stat_delay_guess = 0.0
        if sname in station_offsets_guess:
            stat_delay_guess = station_offsets_guess[sname]
        
        for SSPW in SSPW_list:
            SSPW.prep_fitting( station_locs[SSPW.sname], ant_loc_dict )
            SSPW.ZAT[2] -= stat_delay_guess
            SSPW.pulse_times -= stat_delay_guess
    
    
    #### next, we get the times of planewaves (and referance locations), so that we can pick the best planewaves 
    planewave_time_dict = {}
    for sname, planewave_list in SSPW_dict.items():
        ## find times of planewaves at each station
        ## time is infinity if planewave is already used
        ## do all this with list comprehension, becouse it is faster
        ## I hope I can read this line in the future
        
        SSPW_times = np.array([ (SSPW.ZAT[2] )  for SSPW in planewave_list])
        referance_locs = np.array([ SSPW.ref_location for SSPW in planewave_list])
        
        planewave_time_dict[sname] = [SSPW_times, referance_locs]




    #### some sub-algorithms we will need
    ref_index = None
    def TryLoc_pickSSPW(XYZT):
        """ try a XYZT location, pick the best fitting SSPW"""
        
        ret = 0.0
        num = 0.0
        for sname, (SSPW_times, SSPW_locs) in planewave_time_dict.items():
            
            if len(SSPW_times)==0:
                continue
            
            ## first we compare arrival time to planewave arrival time, in order to pick the best SSPW
            theory_times = np.linalg.norm(SSPW_locs - XYZT[0:3], axis=1)/v_air + XYZT[3]
            
            abs_diff = np.abs( theory_times - SSPW_times )  #### TODO: why is this so large, even for perfect synth data?
            best_planewave_index = abs_diff.argmin()
          

            SSPW = SSPW_dict[sname][best_planewave_index]
            
            delta_X = SSPW.antenna_X - XYZT[0]
            delta_Y = SSPW.antenna_Y - XYZT[1]
            delta_Z = SSPW.antenna_Z - XYZT[2]
                
            delta_X*=delta_X
            delta_Y*=delta_Y
            delta_Z*=delta_Z
        
#            estimated_arrival_times = np.sqrt(delta_X + delta_Y + delta_Z)/C + XYZT[3] 
                
#            diff = estimated_arrival_times - (SSPW.pulse_times - station_delay_dict[sname])
#            diff *= diff
            ##FOR SPEED!!
            estimated_arrival_times = delta_X
            estimated_arrival_times += delta_Y
            estimated_arrival_times += delta_Z
            np.sqrt(estimated_arrival_times, out=estimated_arrival_times)
            estimated_arrival_times *= 1.0/v_air
            estimated_arrival_times += XYZT[3]
            
            diff = estimated_arrival_times
            diff -= SSPW.pulse_times
            diff *= diff
                
            ret += np.sum(diff)
            num += len(diff)
            
        return ret/num
    
    
    
    def GetBestSSPW(XYZT):
        
        ret = {}
        for sname, (SSPW_times, SSPW_locs) in planewave_time_dict.items():
            
            if len(SSPW_times)==0:
                continue
            
            ## first we compare arrival time to planewave arrival time, in order to pick the best SSPW
            theory_times = np.linalg.norm(SSPW_locs - XYZT[0:3], axis=1)/v_air + XYZT[3]
            
            abs_diff = np.abs( theory_times - SSPW_times )
            ret[sname] = abs_diff.argmin()
                
        return ret
    
    
    
    
    class tryradius_pickSSPW(object):
        def __init__(self, ref_sname, ref_SSPW):
            self.ref_sname = ref_sname
            self.ref_SSPW = ref_SSPW
            
        def tst(self, R):
            
            R = np.abs(R) ## there are problems with negative radius
            
            XYZ = self.ref_SSPW.get_point_on_ray( R )
            T = self.ref_SSPW.ZAT[2] - R/v_air
            loc = np.append(XYZ, [T])
            return TryLoc_pickSSPW(loc)
        
        def run_fit(self):            
            
            brute_guess=brute(self.tst, [(500.0, 50000.0)], Ns=10000, finish=False)
            
            self.fit_res=minimize(self.tst, brute_guess, method="Powell", 
              options={"ftol":1.0E-50, "xtol":1.0E-50, "maxiter":1000000, "maxfev":1000000})
            
            XYZ = self.ref_SSPW.get_point_on_ray(self.fit_res.x )
            T = self.ref_SSPW.ZAT[2] - self.fit_res.x/v_air
            loc = np.append(XYZ, [T])
            
            return self.fit_res, loc
        
            
    #### now we loop over referance station and referance SSPW
    pointsource_events = []
    ref_SSPW_list = SSPW_dict[referance_station]
    
    ref_index = -1

    for referance_SSPW in ref_SSPW_list:
        ref_index += 1
        if not np.isfinite( planewave_time_dict[referance_station][0][ref_index] ):
            continue
        
        #### first, try to fit radius from ref_SSPW

        fitter = tryradius_pickSSPW(referance_station, referance_SSPW)
        fit_res, XYZT = fitter.run_fit()
        fit = np.sqrt(fit_res.fun)
        
        log()
        log( "referance SSPW index:", referance_SSPW.unique_index, "pol:", referance_SSPW.polarization)
        log( "  initial fit:", fit, "radius:", fit_res.x)
        
        
        #### now minimize XYZT
#        fit_res=minimize(TryLoc_pickSSPW, XYZT, method="Powell", 
#          options={"ftol":1.0E-20, "xtol":1.0E-50, "maxiter":1000000, "maxfev":1000000})
#        fit = np.sqrt(fit_res.fun)
#        log("  new fit:", fit)
#        XYZT = fit_res.x
        
        log("  initial loc:", XYZT)
        
        picked_planewaveIndex_dict = GetBestSSPW( XYZT )
        picked_planewave_dict = {sname:SSPW_dict[sname][index] for sname, index in picked_planewaveIndex_dict.items()}
        
        #### We will not filter out stations, becouse we want events with all stations
        if fit<final_fit_val and len(picked_planewave_dict)==len(planewave_time_dict):
            
            new_PSE = pointsource_event(ant_loc_dict, picked_planewave_dict, referance_station=referance_station, initial_guess=XYZT, station_offsets_guess=station_offsets_guess)
            

            pointsource_events.append(new_PSE)
            log("Found PSE", new_PSE.unique_index, new_PSE.PolE_RMS, new_PSE.PolO_RMS)
            log("    ", new_PSE.ave_PolE_amp, new_PSE.std_PolE_amp, new_PSE.ave_PolO_amp, new_PSE.std_PolO_amp)
            log("    ", new_PSE.PolE_loc, new_PSE.PolO_loc)
            log("    ", new_PSE.fit_message )
            
            for sname, SSPW in picked_planewave_dict.items():
                log("  using", sname, "SSPW", SSPW.unique_index, "fit", SSPW.fit)
            
            for sname, index in picked_planewaveIndex_dict.items(): ## keep these events from being picked again
                planewave_time_dict[sname][0][index] = np.inf
        else:
            if fit<final_fit_val:
                log("Not enough participating stations")
            else:
                log("not good enough fit")
        log()
        log()
                
    return pointsource_events
    
    
    
if __name__ == "__main__":
        
    ##opening data
    timeID = "D20170929T202255.000Z"
    output_folder = "correlate_SSPW"
    
    SSPW_folder = "SSPW2_tmp"
    
    stations_to_exclude = ["CS028","RS106", "RS305", "RS205", "CS201", "RS407"]#["CS026",  "RS106", "RS205", "RS208", "RS305", "RS306", "RS307", "RS310", "RS406", "RS407", 
                           #"RS409", "RS503", "RS508", "RS509" ] 
    initial_block = 3500
    num_blocks_per_step = 100
    num_steps = 100
    
    station_offsets_guess = {
        "CS002":0.0,
        "CS003":1.0E-6,
        "CS004":0.0,
        "CS005":0.0,
        "CS006":0.0,
        "CS007":0.0,
        "CS011":0.0,
        "CS013":-0.000003,
        "CS017":-7.0E-6,
        "CS021":-8E-7,
        "CS026":-7E-6,
        "CS030":-5.5E-6,
        "CS032":-5.5E-6 + 2E-6 + 1E-7,
        "CS101":-7E-6,
        "CS103":-22.5E-6-20E-8,
        "RS106":35E-6 +30E-8 +12E-7,
        "CS201":-7.5E-6,
        "RS205":25E-6,
        "RS208":8E-5+3E-6,
        "CS301":6.5E-7,
        "CS302":-3.0E-6-35E-7,
        "RS305":-6E-6,
        "RS306":-7E-6,
        "RS307":175E-7+8E-7,
        "RS310":8E-5+6E-6,
        "CS401":-3E-6,
        "RS406":-25E-6,
        "RS407":5E-6 -8E-6 -15E-7,
        "RS409":8E-6,
        "CS501":-12E-6,
        "RS503":-30.0E-8-10E-7,
        "RS508":6E-5+5E-7,
        "RS509":10E-5+15E-7,
        }
    
    ## sorting SSPW
    min_time_between_SSPW_events = 5.0E-5#1.0E-5#0.5E-3  
    
    
    ## correlation SSPW
    referance_station = "CS002"
    min_initial_fitval = 6.0E-5#4.0E-5
    

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
    
    
    print("Time ID:", timeID)
    print("output folder name:", data_dir)
    print("date and time run:", time.strftime("%c") )
    print("SSPW folder:", SSPW_folder)
    print("initial block:", initial_block)
    print("num blocks per step:", num_blocks_per_step)
    print("num steps:", num_steps)
    print("excluded stations", stations_to_exclude)
    print("min. time between SSPW:", min_time_between_SSPW_events)
    print("referance station:", referance_station)
    print("min. initial fit val:", min_initial_fitval)
    
    
    
    pointsource_events = []
    for iter_i in range(num_steps):
        start_block = initial_block + iter_i*num_blocks_per_step
        print()
        print( "Opening Data. Itter:", iter_i, "block", start_block )
        
        planewave_info_dict = read_SSPW_timeID(timeID, SSPW_folder, stations_to_exclude=stations_to_exclude, min_block=start_block, max_block=start_block+1 )
        ant_locs = planewave_info_dict["ant_locations"]
        SSPW_dict = planewave_info_dict["SSPW_dict"]
        del planewave_info_dict["SSPW_dict"] ## so that memory will clean up sooner
        
        
        ## find the strongest SSPW, insuring they are far enough apart
        print("finding stongest SSPW")
        
        Strong_SSPW_Events_dict = {}
        for sname, eventlist in SSPW_dict.items(): 
            print("  ", sname, len(eventlist), "SSPW loaded")
            
            strong_SSPW_events = []
            strong_event_amps = []
                
            eventlist.sort(key=lambda x: x.best_ave_amp(), reverse=True)
            new_event_amps =  [SSPW.best_ave_amp() for SSPW in eventlist]
            
            for new_SSPW, new_amplitude in zip(eventlist, new_event_amps):
                ## the cases
                if len(strong_SSPW_events) == 0:
                    strong_SSPW_events.append(new_SSPW)
                    strong_event_amps.append(new_amplitude)
                    continue
                    
                ### now we know that there is at least one SSPW in strong_SSPW_events, and new_SSPW is strong enough
                ### need to insure that new_SSPW isn't too close to stronger SSPW
                insert_location = bisect.bisect_right(strong_event_amps, new_amplitude)
                good = True ## if there is a stronger event that is too close to the new event, then good will be False
                
                for i in range(insert_location):
                    timeDiff = abs(new_SSPW.ZAT[2] - strong_SSPW_events[i].ZAT[2])
                    if timeDiff < min_time_between_SSPW_events:
                        good = False
                        break
                    
                if not good:
                    continue
                
                ## if we are here, then all stronger events are far enough away. We can garuntee that new_SSPW will be inserted
                ## BUT! we need to find if there are any weaker events that are too close, and should be removed
                
                events_too_close = []
                for i in range(insert_location, len(strong_SSPW_events)):
                    timeDiff = abs(new_SSPW.ZAT[2] - strong_SSPW_events[i].ZAT[2])
                    if timeDiff < min_time_between_SSPW_events:
                        events_too_close.append(i)
                ##remove the events
                events_too_close.sort(reverse=True)
                for i in events_too_close:
                    strong_SSPW_events.pop(i)
                    strong_event_amps.pop(i)
                
                ## add the event!
                strong_SSPW_events.insert(insert_location, new_SSPW)
                strong_event_amps.insert(insert_location, new_amplitude)
            
            print("    ", len(strong_SSPW_events), "strong events")
            Strong_SSPW_Events_dict[sname] = strong_SSPW_events
        del SSPW_dict ## cleanup memory

        #### now we correlate events to find plane waves
        print()
        print()
        print("Now correlating SSPW")
        new_pointsource_events = correlate_stations(ant_locs, Strong_SSPW_Events_dict, referance_station, min_initial_fitval, min_time_between_SSPW_events, station_offsets_guess)

        #### save the PSE!
        print("saving", len(new_pointsource_events), "PSE")
        with open(data_dir + '/point_sources_'+str(start_block), 'wb') as fout:
            
            write_long(fout, 5)
            writeBin_modData_T2(fout, station_delay_dict=planewave_info_dict['stat_delays'], ant_delay_dict=planewave_info_dict["ant_delays"], bad_ant_list=planewave_info_dict['bad_ants'], flipped_pol_list=planewave_info_dict['pol_flips'])
        
            write_long(fout, 2) ## means antenna location data is next
            write_long(fout, len(ant_locs))
            for ant_name, ant_loc in ant_locs.items():
                write_string(fout, ant_name)
                write_double_array(fout, ant_loc)
                    
                
            write_long(fout,3) ## means PSE data is next
            write_long(fout, len(new_pointsource_events))
            for PSE in new_pointsource_events:
                PSE.save_as_binary(fout)
                


        
        
        
    
    
    
    
    
    
    