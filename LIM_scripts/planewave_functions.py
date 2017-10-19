#!/usr/bin/env python3

from os import listdir

##import external packages
import numpy as np
#from matplotlib import pyplot as plt
from scipy.optimize import brute, minimize

##my packages
#from porta_code import code_logger
from utilities import v_air, log, processed_data_dir
from binary_IO import write_long, write_double_array, write_string, write_double, read_long, read_double_array, read_double, read_string, at_eof

from read_PSE import readBin_modData_T2
from read_pulse_data import readBin_modData

singleStation_planewave_next_unique_i=0
class SSPW_event(object):
    def __init__(self, station_info, AntennaPulse_dict, window_start, window_end):
        self.station_info = station_info
        self.AntennaPulse_dict = AntennaPulse_dict
        self.window_start = window_start
        self.window_end = window_end
        self.ref_location = self.station_info.get_station_location()
        self._associated_PS_event = None
        
        data_section = None
        for ant_name, pulse_list in AntennaPulse_dict.items():
            if len(pulse_list) != 0:
                data_section = pulse_list[0].section_number
                break
        
        #### get data for all antennas ####        
        self.ant_names = []
        PolE_pulse_times = []
        PolO_pulse_times = []
        ant_X = []
        ant_Y = []
        ant_Z = []
        polE_is_good = True
        polO_is_good = True
        num_good_E_antennas = 0
        num_good_O_antennas = 0
        for ant_name, pulse_list in AntennaPulse_dict.items():
                
            ant_status = self.station_info.AntennaInfo_dict[ant_name].section_status(data_section) #### this is necisary for the case that pulse_list is empty

            PolE_pulse_times_list = []
            PolO_pulse_times_list = []
            for p in pulse_list:
                PolE_peak_time = p.peak_time(0)
                PolO_peak_time = p.peak_time(1)
                
                PolE_pulse_times_list.append( PolE_peak_time )
                PolO_pulse_times_list.append( PolO_peak_time )
                
                pulse_status = p.antenna_status
                if pulse_status == 1:
                    if ant_status == 0: ant_status = 1
                    elif ant_status == 2: ant_status = 3
                elif pulse_status == 2:
                    if ant_status == 0: ant_status = 2
                    elif ant_status == 1: ant_status = 3
                elif pulse_status == 3:
                    ant_status = 3
                    
            if ant_status == 3: 
                continue
            
            self.ant_names.append(ant_name)
            loc = self.station_info.AntennaInfo_dict[ant_name].location
            
            ant_X.append(loc[0])
            ant_Y.append(loc[1])
            ant_Z.append(loc[2])
            
            if ant_status != 1:
                polE_is_good = polE_is_good and (len(PolE_pulse_times_list)!=0)
                PolE_pulse_times.append( np.array(PolE_pulse_times_list) )
                num_good_E_antennas += 1
            else:
                PolE_pulse_times.append( np.array([np.inf]) )
                
            if ant_status != 2:
                polO_is_good = polO_is_good and (len(PolO_pulse_times_list)!=0)
                PolO_pulse_times.append( np.array(PolO_pulse_times_list) )
                num_good_O_antennas += 1
            else:
                PolO_pulse_times.append( np.array([np.inf]) )
                
        polE_is_good = polE_is_good and num_good_E_antennas>=4
        polO_is_good = polO_is_good and num_good_O_antennas>=4
        if (not polE_is_good) and (not polO_is_good):
            self.solved = False
            return
            
        PolE_pulse_times.append( np.array(  [0.0]*( len(PolE_pulse_times[0])+1 )  ) ) ## a very bad hack to make numpy behave consistantly
        PolO_pulse_times.append( np.array(  [0.0]*( len(PolO_pulse_times[0])+1 )  ) ) 
        
        PolE_pulse_times = np.array(PolE_pulse_times, dtype=object)
        PolO_pulse_times = np.array(PolO_pulse_times, dtype=object)
        self.antenna_X = np.array(ant_X)
        self.antenna_Y = np.array(ant_Y) 
        self.antenna_Z = np.array(ant_Z)
        
        
        
        #### try to fit even polarization ####
        ####: todo: replace fitting functions with least squares?
        PolE_aveAmp = 0.0
        if polE_is_good:
            self.pulse_times = PolE_pulse_times
            brute_guess = brute(self.select_and_test, [[0,np.pi/2], [0,2*np.pi], [window_start, window_end] ], Ns=10, finish=False)
            
            PolE_min_ret = minimize(self.select_and_test, brute_guess, method="Powell", 
                            options={"ftol":1.0E-20, "xtol":1.0E-50, "maxiter":100000, "maxfev":100000})
            
            ## get ave amp for even pol.
            PolE_num_ant = 0
            for i in range(len(self.ant_names)):
                if not np.isfinite(self.pulse_times[i][ self.pulse_indeces_choosen[i] ]): continue
                pulse = self.AntennaPulse_dict[self.ant_names[i]][self.pulse_indeces_choosen[i]]
                PolE_num_ant += 1
                PolE_aveAmp += np.max( pulse.even_antenna_hilbert_envelope )
            PolE_aveAmp /= PolE_num_ant
            
            ### check that we are within the window
            max_time = np.max( self.plane_wave_model(PolE_min_ret.x) )
            if max_time > self.window_end:
                polE_is_good = False
        
        
        
        #### try and fit odd polarization ####
        PolO_aveAmp = 0.0
        if polO_is_good:
            self.pulse_times = PolO_pulse_times
            brute_guess = brute(self.select_and_test, [[0,np.pi/2], [0,2*np.pi], [window_start, window_end] ], Ns=10, finish=False)
            
            PolO_min_ret = minimize(self.select_and_test, brute_guess, method="Powell", 
                            options={"ftol":1.0E-20, "xtol":1.0E-50, "maxiter":100000, "maxfev":100000})
            
            ## get ave amp for odd pol.
            PolO_num_ant = 0
            for i in range(len(self.ant_names)):
                if not np.isfinite(self.pulse_times[i][ self.pulse_indeces_choosen[i] ]): continue
                pulse = self.AntennaPulse_dict[self.ant_names[i]][self.pulse_indeces_choosen[i]]
                PolO_num_ant += 1
                PolO_aveAmp += np.max( pulse.odd_antenna_hilbert_envelope )
            PolO_aveAmp /= PolO_num_ant
            
            ### check that we are within the window
            max_time = np.max( self.plane_wave_model(PolO_min_ret.x) )
            if max_time > self.window_end:
                polO_is_good = False
        
        
        
        #### choose polarization with largest average amp
        if PolO_aveAmp < PolE_aveAmp and polE_is_good: ##odd is weaker, and even has all antennas: choose even
            self.polarization = 0
            min_ret = PolE_min_ret
            self.pulse_times = PolE_pulse_times
        elif polO_is_good:
            self.polarization = 1
            min_ret = PolO_min_ret
            self.pulse_times = PolO_pulse_times
        else:
            self.solved = False
            return
            
            
        self.fit = np.sqrt( min_ret.fun ) ##fit is now RMS
        self.ZAT = min_ret.x
        self.fit_status = min_ret.status
        self.fit_message = min_ret.message
        self.solved = True
        
        
        self.select_and_test( self.ZAT )##insure pulse indeces are set corectly
        pulse_times = np.array([ self.pulse_times[i][self.pulse_indeces_choosen[i]] for i in range(len(self.ant_names))  ])
        
        ## flag bad antennas
        ant_flags = np.array([  1 if np.isfinite(PT) else 0 for PT in pulse_times  ])

        #### filter all important variables by the ant_flags
        self.pulse_dict = {self.ant_names[i] : self.AntennaPulse_dict[self.ant_names[i]][self.pulse_indeces_choosen[i]]    for i in range(len(self.ant_names))   if ant_flags[i]  }
        self.antenna_X = np.array( [self.antenna_X[i] for i in range(len(self.ant_names)) if ant_flags[i]] )
        self.antenna_Y = np.array( [self.antenna_Y[i] for i in range(len(self.ant_names)) if ant_flags[i]] )
        self.antenna_Z = np.array( [self.antenna_Z[i] for i in range(len(self.ant_names)) if ant_flags[i]] )
        self.pulse_times = np.array( [pulse_times[i] for i in range(len(self.ant_names)) if ant_flags[i]] )
        self.ant_names = [self.ant_names[i] for i in range(len(self.ant_names)) if ant_flags[i]]


        cos_z = np.cos(self.ZAT[0])##zenith angle
        sin_z = np.sin(self.ZAT[0])
        cos_a = np.cos(self.ZAT[1])##azimuth angle
        sin_a = np.sin(self.ZAT[1])
        self.normal = np.array([sin_z*cos_a, sin_z*sin_a, cos_z ])
        
        
        ## inform pulses that they have been selected
        for pulse in self.pulse_dict.values():
            pulse.event_data = self
        
        
        ## other variables
        self._average_amplitude_  = None
        self._average_SNR_ = None
        self.nearby_events = [] ## REMOVE THIS!
        
        self.get_average_amplitude()
        
        global singleStation_planewave_next_unique_i
        self.unique_index = singleStation_planewave_next_unique_i
        singleStation_planewave_next_unique_i += 1
        
        log("Found SSPW", self.unique_index, "fit:", self.fit, "ave. SNR:", self._average_SNR_, "POL:", self.polarization)
        log("   ZAT:", self.ZAT)
        log("   ", self.fit_message)
        log()
        
        for ant_name, pulse in self.pulse_dict.items():
            log("  ", ant_name, pulse.section_number, pulse.unique_index, pulse.starting_index, len(pulse.odd_antenna_data) )
        log()
        log()
        
    #### fitting functions
    def plane_wave_model(self, ZAT):
        Z, A, planer_t = ZAT
        
        cos_z = np.cos(Z)##zenith angle
        sin_z = np.sin(Z)
        cos_a = np.cos(A)##azimuth angle
        sin_a = np.sin(A)
        
        ref_dot_normal = self.ref_location[0]*(sin_z*cos_a) + self.ref_location[1]*(sin_z*sin_a) + self.ref_location[2]*cos_z

        ##wierdness here is to speed up the code a little
        d  = self.antenna_X*(sin_z*cos_a)
        d += self.antenna_Y*(sin_z*sin_a) 
        d += self.antenna_Z*cos_z 
        d *= -1.0/v_air ##divide by C to get time,  multiply by -1 becouse zenith and azimuth are angles TOO the source
        d += planer_t + ref_dot_normal/v_air #### So this is equivalent to:  T - d/C.  additional shift (ref_dot_normal), is due to the fact that the antenna XYZ has a different referance than the time
        
        return  d    
    
    def select_and_test(self, ZAT):
        """ find the fit of data to solultion, if there are multple pulses in each antenna and one must chose the best fitting pulse"""
        
        theory_times = self.plane_wave_model(ZAT)
        diffs = self.pulse_times[:-1] - theory_times
        diffs *= diffs
        self.pulse_indeces_choosen = [ D.argmin() for D in diffs ]
        
        sum_dif_sq = 0.0
        N = 0
        for diff,index in zip(diffs, self.pulse_indeces_choosen):
            if np.isfinite(diff[index]):
                N += 1
                sum_dif_sq += diff[index]
            
        return sum_dif_sq/N
    
    #### data functions
    def get_DataTime(self):
        ret=[]
        for ant_name in self.station_info.sorted_antenna_names:
            if ant_name not in self.ant_names:
                ret.append(np.inf)
                continue
            
            i = self.ant_names.index(ant_name)
            ret.append( self.pulse_times[i] )
        return ret
        
    def get_ModelTime(self):
        model_times = self.plane_wave_model(self.ZAT)
        ret=[]
        for ant_name in self.station_info.sorted_antenna_names:
            if ant_name not in self.ant_names:
                ret.append(np.inf)
                continue
            
            i = self.ant_names.index(ant_name)
            ret.append( model_times[i] )
        return ret
    
    def get_average_amplitude(self):
        if self._average_amplitude_ is None:
            self._average_amplitude_ = 0.0
            self._average_SNR_  = 0.0
            for pulse in self.pulse_dict.values():
                max_HE = np.max(pulse.even_antenna_hilbert_envelope) if self.polarization==0 else np.max(pulse.odd_antenna_hilbert_envelope) 
                std =    pulse.even_data_STD if self.polarization==0 else  pulse.odd_data_STD
                self._average_amplitude_ += max_HE
                self._average_SNR_ += max_HE/std
            self._average_amplitude_ /= len(self.pulse_dict)
            self._average_SNR_ /= len(self.pulse_dict)
            
        return self._average_amplitude_
    
    def get_ave_even_amp(self):
        ret = 0.0
        num  = 0.0
        for pulse in self.pulse_dict.values():
            ret += np.max(pulse.even_antenna_hilbert_envelope)
            num += 1
        return ret/num
    
    def get_ave_odd_amp(self):
        ret = 0.0
        num  = 0.0
        for pulse in self.pulse_dict.values():
            ret += np.max(pulse.odd_antenna_hilbert_envelope)
            num += 1
        return ret/num
    
    def save_binary(self, fout):
        write_long(fout, self.unique_index)
        write_string(fout, self.station_info.station_name)
        write_double_array(fout, self.ZAT)
        write_long(fout, self.polarization)
        write_double(fout, self.fit)
        write_long(fout, len(self.ant_names))
        for ant_name, pulse in self.pulse_dict.items():
            
            write_string(fout, ant_name)
            write_double(fout, pulse.peak_time(0))
            write_double(fout, pulse.peak_time(1))
            write_long(fout, pulse.section_number)
            write_long(fout, pulse.unique_index)
            write_long(fout, pulse.starting_index)
            write_long(fout, pulse.antenna_status)
            write_double(fout, pulse.even_data_STD )
            write_double(fout, pulse.odd_data_STD )
            write_double_array(fout, pulse.even_antenna_data)
            write_double_array(fout, pulse.even_antenna_hilbert_envelope)
            write_double_array(fout, pulse.odd_antenna_data)
            write_double_array(fout, pulse.odd_antenna_hilbert_envelope)
            write_double(fout, pulse.PolE_time_offset)
            write_double(fout, pulse.PolO_time_offset)
        
        
#    def spherical_delay_and_fit(self, XYZT_loc, estimate_delay=True):
#        """given a source at a location and time, return RMS fit and estimated station delay. if estimate_delay is false, then the delay is assumed to be perfect"""
#        delta_X = self.antenna_X - XYZT_loc[0]
#        delta_Y = self.antenna_Y - XYZT_loc[1]
#        delta_Z = self.antenna_Z - XYZT_loc[2]
#        
#        antenna_times_from_source = np.sqrt( delta_X*delta_X + delta_Y*delta_Y + delta_Z*delta_Z )/C
#        antenna_times_from_source += XYZT_loc[3] ##now is the theoretical time that pulse arrived at antenna
#    
#        antenna_times_from_source -= self.pulse_times ## note that this creates a different sign convention from our other code. 
#        
#        if estimate_delay:
#            estimated_delay = np.average(antenna_times_from_source)
#            antenna_times_from_source -= estimated_delay
#        else:
#            estimated_delay=0.0
#        
#        RMS_fit = np.sqrt( np.sum(antenna_times_from_source*antenna_times_from_source)/len(antenna_times_from_source) )
#        
#        return RMS_fit, -estimated_delay ##negative sign here accounts for wierd negative sign above
        
    #### set of ray functions ####
    def get_point_on_ray(self, distance):
        """return a XYZ position that is a certain distance along ray to source location"""
        return self.ref_location + self.normal*distance
    
    def ray_intersection(self, SSPW_two):
        """return the distance along this ray that this ray and anouther ray intersect."""
        def bar_operator(vec):
            return vec - SSPW_two.normal * np.dot(SSPW_two.normal, vec)
    
        X1_bar = bar_operator( self.ref_location )
        X2_bar = bar_operator( SSPW_two.ref_location )
        N_bar  = bar_operator( self.normal )
        
        deltaX_bar = X2_bar - X1_bar
    
        denom = np.dot(N_bar, N_bar)
        if abs(denom)<1E-15:
            return None

        R1 = np.dot(deltaX_bar, N_bar)/np.dot(N_bar, N_bar)
        return R1
        
    def distance_onRay_closest_to(self, point):
        """return the distance on this ray, from the ray origin, that is closest to the point"""
        return np.dot(self.normal, point - self.ref_location)
    
    def time_at_spot(self, location):
        """return the time that this planwave passes a certain spot"""
        D = np.dot(self.normal, location-self.ref_location)
        return self.ZAT[2] + D/v_air
    
#    def ray_source_location(self, SSPW_two):
#        """given a second singe-station plane wave, use the time differance between pulses, and the direction information of this ray,
#        return: estimated source location, source time, and RMS fit to SSPW_two. This method ignores ray information of SSPW_two"""
#        
#        distance_delta=(SSPW_two.solution[2] - self.solution[2])*C
#        deltaP=  SSPW_two.ref_location - self.ref_location
#        
#        numerator= np.sum( deltaP*deltaP )  -  distance_delta*distance_delta
#        half_denominator = distance_delta + np.sum( self.normal*deltaP )
#        distance= numerator/(2*half_denominator)
#        time=self.solution[2] - distance/C
#        location_estimate = self.get_point_on_ray(distance)
#        
#        ##get fit to SSPW two
#        SSPW_two_rel_loc = location_estimate - SSPW_two.ref_location
#        delta_X= SSPW_two.antenna_X - SSPW_two_rel_loc[0]
#        delta_Y= SSPW_two.antenna_Y - SSPW_two_rel_loc[1]
#        delta_Z= SSPW_two.antenna_Z - SSPW_two_rel_loc[2]
#        antenna_times_from_source = np.sqrt( delta_X*delta_X + delta_Y*delta_Y + delta_Z*delta_Z )/C
#        antenna_times_from_source += time ##now is the theoretical time that pulse arrived at antenna
#        
#        antenna_times_from_source -= SSPW_two.pulse_times
#        RMS_fit = np.sqrt( np.sum(antenna_times_from_source*antenna_times_from_source)/len(antenna_times_from_source) )
#        
#        return location_estimate, time, RMS_fit

#    def event_too_close(self, SSPW_event):
#        """check if SSPW_event is too close to this event. An event is too close if its referance time is between max and min detected pulse times for this event"""
#
#        max_PT = np.max(self.pulse_times)
#        min_PT = np.min(self.pulse_times)
#        
#        if SSPW_event.ref_time > min_PT and SSPW_event.ref_time < max_PT:
#            self.nearby_events.append(SSPW_event)
#            return True
#        else:
#            return False

    def data_CodeLogPlot(self, CL):    
        model_times = self.get_ModelTime()
        data_times = self.get_DataTime()
        
        for ant_name, model, data in zip(self.station_info.sorted_antenna_names, model_times, data_times):
            
            if ant_name not in self.pulse_dict:
                continue
            
            offset = int(ant_name[-3:])/2
            
            pulse = self.pulse_dict[ant_name]
            
            even_hilbert = np.array( pulse.even_antenna_hilbert_envelope )
            even_data = np.array( pulse.even_antenna_data )
            odd_hilbert = np.array( pulse.odd_antenna_hilbert_envelope )
            odd_data = np.array(  pulse.odd_antenna_data )
            
            amp = max(np.max(even_hilbert), np.max(odd_hilbert))
            even_hilbert /= amp*(3.0)
            odd_hilbert  /= amp*(3.0)
            even_data /=amp*(3.0)
            odd_data /=amp*(3.0)
            even_data -= np.average(even_data)
            odd_data -= np.average(odd_data)
            
            even_T_array = np.arange(len(even_hilbert))*self.station_info.seconds_per_sample + pulse.starting_index*self.station_info.seconds_per_sample + pulse.PolE_time_offset
            CL.add_function("plt.plot", even_T_array, offset+even_hilbert, 'g' )
            CL.add_function("plt.plot", even_T_array, offset+even_data, 'g' )
            
            
            odd_T_array = np.arange(len(odd_hilbert))*self.station_info.seconds_per_sample + pulse.starting_index*self.station_info.seconds_per_sample + pulse.PolO_time_offset
            CL.add_function("plt.plot", odd_T_array, offset+odd_hilbert, 'm' )
            CL.add_function("plt.plot", odd_T_array, offset+odd_data, 'm' )
                
            max_even_T_array = np.max(even_T_array)
            max_odd_T_array = np.max(odd_T_array)
            
            if self.polarization == 0:
                CL.add_function("plt.plot", [data, data], [offset-1.0/3.0, offset+1.0/3.0], 'g')
                CL.add_function("plt.plot", [model,model], [offset-1.0/3.0, offset+1.0/3.0], 'r')
            else:
                CL.add_function("plt.plot", [data, data], [offset-1.0/3.0, offset+1.0/3.0], 'm')
                CL.add_function("plt.plot", [model,model], [offset-1.0/3.0, offset+1.0/3.0], 'r')
            
            max_X = max(max_even_T_array, max_odd_T_array)
            if max_X != -np.inf:
                CL.add_function("plt.annotate", ant_name, xy=[max_X, offset+1.0/3.0], size=15)
        
        
def read_SSPW_timeID(TimeID, analysis_folder, fname_prefix=None, data_loc=None, stations=None, stations_to_exclude=None, min_block=None, max_block=None):
    ## assumes the data is in a particular directory structure:
        ## that the top folder is data_loc (default to "/home/brian/processed_files/")
        ## next folder has the name of TimeID
        ## next folder has the name of the year (found from TimeID)
        ## next folder is analysis_folder
        ## inside is the fname, default to "SSPW_"
        ## also assumes that last 5 letters of the file name is the station name
        
    
    if fname_prefix is None:
        fname_prefix = "SSPW_"
        
    data_loc = processed_data_dir(TimeID, data_loc) + "/" + analysis_folder + "/"
    
    out = {}
    out['ant_locations'] = {}
    out["SSPW_dict"] = {}
    
    out['pol_flips'] = []
    out['bad_ants'] = []
    out["ant_delays"] = {}
    out["stat_delays"] = {}

    for fname in listdir(data_loc):
        if not fname.startswith(fname_prefix):
            continue
        
#        sname = fname[-5:]
        sname = fname.split('_')[1]
        if stations is not None:
            if sname not in stations:
                continue
            
        if stations_to_exclude is not None:
            if sname in stations_to_exclude:
                continue
            
        if min_block is not None or max_block is not None:
            block = int(fname.split('_')[-1])
            
            if min_block is not None and block<min_block:
                continue
            if max_block is not None and block>=max_block:
                continue
        
        with open(data_loc+fname, 'rb') as fin:
            while not at_eof(fin):
                code = read_long(fin)
                                    
                if code == 1: ## mod data
        
                    pol_flips, bad_ants, ant_delays, stat_delays = readBin_modData(fin)
                    
                    out['pol_flips'] += pol_flips
                    out['bad_ants'] += bad_ants
                    for ant_name, delay in ant_delays.items():
                        out["ant_delays"][ant_name] = delay
                    for sname, delay in stat_delays.items():
                        out["stat_delays"][sname] = delay
                        
#                elif code == 5: ## mod data, stored using second format.
#                
#                    pol_flips, bad_ants, ant_delays, stat_delays = readBin_modData_T2(fin)
#                    
#                    out['pol_flips'] += pol_flips
#                    out['bad_ants'] += bad_ants
#                    for ant_name, delay in ant_delays.items():
#                        out["ant_delays"][ant_name] = delay
#                    for sname, delay in stat_delays.items():
#                        out["stat_delays"][sname] = delay
                
                elif code == 2:
                    
                    N_ant = read_long(fin)
                    for i in range(N_ant):
                        ant_name = read_string(fin)
                        ant_loc = read_double_array(fin)
                        out['ant_locations'][ant_name] = ant_loc
                        
                elif code == 5 or code==6: ##code==5 needs to be type 2 mod data
                    
                    if sname not in out["SSPW_dict"]:
                        out["SSPW_dict"][sname] = []
                    SSPW_list = out["SSPW_dict"][sname]
                    
                    num_SSPW = read_long(fin)
                    for x in range(num_SSPW):
                        new_SSPW = SSPW_fromBinary(fin)
                        SSPW_list.append( new_SSPW )
            
            
    return out

def read_SSPW_timeID_multiDir(TimeID, analysis_folders, fname_prefix=None, data_loc=None, stations=None, stations_to_exclude=None, min_block=None, max_block=None):
    out = {}
    out['ant_locations'] = {}
    out["SSPW_multi_dicts"] = []
    
    for folder in analysis_folders:
        folder_out = read_SSPW_timeID(TimeID, folder, fname_prefix, data_loc, stations, stations_to_exclude, min_block, max_block)
        
        out["SSPW_multi_dicts"].append( folder_out["SSPW_dict"] )
        
        ##combine locations
        for ant_name, loc in folder_out['ant_locations'].items():
            out['ant_locations'][ant_name] = loc
            
        ## TODO: combine mod data
            
    return out


### TODO: add a new version that has same data as PSE
    ## also should have a flexible dictionary    
class SSPW_fromBinary:
    class ant_info_obj:
        def __init__(self, fin):
            self.ant_name = read_string(fin)
            self.PolE_peak_time = read_double(fin)
            self.PolO_peak_time = read_double(fin)
            self.pulse_section_number = read_long(fin)
            self.pulse_unique_index = read_long(fin)
            self.pulse_starting_index = read_long(fin)
            self.antenna_status = read_long(fin)
            self.PolE_std = read_double(fin)
            self.PolO_std = read_double(fin)
            self.PolE_antenna_data = read_double_array(fin)
            self.PolE_hilbert_envelope = read_double_array(fin)
            self.PolO_antenna_data = read_double_array(fin)
            self.PolO_hilbert_envelope = read_double_array(fin)
            self.PolE_time_offset = read_double(fin)
            self.PolO_time_offset = read_double(fin)
    
    def __init__(self, fin):
        self.unique_index = read_long(fin)
        self.sname = read_string(fin)
        self.ZAT = read_double_array(fin)
        self.polarization = read_long(fin)
        self.fit = read_double(fin)
        
        n_ant = read_long(fin)
        self.ant_data = {}
        for ant_i in range(n_ant):
            new_ant_data = self.ant_info_obj(fin)
            self.ant_data[ new_ant_data.ant_name ] = new_ant_data
            
            
    def get_DataTime(self, ant_order):
        
        ret=[]
        for ant_name in ant_order:
            if ant_name not in self.ant_data:
                ret.append(np.inf)
                continue
            else:
                ant_data = self.ant_data[ant_name] 
                T = ant_data.PolE_peak_time if self.polarization==1 else ant_data.PolO_peak_time
                ret.append( T )
        return ret
    
    def best_ave_amp(self):
        PolE_tot = 0.0
        PolE_num_ant = 0
        
        PolO_tot = 0.0
        PolO_num_ant = 0
        
        for ant_info in self.ant_data.values():
            if ant_info.antenna_status == 0 or ant_info.antenna_status == 2:
                PolE_tot += np.max( ant_info.PolE_hilbert_envelope )
                PolE_num_ant += 1
                
            if ant_info.antenna_status == 0 or ant_info.antenna_status == 1:
                PolO_tot += np.max( ant_info.PolO_hilbert_envelope )
                PolO_num_ant += 1
    
        return np.max( [PolE_tot/PolE_num_ant, PolO_tot/PolO_num_ant] ) 

    def prep_fitting(self, ref_location, ant_locs):
        self.ref_location = ref_location
        
        ant_X = []
        ant_Y = []
        ant_Z = []
        
        pulse_times = []
        
        for pulse_info in self.ant_data.values():
            if self.polarization==0 and (pulse_info.antenna_status==1 or pulse_info.antenna_status==3):
                continue
            if self.polarization==1 and (pulse_info.antenna_status==2 or pulse_info.antenna_status==3):
                continue
            
            loc = ant_locs[ pulse_info.ant_name ]
            ant_X.append( loc[0] )
            ant_Y.append( loc[1] )
            ant_Z.append( loc[2] )
            
            pulse_times.append(  pulse_info.PolE_peak_time if self.polarization==0 else pulse_info.PolO_peak_time  )
            
        self.antenna_X = np.array( ant_X )
        self.antenna_Y = np.array( ant_Y )
        self.antenna_Z = np.array( ant_Z )
        self.pulse_times = np.array( pulse_times )
        
        cos_z = np.cos(self.ZAT[0])##zenith angle
        sin_z = np.sin(self.ZAT[0])
        cos_a = np.cos(self.ZAT[1])##azimuth angle
        sin_a = np.sin(self.ZAT[1])
        self.normal = np.array([sin_z*cos_a, sin_z*sin_a, cos_z ])
        
    ## ray operations
    def get_point_on_ray(self, distance):
        """return a XYZ position that is a certain distance along ray to source location"""
        return self.ref_location + self.normal*distance
    
    def ray_intersection(self, SSPW_two):
        """return the distance along this ray that this ray and anouther ray intersect."""
        def bar_operator(vec):
            return vec - SSPW_two.normal * np.dot(SSPW_two.normal, vec)
    
        X1_bar = bar_operator( self.ref_location )
        X2_bar = bar_operator( SSPW_two.ref_location )
        N_bar  = bar_operator( self.normal )
        
        deltaX_bar = X2_bar - X1_bar
    
        denom = np.dot(N_bar, N_bar)
        if abs(denom)<1E-15:
            return None

        R1 = np.dot(deltaX_bar, N_bar)/np.dot(N_bar, N_bar)
        return R1
        
    def distance_onRay_closest_to(self, point):
        """return the distance on this ray, from the ray origin, that is closest to the point"""
        return np.dot(self.normal, point - self.ref_location)
    
    def time_at_spot(self, location):
        """return the time that this planwave passes a certain spot"""
        D = np.dot(self.normal, location-self.ref_location)
        return self.ZAT[2] + D/v_air
    
#    def save_binary(self, fout):
#        write_long(fout, self.unique_index)
#        write_string(fout, self.sname)
#        write_double_array(fout, self.ZAT)
#        write_long(fout, self.polarization)
#        write_double(fout, self.fit)
#        write_long(fout, len(self.ant_names))
#        for ant_name, even_T, odd_T, pulse in zip(self.ant_names, self.even_pulse_times, self.odd_pulse_times, self.pulse_info):
#            
#            write_string(fout, ant_name)
#            write_double(fout, even_T)
#            write_double(fout, odd_T)
#            write_long(fout, pulse.section_number)
#            write_long(fout, pulse.unique_index)
#            write_long(fout, pulse.starting_index)
#            write_long(fout, pulse.antenna_status)
        
        
        
## Old algorithm. One further down is now preferd. (simpler and slightly better)

#def find_planewave_events(station_info, AntennaPulse_dict ):
#    """ find potential SSPW events in one station, and fit them to a planewave solution"""
#        
##    num_antennas = len(AntennaPulse_dict)
#    
#    log( station_info.station_name )
#
#    ##first, find diameter of station, to find width of window
#    station_center = station_info.get_station_location()
#    radius=0
#    for ant in station_info.AntennaInfo_dict.values():
#        AD= np.linalg.norm( ant.location-station_center )
#        if AD>radius:
#            radius = AD
#
#    half_window_width = 2*radius/v_air
#
#    ##start and stop times
#    start_time = np.inf
#    end_time = -np.inf
#    for pulse_list in AntennaPulse_dict.values():
#        for pulse in pulse_list:
#            T = pulse.peak_time()
#            
#            if T < start_time:
#                start_time = T
#                
#            if T > end_time:
#                end_time = T
#                
#    windowA_start_time = start_time - half_window_width
#    windowA_end_time = windowA_start_time + 2*half_window_width
#    
#    AntIndex_dict={ant_name:0 for ant_name in station_info.sorted_antenna_names} ### the index is the index of the first pulse still in the window
#    def count_antennas_window(start_time, stop_time):
#        """count number of antennas that have pulses in a window"""
#        count = 0
#        resulting_pulse_dict = {}
#        for ant_name, pulselist in AntennaPulse_dict.items():
#            resulting_pulse_dict[ant_name] = []
#            
#            found = False
#            for pulse_index in range( AntIndex_dict[ant_name],  len(pulselist) ):
#                pulse = pulselist[pulse_index]
#                pulse_T = pulse.peak_time()
#
#                if pulse_T < start_time:
#                    AntIndex_dict[ant_name] = pulse_index
#                elif (pulse_T < stop_time) and (pulse.event_data is None):
#                    if not found:
#                        found = True
#                        count += 1
#                    resulting_pulse_dict[ant_name].append( pulse )
#                    
#                elif pulse_T > stop_time:
#                    break
#
#        return count, resulting_pulse_dict
#
#    windowA_count, windowA_pulse_dict = count_antennas_window(windowA_start_time, windowA_end_time)
#
#    ## count following windows
#    SSPW_event_list=[]
##    last_event = None
#    while windowA_end_time-half_window_width < end_time:
#
#        
#        windowB_start_time= windowA_start_time + half_window_width
#        windowB_end_time  = windowA_end_time + half_window_width
#        windowB_count, windowB_pulse_dict = count_antennas_window(windowB_start_time, windowB_end_time)
#        
##        if (windowA_count >= 10) and not (windowA_count >= num_antennas):
#        
#        while (windowA_count >= 4):# num_antennas): ##NOTE: the requirment of using all antennas in station
#            
#            if windowA_count >= windowB_count:
#                new_event = SSPW_event(station_info, windowA_pulse_dict, windowA_start_time, windowA_end_time)
#            else:
#                new_event = SSPW_event(station_info, windowB_pulse_dict, windowB_start_time, windowB_end_time)
#                
#            if not new_event.solved:
#                log("Internal throw")
#                break
#
#            SSPW_event_list.append(new_event)
#            
#            windowA_count, windowA_pulse_dict = count_antennas_window(windowA_start_time, windowA_end_time)
#            windowB_count, windowB_pulse_dict = count_antennas_window(windowB_start_time, windowB_end_time)
#        
#            #if next_unique_i>=1:
#                #break
#
#        windowA_start_time = windowB_start_time
#        windowA_end_time = windowB_end_time
#        windowA_count = windowB_count
#        windowA_pulse_dict = windowB_pulse_dict
#       
#    print(len(SSPW_event_list))
#        
#    return SSPW_event_list


def find_planewave_events_smallWindow(station_info, AntennaPulse_dict ):
    """ find potential SSPW events in one station, and fit them to a planewave solution"""
    
    print( "finding PSE on ", station_info.station_name )

    ##first, find diameter of station, to find width of window
    station_center = station_info.get_station_location()
    radius=0
    for ant in station_info.AntennaInfo_dict.values():
        AD= np.linalg.norm( ant.location-station_center )
        if AD>radius:
            radius = AD

    window_width = 2*radius/v_air

    ##start and stop times
    start_time = np.inf
    end_time = -np.inf
    for pulse_list in AntennaPulse_dict.values():
        for pulse in pulse_list:
            
            if (pulse.antenna_status == 0 or pulse.antenna_status == 2):
                T = pulse.PolE_peak_time
                if T < start_time:
                    start_time = T
                if T > end_time:
                    end_time = T
                    
            if (pulse.antenna_status == 0 or pulse.antenna_status == 1):
                T = pulse.PolO_peak_time
                if T < start_time:
                    start_time = T
                if T > end_time:
                    end_time = T
                
    current_time = start_time
    
    AntIndex_dict={ant_name:0 for ant_name in station_info.sorted_antenna_names} ### the index is the index of the first pulse still after current_time
    def count_antennas_window(window_start_time, window_stop_time):
        """count number of antennas that have pulses in a window"""

        resulting_pulse_dict = {}
        for ant_name, pulselist in AntennaPulse_dict.items():
            resulting_pulse_dict[ant_name] = []
            
            for pulse_index in range( AntIndex_dict[ant_name],  len(pulselist) ):
                pulse = pulselist[pulse_index]
                
                pulse_in_window = False
                
                if (pulse.antenna_status == 0 or pulse.antenna_status == 2):
                    T = pulse.PolE_peak_time
                    if window_start_time <= T < window_stop_time:
                        pulse_in_window = True
                        
                if (pulse.antenna_status == 0 or pulse.antenna_status == 1) and not pulse_in_window:
                    T = pulse.PolO_peak_time
                    if window_start_time <= T < window_stop_time:
                        pulse_in_window = True
                        
                if pulse_in_window and (pulse.event_data is None):
                    resulting_pulse_dict[ant_name].append( pulse )
                    
                    
        count = 0
        for pulse_list in resulting_pulse_dict.values():
            if len(pulse_list) != 0:
                count += 1

        return count, resulting_pulse_dict
    
    
    
    SSPW_event_list = []
    while current_time < end_time - 0.5*window_width:
        num_ant_with_pulses, pulse_dict = count_antennas_window(current_time, current_time+window_width)
        
        ## see if we have a planewave
        if num_ant_with_pulses >= 4:
            
            new_event = SSPW_event(station_info, pulse_dict, current_time, current_time+window_width)
            
            if not new_event.solved:
                log("Internal throw")
                
            else:
                SSPW_event_list.append(new_event)
                
        ## update the current time to the earliest pulse after the present time
        ## also update all AntIndex_dict[ant_name] to be after the current time
        new_time = np.inf
        for ant_name, pulselist in AntennaPulse_dict.items():
            for pulse_index in range( AntIndex_dict[ant_name],  len(pulselist) ):
                pulse = pulselist[pulse_index]
                
                T_found = None
                
                if pulse.event_data is None:
                    if (pulse.antenna_status == 0 or pulse.antenna_status == 2):
                        if pulse.PolE_peak_time > current_time:
                            T_found = pulse.PolE_peak_time
                            
                    if (pulse.antenna_status == 0 or pulse.antenna_status == 3):
                        if  (T_found is None or pulse.PolO_peak_time<T_found) and (current_time<pulse.PolO_peak_time):
                            T_found = pulse.PolO_peak_time
                            
                if T_found is not None:
                    AntIndex_dict[ant_name] = pulse_index
                    if T_found < new_time:
                        new_time = T_found
                    break
                
        current_time = new_time
                
    print("found", len(SSPW_event_list), "SSPW")
    return SSPW_event_list
                            
                        
                

###### use the planewaves and find intersections between them ####
def multi_planewave_intersections(planewave_dict, referance_station=None, do_station_delays=True):
    """use the data from the planewaves to estimate source location and station delays"""
    
    station_order = planewave_dict.keys()
    planewaves = [ planewave_dict[sname] for sname in station_order]
    if referance_station is None:
        referance_station = station_order[0]
    elif referance_station not in planewave_dict:
        print( "ERROR!  referance station not in planewaves!" )
        quit()
        
    location_guesses = []
    for PW_index in range(len(planewaves)-1):
        PW_a = planewaves[PW_index]
        for PW_b in planewaves[PW_index+1:]:
            ## loop over each pair of planewaves
            ##find intersection
            distance_A = PW_a.ray_intersection(PW_b)
            if distance_A is not None: ## result could be None if the planewaves are nearly parrellel
                location_guesses.append( PW_a.get_point_on_ray(distance_A) )
              
    if len( location_guesses ) < 2:
        return None, None, None, None
            
    source_location = np.average(location_guesses, axis=0)
    referance_location = planewave_dict[referance_station].ref_location
    referance_distance = np.linalg.norm(source_location - referance_location)
    referance_time = planewave_dict[referance_station].ZAT[2]
    
    source_time = referance_time - referance_distance/v_air
    
    if do_station_delays:
        station_delays = {}
        for sname, planewave in planewave_dict.items():
            
            if sname == referance_station:
                station_delays[sname]=0
            else:
                distance = np.linalg.norm(source_location - planewave.ref_location)
                delta_T = (distance-referance_distance)/v_air
                measured_delta_T = planewave.ZAT[2] - referance_time
                delay = measured_delta_T - delta_T
                station_delays[sname] = delay
    else:
        station_delays = None
            
    return source_location, source_time, station_delays, referance_station

