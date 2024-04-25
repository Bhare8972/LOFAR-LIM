#!/usr/bin/env python3

#python
import time
from os import mkdir, listdir
from os.path import isdir, isfile
from itertools import chain
from datetime import date
#from pickle import load

#external
import numpy as np
np.set_printoptions(precision=10, threshold=np.inf)
from scipy.optimize import least_squares
from matplotlib import pyplot as plt

import h5py


#mine
from LoLIM.prettytable import PrettyTable
from LoLIM.utilities import logger, processed_data_dir, v_air, SId_to_Sname, Sname_to_SId_dict, RTD, \
    even_antName_to_odd, antName_is_even, antName_to_even, antName_to_odd

from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1

from LoLIM.stationTimings.autoCorrelator_tools import  delay_fitter_polT

from LoLIM.NumLib.LeastSquares import GSL_LeastSquares 


## TODO: 
# RMS by station
# whitelist of stations
# cross correlations
# fix error estimate (later comment: what is wrong with it??)
# different weights for different antennas
# atmosphere
# Kalman filtter bit. Maybe if location is none, follow kalman prodedure untill a chi-square is too high
# add a method to output source locations as SPSF format!

class Part1_input_manager:
    def __init__(self, processed_data_folder, input_files):
        self.max_num_input_files = 10
        if len(input_files) > self.max_num_input_files:
            print("TOO MANY INPUT FOLDERS!!!")
            quit()
        
        self.input_files = input_files
        
        self.input_data = []
        for folder_i, folder in enumerate(input_files):
            input_folder = processed_data_folder + "/" + folder +'/'
            
            file_list = [(int(f.split('_')[1][:-3])*self.max_num_input_files+folder_i ,input_folder+f) for f in listdir(input_folder) if f.endswith('.h5')] ## get all file names, and get the 'ID' for the file name
            file_list.sort( key=lambda x: x[0] ) ## sort according to ID
            self.input_data.append( file_list )
        
    def known_source(self, ID):
        
        file_i = int(ID/self.max_num_input_files)
        folder_i = ID - file_i*self.max_num_input_files
        file_list = self.input_data[ folder_i ]
        
        return [info for info in file_list if info[0]==ID][0]

class source_object():

    def __init__(self, ID, input_fname, stations_to_exclude, antennas_to_exclude, StationSampleTime, input_antenna_delays=None ):
        self.ID = ID
        self.stations_to_exclude = stations_to_exclude
        self.antennas_to_exclude = antennas_to_exclude
        self.data_file = h5py.File(input_fname, "r")
        self._num_data = 0
        self.input_antenna_delays = input_antenna_delays
        self.StationSampleTime = StationSampleTime  ## dictionary where key is sname, value is sample_number*5.0e-9

                 
    def prep_for_fitting(self, polarization, info, pol_flips):
        self.polarization = polarization ## 0 is even, 1 is odd, 2 is both
        
        self.pulse_times = np.empty( len(info.sorted_antenna_names), dtype=np.double )
        self.used_antenna_delays = np.empty( len(info.sorted_antenna_names), dtype=np.double )
        self.pulse_times[:] = np.nan
        self.used_antenna_delays[:] = np.nan
        self.waveforms = [None]*len(self.pulse_times)
        self.waveform_startTimes = [None]*len(self.pulse_times)
        
        #### first add times from referance_station
        for sname, ant_range in info.station_to_antenna_index_dict.items():
            if (sname in self.stations_to_exclude) or (sname not in self.data_file):
                continue
            
            
            station_group = self.data_file[sname]
            
            for ant_i in range(ant_range[0], ant_range[1]):
                ant_name = info.sorted_antenna_names[ant_i]
                
                if (not antName_is_even(ant_name)) or (ant_name not in station_group):
#                if (ant_name in self.antennas_to_exclude) or (ant_name in bad_antennas) :
                    continue
                
                even_ant_name = ant_name
                odd_ant_name = even_antName_to_odd( ant_name )

                do_pol_flip = even_ant_name in pol_flips
                
                ant_data = station_group[ant_name]
                start_index = ant_data.attrs['starting_index']
                


                ## even data
                E_peak_time = ant_data.attrs['PolE_peakTime']
                filePolE_timeOffset = ant_data.attrs['PolE_timeOffset_CS'] + self.StationSampleTime[sname]

                if self.input_antenna_delays is not None:
                    if even_ant_name in self.input_antenna_delays:
                        polE_timeOffset = self.input_antenna_delays[ even_ant_name ]
                    else:
                        polE_timeOffset = 0.0
                else:
                    polE_timeOffset = filePolE_timeOffset

                E_waveform = ant_data[0,:]
                E_HE_waveform = ant_data[1,:]
                E_amp = np.max( E_HE_waveform )


                ## odd data
                O_peak_time = ant_data.attrs['PolO_peakTime']

                filePolO_timeOffset = ant_data.attrs['PolO_timeOffset_CS']  + self.StationSampleTime[sname]
                if self.input_antenna_delays is not None:
                    if odd_ant_name in self.input_antenna_delays:
                        polO_timeOffset = self.input_antenna_delays[ odd_ant_name ]
                    else:
                        polO_timeOffset = 0.0
                else:
                    polO_timeOffset = filePolO_timeOffset

                O_waveform = ant_data[2,:]
                O_HE_waveform = ant_data[3,:]
                O_amp = np.max( O_HE_waveform )



                if do_pol_flip:
                    E_waveform, O_waveform           = O_waveform, E_waveform
                    polE_timeOffset, polO_timeOffset = polO_timeOffset, polE_timeOffset
                    E_peak_time, O_peak_time         = O_peak_time, E_peak_time
                    filePolE_timeOffset, filePolO_timeOffset = filePolO_timeOffset, filePolE_timeOffset
                    E_waveform, O_waveform           = O_waveform, E_waveform
                    E_amp, O_amp                     = O_amp, E_amp



                ## actually set data if is good
                even_is_bad = (even_ant_name in self.antennas_to_exclude) or (even_ant_name in info.bad_ants)
                if (not even_is_bad) and (polarization==0 or polarization==2 or polarization==3):
                    if np.isfinite(E_peak_time) and E_amp >= info.min_ant_amplitude:

                        self.used_antenna_delays[ant_i] = polE_timeOffset
                        self.pulse_times[ ant_i ] = E_peak_time + (filePolE_timeOffset - polE_timeOffset)
                        self.waveforms[ ant_i ] = E_waveform
                        self.waveform_startTimes[ ant_i ] = start_index*5.0E-9 - ( polE_timeOffset - self.StationSampleTime[sname] )

                        self._num_data += 1

                ## odd
                odd_is_bad = (odd_ant_name in self.antennas_to_exclude) or (odd_ant_name in info.bad_ants)
                if (not odd_is_bad) and (polarization==1 or polarization==2 or polarization==3):
                    if np.isfinite(O_peak_time) and O_amp >= info.min_ant_amplitude:

                        self.used_antenna_delays[ant_i+1] = polO_timeOffset
                        self.pulse_times[ ant_i+1 ] = O_peak_time + (filePolO_timeOffset - polO_timeOffset)
                        self.waveforms[ ant_i+1 ] = O_waveform
                        self.waveform_startTimes[ ant_i+1 ] = start_index*5.0E-9 - ( polO_timeOffset - self.StationSampleTime[sname] )

                        self._num_data += 1


                    
    def num_data(self):
        return self._num_data
                
                

class run_fitter:
    def __init__(self, timeID, output_folder, pulse_input_folders, guess_timings, sources_to_fit, guess_source_locations,
               source_polarizations, source_stations_to_exclude, source_antennas_to_exclude, bad_ants,
               antennas_to_recalibrate={}, min_ant_amplitude=10, ref_station="CS002", input_antenna_delays=None, new_pol_flips=[]):
        
        self.timeID = timeID
        self.output_folder = output_folder
        self.pulse_input_folders = pulse_input_folders
        self.guess_timings = guess_timings
        self.sources_to_fit = sources_to_fit
        self.guess_source_locations = guess_source_locations
        self.source_polarizations = source_polarizations
        self.source_stations_to_exclude = source_stations_to_exclude
        self.source_antennas_to_exclude = source_antennas_to_exclude
        self.bad_ants = bad_ants
        self.antennas_to_recalibrate = antennas_to_recalibrate
        self.min_ant_amplitude = min_ant_amplitude
        self.ref_station = ref_station
        self.input_antenna_delays = input_antenna_delays ## this overrides the antenna delays in the pulse files. SHould be dict of antenna delays.


        ## make sure new_pol_flips is all even
        self.new_pol_flips = [ antName_to_even(PF) for PF in new_pol_flips ]
           ## this flips all known antenna delays, but NOT antennas_to_recalibrate


        
        # variables to change
        self.num_stat_per_table = 10


    def setup(self):
        if self.ref_station in self.guess_timings:
            ref_T = self.guess_timings[self.ref_station]
            self.guess_timings = {station:T-ref_T for station,T in self.guess_timings.items() if station != self.ref_station}
 
        processed_data_folder = processed_data_dir( self.timeID )
    
        data_dir = processed_data_folder + "/" + self.output_folder
        if not isdir(data_dir):
            mkdir(data_dir)
        self.data_dir = data_dir

        #Setup logger
        logging_folder = data_dir# + '/logs_and_plots'
        # if not isdir(logging_folder):
        #     mkdir(logging_folder)
        self.logging_folder = logging_folder
            
        self.log = logger()
        self.log.set(logging_folder + "/log_out.txt") ## TODo: save all output to a specific output folder
        self.log.take_stderr()
        self.log.take_stdout()
        
        print("timeID:", self.timeID)
        print("date and time run:", time.strftime("%c") )
        print("input folders:", self.pulse_input_folders)
        print("source IDs to fit:", self.sources_to_fit)
        print("guess locations:", self.guess_source_locations)
        print("polarization to use:", self.source_polarizations)
        print("source stations to exclude:", self.source_stations_to_exclude)
        print("bad antennas:", self.bad_ants)
        print("referance station:", self.ref_station)
        print("guess delays:", self.guess_timings)
        print()
        
        
        #### open data and data processing stuff ####
        print("loading data")
        
        ## needed for ant locs
        raw_fpaths = filePaths_by_stationName( self.timeID )
        raw_data_files = {sname:MultiFile_Dal1(fpaths, force_metadata_ant_pos=True) for sname,fpaths in raw_fpaths.items() if sname in chain(self.guess_timings.keys(), [self.ref_station]) }
    
        #### sort antennas and stations ####
        self.station_order = list(self.guess_timings.keys()) + [self.ref_station]
        self.sorted_antenna_names = []
        self.station_to_antenna_index_dict = {}
        self.ant_loc_dict = {}
        self.StationSampleTime = {}

        if self.input_antenna_delays is not None:
            self.used_antenna_delays = []
        
        for sname in self.station_order:
            first_index = len(self.sorted_antenna_names)
            
            stat_data = raw_data_files[sname]
            self.StationSampleTime[sname] = stat_data.get_nominal_sample_number()*5.0e-9

            ant_names = stat_data.get_antenna_names()
            stat_ant_locs = stat_data.get_LOFAR_centered_positions()
            
            self.sorted_antenna_names += ant_names
            
            for ant_name, ant_loc in zip(ant_names,stat_ant_locs):
                self.ant_loc_dict[ant_name] = ant_loc

                if self.input_antenna_delays is not None:
                    if ant_name in self.input_antenna_delays:
                        self.used_antenna_delays.append( self.input_antenna_delays[ant_name] )
                    else:
                        self.used_antenna_delays.append( 0.0 )
                    
            self.station_to_antenna_index_dict[sname] = (first_index, len(self.sorted_antenna_names))



        if self.input_antenna_delays is not None:
            self.used_antenna_delays = np.array( self.used_antenna_delays )

            ## account for pol flips
            for even_pf in self.new_pol_flips:
                if even_pf in self.sorted_antenna_names:
                    ant_i = self.sorted_antenna_names.index( even_pf )
                    A = self.used_antenna_delays[ant_i]
                    self.used_antenna_delays[ant_i] = self.used_antenna_delays[ant_i+1]
                    self.used_antenna_delays[ant_i+1] = A


        
        self.ant_locs = np.zeros( (len(self.sorted_antenna_names), 3))
        for i, ant_name in enumerate(self.sorted_antenna_names):
            self.ant_locs[i] = self.ant_loc_dict[ant_name]
        
        self.station_locations = {sname:self.ant_locs[self.station_to_antenna_index_dict[sname][0]] for sname in self.station_order }
        self.station_to_antenna_index_list = [ self.station_to_antenna_index_dict[sname] for sname in self.station_order ]
        
        
        #### sort the delays guess, and account for station locations ####
        self.current_delays_guess = np.array([ self.guess_timings[sname] for sname in self.station_order[:-1] ])
        self.original_delays = np.array( self.current_delays_guess )
        self.ant_recalibrate_order = np.array( [ i for i,antname in enumerate( self.sorted_antenna_names ) if antname in self.antennas_to_recalibrate.keys() ], dtype=int)
        self.ant_recalibrate_guess = np.array( [self.antennas_to_recalibrate[ self.sorted_antenna_names[i] ] for i in self.ant_recalibrate_order] )

        self.station_indeces = np.empty( len( self.sorted_antenna_names ), dtype=int )
        for station_index, index_range in enumerate(self.station_to_antenna_index_list): ## note this DOES include ref stat
            first,last = index_range
            self.station_indeces[first:last] = station_index
    
    
        #### now we open the pulses ####
        input_manager = Part1_input_manager( processed_data_folder, self.pulse_input_folders )
        
        self.current_sources = []
        self.num_XYZT_params = 0
        if self.input_antenna_delays is None:
            self.used_antenna_delays = None
            
        for knownID in self.sources_to_fit:
            source_ID, input_name = input_manager.known_source( knownID )
            
            print("prep fitting:", source_ID)
            
            polarity = self.source_polarizations[source_ID]
            
            source_to_add = source_object(source_ID, input_name,
                                          self.source_stations_to_exclude[source_ID], 
                                          self.source_antennas_to_exclude[source_ID],
                                          StationSampleTime = self.StationSampleTime,
                                          input_antenna_delays = self.input_antenna_delays )
            source_to_add.prep_for_fitting( polarity, self, self.new_pol_flips)
            print("  n. ants:", source_to_add.num_data() )


            if self.used_antenna_delays is None:
                self.used_antenna_delays = np.array( source_to_add.used_antenna_delays )   ### this already accounts for pol-flips
            elif self.input_antenna_delays is None:
                diff = self.used_antenna_delays - source_to_add.used_antenna_delays
                IS_FINITE = np.isfinite(diff)
                if np.sum( IS_FINITE ) > 0:
                    maxdiff = np.max( np.abs( diff[ IS_FINITE ] ) )
                    if maxdiff > 1e-11:
                        print('WARNING: antenna cal difference! max:', maxdiff )

                for antI in range(len( self.sorted_antenna_names )):
                    if (not np.isfinite( self.used_antenna_delays[antI] ) ) and np.isfinite( source_to_add.used_antenna_delays[antI] ):
                        self.used_antenna_delays[antI] = source_to_add.used_antenna_delays[antI]

            self.num_XYZT_params += 4
            if polarity == 2:
                self.num_XYZT_params += 1
            elif polarity == 3:
                self.num_XYZT_params += 4
            
            self.current_sources.append( source_to_add )
            
            


            
        self.num_sources = len( self.current_sources )
        self.num_delays = len( self.original_delays )
        self.num_RecalAnts = len( self.ant_recalibrate_order )
        self.num_antennas = len( self.sorted_antenna_names )
        self.num_measurments = self.num_sources*self.num_antennas
        
        self.current_solution = np.zeros( self.num_delays + self.num_RecalAnts + self.num_XYZT_params )
        self.current_solution[ :self.num_delays ] = self.current_delays_guess
        self.current_solution[ self.num_delays: self.num_delays + self.num_RecalAnts] = self.ant_recalibrate_guess
        
        self.fitter = delay_fitter_polT( self.ant_locs,  self.station_indeces,  self.ant_recalibrate_order,  
                    self.num_sources, self.num_delays )
        
        param_i = self.num_delays + self.num_RecalAnts
        self.num_DOF = -self.num_delays -self.num_RecalAnts - self.num_XYZT_params
        for i,PSE in enumerate(self.current_sources):
            
            guess_loc = self.guess_source_locations[ PSE.ID ]
            self.current_solution[ param_i : param_i+4 ] = guess_loc[:4]
            param_i += 4## NOTE!
            
            if PSE.polarization == 2:
                if len(guess_loc) == 4:
                    self.current_solution[ param_i ] = guess_loc[3]
                elif len(guess_loc) == 5:
                    self.current_solution[ param_i ] = guess_loc[4]
                elif len(guess_loc) == 8:
                    self.current_solution[ param_i ] = guess_loc[7]
                param_i += 1
            elif PSE.polarization == 3:
                if (len(guess_loc) == 4) or (len(guess_loc) == 5):
                    self.current_solution[ param_i : param_i+4 ] = guess_loc[:4]
                elif len(guess_loc) == 8:
                    self.current_solution[ param_i : param_i+4 ] = guess_loc[4:8]
                param_i += 4
                
            self.num_DOF += PSE.num_data()
            self.fitter.set_event( PSE.pulse_times, PSE.polarization )
            
    def fit(self, num_stoch_iters, num_switch_iters, max_s_itters=np.inf, randomness=10E-9, ant_randomness=0.5e-9, 
        min_num_iters=10, max_num_iters=1000, xtol=1e-16, ftol=1e-16, gtol=1e-16):
        """if num_switch_iters is negative, then it is minimum itterations, only ending once RMS increases between runs"""

        num_switch_iters_isMinimum = False
        if num_switch_iters < 0:
            num_switch_iters = np.abs(num_switch_iters)
            num_switch_iters_isMinimum = True
        max_s_itters = max(max_s_itters, num_switch_iters)

        initial_RMS = self.fitter.RMS( self.current_solution, self.num_DOF )
        print("initial RMS:", initial_RMS)
        
        print('fitting:')
        best_sol = np.array( self.current_solution )
        best_RMS = initial_RMS
        number_LS_runs = 0
        num_LS_runs_needMoreItters = 0

        fitterSQ= GSL_LeastSquares( self.fitter.get_num_parameters(),  self.fitter.get_num_measurments(), self.fitter.objective_fun_sq, jacobian=self.fitter.objective_fun_sq_jacobian )
        fitter = GSL_LeastSquares( self.fitter.get_num_parameters(),  self.fitter.get_num_measurments(), self.fitter.objective_fun, jacobian=self.fitter.objective_fun_jacobian )

        for i in range( num_stoch_iters ):
            
                current_solution = np.array( best_sol )
                current_solution[ : self.num_delays] += np.random.normal(scale=randomness, size=self.num_delays )
                current_solution[ self.num_delays : self.num_delays+self.num_RecalAnts] += np.random.normal(scale=ant_randomness, size=self.num_RecalAnts )
                current_solution[self.num_delays+self.num_RecalAnts:] += np.random.normal(scale=randomness, size=len(current_solution)-(self.num_delays+self.num_RecalAnts) )

                print(i)

                switch_itter = 0
                previous_RMS = np.inf
                RMS_had_increased = False
                while num_switch_iters_isMinimum or switch_itter<num_switch_iters:
                    # fit_res = least_squares( self.fitter.objective_fun_sq, current_solution, jac='2-point', method='lm', xtol=3.0E-16, ftol=3.0E-16, gtol=3.0E-16, x_scale='jac', max_nfev=max_nfev)
                    # fit_res = least_squares( self.fitter.objective_fun_sq, current_solution,  jac=self.fitter.objective_fun_sq_jacobian, method='lm', xtol=3.0E-16, ftol=3.0E-16, gtol=3.0E-16, x_scale='jac', max_nfev=max_nfev)
                    
                    fitterSQ.reset( current_solution )
                    code, text = fitterSQ.run(min_num_iters, max_itters=max_num_iters, xtol=xtol, gtol=gtol, ftol=ftol)
                    current_solution = fitterSQ.get_X()
                    # num_runs = fitterSQ.get_num_iters()

                    if code == 4:
                        num_LS_runs_needMoreItters += 1
                    
                    ARMS = self.fitter.RMS( current_solution, self.num_DOF )
                    
                    # fit_res = least_squares( self.fitter.objective_fun, current_solution, jac='2-point', method='lm', xtol=3.0E-16, ftol=3.0E-16, gtol=3.0E-16, x_scale='jac', max_nfev=max_nfev)
                    # fit_res = least_squares( self.fitter.objective_fun, current_solution, jac=self.fitter.objective_fun_sq_jacobian, method='lm', xtol=3.0E-16, ftol=3.0E-16, gtol=3.0E-16, x_scale='jac', max_nfev=max_nfev)
                    # fit_res = least_squares( self.fitter.objective_fun, current_solution, jac=self.fitter.objective_fun_jacobian, method='lm', xtol=3.0E-16, ftol=3.0E-16, gtol=3.0E-16, x_scale='jac', max_nfev=max_num_iters)
                    
                    fitter.reset( current_solution )
                    code, text = fitter.run(min_num_iters, max_itters=max_num_iters, xtol=xtol, gtol=gtol, ftol=ftol)
                    current_solution = fitter.get_X()
                    num_runs = fitter.get_num_iters()

                    if code == 4:
                        num_LS_runs_needMoreItters += 1

                    number_LS_runs += 2
                    RMS = self.fitter.RMS( current_solution, self.num_DOF )
                    
                    print("  ", i, switch_itter,  ":", RMS, '(', ARMS, ')', text, 'N:', num_runs)

                    if RMS>previous_RMS:
                        RMS_had_increased = True
                    
                    if RMS < best_RMS:
                        best_RMS = RMS
                        best_sol = np.array( current_solution )
                        print('IMPROVEMENT!')
                    else:
                        if num_switch_iters_isMinimum and (switch_itter>=num_switch_iters) and RMS_had_increased:
                            break
                        if switch_itter>=max_s_itters:
                            break

                    previous_RMS = RMS
                    switch_itter += 1

        self.current_solution = best_sol
        self.GSL_covariance_matrix = fitter.get_covariance_matrix()
        print('frac. runs need more itters:', num_LS_runs_needMoreItters/number_LS_runs)
        print()
        print()
        
        print('initial RMS:', initial_RMS)
        print("best RMS:", best_RMS)
            
        
        print('fits:')
                
        stations_to_print = self.station_order
        PSE_RMS_data = [ self.fitter.event_SSqE( source_i, best_sol ) for source_i in range(len(self.current_sources)) ]
        while len(stations_to_print) > 0:
            stations_this_run = stations_to_print[:self.num_stat_per_table]
            stations_to_print = stations_to_print[len(stations_this_run):]
            
            fit_table = PrettyTable()
            fit_table.field_names = ['id'] + stations_this_run + ['total']
            fit_table.float_format = '.2E'
            
            for source_i,PSE in enumerate(self.current_sources):
                new_row = ['']*len(fit_table.field_names)
                new_row[0] = PSE.ID
                
#                PSE_SSqE, total_ants = self.fitter.event_SSqE( source_i, best_sol )
                new_row[-1] = np.sqrt( PSE_RMS_data[source_i][0] / (PSE_RMS_data[source_i][1]-4) )
                
                for i,sname in enumerate( stations_this_run ):
                    stat_ant_range = self.station_to_antenna_index_dict[sname]
                    SSqE, numants = self.fitter.event_SSqE( source_i, best_sol, stat_ant_range  )
                    if numants > 0:
                        new_row[i+1] = np.sqrt( SSqE/numants )
                    else:
                        new_row[i+1] = ''
                        
                fit_table.add_row( new_row )
                
            print( fit_table )
            print()
            
        print()
        print()
        
        station_delays = best_sol[:self.num_delays]
        antenna_delays = best_sol[ self.num_delays:self.num_delays+self.num_RecalAnts ]
        source_locs = best_sol[ self.num_delays+self.num_RecalAnts: ]
        
        print("stat delays:")
        for sname, delay, original in zip( self.station_order, station_delays, self.original_delays ):
            print("'"+sname+"' :", delay, ", ## diff to guess:", delay-original)
                  
        print()
        print()
        
        print("ant delays:")
        for ant_i, delay in zip(self.ant_recalibrate_order, antenna_delays):
            ant_name = self.sorted_antenna_names[ant_i]
            print("'"+ant_name+"' : ", delay, ', #', SId_to_Sname[ int(ant_name[:3]) ])
                  
        print()
        print()
            
        print("locations:")
        offset = 0
        for i,source in enumerate(self.current_sources):
            
            print(source.ID,':[', source_locs[offset+0], ',', source_locs[offset+1], ',', np.abs(source_locs[offset+2]), ',', source_locs[offset+3], end=' ')# '],')
            
            offset += 4
            if source.polarization == 2:
                print( ',',  source_locs[offset], '],')
                offset += 1
            elif  source.polarization == 3:
                print(',')
                print("   ", source_locs[offset+0], ',', source_locs[offset+1], ',', np.abs(source_locs[offset+2]), ',', source_locs[offset+3], '],' )
                offset += 4
            else: ## end pol = 0 or 1
                print('],')
            
        print()
        print()
        
        print("REL LOCS")
        refX, refY, refZ, refT = source_locs[:4]
        offset = 0
        for i,source in enumerate(self.current_sources):
            X,Y,Z,T = source_locs[offset:offset+4]
            print(source.ID,':[', X-refX, ',',  Y-refY, ',', np.abs(Z)-np.abs(refZ), ',', T-refT, '],')
            offset +=4
            if source.polarization == 2:
                offset += 1
            elif  source.polarization == 3:
                offset += 4
                
                
    def print_antenna_info(self, antname):
        self.fitter.print_antenna_info( self.sorted_antenna_names.index( antname ), self.current_solution )

    def print_antenna_RMS(self):
        print("goodness of fit by antenna")
        print("antenna   num_fits  RMS<2.0E-9   RMS")
        print()
        print("  recalibrated antennas:")
        
        antenna_SSqE, antenna_num_fits = self.fitter.fit_by_antenna( self.current_solution )
        
        for ant_i in self.ant_recalibrate_order:
            print( self.sorted_antenna_names[ant_i],  antenna_num_fits[ant_i], end=' ')
            if antenna_num_fits[ant_i] == 0:
                print()
            else:
                RMS = np.sqrt( antenna_SSqE[ant_i]/antenna_num_fits[ant_i] )
                print(RMS<2.0E-9, RMS)
                
        print()
        print("  all antennas:")
        RMSs = np.sqrt( antenna_SSqE/ (antenna_num_fits+1.0e-10) )
        sorter = np.argsort( RMSs )[::-1]
        for ant_i in sorter: #range(len(self.sorted_antenna_names)):
            print( self.sorted_antenna_names[ant_i],  antenna_num_fits[ant_i], end=' ')
            if antenna_num_fits[ant_i] == 0:
                print()
            else:
                RMS = np.sqrt( antenna_SSqE[ant_i]/antenna_num_fits[ant_i] )
                print(RMS<2.0E-9, RMS)
            
    def print_cov_diag(self, useGSL=True):
        if useGSL:
            cov = self.GSL_covariance_matrix
        else:
            cov = self.fitter.analytical_covariance_matrix( self.current_solution )
        diag_cov = np.sqrt( cov.diagonal() )

        stat_delays = diag_cov[:self.num_delays]
        ant_delays = diag_cov[ self.num_delays:self.num_delays+self.num_RecalAnts]
        source_locs = diag_cov[ self.num_delays+self.num_RecalAnts: ]
        
        print("### COVARIANCE DIAGONALS ###")
        
        print("stat delay error:")
        for sname, err in zip( self.station_order, stat_delays ):
            print(sname, err)
                  
        print()
        print()
        
        print("ant delay error:")
        for ant_i, err in zip(self.ant_recalibrate_order, ant_delays):
            ant_name = self.sorted_antenna_names[ant_i]
            print(ant_name, err)
                  
        print()
        print()
        
        print("average error of whole flash:")
        ave_operator = np.zeros( (4, len(self.current_solution)), dtype=np.double ) ## gets the average of X, Y, and Z
        diff_ave_operator = np.zeros( (4*self.num_sources, len(self.current_solution)), dtype=np.double ) ## gets distance from average for each source
        w = 1.0/len(self.current_sources)
        offset = self.num_delays + self.num_RecalAnts
        for i, source_i in enumerate(self.current_sources):
            ave_operator[0, offset + 0] = w
            ave_operator[1, offset + 1] = w
            ave_operator[2, offset + 2] = w
            ave_operator[3, offset + 3] = w
            
            offset_j = 0
            for j,source_j in enumerate(self.current_sources):
                diff = 0
                if i==j:
                    diff -= 1
                diff_ave_operator[offset_j + 0, offset + 0] = w + diff
                diff_ave_operator[offset_j + 1, offset + 1] = w + diff
                diff_ave_operator[offset_j + 2, offset + 2] = w + diff
                diff_ave_operator[offset_j + 3, offset + 3] = w + diff
                
                offset_j += 4
                if source_j.polarization==2:
                    offset_j += 1
                
            offset += 4
            if source_i.polarization==2:
                offset += 1
            
        ave_cov = np.dot(  np.dot(ave_operator, cov), ave_operator.T) 
        print(np.sqrt( np.diag(ave_cov) ))
        
        print()
        print()
            
        print("location error (rel. err):")
        print("raw X Y Z T std")
        print("  processed X Y Z T std")
        
        diff_ave_cov = np.dot(  np.dot(diff_ave_operator, cov), diff_ave_operator.T)
        rel_errors = np.sqrt( np.diag(diff_ave_cov) )
        offset = 0
        for i,source in enumerate(self.current_sources):
            print(source.ID, source_locs[offset+0], source_locs[offset+1], source_locs[offset+2], source_locs[offset+3])
            print("  ", rel_errors[4*i+0], rel_errors[4*i+1], rel_errors[4*i+2], rel_errors[4*i+3] ) ## note that rel_errors only looks at one T!
            print()
            
            offset += 4
            if source.polarization == 2:
                offset += 1
            
        print()
        print()


    def output_totalcal(self, previous_PolFlips=[], previous_SignFlips=[] ):
        """ outputs total cal file. TO be consistant it needs to know previous_PolFlips and previous_SignFlips, typically acquired from a different totalCal file.
         previous_PolFlips should be a list of even antenna names corresponding to dipole pairs that should be flipped.
         previous_SignFlips should be a list of dipole names that need a sign flip.
         """

        station_delays = self.current_solution[:self.num_delays]
        antenna_delays = self.current_solution[ self.num_delays:self.num_delays+self.num_RecalAnts ]


        with open( self.data_dir + '/totalCalOut.txt', 'w' ) as fout:
            fout.write('v1\n')
            fout.write('# timeid ')
            fout.write(self.timeID)
            fout.write('\n')
            fout.write('#vair 299792458.0/1.000293')
            fout.write('\n')
            fout.write('# made')
            fout.write( date.today().strftime('%d-%b-%Y') )
            fout.write('\n')

            fout.write('bad_antennas\n')
            for antname in self.bad_ants:
                fout.write(antname)
                fout.write('\n')

            antenna_SSqE, antenna_num_fits = self.fitter.fit_by_antenna(self.current_solution)
            for ant_i in range(len(self.sorted_antenna_names)):
                if antenna_num_fits[ant_i] == 0:
                    name = self.sorted_antenna_names[ant_i]
                    if name not in self.bad_ants:
                        fout.write(name)
                        fout.write('\n')





            fout.write('antenna_delays\n')  ## these should be AFTER pol flips, so we're good!
            for ant_i in range(len(self.sorted_antenna_names)):
                name = self.sorted_antenna_names[ant_i]
                if (not name in self.bad_ants) and (antenna_num_fits[ant_i] >0):
                    delay = self.used_antenna_delays[ant_i]

                    if ant_i in self.ant_recalibrate_order:
                        I = np.where( self.ant_recalibrate_order == ant_i)[0][0]
                        delay += antenna_delays[I]

                    fout.write(name)
                    fout.write(' ')
                    fout.write(str(delay))
                    fout.write('\n')





            fout.write('station_delays\n')
            for sname, delay in zip( self.station_order, station_delays ):
                fout.write(sname)
                fout.write(' ')
                fout.write(str(delay))
                fout.write('\n')
            fout.write(self.ref_station)
            fout.write(' 0.0\n')

            fout.write('pol_flips\n')
            for ant in self.new_pol_flips:
                fout.write(ant)
                fout.write('\n')

            for ant in previous_PolFlips:
                if ant not in self.new_pol_flips:
                    fout.write(ant)
                    fout.write('\n')

            fout.write('sign_flips\n')
            for ant in previous_SignFlips:
                fout.write(ant)
                fout.write('\n')
    
    