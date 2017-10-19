#!/usr/bin/env python3

#python
import time
from os import mkdir
from os.path import isdir

#external
import numpy as np
from scipy.optimize import least_squares

#mine
from utilities import log, processed_data_dir, v_air
from binary_IO import read_long

from read_PSE import read_PSE_timeID
#from read_pulse_data import read_station_info
from planewave_functions import read_SSPW_timeID


class fitting_PSE:
    def __init__(self, antenna_locations, PSE):
        self.PSE = PSE
        self.PSE.load_antenna_data()
        
        if PSE.PolE_RMS<PSE.PolO_RMS:
            self.polarity = 0
        else:
            self.polarity = 1
            
        self.XYZT = PSE.PolE_loc if self.polarity==0 else PSE.PolO_loc
            
        ant_X = []
        ant_Y = []
        ant_Z = []
        pulse_times = []
        
        for ant_name, ant_data in self.PSE.antenna_data.items():
            pt = ant_data.PolE_peak_time if self.polarity==0 else ant_data.PolO_peak_time
            if np.isfinite(pt):
                loc = antenna_locations[ant_name]
                ant_X.append( loc[0] ) 
                ant_Y.append( loc[1] ) 
                ant_Z.append( loc[2] )
                
                pulse_times.append( pt )
            
            
        self.PSE_ant_X = np.array( ant_X )
        self.PSE_ant_Y = np.array( ant_Y )
        self.PSE_ant_Z = np.array( ant_Z )
        self.PSE_pulse_times = np.array( pulse_times )
        
    def set_SSPW(self, antenna_locs, SSPW):
            
        ant_X = []
        ant_Y = []
        ant_Z = []
        pulse_times = []
        
        for ant_name, ant_data in SSPW.ant_data.items():
            pt = ant_data.PolE_peak_time if self.polarity==0 else ant_data.PolO_peak_time 
            if np.isfinite(pt):
                X,Y,Z = antenna_locs[ant_name]
                ant_X.append(X)
                ant_Y.append(Y)
                ant_Z.append(Z)
                pulse_times.append( pt )
            
        self.SSPW_ant_X = np.array( ant_X )
        self.SSPW_ant_Y = np.array( ant_Y )
        self.SSPW_ant_Z = np.array( ant_Z )
        self.SSPW_pulse_times = np.array( pulse_times )
            
        ### get the pre-fit values ##
        delta_X = self.SSPW_ant_X - self.XYZT[0]
        delta_Y = self.SSPW_ant_Y - self.XYZT[1]
        delta_Z = self.SSPW_ant_Z - self.XYZT[2]
        
        delta_X *= delta_X
        delta_Y *= delta_Y
        delta_Z *= delta_Z
        
        model_T = delta_X
        model_T += delta_Y
        model_T += delta_Z
        np.sqrt(model_T, out=model_T)
        model_T *= 1.0/v_air
        model_T += self.XYZT[3]
        
        model_T -= self.SSPW_pulse_times
        
        delay = np.average(model_T)
        model_T -= delay
        model_T *= model_T
        
        return -delay, np.sqrt(np.sum(model_T)/float(len(model_T)))
    
    def try_location_LS(self, delay, XYZT, out):
        X,Y,Z,T = XYZT
        
        delta_X_sq = self.PSE_ant_X-X
        delta_Y_sq = self.PSE_ant_Y-Y
        delta_Z_sq = self.PSE_ant_Z-Z
        
        SSPWdelta_X_sq = self.SSPW_ant_X-X
        SSPWdelta_Y_sq = self.SSPW_ant_Y-Y
        SSPWdelta_Z_sq = self.SSPW_ant_Z-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
        SSPWdelta_X_sq *= SSPWdelta_X_sq
        SSPWdelta_Y_sq *= SSPWdelta_Y_sq
        SSPWdelta_Z_sq *= SSPWdelta_Z_sq
        
            
        out[:len(self.PSE_ant_X)] = T - self.PSE_pulse_times
        out[len(self.PSE_ant_X):] = (T+delay) - self.SSPW_pulse_times
                
                
        out *= v_air
        out *= out ##this is now delta_t^2 *C^2
        
        out[:len(self.PSE_ant_X)] -= delta_X_sq
        out[:len(self.PSE_ant_X)] -= delta_Y_sq
        out[:len(self.PSE_ant_X)] -= delta_Z_sq
        
        out[len(self.PSE_ant_X):] -= SSPWdelta_X_sq
        out[len(self.PSE_ant_X):] -= SSPWdelta_Y_sq
        out[len(self.PSE_ant_X):] -= SSPWdelta_Z_sq
    
    def try_location_JAC(self, delay, XYZT, out_delay, out_loc):
        X,Y,Z,T = XYZT
        
        out_loc[:,0] = X
        out_loc[:len(self.PSE_ant_X),0] -= self.PSE_ant_X
        out_loc[len(self.PSE_ant_X):,0] -= self.SSPW_ant_X
        out_loc[:,0] *= -2.0
        
        out_loc[:,1] = Y
        out_loc[:len(self.PSE_ant_X),1] -= self.PSE_ant_Y
        out_loc[len(self.PSE_ant_X):,1] -= self.SSPW_ant_Y
        out_loc[:,1] *= -2.0
        
        out_loc[:,2] = Z
        out_loc[:len(self.PSE_ant_X),2] -= self.PSE_ant_Z
        out_loc[len(self.PSE_ant_X):,2] -= self.SSPW_ant_Z
        out_loc[:,2] *= -2.0
        
        
        out_loc[:len(self.PSE_ant_X),3] = self.PSE_pulse_times - T
        out_loc[len(self.PSE_ant_X):,3] = self.SSPW_pulse_times - T - delay
        out_loc[:,3] *= -2*v_air*v_air
        
        out_delay[len(self.PSE_ant_X):] = out_loc[len(self.PSE_ant_X):,3]    
        
    def RMS_without(self, XYZT):
        X,Y,Z,T = XYZT
        
        delta_X_sq = self.PSE_ant_X-X
        delta_Y_sq = self.PSE_ant_Y-Y
        delta_Z_sq = self.PSE_ant_Z-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
        distance = delta_X_sq
        distance += delta_Y_sq
        distance += delta_Z_sq
        
        np.sqrt(distance, out=distance)
        distance *= 1.0/v_air
        distance += T
        distance -= self.PSE_pulse_times
        
        distance *= distance
        
        return np.sqrt( np.sum(distance)/float(len(distance)) )
    
    def RMS_with(self, delay, XYZT):
        X,Y,Z,T = XYZT
        
        delta_X_sq = self.PSE_ant_X-X
        delta_Y_sq = self.PSE_ant_Y-Y
        delta_Z_sq = self.PSE_ant_Z-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
        distance = delta_X_sq
        distance += delta_Y_sq
        distance += delta_Z_sq
        
        np.sqrt(distance, out=distance)
        distance *= 1.0/v_air
        distance += T
        distance -= self.PSE_pulse_times
        
        distance *= distance
        
        
        delta_X_sq = self.SSPW_ant_X-X
        delta_Y_sq = self.SSPW_ant_Y-Y
        delta_Z_sq = self.SSPW_ant_Z-Z
        
        delta_X_sq *= delta_X_sq
        delta_Y_sq *= delta_Y_sq
        delta_Z_sq *= delta_Z_sq
        
        SSPWdistance = delta_X_sq
        SSPWdistance += delta_Y_sq
        SSPWdistance += delta_Z_sq
        
        np.sqrt(SSPWdistance, out=SSPWdistance)
        SSPWdistance *= 1.0/v_air
        SSPWdistance += T+delay
        SSPWdistance -= self.SSPW_pulse_times
        
        SSPWdistance *= SSPWdistance
        
        return np.sqrt(  (np.sum(distance) + np.sum(SSPWdistance))/(len(distance)+len(SSPWdistance))  )
        
    
        

if __name__=="__main__":
    ##opening data
    timeID = "D20160712T173455.100Z"
    output_folder = "autoCorrelate_RS406"
    
    SSPW_folder = "excluded_planewave_data" 
    PSE_folder = "allPSE"
    
    station_to_correlate = "RS406"
    
    max_RMS = 9.0E-8
    max_delay = 1.0E-5
    max_pdiff = 0.1
    
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
    log("SSPW folder:", SSPW_folder)
    log("PSE folder:", PSE_folder)
    log("station to correlate:", station_to_correlate)
    log("maximum RMS to consider:", max_RMS)
    log("maximum delay to consider:", max_delay)
    log("maximum pdiff:", max_pdiff)

    PSE_data_dict = read_PSE_timeID(timeID, PSE_folder, data_loc="/home/brian/processed_files")
    SSPW_data_dict = read_SSPW_timeID(timeID, SSPW_folder, data_loc="/home/brian/processed_files", stations=[station_to_correlate]) ## still need to implement this...
    
    PSE_ant_locations = PSE_data_dict["ant_locations"]
    fitting_PSE_list = [fitting_PSE(PSE_ant_locations, PSE) for PSE in PSE_data_dict["PSE_list"] ]
    
    SSPW_list = SSPW_data_dict["SSPW_dict"][station_to_correlate]
    SSPW_locs_dict = SSPW_data_dict["ant_locations"]

    
    for PSE_A in fitting_PSE_list:
        for PSE_B in fitting_PSE_list:
            if PSE_B.PSE.unique_index <= PSE_A.PSE.unique_index:
                    continue
                
            print("PSE A", PSE_A.PSE.unique_index)
            print("    initial fit:", PSE_A.RMS_without(PSE_A.XYZT))
            print("PSE B", PSE_B.PSE.unique_index)
            print("    initial fit:", PSE_B.RMS_without(PSE_B.XYZT))
            
            previous_found = False
            for SSPW_A in SSPW_list:
                    
                found_sol = False
                initial_Delay_A, initial_fit_A = PSE_A.set_SSPW(SSPW_locs_dict, SSPW_A)
                for SSPW_B in SSPW_list:
                    print()
                    print("   PSE", PSE_A.PSE.unique_index, '-> SSPW',SSPW_A.unique_index)
                    print("   PSE", PSE_B.PSE.unique_index, '-> SSPW',SSPW_B.unique_index)
                    
                    initial_Delay_B, initial_fit_B = PSE_B.set_SSPW(SSPW_locs_dict, SSPW_B)
                    
                    pdiff =  np.abs(2.0*(initial_Delay_A-initial_Delay_B)/(initial_Delay_A+initial_Delay_B) )
                    if pdiff > max_pdiff:
                        print(" pdiff too large:", pdiff)
                        continue
    
                    initial_guess = np.zeros(9, dtype=np.double)
                    initial_guess[1:5] = PSE_A.XYZT
                    initial_guess[5:9] = PSE_B.XYZT
                   
                    N_ant_A = len(PSE_A.PSE_ant_X) + len(PSE_A.SSPW_ant_X)
                    N_ant_B = len(PSE_B.PSE_ant_X) + len(PSE_B.SSPW_ant_X)
                    N_ant = N_ant_A+N_ant_B
                    
                    workspace_sol = np.zeros(N_ant, dtype=np.double)
                    workspace_jac = np.zeros((N_ant, 9), dtype=np.double)
            
            
                    def objective_fun(sol):
                        delay = sol[0]
                        XYZT_A = sol[1:5]
                        XYZT_B = sol[5:9]
                        PSE_A.try_location_LS(delay, XYZT_A, workspace_sol[:N_ant_A])
                        PSE_B.try_location_LS(delay, XYZT_B, workspace_sol[N_ant_A:])
                        return workspace_sol
                
                    def objective_jac(sol):
                        delay = sol[0]
                        XYZT_A = sol[1:5]
                        XYZT_B = sol[5:9]
                        PSE_A.try_location_JAC(delay, XYZT_A, workspace_jac[:N_ant_A,0], workspace_jac[:N_ant_A,1:5])
                        PSE_B.try_location_JAC(delay, XYZT_B, workspace_jac[N_ant_A:,0], workspace_jac[N_ant_A:,5:9])
                        return workspace_jac
                
                    fit_res = least_squares(objective_fun, initial_guess, jac=objective_jac, method='lm', xtol=1.0E-15, ftol=1.0E-15, gtol=1.0E-15, x_scale='jac')
    
                    new_RMS_A = PSE_A.RMS_with(fit_res.x[0], fit_res.x[1:5])
                    new_RMS_B = PSE_B.RMS_with(fit_res.x[0], fit_res.x[5:9])
                    if new_RMS_A<max_RMS and new_RMS_B<max_RMS and np.abs(fit_res.x[0])<max_delay:
                        print("   initial Delay A:", initial_Delay_A)
                        print("   initial Delay B:", initial_Delay_B)
                        print("   pdiff:", pdiff) 
                        print("   fitA:", new_RMS_A)
                        print("   fitB:", new_RMS_B)
                        print("   delay:", fit_res.x[0])
                        print("   fit msg:", fit_res.message)
                        print()
                        print()
                        found_sol = True
                        previous_found = True
                        log.flush()
                    else:
                        print("  not found")
                        print("    fitA:", new_RMS_A)
                        print("    fitB:", new_RMS_B)
                        print("    delay:", fit_res.x[0])
                        print("    pdiff", pdiff)
                        if found_sol:
                            break
                if previous_found and not found_sol:
                    break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    