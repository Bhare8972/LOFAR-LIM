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
    timeID = "D20160712T173455.100Z"
    output_folder = "MCerror_StochasticFitter_65PSE_2ns"
    
    SSPW_folders = ['excluded_SSPW_delay_fixed', 'excluded_SSPW_delay_fixed_later', 'old_stats_SSPW_early', 'old_stats_SSPW_late']
    PSE_folder = "allPSE"
    
    num_itter = 1000
    timing_error = 2.0E-9
    
    referance_station = "CS002"
    stations_to_keep = ['CS001', 'CS002', 'CS013', 'CS006', 'CS031', 'CS028', 'CS011', 'CS030', 'CS026', 'CS302', 'CS032', 'CS021', 'CS004', 'RS106', 'RS305', 'RS306', 'RS407', 'RS508'] ## to keep from the PSE
    stations_to_correlate = ['RS205', 'RS208', 'RS307', 'RS406', 'RS503', 'RS509']  ## to correlate from the SSPW
    
    ## to find delays
    stations_to_find_offsets = stations_to_correlate + stations_to_keep

    correlation_table =  {
            63: [3482, 2703, 330, 1151, 1957, 1487], 
            52: [3341, 2584, 220, 1003, 1817, 1349], 
            69: [3529, 2747, 372, 1198, 2006, None], 
            28: [3223, 2469, 124, 863, 1685, 1211], 
            21: [3306, 2548, 194, 968, 1783, 1310], 
            22: [3322, 2565, 208, 983, 1799, 1326], 
            
            456: [5914, 4607, 730, 2032, 3479, None], 
            362: [5191, 4102, None, None, 2758, 2026], 
            497: [6306, 4890, 1034, 2415, 3868, None], 
            429: [5507, 4317, None, 1658, 3072, 2330], 
            501: [6162, 4782, 913, 2272, 3720, 2949], 
            509: [6297, 4885, 1025, 2404, 3856, None], 
            
            0: [3051, 2334, 0, 680, 1506, 1027], 
            1: [3067, 2343, 12, 698, 1526, 1048], 
            3: [3114, 2377, 44, 742, 1571, 1094], 
            4: [3086, 2359, 27, 718, 1549, 1065], 
            7: [3069, 2344, 13, 699, 1528, 1050], 
            8: [3141, 2404, 66, None, 1601, None], 
            
            9: [3133, 2398, 60, 770, 1592, 1119], 
            10: [3084, 2358, 26, 716, 1547, None], 
            11: [3148, 2412, 70, 786, 1608, 1131], 
            17: [3143, None, 68, 780, None, None], 
            18: [3210, 2459, 115, 848, None, 1198], 
            19: [3187, 2441, 101, 828, 1653, 1171], 
            
            20: [3303, 2546, 192, 966, 1780, 1307], 
            418: [5514, 4325, 421, 1664, 3078, None], 
            413: [5508, 4321, 417, 1660, 3073, None], 
            346: [5153, 4073, 151, 1303, 2710, None], 
            434: [5427, 4263, 359, 1579, 2991, None], 
            440: [5709, 4462, 567, 1833, 3266, None], 
            
            449: [None, 4556, 673, 1963, 3409, 2626], 
            344: [5301, 4173, 260, 1451, 2866, 2125], 
            495: [6095, 4743, 870, 2219, 3667, None], 
            379: [5602, None, 490, 1739, 3163, None], 
            334: [5201, 4109, 188, 1358, 2767, 2034], 
            339: [5206, 4111, 192, 1363, 2771, None], 
            
            338: [5249, 4135, 219, 1401, 2814, None], 
            292: [4996, 3977, 41, 1153, 2555, 1854], 
            298: [5047, 4003, 75, 1209, 2612, None], 
            291: [4988, 3971, 34, 1146, 2546, 1847], 
            296: [5070, 4019, 92, 1232, 2634, None], 
            458: [5814, 4539, 646, 1939, 3382, None], 
            
            455: [5956, 4639, 767, 2079, 3524, None], 
            341: [5322, 4189, 278, 1472, 2888, None], 
            420: [5424, 4258, 356, 1572, 2986, None], 
            468: [5949, 4628, 756, None, 3515, 2738], 
            463: [5992, None, 796, 2112, 3560, None], 
            82: [3830, 3016, 644, 1466, 2291, 1892], 
            
            24: [3315, 2558, 203, 977, 1792, 1321], 
            77: [3805, 2998, 625, 1444, 2268, 1860], 
            62: [3410, 2642, 273, 1079, 1887, 1418], 
            80: [3820, 3009, 639, 1459, 2283, 1882], 
            57: [3443, 2668, 301, 1116, 1919, 1448], 
            23: [3316, 2559, 204, 978, 1793, 1322], 
            
            79: [3754, 2957, 580, 1394, 2214, 1807], 
            60: [3330, 2573, 212, 991, 1805, 1335], 
            65: [3419, 2649, 279, 1086, 1894, 1426], 
            30: [None, None, 109, 840, 1665, 1189], 
            38: [3182, 2437, 97, 823, 1646, 1165], 
            56: [3433, 2660, 293, 1102, 1910, 1435], 
            
            64: [3486, 2706, 331, 1156, 1960, 1491], 
            58: [3408, 2640, 270, 1078, 1884, 1415], 
            39: [3191, None, None, 830, 1655, 1176], 
            49: [3413, None, 275, 1082, 1889, 1422], 
            45: [3259, 2510, 158, 918, 1737, 1256]}
#           RS205 RS208 RS307 RS406 RS503 RS509

    correlation_table_SSPW_group = { ### since we are loading multiple SSPW files, the SSPW unique IDs are not actually unique...
            63: [0, 0, 0, 0, 0, 2], 
            52: [0, 0, 0, 0, 0, 2], 
            69: [0, 0, 0, 0, 0, 2], 
            28: [0, 0, 0, 0, 0, 2], 
            21: [0, 0, 0, 0, 0, 2], 22: [0, 0, 0, 0, 0, 2], 456: [1, 1, 1, 1, 1, 3], 362: [1, 1, 1, 1, 1, 3], 497: [1, 1, 1, 1, 1, 3], 429: [1, 1, 1, 1, 1, 3], 501: [1, 1, 1, 1, 1, 3], 509: [1, 1, 1, 1, 1, 3], 0: [0, 0, 0, 0, 0, 2], 1: [0, 0, 0, 0, 0, 2], 3: [0, 0, 0, 0, 0, 2], 4: [0, 0, 0, 0, 0, 2], 7: [0, 0, 0, 0, 0, 2], 8: [0, 0, 0, 0, 0, 2], 9: [0, 0, 0, 0, 0, 2], 10: [0, 0, 0, 0, 0, 2], 11: [0, 0, 0, 0, 0, 2], 17: [0, 0, 0, 0, 0, 2], 18: [0, 0, 0, 0, 0, 2], 19: [0, 0, 0, 0, 0, 2], 20: [0, 0, 0, 0, 0, 2], 418: [1, 1, 1, 1, 1, 3], 413: [1, 1, 1, 1, 1, 3], 346: [1, 1, 1, 1, 1, 3], 434: [1, 1, 1, 1, 1, 3], 440: [1, 1, 1, 1, 1, 3], 449: [1, 1, 1, 1, 1, 3], 344: [1, 1, 1, 1, 1, 3], 495: [1, 1, 1, 1, 1, 3], 379: [1, 1, 1, 1, 1, 3], 334: [1, 1, 1, 1, 1, 3], 339: [1, 1, 1, 1, 1, 3], 338: [1, 1, 1, 1, 1, 3], 292: [1, 1, 1, 1, 1, 3], 298: [1, 1, 1, 1, 1, 3], 291: [1, 1, 1, 1, 1, 3], 296: [1, 1, 1, 1, 1, 3], 458: [1, 1, 1, 1, 1, 3], 455: [1, 1, 1, 1, 1, 3], 341: [1, 1, 1, 1, 1, 3], 420: [1, 1, 1, 1, 1, 3], 468: [1, 1, 1, 1, 1, 3], 463: [1, 1, 1, 1, 1, 3], 82: [0, 0, 0, 0, 0, 2], 24: [0, 0, 0, 0, 0, 2], 77: [0, 0, 0, 0, 0, 2], 62: [0, 0, 0, 0, 0, 2], 80: [0, 0, 0, 0, 0, 2], 57: [0, 0, 0, 0, 0, 2], 23: [0, 0, 0, 0, 0, 2], 79: [0, 0, 0, 0, 0, 2], 60: [0, 0, 0, 0, 0, 2], 65: [0, 0, 0, 0, 0, 2], 30: [0, 0, 0, 0, 0, 2], 38: [0, 0, 0, 0, 0, 2], 56: [0, 0, 0, 0, 0, 2], 64: [0, 0, 0, 0, 0, 2], 58: [0, 0, 0, 0, 0, 2], 39: [0, 0, 0, 0, 0, 2], 49: [0, 0, 0, 0, 0, 2], 45: [0, 0, 0, 0, 0, 2]}
    
    
    
    initial_guess = np.array([  
   8.95586087e-08,   1.28237516e-07,   2.68804730e-08,   1.21275736e-07,
   4.30019247e-08,  -9.17438250e-08,   1.45847162e-09,   3.66247929e-09,
  -1.14850014e-09,   1.00924553e-08,  -1.22278859e-10,  -3.44331172e-09,
   9.13698791e-09,  -1.01954214e-08,   1.11357876e-08,   7.47272969e-09,
   6.58799125e-09,   6.44090774e-10,  -1.65764371e-07,   5.56037437e-08,
   7.18077778e-08,  -1.15865721e-07,  -9.37889376e-07,   3.21295163e+04,
   2.82279468e+04,   3.46872191e+03,   3.82989340e+00,   3.22332014e+04,
   2.84367908e+04,   4.07810030e+03,   3.82738436e+00,   3.21956491e+04,
   2.83121260e+04,   3.23438325e+03,   3.83103439e+00,   3.13336000e+04,
   2.85809365e+04,   4.56617675e+03,   3.82450756e+00,   3.20490980e+04,
   2.83526201e+04,   4.23222690e+03,   3.82664692e+00,   3.20467523e+04,
   2.83442559e+04,   4.19599079e+03,   3.82697834e+00,   2.91906853e+04,
   2.60772841e+04,   2.62107227e+03,   3.87087677e+00,   2.98241706e+04,
   2.54413422e+04,   2.76151181e+03,   3.86152731e+00,   2.92079429e+04,
   2.64968099e+04,   2.78197974e+03,   3.87547632e+00,   3.16599943e+04,
   2.91705679e+04,   2.66289662e+03,   3.86592608e+00,   3.12200166e+04,
   2.80713416e+04,   6.32304355e+03,   3.87383893e+00,   2.94468790e+04,
   2.65314420e+04,   2.58984283e+03,   3.87533879e+00,   3.14650080e+04,
   2.80313540e+04,   5.32796336e+03,   3.81917148e+00,   3.13998026e+04,
   2.81853848e+04,   5.11558488e+03,   3.82029004e+00,   3.14932448e+04,
   2.83537896e+04,   4.76633170e+03,   3.82156557e+00,   3.12943436e+04,
   2.82896033e+04,   5.05435576e+03,   3.82091056e+00,   3.14490824e+04,
   2.80727257e+04,   5.14447538e+03,   3.82029922e+00,   3.14367637e+04,
   2.85284613e+04,   4.76041214e+03,   3.82245242e+00,   3.15033092e+04,
   2.83863716e+04,   4.71055589e+03,   3.82223943e+00,   3.13280016e+04,
   2.82648542e+04,   5.04239787e+03,   3.82084694e+00,   3.14251579e+04,
   2.85560859e+04,   4.75058349e+03,   3.82262140e+00,   3.17298470e+04,
   2.83758074e+04,   4.71990389e+03,   3.82247876e+00,   3.18974693e+04,
   2.83800043e+04,   4.67835150e+03,   3.82424272e+00,   3.15261175e+04,
   2.82547730e+04,   4.59922034e+03,   3.82367261e+00,   3.21954939e+04,
   2.84802122e+04,   4.24595894e+03,   3.82657674e+00,   3.12761736e+04,
   2.78941633e+04,   6.06377294e+03,   3.86602562e+00,   2.95178415e+04,
   2.56340524e+04,   2.79861165e+03,   3.86596354e+00,   2.98395356e+04,
   2.53665185e+04,   2.66935809e+03,   3.86080982e+00,   2.95904827e+04,
   2.54937372e+04,   2.88103507e+03,   3.86489837e+00,   2.94989749e+04,
   2.58985910e+04,   2.72739056e+03,   3.86834432e+00,   3.12416249e+04,
   2.78629805e+04,   6.06248166e+03,   3.87006261e+00,   2.95892643e+04,
   2.53454754e+04,   2.52670411e+03,   3.86324434e+00,   3.29673982e+04,
   2.66064039e+04,   2.63025115e+03,   3.87307161e+00,   2.94691481e+04,
   2.57677323e+04,   2.79135628e+03,   3.86715456e+00,   3.13538007e+04,
   2.80665693e+04,   5.99216270e+03,   3.86169204e+00,   2.97472616e+04,
   2.53936333e+04,   2.64177147e+03,   3.86178283e+00,   2.96954805e+04,
   2.53722476e+04,   2.55654663e+03,   3.86248493e+00,   3.22220891e+04,
   2.85642091e+04,   4.23912090e+03,   3.85765708e+00,   3.00477463e+04,
   2.53220254e+04,   2.68182428e+03,   3.85885524e+00,   3.13588662e+04,
   2.79999757e+04,   6.05069319e+03,   3.85750060e+00,   3.13041820e+04,
   2.79206474e+04,   6.04084941e+03,   3.85939808e+00,   2.92029845e+04,
   2.59270270e+04,   2.72053170e+03,   3.86971687e+00,   2.95293416e+04,
   2.48930083e+04,   3.35405704e+03,   3.87140690e+00,   2.96531723e+04,
   2.54084873e+04,   2.77844839e+03,   3.86353192e+00,   2.95848660e+04,
   2.54889654e+04,   2.87868857e+03,   3.86482656e+00,   2.95306904e+04,
   2.60887331e+04,   2.68209659e+03,   3.87128741e+00,   2.90764506e+04,
   2.62175936e+04,   2.70170112e+03,   3.87182169e+00,   3.26126193e+04,
   2.69385301e+04,   2.91539372e+03,   3.83710396e+00,   3.22029993e+04,
   2.85403257e+04,   4.24249951e+03,   3.82685769e+00,   3.25884481e+04,
   2.70755117e+04,   2.97426139e+03,   3.83673799e+00,   3.23429567e+04,
   2.82696437e+04,   3.98761843e+03,   3.82873670e+00,   3.26192411e+04,
   2.69627846e+04,   2.90777706e+03,   3.83696354e+00,   3.17105695e+04,
   2.78781924e+04,   4.79540428e+03,   3.82924040e+00,   3.19044159e+04,
   2.81397302e+04,   4.38674785e+03,   3.82688376e+00,   3.06801542e+04,
   2.76127086e+04,   3.00255771e+03,   3.83600944e+00,   3.20488822e+04,
   2.81051359e+04,   4.47767737e+03,   3.82710521e+00,   3.24877628e+04,
   2.85774712e+04,   3.99331671e+03,   3.82887827e+00,   3.13539248e+04,
   2.85675002e+04,   4.58803989e+03,   3.82400102e+00,   3.15254596e+04,
   2.82736831e+04,   4.62750830e+03,   3.82350774e+00,   3.21606230e+04,
   2.83035960e+04,   3.80972568e+03,   3.82908254e+00,   3.18378171e+04,
   2.83910591e+04,   4.65630566e+03,   3.82997898e+00,   3.23126632e+04,
   2.84280713e+04,   3.91134638e+03,   3.82868102e+00,   3.18640104e+04,
   2.83971366e+04,   4.65427406e+03,   3.82373717e+00,   3.24691049e+04,
   2.85731863e+04,   3.96539622e+03,   3.82880689e+00,   3.16420992e+04,
   2.80233674e+04,   4.56524629e+03,   3.82558914e+00])
    

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
        
        new_PSE_to_fit = fitting_PSE( found_PSE )
        
        PSE_to_correlate.append(new_PSE_to_fit)
            
        ## correlate SSPW
        group_indeces = correlation_table_SSPW_group[PSE_index]
        
        for sname, SSPW_index, SSPW_group_index in zip(stations_to_correlate, SSPW_indeces, group_indeces):
            SSPW_dict = SSPW_multiple_dict[SSPW_group_index]
            
            if SSPW_index is None:
                continue
            for SSPW in SSPW_dict[sname]:
                if SSPW.unique_index==SSPW_index:
                    new_PSE_to_fit.add_SSPW(SSPW)
                    break
                    
        ## prep the PSE
        new_PSE_to_fit.fitting_prep(stations_to_find_offsets, stations_to_keep, PSE_ant_locs, SSPW_locs_dict)
        print()
        
            
            
    #### prep for fitting! ####
    N_delays = len(stations_to_find_offsets)

    i = N_delays
    N_ant = 0
    for PSE in PSE_to_correlate:
        N_ant += len(PSE.antenna_X)
        
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
        
    
    
    
    
    