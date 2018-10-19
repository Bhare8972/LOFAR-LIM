#!/usr/bin/env python3

##ON APP MACHINE

##import python packages
import glob
import struct

##import external packages
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import blended_transform_factory

##import own tools
from LoLIM.IO.binary_IO import read_string, read_double, read_long, read_double_array, write_long, write_double, write_string
import LoLIM.IO.metadata as md
from LoLIM.porta_code import code_logger

from LoLIM.utilities import Fname_data, processed_data_dir

class station_metadata(object):
    """object used to store info about a single station"""
    
    class antenna_metadata(object):
        """ object used to store info about a pair of antennas"""
        def __init__(self, fin, locations_are_local=True):
            
            self.even_antenna_name = read_string(fin)
            self.odd_antenna_name = read_string(fin)
            
            self.location = read_double_array(fin)
            
            if not locations_are_local: ##remove this soon
                self.location = md.convertITRFToLocal( np.array([ self.location ]) )[0]
            
            self.even_index = int(self.even_antenna_name[-3:])
            
            if self.even_index==1:
                print(self.even_antenna_name, self.odd_antenna_name)
            
            self.status = 0 ##0 means both operation. 1 means even is bad, 2 means odd is bad. 3 means both are bad
            #### antenna status is only set by external mod files ####
            
            ## calibration delays are set by callibration files ##
            self.even_callibration_delay = 0.0
            self.odd_callibration_delay = 0.0
            
            ## additional errors set by mod files
            self.even_additional_timing_error = 0 
            self.odd_additional_timing_error = 0
           
            ## ...also set by mod files ...
            self.flip_polarities = False
            
        def set_bad_pol(self, bad_polarization):
            """ set a polarization to be bad. 0 if even is bad, 1 if odd is bad"""
            
            if self.status == 0:
                if bad_polarization == 0:
                    self.status = 1
                else:
                    self.status = 2
            elif self.status == 1:
                if bad_polarization == 1:
                    self.status = 3
            elif self.status == 2:
                if bad_polarization == 0:
                    self.status = 3
                    
        def set_section_averages(self, section_ave_data):
            """section averages are the averge of a hilbert envelope for a section. Helps determine bad data """
            self.section_ave_data = section_ave_data
            
        def section_status(self, section):
            """get the status of this antenna during a section of the data. Based on section averages"""
            
            even_ave, odd_ave = self.section_ave_data[section]
            
            status = self.status
            
            even_good = even_ave > 1.0
            if status == 0:
                if not even_good:
                    status = 1
            elif status == 2:
                if not even_good:
                    status = 3
            
            odd_good = odd_ave > 1.0
            if status == 0:
                if not odd_good:
                    status = 2
            elif status == 1:
                if not odd_good:
                    status = 3
                    
            return status
            
    
    def __init__(self, header_fpath, locations_are_local=True):
        self.header_paths = []
        self.num_antenna_sets = 0
        self.AntennaInfo_dict = {} ##a dictionary of antenna set info
        self._station_location = None ##average of all antenna_locations
        self.sorted_antenna_names = []
        
        self.locations_are_local = locations_are_local
        
        self.add_header( header_fpath )
        
        self.known_station_delay = -self.clock_offset - self.SAMPLE_NUMBER*5.0E-9
        self.additional_station_delay = 0.0 ## additional delay set by mod file
        
    def add_header(self, header_fpath):
        self.header_paths.append( header_fpath )
        
        with open(header_fpath, 'rb') as fin:
            ##read station data##
            ## assume this is the same for all headers
            self.station_name = read_string(fin)
            self.start_time = read_long(fin)
            self.observation_ID = read_string(fin)
            self.SAMPLE_NUMBER = read_long(fin) ##divide by 200.0e6 to get subsecond from UTC
            self.clock_offset = read_double(fin)  ##delay, in seconds, from central clock to the station. Also need differances in sample_number to calculate delays
            self.seconds_per_sample = read_double(fin)
            self.num_extra_points = read_long(fin)
            
            ##read antenna data##
            num_new_antennas = read_long(fin)
            self.num_antenna_sets += num_new_antennas
            n=0
            for x in range(num_new_antennas):
                new_antenna_info = station_metadata.antenna_metadata(fin, self.locations_are_local)
                n+=1
                if new_antenna_info.even_antenna_name in self.AntennaInfo_dict:
                    print("ARG ERROR!")

                self.AntennaInfo_dict[new_antenna_info.even_antenna_name] = new_antenna_info
                
            self.sorted_antenna_names = list(self.AntennaInfo_dict.keys()) ### this will get remade every time a header is added
            self.sorted_antenna_names.sort(key=lambda ant: self.AntennaInfo_dict[ant].even_index)
                
            if len(self.sorted_antenna_names)!=self.num_antenna_sets:
                print("error in station:", self.station_name)
                print("end:", self.num_antenna_sets, len(self.sorted_antenna_names), n, self.sorted_antenna_names)
                quit()
        
        
        ## open section average files
        SAF_fpath = header_fpath[:-3] + "saf"
        SAF_data = {}
        with open(SAF_fpath, 'rb') as SAF_fin:
            while True:
                try:
                    ant = read_string(SAF_fin)
                    block = read_long(SAF_fin)
                    even_ant_ave = read_double(SAF_fin)
                    odd_ant_ave = read_double(SAF_fin)
                    
                except struct.error:
                    break
                else:
                    if ant not in SAF_data:
                        SAF_data[ant] = {}
                    SAF_data[ant][block] = [even_ant_ave, odd_ant_ave]
            
        for ant_name, ant_SAF_data in SAF_data.items():
            self.AntennaInfo_dict[ant_name].set_section_averages(ant_SAF_data)
              
    def get_station_location(self):
        
        if self._station_location is None:
            self._station_location = np.zeros(3, dtype=np.double)
            N=0.0
            for ant in self.AntennaInfo_dict.values():
                self._station_location+=ant.location
                N+=1.0
            self._station_location *= 1.0/N
            
        return self._station_location

    def set_antenna_calibration_delays(self, txt_cal_table_folder):
        calibration_delays = {}
        with open(txt_cal_table_folder+'/'+self.station_name+'.txt', 'r') as fin:
            for line in fin:
                ant_index, delay = line.split()
                calibration_delays[int(ant_index)] = float(delay)
                
        for ant_data in self.AntennaInfo_dict.values():
            ant_data.even_callibration_delay = calibration_delays[ant_data.even_index]
            ant_data.odd_callibration_delay = calibration_delays[ant_data.even_index+1]
        
    #### set mod data
    def set_antenna_timing_delay(self, antenna_delay_dict):
        """account for systematic errors in antenna timing. keys are antenna names, values are delays. each delay should be a list with first value is delay of even antenna and second value is delay of odd antenna"""
        
        for ant_name, ant_data in self.AntennaInfo_dict.items():
            if ant_name in antenna_delay_dict:
                ant_data.even_additional_timing_error = antenna_delay_dict[ant_name][0]
                ant_data.odd_additional_timing_error = antenna_delay_dict[ant_name][1]
                
    def adjust_station_delay(self, station_delay):
        """add an additional delay to all data in this station"""
        self.additional_station_delay += station_delay
        
    def set_inverted_antennas(self, inverted_antenna_names):
        """invert the polarities of certain antennas"""
        
        for ant_name, ant_data in self.AntennaInfo_dict.items():
            if ant_name in inverted_antenna_names:
                ant_data.flip_polarities = True
                
    def set_bad_antennas(self, antennas):
        for ant_name, pol in antennas:
            if ant_name in self.AntennaInfo_dict:
                self.AntennaInfo_dict[ant_name].set_bad_pol(pol)
                
    ##### retrieve mod data ####
    def get_antenna_delays(self):
        antenna_delay_dict = {ant_name:(ant_data.even_additional_timing_error, ant_data.odd_additional_timing_error) for ant_name, ant_data in self.AntennaInfo_dict.items()}
        return antenna_delay_dict
    
    def get_bad_antennas(self):
        bad_antennas = []
        for ant_name, ant_data in self.AntennaInfo_dict.items():
            if ant_data.status == 1:
                bad_antennas.append([ant_name, 0])
            elif ant_data.status == 2:
                bad_antennas.append([ant_name, 1])
            elif ant_data.status == 3:
                bad_antennas.append([ant_name, 0])
                bad_antennas.append([ant_name, 1])
        return bad_antennas
        
    def get_flipped_polarities(self):
        flipped_pols = []
        for ant_name, ant_data in self.AntennaInfo_dict.items():
            if ant_data.flip_polarities:
                flipped_pols.append(ant_name)
        return flipped_pols
        
    
    #### read pulse data
    def read_pulse_data(self, approx_min_block=-np.inf, approx_max_block=np.inf):
        """opens pulse data associated with this station between the two blocks. Returns dictionary where keys are antenna names and values are lists of pulse_data. approx_max_block is not included in opened data"""
    
        AntennaPulse_dict = {}
        for ant_name, ant_info in self.AntennaInfo_dict.items():
            if ant_info.status != 3:
                AntennaPulse_dict[ant_name] = []
            
        for header_fpath in self.header_paths:
            base_fname = header_fpath[:-4]
                
            pulse_fnames = glob.glob(base_fname + '*.pls')
            pulse_fnames = [fname for fname in pulse_fnames if approx_min_block <= int(fname.split('_')[-1][:-4]) < approx_max_block  ]
            
            for fname in pulse_fnames:
                with open(fname, 'rb') as fin:
                    while True:
                        cmd = read_long(fin) ##check if we are at end of file
                        if cmd != 1:
                            break
                        
                        new_pulse = pulse_data(fin, self)
                        
                        if new_pulse.antenna_status != 3:
                            even_antenna = new_pulse.even_antenna_name
                            AntennaPulse_dict[even_antenna].append(new_pulse)
                    
            
        for ant_name, pulse_list in AntennaPulse_dict.items():
            pulse_list.sort(key=lambda x: x.peak_time())
            
        return AntennaPulse_dict
    
    
    def read_pulse_data_inBlockList(self, blockList):
        
        
        AntennaPulse_dict = {}
        for ant_name, ant_info in self.AntennaInfo_dict.items():
            if ant_info.status != 3:
                AntennaPulse_dict[ant_name] = []
            
            
        for header_fpath in self.header_paths:
            base_fname = header_fpath[:-4]
                
            pulse_fnames = glob.glob(base_fname + '*.pls')
            
            pulse_fnames.sort(key=lambda fname: int(fname.split('_')[-1][:-4])) ##sort fnames by block
            
            for f_i in range(len(pulse_fnames)):
                starting_block = int( pulse_fnames[f_i].split('_')[-1][:-4] )
                
                if f_i == len(pulse_fnames)-1:
                    ending_block = np.inf
                else:
                    ending_block = int( pulse_fnames[f_i+1].split('_')[-1][:-4] )
                    
                    
                file_needed = False
                for block_needed in blockList:
                    if starting_block <= block_needed < ending_block:
                        file_needed = True
                        break
                    
                    
                if file_needed:
                    with open(pulse_fnames[f_i], 'rb') as fin:
                        while True:
                            cmd = read_long(fin) ##check if we are at end of file
                            if cmd != 1:
                                break
                            
                            new_pulse = pulse_data(fin, self)
                            
                            even_antenna = new_pulse.even_antenna_name
                            AntennaPulse_dict[even_antenna].append(new_pulse)
                    
            
        for ant_name, pulse_list in AntennaPulse_dict.items():
            pulse_list.sort(key=lambda x: x.peak_time())
            
        return AntennaPulse_dict
                    
#### global data for finding peak times ####

##TODO: replace with code from signal processing
n_points = 5
tmp_matrix = np.zeros((n_points,3), dtype=np.double)
for n_i in range(n_points):
    tmp_matrix[n_i, 0] = n_i**2
    tmp_matrix[n_i, 1] = n_i
    tmp_matrix[n_i, 2] = 1.0
peak_time_matrix = np.linalg.pinv( tmp_matrix ) #### this matrix, when multilied by 5 points, will give a parabolic fit(A,B,C): A*x*x + B*x + C
                
                
hessian = np.zeros((3,3))
for n_i in range(n_points):
    hessian[0,0] += n_i**4
    hessian[0,1] += n_i**3
    hessian[0,2] += n_i**2
    
    hessian[1,1] += n_i**2
    hessian[1,2] += n_i
    
    hessian[2,2] +=1
            
hessian[1,0] = hessian[0,1]
hessian[2,0] = hessian[0,2]
hessian[2,1] = hessian[1,2]
    
inverse = np.linalg.inv(hessian)
#### these error ratios, for each value in the parabolic fit, give teh std of the error in that paramter when mupltipled by the std of the noise of the data ####
error_ratio_A = np.sqrt(inverse[0,0])
error_ratio_B = np.sqrt(inverse[1,1])
error_ratio_C = np.sqrt(inverse[2,2])

class pulse_data(object):
    """ object used to store data about single pulse"""
    def __init__(self, fin, station_info):
        
        self.station_info = station_info
        
        self.section_number = read_long(fin)
        self.unique_index = read_long(fin)
        
        self.even_antenna_name = read_string(fin)
        self.starting_index = read_long(fin)
        
        self.antenna_info = self.station_info.AntennaInfo_dict[self.even_antenna_name]
        
        self.even_envelope_average = read_double(fin)
        self.odd_envelope_average = read_double(fin)
        
        self.even_data_STD = read_double(fin)
        self.odd_data_STD = read_double(fin)
        self.even_data_STD = 1.0 ##BAD HACK
        self.odd_data_STD = 1.0
        
        self.even_antenna_data = read_double_array(fin)
        self.even_antenna_hilbert_envelope = read_double_array(fin)
        
        self.odd_antenna_data =  read_double_array(fin)
        self.odd_antenna_hilbert_envelope =  read_double_array(fin)
        
        if self.antenna_info.flip_polarities:
            even_ave_store = self.even_envelope_average
            even_STD_store = self.even_data_STD
            even_data_store = self.even_antenna_data
            even_HE_store = self.even_antenna_hilbert_envelope
            
            self.even_envelope_average = self.odd_envelope_average
            self.even_data_STD = self.odd_data_STD
            self.even_antenna_data = self.odd_antenna_data
            self.even_antenna_hilbert_envelope = self.odd_antenna_hilbert_envelope
            
            self.odd_envelope_average = even_ave_store
            self.odd_data_STD = even_STD_store
            self.odd_antenna_data = even_data_store
            self.odd_antenna_hilbert_envelope = even_HE_store
                
        self.event_data = None
        self._max_SNR_ = None
        

        #### check antenna status ####
        self.antenna_status = self.antenna_info.status  ##0 means both operation. 1 means even is bad, 2 means odd is bad. 3 means both are bad
        
        if len(self.even_antenna_hilbert_envelope)==0 or len(self.odd_antenna_hilbert_envelope)==0:
            self.antenna_status = 3
            return
            
        even_status = (np.max(self.even_antenna_hilbert_envelope)  > self.even_envelope_average) and self.even_envelope_average>0.00000000001
        odd_status =  (np.max(self.odd_antenna_hilbert_envelope)   > self.odd_envelope_average) and self.odd_envelope_average>0.00000000001
        if self.antenna_status == 0: ##both antennas should be opperational
            if (not even_status) and (not odd_status):
                self.antenna_status = 3
            elif not even_status:
                self.antenna_status = 1
            elif not odd_status:
                self.antenna_status = 2
        elif self.antenna_status == 1:
            if not odd_status:
                self.antenna_status = 3
        elif self.antenna_status == 2:
            if not even_status:
                self.antenna_status = 3
                
                
        #### peak location information ####
        self.PolO_time_offset = - self.antenna_info.odd_callibration_delay - self.antenna_info.odd_additional_timing_error - self.station_info.known_station_delay - self.station_info.additional_station_delay
        self.PolE_time_offset = - self.antenna_info.even_callibration_delay - self.antenna_info.even_additional_timing_error - self.station_info.known_station_delay - self.station_info.additional_station_delay
        
        
        peak_index = np.argmax(self.even_antenna_hilbert_envelope) #### TODO: fix the tst that needs this
        if (self.antenna_status == 0 or self.antenna_status == 2) and np.max(self.even_antenna_hilbert_envelope)>self.even_envelope_average and len(self.even_antenna_hilbert_envelope[peak_index-2:peak_index+3])==5: ##even antenna is good and has a peak
#            peak_index = np.argmax(self.even_antenna_hilbert_envelope)
#            print( len(self.even_antenna_hilbert_envelope[peak_index-2:peak_index+3]), len(self.even_antenna_hilbert_envelope) )
#            if len(self.even_antenna_hilbert_envelope[peak_index-2:peak_index+3]) < 5:
#            plt.plot(self.even_antenna_data)
#            plt.plot(self.odd_antenna_data)
#            plt.plot(np.arange(len(self.even_antenna_data)), [self.even_envelope_average]*len(self.even_antenna_data), 'g')
#            plt.plot(np.arange(len(self.even_antenna_data)), [self.odd_envelope_average]*len(self.even_antenna_data), 'r')
#            plt.plot(self.even_antenna_hilbert_envelope, 'g')
#            plt.plot(self.odd_antenna_hilbert_envelope, 'r')
#            plt.show()
            
            
            self.PolE_parabola_A, self.PolE_parabola_B, self.PolE_parabola_C = np.dot( peak_time_matrix, self.even_antenna_hilbert_envelope[peak_index-2:peak_index+3] )
            
            self.PolE_parabola_start_index = peak_index-2 ##this is the offset (in units of samples) of when the parabola X==0
            self.PolE_peak_index = -self.PolE_parabola_B/(2.0*self.PolE_parabola_A) - 2 + peak_index
            
            self.PolE_peak_time = (self.PolE_peak_index+self.starting_index)*self.station_info.seconds_per_sample  + self.PolE_time_offset
        
            frac_error_A = error_ratio_A/self.PolE_parabola_A
            frac_error_B = error_ratio_B/self.PolE_parabola_B
    
    
#            print(np.sqrt( frac_error_A*frac_error_A  +  frac_error_B*frac_error_B ))
#            print(( -self.PolE_parabola_B/(2.0*self.PolE_parabola_A) ))
#            print((0.655*self.even_data_STD))
            self.PolE_estimated_timing_error = np.sqrt( frac_error_A*frac_error_A  +  frac_error_B*frac_error_B ) * ( -self.PolE_parabola_B/(2.0*self.PolE_parabola_A) ) * (0.655*self.even_data_STD)
        else:
            self.PolE_peak_time = np.inf
            self.PolE_estimated_timing_error = 1
            
            
        peak_index = np.argmax(self.odd_antenna_hilbert_envelope) #### TODO: fix the tst that needs this
        if (self.antenna_status == 0 or self.antenna_status == 1) and np.max(self.odd_antenna_hilbert_envelope)>self.odd_envelope_average and len(self.odd_antenna_hilbert_envelope[peak_index-2:peak_index+3])==5: ##odd antenna is good and has a peak
#            peak_index = np.argmax(self.odd_antenna_hilbert_envelope)
            self.PolO_parabola_A, self.PolO_parabola_B, self.PolO_parabola_C = np.dot( peak_time_matrix, self.odd_antenna_hilbert_envelope[peak_index-2:peak_index+3] )
            
            self.PolO_parabola_start_index = peak_index-2 ##this is the offset (in units of samples) of when the parabola X==0
            self.PolO_peak_index = -self.PolO_parabola_B/(2.0*self.PolO_parabola_A) - 2 + peak_index
            
            self.PolO_peak_time = (self.PolO_peak_index+self.starting_index)*self.station_info.seconds_per_sample  + self.PolO_time_offset
        
            frac_error_A = error_ratio_A/self.PolO_parabola_A
            frac_error_B = error_ratio_B/self.PolO_parabola_B
    
            self.PolO_estimated_timing_error = np.sqrt( frac_error_A*frac_error_A  +  frac_error_B*frac_error_B ) * ( -self.PolO_parabola_B/(2.0*self.PolO_parabola_A) ) * (0.655*self.odd_data_STD)
        else:
            self.PolO_peak_time = np.inf
            self.PolO_estimated_timing_error = 1

    def best_SNR(self):
        """gives the best SNR of even or odd polarization."""
        
        if self._max_SNR_ is None:
        
            if self.antenna_status == 2:
                self._max_SNR_ = np.max(self.even_antenna_hilbert_envelope)/self.even_data_STD
                
            elif self.antenna_status == 1:
                self._max_SNR_ = np.max(self.odd_antenna_hilbert_envelope)/self.odd_data_STD
                
            elif self.antenna_status == 0:
                even_SNR = np.max(self.even_antenna_hilbert_envelope)/self.even_data_STD
                odd_SNR = np.max(self.odd_antenna_hilbert_envelope)/self.odd_data_STD
                self._max_SNR_ = max(even_SNR, odd_SNR)
                
            else:
                self._max_SNR_ = 0
            
        return self._max_SNR_
        
    def above_thresh(self, num_std):
        """returns true if peak hilbert envelope is above a number of standerd deviations, False otherwise"""
        SNR = self.best_SNR()
        if SNR>num_std:
            return True
        return False
    
    def meets_pol_requirement(self, pol_requirement):
        if pol_requirement == 0:
            return self.antenna_status != 3
        elif pol_requirement == 1:
            return self.antenna_status==0 or self.antenna_status==2
        elif pol_requirement == 2:
            return self.antenna_status==0 or self.antenna_status==1
        elif pol_requirement == 3:
            return self.antenna_status==0
                
    
    
    #### depreciated functions, still needed for finding SSPW ####
    
    def best_polarization(self):
        """ return 0 is even polarization is best, 1 if odd polarization is best."""
        if self.antenna_status==1:
            return 1
        elif self.antenna_status==2:
            return 0
        elif self.antenna_status==3:
            return None
        
        even_peak = np.max( self.even_antenna_hilbert_envelope )
        odd_peak = np.max( self.odd_antenna_hilbert_envelope )
        
        if even_peak>=odd_peak:
            return 0
        else:
            return 1
        
    def peak_time(self, polarization=None):
        """returns the peak time for a polarization. Default is best polarization. 0 for even, 1 for odd. Returns None if not avaliable"""
        
        if polarization is None:
            polarization = self.best_polarization()
            
        if polarization is None:
            return None
        
        if polarization==0:
            return self.PolE_peak_time
        else:
            return self.PolO_peak_time
    
    
    
#### functions for reading, writing, and getting mod data ####
def read_antenna_pol_flips(fname):
    antennas_to_flip = []
    with open(fname) as fin:
        for line in fin:
            ant_name = line.split()[0]
            antennas_to_flip.append( ant_name )
    return antennas_to_flip

def read_bad_antennas(fname):
    bad_antenna_data = []
    with open(fname) as fin:
        for line in fin:
            ant_name, pol = line.split()[0:2]
            bad_antenna_data.append((ant_name,int(pol)))
    return bad_antenna_data

def read_antenna_delays(fname):
    additional_ant_delays = {}
    with open(fname) as fin:
        for line in fin:
            ant_name, pol_E_delay, pol_O_delay = line.split()[0:3]
            additional_ant_delays[ant_name] = [float(pol_E_delay), float(pol_O_delay)]
    return additional_ant_delays

def read_station_delays(fname):
    station_delays = {}
    with open(fname) as fin:
        for line in fin:
            sname, delay = line.split()[0:2]
            station_delays[sname] = float(delay)
    return station_delays



def writeTXT_antenna_pol_flips(fname, pol_flips):
    with open(fname, 'w') as fout:
        for ant_name in pol_flips:
            fout.write(ant_name)
            fout.write('\n')
            
def writeTXT_bad_antennas(fname, bad_antennas):
    with open(fname, 'w') as fout:
        for ant_name, pol in bad_antennas:
            fout.write( ant_name )
            fout.write(" ")
            fout.write(str(pol))
            fout.write("\n")
            
def writeTXT_ant_delays(fname, antenna_delay_dict):
    with open(fname, 'w') as fout:
        for ant_name, (PolE_delay, PolO_delay) in antenna_delay_dict.items():
            fout.write(ant_name)
            fout.write(" ")
            fout.write( str(PolE_delay) )
            fout.write(" ")
            fout.write( str(PolO_delay) )
            fout.write("\n")
            
def writeTXT_station_delays(fname, station_delay_dict):
    with open(fname, 'w') as fout:
        for sname, delay in station_delay_dict.items():
            fout.write(sname)
            fout.write(" ")
            fout.write( str(delay) )
            fout.write("\n")
            
def getNwriteBin_modData(fout, stationInfo_dict, stations_to_write=None):
    
    if stations_to_write is None:
        stations_to_write = list( stationInfo_dict.keys() )
    
    write_long(fout, len(stations_to_write))
    
    for sname in stations_to_write:
        sinfo = stationInfo_dict[sname]
        
        write_string(fout, sname)
        write_double(fout, sinfo.additional_station_delay)

        ant_delay_dict = sinfo.get_antenna_delays()
        write_long(fout, len(ant_delay_dict))
        for ant_name, (PolE_delay, PolO_delay) in ant_delay_dict.items():
            write_string(fout, ant_name)
            write_double(fout, PolE_delay)
            write_double(fout, PolO_delay)
            
            
        bad_ant = sinfo.get_bad_antennas()
        write_long(fout, len(bad_ant))
        for ant_name, pol in bad_ant:
            write_string(fout, ant_name)
            write_long(fout, pol)
            
        flipped_pols = sinfo.get_flipped_polarities()
        write_long(fout, len(flipped_pols))
        for ant_name in flipped_pols:
            write_string(fout, ant_name)
            
def readBin_modData(fin):
    
    station_delay_dict = {}
    ant_delay_dict = {}
    bad_ant_list = []
    flipped_pol_list = []
    
    num_stations = read_long(fin)
    for si in range(num_stations):
        sname = read_string(fin)
        delay = read_double(fin)
        station_delay_dict[sname] = delay
        
        
        num_ant_delays = read_long(fin)
        for adi in range(num_ant_delays):
            ant_name = read_string(fin)
            PolE_delay = read_double(fin)
            PolO_delay = read_double(fin)
            ant_delay_dict[ant_name] = [PolE_delay, PolO_delay]
            
            
        num_bad_ant = read_long(fin)
        for bai in range(num_bad_ant):
            ant_name = read_string(fin)
            pol = read_long(fin)
            bad_ant_list.append([ant_name, pol])
            
        
        num_flipped_pol = read_long(fin)
        for fpi in range(num_flipped_pol):
            ant_name = read_string(fin)
            flipped_pol_list.append(ant_name)
            
    return flipped_pol_list, bad_ant_list, ant_delay_dict, station_delay_dict
    
    
    
def read_station_info(timeID, input_folder=None, station_names=None, stations_to_exclude=[], data_directory=None, data_directory_append=None,  ant_delays_fname=None, bad_antennas_fname=None, 
                      pol_flips_fname=None, station_delays_fname=None, txt_cal_table_folder=None, locations_are_local=True):
    """opens all station info with a trigger (at a UTC time). Can filter by station names. 
    Returns a dictionary of station_metadata."""
    
    #### locations_are_local is due to a code error, where locations were saved in ITRF and not local. This is fixed. Future code should not need it.
    
    if input_folder is None:
        input_folder = '/pulse_data'
    
    #### get list of header files
    if data_directory == None:
        data_directory=processed_data_dir(timeID) + input_folder
        
    if data_directory_append != None:
        data_directory += data_directory_append
        
    header_fpaths = glob.glob(data_directory + '/*.hdr')
    
    StationInfo_dict = {}
    for header_fpath in header_fpaths:
        
        #### FIX THIS TO USE file_number
        UTC_time, station_name, Fpath, file_number = Fname_data(header_fpath)
        if (station_names is not None) and (station_name not in station_names):
            continue
        if station_name in stations_to_exclude:
            continue
        
        if station_name not in StationInfo_dict:
            station_info = station_metadata(header_fpath, locations_are_local=locations_are_local)
            StationInfo_dict[station_name] = station_info
        else:
            StationInfo_dict[station_name].add_header(header_fpath)
            
            
    ### adjust the data for various affects ###
    if ant_delays_fname is not None:
        ant_delays = read_antenna_delays( ant_delays_fname )
        for sdata in StationInfo_dict.values():
            sdata.set_antenna_timing_delay( ant_delays )
            
            
    if bad_antennas_fname is not None:
        bad_antennas = read_bad_antennas( bad_antennas_fname )
        for sdata in StationInfo_dict.values():
            sdata.set_bad_antennas( bad_antennas )
            
            
    if pol_flips_fname is not None:
        pol_flips = read_antenna_pol_flips( pol_flips_fname )
        for sdata in StationInfo_dict.values():
            sdata.set_inverted_antennas( pol_flips )
            
    
    if station_delays_fname is not None:
        station_delays = read_station_delays(station_delays_fname)
        for sname, sdata in StationInfo_dict.items():
            if sname in station_delays:
                sdata.adjust_station_delay( station_delays[sname] )
        
    if txt_cal_table_folder is not None:
        for sdata in StationInfo_dict.values():
            sdata.set_antenna_calibration_delays( txt_cal_table_folder )
        
        
            
    return StationInfo_dict
    


    
class curtain_plot(object):
    def __init__(self, StationInfo_dict):
        self.StationInfo_dict = StationInfo_dict
        self.sorted_station_names = list(self.StationInfo_dict.keys())
        self.sorted_station_names.sort(key=lambda sn: int(sn[2:]))
        
        self.station_addedData = {sname:False for sname in self.sorted_station_names} ##true if we have data from a station, False otherwise
        
        self.station_offsets = {}
        self.antenna_offsets = {}
        next_offset=0
        for sname in self.sorted_station_names:
            sdata = self.StationInfo_dict[sname]
            offsets = np.arange(sdata.num_antenna_sets)+next_offset
            next_offset += sdata.num_antenna_sets
            self.station_offsets[sname] = offsets
            
            for i in range(sdata.num_antenna_sets):
                self.antenna_offsets[ sdata.sorted_antenna_names[i] ] = offsets[i]
        
        self.min_time = np.inf
        
    def add_AntennaTime_dict(self, station_name, AntennaTime_dict, color='b', marker='o', size=20):
        self.station_addedData[station_name] = True
        ant_i = -1
        for ant_name in self.StationInfo_dict[station_name].sorted_antenna_names:
            ant_i += 1
            if ant_name not in AntennaTime_dict:
                continue
            
            if len(AntennaTime_dict[ant_name])==0:
                continue
            
            plt.scatter(AntennaTime_dict[ant_name], self.station_offsets[station_name][ant_i]*np.ones(len(AntennaTime_dict[ant_name])), c=color, marker=marker, s=size )
        
            min = np.min(AntennaTime_dict[ant_name])
            if min<self.min_time:
                self.min_time=min
        
    def addEventList(self, station_name, event_list, color='b', marker='o', size=20, annotation_list=None, annotation_size=15):
        self.station_addedData[station_name]=True
        for event_i in range(len(event_list)):
            event = event_list[event_i]
            
            plt.scatter(event, self.station_offsets[station_name], c=color, marker=marker, s=size)
            
            min_t = np.min(event)
            if min_t<self.min_time:
                self.min_time=min_t
           
            if annotation_list is not None:
                min_y=np.min(self.station_offsets[station_name])
                
                plt.annotate(str(annotation_list[event_i]), xy=(min_t,min_y), size=annotation_size)
    
    def annotate_station_names(self, size=15, t_offset=-0.0005, xt=0):
        ax = plt.gca()
        x_transform = ax.transData
        y_transform = ax.transData
        time_loc = self.min_time+t_offset
        if xt==1:
            x_transform = ax.transAxes
            time_loc=t_offset
        transform = blended_transform_factory(x_transform, y_transform)
        
        for stat_name,offsets in self.station_offsets.items():
            if self.station_addedData[stat_name]:
                ave_offset=(offsets[0]+offsets[-1])/2.0
                plt.annotate(str(stat_name), xy=(time_loc, ave_offset), size=size, textcoords=transform, xycoords=transform)
                
    #### TODO: add func to annotate antenna names
    
class curtain_plot_CodeLog(object):
    def __init__(self, StationInfo_dict, fname):
        self.CL = code_logger(fname)
        self.CL.add_statement("import numpy as np")
        self.CL.add_statement("import matplotlib.pyplot as plt")
        
        self.StationInfo_dict = StationInfo_dict
        self.sorted_station_names = list(self.StationInfo_dict.keys())
        self.sorted_station_names.sort(key=lambda sn: int(sn[2:]))
        
        self.station_addedData = {sname:False for sname in self.sorted_station_names} ##true if we have data from a station, False otherwise
        
        self.station_offsets = {}
        self.antenna_offsets = {}
        next_offset=0
        for sname in self.sorted_station_names:
            sdata = self.StationInfo_dict[sname]
            offsets = np.arange(sdata.num_antenna_sets)+next_offset
            next_offset += sdata.num_antenna_sets
            self.station_offsets[sname] = offsets
            
            for i in range(sdata.num_antenna_sets):
                self.antenna_offsets[ sdata.sorted_antenna_names[i] ] = offsets[i]
        
        self.min_time = np.inf
        
    def add_AntennaTime_dict(self, station_name, AntennaTime_dict, color='b', marker='o', size=20):
        self.station_addedData[station_name] = True
        ant_i = -1
        for ant_name in self.StationInfo_dict[station_name].sorted_antenna_names:
            ant_i += 1
            if ant_name not in AntennaTime_dict:
                continue
            
            if len(AntennaTime_dict[ant_name])==0:
                continue
            
            self.CL.add_function("plt.scatter", AntennaTime_dict[ant_name], self.station_offsets[station_name][ant_i]*np.ones(len(AntennaTime_dict[ant_name])), c=color, marker=marker, s=size )
#            plt.scatter(AntennaTime_dict[ant_name], self.station_offsets[station_name][ant_i]*np.ones(len(AntennaTime_dict[ant_name])), c=color, marker=marker, s=size )
        
            min = np.min(AntennaTime_dict[ant_name])
            if min<self.min_time:
                self.min_time=min
        
    def addEventList(self, station_name, event_list, color='b', marker='o', size=20, annotation_list=None, annotation_size=15):
        self.station_addedData[station_name]=True
        for event_i in range(len(event_list)):
            event = event_list[event_i]
            
            self.CL.add_function("plt.scatter", event, self.station_offsets[station_name], c=color, marker=marker, s=size)
#            plt.scatter(event, self.station_offsets[station_name], c=color, marker=marker, s=size)
            
            min_t = np.min(event)
            if min_t<self.min_time:
                self.min_time=min_t
           
            if annotation_list is not None:
                min_y = np.min(self.station_offsets[station_name])
                
                self.CL.add_function("plt.annotate", str(annotation_list[event_i]), xy=(min_t,min_y), size=annotation_size)
#                plt.annotate(str(annotation_list[event_i]), xy=(min_t,min_y), size=annotation_size)
    
    def annotate_station_names(self, size=15, t_offset=-0.0005, xt=0):
#        ax = plt.gca()
#        x_transform = ax.transData
#        y_transform = ax.transData
        time_loc = self.min_time+t_offset
#        if xt==1:
#            x_transform = ax.transAxes
#            time_loc=t_offset
#        transform = blended_transform_factory(x_transform, y_transform)
        
        for stat_name,offsets in self.station_offsets.items():
            if self.station_addedData[stat_name]:
                ave_offset=(offsets[0]+offsets[-1])/2.0
                self.CL.add_function("plt.annotate", str(stat_name), xy=(time_loc, ave_offset), size=size)

    def save(self):
        self.CL.add_statement( "plt.tick_params(axis='both', which='major', labelsize=30)" )
        self.CL.add_statement( "plt.show()" )
        self.CL.save()
                
    #### TODO: add func to annotate antenna names
                
def AntennaPulse_dict__TO__AntennaTime_dict(AntennaPulse_dict, station_data, polarization=None):
    out={}
    for antname, pulses in AntennaPulse_dict.items():
        time_list=[]
        
        for pulse in pulses:
            time_list.append(  pulse.peak_time(polarization) )
            
        out[antname] = time_list
    return out

def refilter_pulses(AntennaPulse_dict, new_num_std, bad_antennas=[], pol_requirement=0):
    """ given a dictionary of list of pulses, filter out pulses without a large enough signal. 
    pol_requirement is good polarizations 0 (default) means either polarization can be good. 1 means even mus be good. 2 means odd must be good, 3 means both must be good"""
    return {ant_name:[pulse for pulse in pulse_list if pulse.above_thresh(new_num_std) and pulse.meets_pol_requirement(pol_requirement)] for ant_name, pulse_list in  AntennaPulse_dict.items() if ant_name not in bad_antennas}

def plot_pulses():
    ##opening data
    timeID = "D20170929T202255.000Z"
    input_folder_name = "/pulse_data"
    station = "CS002"
    
    first_block = 3500
    num_blocks = 100
    
    #### additional data files
    ant_timing_calibrations = "cal_tables/TxtAntDelay"
    polarization_flips = "/polarization_flips.txt"
    bad_antennas = "/bad_antennas.txt"
    additional_antenna_delays = "/ant_delays.txt"
    
    
    
    processed_dir = processed_data_dir(timeID)
    
    polarization_flips = processed_dir + '/' + polarization_flips
    bad_antennas = processed_dir + '/' + bad_antennas
    ant_timing_calibrations = processed_dir + '/' + ant_timing_calibrations
    additional_antenna_delays = processed_dir + '/' + additional_antenna_delays
    
    print("A")
    
    StationInfo_dict = read_station_info(timeID, input_folder_name, station_names=[station], ant_delays_fname=additional_antenna_delays, 
                                         bad_antennas_fname=bad_antennas, pol_flips_fname=polarization_flips, txt_cal_table_folder=ant_timing_calibrations)
    
    
    print("B")
    antennaPulse_dict = StationInfo_dict[station].read_pulse_data(approx_min_block=first_block, approx_max_block=first_block+num_blocks )
    
    print("C")
    H = 0
    for ant_name,pulses in antennaPulse_dict.items():
        print(ant_name, len(pulses))
        num_plotted = 0
        for pulse in pulses:
            
            block = pulse.section_number/2
            if block<first_block or block>(first_block+num_blocks):
                continue
            
            even_HE = pulse.even_antenna_hilbert_envelope
            odd_HE = pulse.odd_antenna_hilbert_envelope
            
            M = 2*max(np.max(even_HE), np.max(odd_HE))
            
            even_HE /= M
            odd_HE /= M
            
            plt.plot( np.arange(len(even_HE))+pulse.starting_index,  even_HE+H)
            plt.plot( np.arange(len(even_HE))+pulse.starting_index,  odd_HE+H)
            num_plotted += 1
            
        print("  ",num_plotted)
            
        H += 1
        
    plt.show()
    
    print("D")
    
#    station = "CS001"
#    StationInfo_dict = read_station_info("D20160712T173455.100Z", [station])
#    station_info = StationInfo_dict[station]
#    
#    antenna_pulse_dict = station_info.read_pulse_data(2**4, approx_min_block=90800, approx_max_block=90900)
#    ATD = AntennaPulse_dict__TO__AntennaTime_dict(antenna_pulse_dict, station_info)
#    
#    CP = curtain_plot_CodeLog(StationInfo_dict, "CS001_CP")
#    CP.add_AntennaTime_dict( station, ATD )
#    CP.save()
#    print( "DONE" )
    
    
def do_curtain_plot():
    ##opening data
    
    station_offsets = {
            }
    
    timeID = "D20180308T165753.250Z"
    input_folder_name = "/pulse_data"
    
    first_block = 700
    num_blocks = 60
    
    filter_fraction = 0.0
    
    stations_to_exclude = ["RS407", "CS028", "CS007", "CS101", "RS210",  "RS310", "CS401"]
    
    #### additional data files
    ant_timing_calibrations = "cal_tables/TxtAntDelay"
    polarization_flips = "/polarization_flips.txt"
    bad_antennas = "/bad_antennas.txt"
    additional_antenna_delays = "/ant_delays.txt"
    
    
    
    processed_dir = processed_data_dir(timeID)
    
    polarization_flips = processed_dir + '/' + polarization_flips
    bad_antennas = processed_dir + '/' + bad_antennas
    ant_timing_calibrations = processed_dir + '/' + ant_timing_calibrations
    additional_antenna_delays = processed_dir + '/' + additional_antenna_delays
    
    
    
    print("A")
    StationInfo_dict = read_station_info(timeID, input_folder_name, stations_to_exclude=stations_to_exclude, ant_delays_fname=additional_antenna_delays, 
                                         bad_antennas_fname=bad_antennas, pol_flips_fname=polarization_flips, txt_cal_table_folder=ant_timing_calibrations)
    
    
    
    print("B")
    CP = curtain_plot( StationInfo_dict )
    for sname, sinfo in StationInfo_dict.items():
        print(sname)
        if sname in station_offsets:
            sinfo.adjust_station_delay( station_offsets[sname] )
            print("    ", station_offsets[sname] )
        
        antennaPulse_dict = sinfo.read_pulse_data(approx_min_block=first_block, approx_max_block=first_block+num_blocks )

        
        ## find maximum amplitude
        max_amp = 0.0
        for ant_name, pulse_list in antennaPulse_dict.items():
            for pulse in pulse_list:
                Even_amp = np.max( pulse.even_antenna_hilbert_envelope )
                Odd_amp = np.max( pulse.odd_antenna_hilbert_envelope )
                max_amp = max(max_amp, Even_amp, Odd_amp)
                
        ##filter the pulses
        filtered_pulse_dict = {}
        for ant_name, pulse_list in antennaPulse_dict.items():
            
            new_pulse_list = []
            for pulse in pulse_list:
                amp = max( np.max( pulse.even_antenna_hilbert_envelope ), np.max( pulse.odd_antenna_hilbert_envelope ) )
                if amp > max_amp*filter_fraction:
                    new_pulse_list.append( pulse )
                    
            filtered_pulse_dict[ant_name] = new_pulse_list
                
        
        
        
        ant_time_dict = AntennaPulse_dict__TO__AntennaTime_dict(filtered_pulse_dict, sinfo, 0 )
        
        CP.add_AntennaTime_dict(sname, ant_time_dict)
        
    CP.annotate_station_names(t_offset=0.00007)
    plt.axvline(x=1.15321885)
    plt.show()
    
    
def do_amplitude_histogram():
    timeID = "D20170929T202255.000Z"
    input_folder_name = "/pulse_data"
    
    first_block = 3500
    num_blocks = 50
    
    stations_to_exclude = ["CS028"]
    
    #### additional data files
    ant_timing_calibrations = "cal_tables/TxtAntDelay"
    polarization_flips = "/polarization_flips.txt"
    bad_antennas = "/bad_antennas.txt"
    additional_antenna_delays = "/ant_delays.txt"
    
    
    
    processed_dir = processed_data_dir(timeID)
    
    polarization_flips = processed_dir + '/' + polarization_flips
    bad_antennas = processed_dir + '/' + bad_antennas
    ant_timing_calibrations = processed_dir + '/' + ant_timing_calibrations
    additional_antenna_delays = processed_dir + '/' + additional_antenna_delays
    
   
    StationInfo_dict = read_station_info(timeID, input_folder_name, stations_to_exclude=stations_to_exclude, ant_delays_fname=additional_antenna_delays, 
                                         bad_antennas_fname=bad_antennas, pol_flips_fname=polarization_flips, txt_cal_table_folder=ant_timing_calibrations)
    
    for sname, sinfo in StationInfo_dict.items():
        print(sname)
        
        antennaPulse_dict = sinfo.read_pulse_data(approx_min_block=first_block, approx_max_block=first_block+num_blocks )

        
        pulse_amplitudes = []
        for ant_name, pulse_list in antennaPulse_dict.items():
            for pulse in pulse_list:
                Even_amp = np.max( pulse.even_antenna_hilbert_envelope )
                if Even_amp >10:
                    pulse_amplitudes.append( Even_amp )
                    
        plt.hist(pulse_amplitudes, bins=100)
        plt.show()
    
    
if __name__ == "__main__":
    do_curtain_plot()
#    do_amplitude_histogram()
    
    
