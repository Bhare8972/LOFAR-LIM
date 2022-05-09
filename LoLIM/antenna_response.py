#!/usr/bin/env python

""" A set of tools for modelling the antenna reponse and for callibrating the amplitude vs frequency of the antennas
based on pyCRtools. see Schellart et al. Detecting cosmic rays with the LOFAR radio telescope,  and Nelles et al. Calibrating the absolute amplitude scale for air showers measured at LOFAR

Note: LBA_ant_calibrator still needs some work.

author: Brian hare
"""

##internal
import glob
from pickle import load
import datetime
from os.path import  dirname, abspath

##external 
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator, pchip_interpolate, PchipInterpolator
from scipy.io import loadmat



##mine
from LoLIM.utilities import processed_data_dir, MetaData_directory, RTD, SId_to_Sname, v_air
import LoLIM.transmogrify as tmf
import LoLIM.IO.metadata as md



# def get_absolute_calibration(frequencies, freq_mode='30-90'):
#     """this function returns the absolute calibration given a set of frequencies (in Hz). Note that this is the INVERSE of the factor in pyCRtools. You should multiply this by the jones matrix to set the absolute correction
#     Note this is spefically derived for 80m cable lengths, which is consistant with relative calibraiton below.
    
#     If frequency is outside 30-80 MHz. A is assumed to be constant outside 30-80. Cable lengths and RCU gain are correct between 10-90 MHz. Assumed to be constant outside 10-90.
    
#     Returns 0 where frequencies<0
    
#     freq-mode: can be '30-90' or '10-90'. This sets which RCU setting is used. This should probably be the same as the data set being used, but probably doesn't matter.
#           The calibration was derived with 30-80, which is default."""
          
#     A = [
    
#         ]
    


class aartfaac_LBA_model:

    def __init__(self, model_loc=None, R=700, C=15e-12, mode="LBA_OUTER"):
        if model_loc is None:
            model_loc = dirname(abspath(__file__)) + '/AARTFAAC_LBA_MODEL/'
        
        self.model_loc = model_loc
        self.antenna_mode = mode
        
        self.fine_model_loc = self.model_loc + "LBA_core_fine"
        self.course_model_loc = self.model_loc + "LBA_core"
        
        ## get base data, assume is same between models
        data_fine = loadmat( self.fine_model_loc , variable_names=['Positions', 'Theta', 'Phi', 'Freq', 'Zant', 'g_z'] ) ## Zant is antenna impedences, g_z is angular dependence
        self.AART_ant_positions = data_fine['Positions']
        self.AART_thetas = data_fine['Theta'][0]  ## zenithal angle in degrees
        self.AART_Phis = data_fine['Phi'][0]      ## azimuthal angle in degrees
        
        self.AART_fineFreqs = data_fine['Freq'][0]
        fine_gz = data_fine['g_z']
        fine_Zant =  data_fine['Zant']
        
        data_course = loadmat( self.course_model_loc , variable_names=['Freq', 'Zant', 'g_z'] )
        self.AART_courseFreqs = data_course['Freq'][0]
        course_gz = data_course['g_z']
        course_Zant =  data_course['Zant']
        
        
        
        
        
        ### figure out how to fold fine and course models together
        fineFreq_low = self.AART_fineFreqs[0]
        # fineFreq_high = self.AART_fineFreqs[-1]
        
        freqs = []
        course_range_1 = [0]
        fine_range = []
        course_range_2 = []
        
        ## first we search AART_courseFreqs to find highest frequency lower than fine-range, adding them to the frequency list
        for i in range(len(self.AART_courseFreqs)): 
            Fc = self.AART_courseFreqs[i]
            if Fc < fineFreq_low:
                freqs.append(Fc)
            else:
                break
        course_range_1.append(i)
        fine_range.append(i)
        
        ## add all fine frequencies
        freqs += list(self.AART_fineFreqs)
        fine_range.append(len(freqs))
        course_range_2.append(len(freqs))
        
        ## find first course frequency greater than fine frequencies
        for i in range(i, len(self.AART_courseFreqs)):
            if self.AART_courseFreqs[i] > freqs[-1]:
                break
            
        ## add rest of course frequencies to list
        freqs += list(self.AART_courseFreqs[i:]) 
        course_range_2.append(len(freqs))
        
        self.all_frequencies = np.array(freqs, dtype=np.double)
        self.course_frequencyRange_low = np.array(course_range_1)
        self.fine_frequencyRange = np.array(fine_range)
        self.course_frequencyRange_high = np.array(course_range_2)
        
        
        
        #### combine antenna impedences
        num_ants = len(fine_Zant)
        num_thetas = len(self.AART_thetas)
        num_phis = len(self.AART_Phis)
        num_freqs = len(self.all_frequencies)
        
        self.antenna_Z = np.empty([num_ants, num_ants, num_freqs ], dtype=np.complex)
        
        self.antenna_Z[:,:, :self.course_frequencyRange_low[1]] = course_Zant[:,:,  :self.course_frequencyRange_low[1]]
        self.antenna_Z[:,:, self.fine_frequencyRange[0]:self.fine_frequencyRange[1]] = fine_Zant[:,:,  :]
        self.antenna_Z[:,:, self.fine_frequencyRange[1]:] = course_Zant[:,:, self.fine_frequencyRange[1]-len(self.all_frequencies):]
        
        
        self.total_impedence = np.empty((num_ants, num_ants, num_freqs), dtype=np.complex)
        self.set_RC( R, C ) ## this sets self.total_impedence
        
        
        
        
        
        ### now we combine total G_z, which is voltage on antenna
        self.total_Gz = np.empty( (num_ants,2,num_thetas,num_phis,num_freqs), dtype=np.complex )
        
        # for ant_i in range(num_ants):
            # for pol_i in range(2):
        self.total_Gz[:,:, :,:,                             : self.course_frequencyRange_low[1]] = course_gz[:,:, :,:,    : self.course_frequencyRange_low[1]]
        self.total_Gz[:,:, :,:, self.fine_frequencyRange[0] : self.fine_frequencyRange[1]]       = fine_gz  [:,:, :,:,    : ]
        self.total_Gz[:,:, :,:, self.fine_frequencyRange[1] :]                                   = course_gz[:,:, :,:, self.fine_frequencyRange[1]-len(self.all_frequencies): ]
        
    
    def get_LNA_filter(self):
        """given the antenna mode, return two things. First is np.array of indecies of LNAs that are on, second is np.array of LNAs that are off"""
        
        station_ant_positions = []
        for station in ['CS002', 'CS003', 'CS004', 'CS005', 'CS006', 'CS007']:
            station_ant_positions.append(  md.convertITRFToLocal( md.getItrfAntennaPosition(station, self.antenna_mode) )[::2, :] )
        
        found_indeces = []
        unfound_indeces = []
        
        AARFAACC_X_locations = self.AART_ant_positions[0]
        AARFAACC_Y_locations = self.AART_ant_positions[1]
        
        err2 = 0.1*0.1
        for AAR_i in range(len(AARFAACC_X_locations)):
            
            found = False
            for stat_ant_XYZ in station_ant_positions:
                for XYZ in stat_ant_XYZ:
                    KX, KY, KZ = XYZ
                    AX = AARFAACC_X_locations[AAR_i]
                    AY = AARFAACC_Y_locations[AAR_i]
                    R2 = (AX-KX)**2 + (AY-KY)**2
                    if R2<err2:
                        found = True
                        break
                if found:
                    break
                
            if found:
                found_indeces.append(AAR_i)
            else:
                unfound_indeces.append(AAR_i)
                
        return np.array(found_indeces), np.array(unfound_indeces)
    
    def set_RC(self, R, C ):
        self.R = R
        self.C = C
        
        
        num_ants = len(self.antenna_Z)
        num_freqs = len(self.all_frequencies)
        
        # functioning_indeces, nonfunctioning_indeces = self.get_LNA_filter()
        # LNA_filter = np.zeros(num_ants, dtype=np.int)
        # LNA_filter[ functioning_indeces ] = 1
        # LNA_Off_filter = np.zeros(num_ants, dtype=np.int)
        # LNA_Off_filter[ nonfunctioning_indeces ] = 1
        
        
        
        tmp_mat = np.empty((num_ants, num_ants), dtype=np.complex)
        ZLMA_matrix =  np.zeros((num_ants, num_ants), dtype=np.complex)
        
        for Fi in range(num_freqs):
            # set LNA matrix
            LNA = R/(1+self.all_frequencies[Fi]*(2*np.pi*1j*R*C)) # LNA_filter*(R/(1+self.all_frequencies[Fi]*(2*np.pi*1j*R*C)))  +  LNA_Off_filter*( 1/(self.all_frequencies[Fi]*2*np.pi*1j*C) )
            np.fill_diagonal(ZLMA_matrix,  LNA)
            
            ## caclulate denominator bit
            tmp_mat[:,:] = self.antenna_Z[:,:,Fi]
            tmp_mat += ZLMA_matrix
            
            ## try sign difference
            # tmp_mat *= -1
            # np.fill_diagonal(tmp_mat,  np.diagonal(tmp_mat)*-1)
            
            ## invert and multiple
            inv = np.linalg.inv( tmp_mat )
            np.matmul(ZLMA_matrix, inv, out=tmp_mat)
            
            # finally
            self.total_impedence[:,:,Fi] = tmp_mat
            

    def loc_to_anti(self, antenna_XY):
        """given an antenna XY location, return internal index of X-dipole of that antenna"""
        
        KX, KY = antenna_XY
        
        AARFAACC_X_locations = self.AART_ant_positions[0, 0::2]
        AARFAACC_Y_locations = self.AART_ant_positions[1, 0::2]
        
        found_index = None
        err2 = 0.1*0.1
        for AAR_i in range(len(AARFAACC_X_locations)):
            AX = AARFAACC_X_locations[AAR_i]
            AY = AARFAACC_Y_locations[AAR_i]
            R2 = (AX-KX)**2 + (AY-KY)**2
            if R2<err2:
                found_index = AAR_i*2
                break
        if found_index is None:
            print("ERROR! AARTFAAC model cannot find your antenna loc:", antenna_XY)
            quit()
            
        X_ant_i = found_index
        return X_ant_i
        
    def RCUname_to_anti(self, antenna_name):
        """given RCUid (and known mode), return internal index of the X-dipole of that antenna"""
        
        station = SId_to_Sname[ int(antenna_name[:3]) ]
        known_positions = md.convertITRFToLocal( md.getItrfAntennaPosition(station, self.antenna_mode) )
        this_antenna_location = known_positions[ int(antenna_name[-3:]) ]
        
        return self.loc_to_anti( this_antenna_location[0:2] )
    
    
    
    class single_LBA_model:
        def __init__(self, jones_functions, freq_bounds):
            self.jones_functions = jones_functions
            self.freq_bounds = freq_bounds
            
        def Jones_Matrices(self, frequencies, zenith, azimuth, freq_fill=1.0):
            """ if frequencies is numpy array in Hz, zenith and azimuth in degrees, than return numpy array of jones matrices,
            that when doted with [zenithal,azimuthal] component of incidident E-field, then will give [X,Y] voltages on dipoles"""
            
            return_matrices = np.empty( (len(frequencies), 2,2), dtype=np.complex )
            
            if zenith<0:
                zenith = 0
            elif zenith > 90:
                zenith = 90
                
            while azimuth<0:
                azimuth += 360
            while azimuth>360:
                azimuth -= 360
                
            # good_frequency_filter = np.logical_and(frequencies>=self.freq_bounds[0], frequencies<self.freq_bounds[-1])
            
            # points = np.empty( (np.sum(good_frequency_filter), 3), dtype=np.double )
            
            
            J00,J01 = self.jones_functions[0]
            J10,J11 = self.jones_functions[1]
            
            for fi,f in enumerate(frequencies):
                if f<=self.freq_bounds[0] or f>=self.freq_bounds[-1]:
                    return_matrices[fi, 0,0] = freq_fill
                    return_matrices[fi, 1,1] = freq_fill
                    return_matrices[fi, 0,1] = 0.0
                    return_matrices[fi, 1,0] = 0.0
                else:
                    return_matrices[fi, 0,0] = J00( (zenith,azimuth,f) )
                    return_matrices[fi, 0,1] = J01( (zenith,azimuth,f) )
                    return_matrices[fi, 1,0] = J10( (zenith,azimuth,f) )
                    return_matrices[fi, 1,1] = J11( (zenith,azimuth,f) )
                    
                    # print(fi)
                    # M = return_matrices[fi]
                    # print("J:", M)
                    # print( M[ 0,0]*M[ 1,1] - M[  0,1]*M[  1,0] )
                    # print()
                    
            return return_matrices
        
    def get_antenna_model(self, ant_i):
        """note: ant_i should be internal index!"""
        
        
        X_ant_i = int(ant_i/2)*2
        
        KX, KY = self.AART_ant_positions[:, X_ant_i]
        
        num_thetas = len(self.AART_thetas)
        num_phis = len(self.AART_Phis)
        num_frequencies = len( self.all_frequencies)
        num_antennas = len( self.total_Gz )
        
    #### some setup
        ## phase shifter to remove geometric delay
        shifter = np.empty([num_thetas, num_phis, num_frequencies ], dtype=np.complex)
        for zi in range( num_thetas ):
            for ai in range( num_phis ):
                Zenith = self.AART_thetas[zi]/RTD
                Azimuth = self.AART_Phis[ai]/RTD
                dt = ( KX*np.sin(Zenith)*np.cos(Azimuth) + KY*np.sin(Zenith)*np.sin(Azimuth))/v_air
                np.exp( self.all_frequencies*(1j*2*np.pi*(-dt)), out=shifter[zi,ai] )
                

    
        ## frequency interpolation
        frequency_bin = 0.5e6
        minF = self.all_frequencies[0]
        maxF = self.all_frequencies[-1]
        num_interpolant_freqs = int((maxF+minF)/frequency_bin )
        interpolant_frequencies = np.linspace(minF, maxF, num_interpolant_freqs)
        
        ## memory
        tmp1 = np.empty( (num_antennas,num_frequencies), dtype=np.complex )
        tmp2 = np.empty(num_frequencies, dtype=np.complex )
        tmp3 = np.empty(num_frequencies, dtype=np.double )

        def make_interpolant(antenna, polarization):
            """ antenna is 0 or 1 for X or Y, pol is 0 or 1 for zenith or azimithal component"""
            nonlocal tmp1, tmp2, tmp3
            
            antenna = antenna + X_ant_i
            
            grid = np.empty([num_thetas, num_phis, len(interpolant_frequencies) ], dtype=np.complex)
        
            for theta_i in range(num_thetas):
                for phi_i in range(num_phis):
                    
                    ## dot product between voltages and impedences
                    tmp1[:,:]  = self.total_Gz[:, polarization,theta_i,phi_i, :] 
                    tmp1 *= self.total_impedence[antenna, :, :] # this shouldn't matter??
                    # tmp1 *= self.total_impedence[:, antenna, :]
                    
                    
                    
                    np.sum( tmp1, axis=0, out = tmp2 )
                    
                    ## shift phase due to arival direction
                    tmp2 *= shifter[theta_i,phi_i]
                    
                    ## interpolate amplitude and phase
                    np.abs(tmp2, out=tmp3)
                    interp_ampltude = pchip_interpolate(self.all_frequencies,tmp3, interpolant_frequencies)
                    
                    # if theta_i==0 and phi_i==0:
                        # print('amp')
                        # plt.plot( interpolant_frequencies, interp_ampltude, 'o' )
                        # plt.plot( self.all_frequencies, tmp3, 'o' )
                        # plt.show()
                    
                    angles = np.angle(tmp2)
                    angles = np.unwrap(angles)
                    
                    interp_angle = pchip_interpolate(self.all_frequencies,angles, interpolant_frequencies)
                    
                    # if theta_i==0 and phi_i==0:
                    #     print('angle')
                    #     plt.plot( interpolant_frequencies, interp_angle, 'o' )
                    #     plt.plot( self.all_frequencies, angles, 'o' )
                    #     plt.show()
                    

                        
                    ## now convert back to real and imag
                    interp_angle = interp_angle*1j
                    np.exp( interp_angle, out=grid[theta_i,phi_i] )
                    grid[theta_i,phi_i] *= interp_ampltude
                    
            ## correct for different in definition
            if polarization==0: ## zenith points in oppisite directions in two models??
                grid *= -1
            ## and final angle-frequency interpolant
            interpolant = RegularGridInterpolator((self.AART_thetas, self.AART_Phis, interpolant_frequencies),  grid,  bounds_error=False,fill_value=0.0)
            return interpolant
        
        J00_interpolant = make_interpolant(0, 0)
        J01_interpolant = make_interpolant(0, 1)
        J10_interpolant = make_interpolant(1, 0)
        J11_interpolant = make_interpolant(1, 1)
        
        return self.single_LBA_model([[J00_interpolant,J01_interpolant],[J10_interpolant,J11_interpolant]],  [interpolant_frequencies[0],interpolant_frequencies[-1]])
    
    def get_average_antenna_model(self):
        """return model averaged over antenna set"""
        
        if self.antenna_mode == "LBA_OUTER":
            start_i = 576
            end_i = 576*2
        elif self.antenna_mode == "LBA_INNER":
            start_i = 0
            end_i = 576
        else:
            print("unknown mode:", self.antenna_mode )
    
        total_num_antennas = len( self.total_Gz )
        num_frequencies = len(self.all_frequencies)
        num_zeniths = len(self.AART_thetas)
        num_azimuths = len(self.AART_Phis)
    
        ### define a utility function for extracting grid data
        temp_matrix = np.empty( (num_zeniths, num_azimuths, num_frequencies), dtype=np.complex )
        shifterTMP = np.empty( [ num_frequencies ], dtype=np.complex)
        tmp1 = np.empty( (total_num_antennas,num_frequencies), dtype=np.complex )
        def get_ant_i(ant_i, pol_i):
            """given internal antenna index, and polarization, fill temp_matrix with correct response"""
            nonlocal  shifterTMP, tmp1, temp_matrix
            
            KX, KY = self.AART_ant_positions[:, ant_i]
            for zi, ze in enumerate(self.AART_thetas):
                for ai, az in enumerate(self.AART_Phis):
                    
                    ## dot product between voltages and impedences
                    tmp1[:,:]  = self.total_Gz[:, pol_i,zi,ai, :] 
                    tmp1 *= self.total_impedence[ant_i, :, :] # this shouldn't matter??
                    # tmp1 *= self.total_impedence[:, antenna, :]
                    
                    np.sum( tmp1, axis=0, out = temp_matrix[zi,ai, : ] )
                    
                    ## caculate phase shifts
                    Zenith = ze/RTD
                    Azimuth = az/RTD
                    dt = ( KX*np.sin(Zenith)*np.cos(Azimuth) + KY*np.sin(Zenith)*np.sin(Azimuth))/v_air
                    shifterTMP[:] = self.all_frequencies
                    shifterTMP *= -1j*2*np.pi*dt
                    np.exp( shifterTMP, out=shifterTMP )
                    
                    ## apply the shits
                    temp_matrix[zi,ai, : ] *= shifterTMP
                    
            if pol_i==0: ## zenith points in oppisite directions in two models??
                temp_matrix *= -1
                    
        ### now define a utility function for calculating the average, and interpolating
        
        ## frequency interpolation
        frequency_bin = 0.5e6
        minF = self.all_frequencies[0]
        maxF = self.all_frequencies[-1]
        num_interpolant_freqs = int((maxF+minF)/frequency_bin )
        interpolant_frequencies = np.linspace(minF, maxF, num_interpolant_freqs)
        
        temp_AVE_matrix = np.empty( (num_zeniths, num_azimuths, num_frequencies), dtype=np.complex )
        tmp3 = np.empty(num_frequencies, dtype=np.double )
        def calc_func(antenna_pol, field_pol):
            """antenna_pol should be 0 or 1 for X or Y antenna, and field_pol is 0 or 1 for zenithal or azimuthal field"""
            nonlocal temp_AVE_matrix, tmp3
            
            ## first average
            Npairs = int( (end_i-start_i)/2 )
            for pair_i in range(Npairs):
                ant_i = 2*pair_i + antenna_pol + start_i
                get_ant_i(ant_i, field_pol) ## this fills temp_matrix
                temp_AVE_matrix += temp_matrix ## error prone, but should be okay
                
                if not np.all( np.isfinite(temp_AVE_matrix) ):
                    print('YO problem!', pair_i, np.all( np.isfinite(temp_matrix) ) )
                    quit()
                
            temp_AVE_matrix /= Npairs
        
            
            upsample_grid = np.empty([num_zeniths, num_azimuths, len(interpolant_frequencies) ], dtype=np.complex)
        
            ## now interpolate frequencies 
            for zi, ze in enumerate(self.AART_thetas):
                for ai, az in enumerate(self.AART_Phis):
                    
                    np.abs(temp_AVE_matrix[zi,ai, : ], out=tmp3)
                    interp_ampltude = pchip_interpolate(self.all_frequencies, tmp3, interpolant_frequencies)
                    
                    
                    
                    angles = np.angle(temp_AVE_matrix[zi,ai, : ])
                    angles = np.unwrap(angles)
                    
                    interp_angle = pchip_interpolate(self.all_frequencies, angles, interpolant_frequencies)
                    
                    ## now convert back to real and imag
                    interp_angle = interp_angle*1j
                    np.exp( interp_angle, out = upsample_grid[zi,ai, :] )
                    upsample_grid[zi,ai, :] *= interp_ampltude
                    
            
            ## and final angle-frequency interpolant
            return RegularGridInterpolator((self.AART_thetas, self.AART_Phis, interpolant_frequencies),  upsample_grid,  bounds_error=False, fill_value=0.0)
        
        
        ### now we actually make the responses
        J00 = calc_func(0, 0)
        J01 = calc_func(0, 1)
        J10 = calc_func(1, 0)
        J11 = calc_func(1, 1)
        return self.single_LBA_model([[J00,J01],[J10,J11]],  [interpolant_frequencies[0],interpolant_frequencies[-1]])
                    
                    
    
    def get_antenna_locs(self):
        """return two arrays. First is X locations of all antennas. Second is Y locations. Each location is in pairs, the first is the X dipole, second is Y dipole"""
        return self.AART_ant_positions[0], self.AART_ant_positions[1]
        
    def get_grid_info(self):
        """return info about the internal grid. returns three real-valued numpy arrays: frequencies [Hz], zenithal [degrees], and azimuthal [degrees]"""
        return self.all_frequencies, self.AART_thetas, self.AART_Phis
    
    def get_response_grid(self, antenna_i, frequency_i, ant_pol_i, field_pol_i):
        """given an antenna_i, and frequency_i, return response of ant_pol_i (0 for X 1 for Y dipole) to field_pol_i (0 is zenithal, 1 is azimuthal). Response is a 2D matrix, first index is zenith angle, second is azimuthal"""
        
        total_antenna_i = int(antenna_i/2)*2 + ant_pol_i
        
        
        #### some setup
        KX, KY = self.AART_ant_positions[:, total_antenna_i]
        
        num_thetas = len(self.AART_thetas)
        num_phis = len(self.AART_Phis)
        
        frequency = self.all_frequencies[ frequency_i ]
        
        ret = np.empty([num_thetas, num_phis], dtype=np.complex)
        
        ## the calculation
        for zenith_i in range( num_thetas ):
            for azimuth_i in range( num_phis ):
                
                
                ## phase shifter to remove geometric delay
                Zenith = self.AART_thetas[zenith_i]/RTD
                Azimuth = self.AART_Phis[azimuth_i]/RTD
                dt = ( KX*np.sin(Zenith)*np.cos(Azimuth) + KY*np.sin(Zenith)*np.sin(Azimuth))/v_air
                phase_shift = np.exp( frequency*(1j*2*np.pi*(-dt)) )
                
                
                ## the antenna response
                response = np.dot( self.total_impedence[total_antenna_i, :, frequency_i] , self.total_Gz[:, field_pol_i, zenith_i, azimuth_i, frequency_i] )
                
                
                ret[ zenith_i,  azimuth_i] = response*phase_shift
        
        
        
        if field_pol_i==0: ## zenith points in oppisite directions in two models??
            ret *= -1
            
        return ret

class aartfaac_average_LBA_model:

    def __init__(self, mode="LBA_OUTER"):
        if mode == "LBA_OUTER":
            fname = MetaData_directory+"/lofar/antenna_response_model/AARTFAAC_AVE_LBAOUTER_R700_C17.npz"
        else:
            print('mode unknown:', mode)
            quit()

        data = np.load(fname)
        initial_frequencies = data['freqs']
        zeniths = data['zeniths']
        azimuths = data['azimuths']
        
        original_J00 = data['J00']
        original_J01 = data['J01']
        original_J10 = data['J10']
        original_J11 = data['J11']
        
        
        num_zeniths  = len(zeniths)
        num_azimuths = len(azimuths)
        
        ## frequency interpolation
        frequency_bin = 0.5e6
        minF = initial_frequencies[0]
        maxF = initial_frequencies[-1]
        num_interpolant_freqs = int((maxF+minF)/frequency_bin )
        interpolant_frequencies = np.linspace(minF, maxF, num_interpolant_freqs)
        
        
        # print('arg', initial_frequencies.shape, original_J00.shape)
        
        tmp = np.empty(len(initial_frequencies), dtype=np.double )
        def upsample_and_interpolate(GRID):
            nonlocal tmp
            
            upsample_grid = np.empty([num_zeniths, num_azimuths, num_interpolant_freqs ], dtype=np.complex)
        
            ## now interpolate frequencies 
            for zi, ze in enumerate(zeniths):
                for ai, az in enumerate(azimuths):
                    
                    np.abs(GRID[zi,ai, : ], out=tmp)
                    interp_ampltude = pchip_interpolate(initial_frequencies, tmp, interpolant_frequencies)
                    
                    
                    
                    angles = np.angle(GRID[zi,ai, : ])
                    angles = np.unwrap(angles)
                    
                    interp_angle = pchip_interpolate(initial_frequencies, angles, interpolant_frequencies)
                    
                    ## now convert back to real and imag
                    interp_angle = interp_angle*1j
                    np.exp( interp_angle, out = upsample_grid[zi,ai, :] )
                    upsample_grid[zi,ai, :] *= interp_ampltude
                    
            
            ## and final angle-frequency interpolant
            
            return RegularGridInterpolator((zeniths, azimuths, interpolant_frequencies),  upsample_grid,  bounds_error=False, fill_value=0.0)
        
        self.freq_bounds = [ interpolant_frequencies[0], interpolant_frequencies[-1] ]
        
        self.J00 = upsample_and_interpolate( original_J00 )
        self.J01 = upsample_and_interpolate( original_J01 )
        self.J10 = upsample_and_interpolate( original_J10 )
        self.J11 = upsample_and_interpolate( original_J11 )
            
    def Jones_Matrices(self, frequencies, zenith, azimuth, freq_fill=1.0):
        """ if frequencies is numpy array in Hz, zenith and azimuth in degrees, than return numpy array of jones matrices,
        that when doted with [zenithal,azimuthal] component of incidident E-field, then will give [X,Y] voltages on dipoles"""
        
        return_matrices = np.empty( (len(frequencies), 2,2), dtype=np.complex )
        
        if zenith<0:
            zenith = 0
        elif zenith > 90:
            zenith = 90
            
        while azimuth<0:
            azimuth += 360
        while azimuth>360:
            azimuth -= 360
            
        # good_frequency_filter = np.logical_and(frequencies>=self.freq_bounds[0], frequencies<self.freq_bounds[-1])
        
        # points = np.empty( (np.sum(good_frequency_filter), 3), dtype=np.double )
        
    
        
        for fi,f in enumerate(frequencies):
            if f<self.freq_bounds[0] or f>self.freq_bounds[-1]:
                return_matrices[fi, 0,0] = freq_fill
                return_matrices[fi, 1,1] = freq_fill
                return_matrices[fi, 0,1] = 0.0
                return_matrices[fi, 1,0] = 0.0
            else:
                return_matrices[fi, 0,0] = self.J00( (zenith,azimuth,f) )
                return_matrices[fi, 0,1] = self.J01( (zenith,azimuth,f) )
                return_matrices[fi, 1,0] = self.J10( (zenith,azimuth,f) )
                return_matrices[fi, 1,1] = self.J11( (zenith,azimuth,f) )
                
                # print(fi)
                # M = return_matrices[fi]
                # print("J:", M)
                # print( M[ 0,0]*M[ 1,1] - M[  0,1]*M[  1,0] )
                # print()
                
        return return_matrices
    
    def save_to_text(self):
        
        freq_grid = np.linspace(10,90, num=int((90-10)/1)+1 ) 
        zenith_grid = np.linspace(0,90, num=int(90/5)+1 )
        azimuth_grid = np.linspace(0,360, num=int(360/10)+1 )
        
        J00_out = open('./J00_text.txt', 'w')
        J01_out = open('./J01_text.txt', 'w')
        J10_out = open('./J10_text.txt', 'w')
        J11_out = open('./J11_text.txt', 'w')
        
        J00_out.write('voltage on X-dipole to zenithal field. Azimuthal=0 points 45 degrees south from East.\n')
        J00_out.write('f (MHz) Zenith(deg.) Azimuth(deg.) real(Vout) imag(Vout)\n')
        
        J01_out.write('voltage on X-dipole to azimuthal field. Azimuthal=0 points 45 degrees south from East.\n')
        J01_out.write('f (MHz) Zenith(deg.) Azimuth(deg.) real(Vout) imag(Vout)\n')
        
        J10_out.write('voltage on Y-dipole to zenithal field\. Azimuthal=0 points 45 degrees south from East.n')
        J10_out.write('f (MHz) Zenith(deg.) Azimuth(deg.) real(Vout) imag(Vout)\n')
        
        J11_out.write('voltage on Y-dipole to azimuthal field. Azimuthal=0 points 45 degrees south from East.\n')
        J11_out.write('f (MHz) Zenith(deg.) Azimuth(deg.) real(Vout) imag(Vout)\n')
        
        for f in freq_grid:
            for z in zenith_grid:
                for a in azimuth_grid:
                    jonesy = self.Jones_Matrices( [f*1e6],z,a+45 )
                    start = str(f)+ ' ' + str(z)+' '+str(a)+' '
                    J00_out.write(start+str(np.real(jonesy[0,0,0])) + ' ' + str(np.imag(jonesy[0,0,0])) +'\n')
                    J01_out.write(start+str(np.real(jonesy[0,0,1])) + ' ' + str(np.imag(jonesy[0,0,1])) +'\n')
                    J10_out.write(start+str(np.real(jonesy[0,1,0])) + ' ' + str(np.imag(jonesy[0,1,0])) +'\n')
                    J11_out.write(start+str(np.real(jonesy[0,1,1])) + ' ' + str(np.imag(jonesy[0,1,1])) +'\n')
        
        
    
class calibrated_AARTFAAC_model:
    """returns the AARTFAAC model multiplied by katies cal."""
    
    def __init__(self):
        self.AARTFAAC = aartfaac_average_LBA_model()
        
        calibration = [42484.88872879, 41519.47733373, 39694.16854372, 38435.36963118,
                       36974.13596039, 35819.34454985, 35072.53901876, 33960.74197721,
                       32944.65142405, 32112.44688046, 31590.52174516, 30433.52586868,
                       29756.92041985, 28880.95581826, 28204.37711364, 27646.83132496,
                       27058.11219529, 26693.43614747, 25952.89684011, 25517.72346825,
                       25220.5602153 , 24782.62398452, 24349.88724441, 23755.78027908,
                       23289.45439997, 22800.30039329, 22176.17569116, 21619.22598549,
                       21341.84386379, 21368.96227069, 22032.38713345, 22127.96304476,
                       22678.25184787, 24670.14420208, 25703.00236409, 24792.9736945,
                       23817.8752368 , 22912.1048524 , 22393.11267467, 21844.38535201,
                       20831.10515113, 20220.27315072, 19223.64518551, 18099.23669312,
                       17569.76228552, 16298.99230863, 14845.6897604 , 13488.79405579,
                       11966.53709451, 10895.66530218,  9703.34502309]
    
        cal_frequencies = np.arange(30e6, 80.5e6, 1e6)
        self.calibration_interpolator = PchipInterpolator(cal_frequencies, calibration, extrapolate=True  )

    def get_calibrator(self, frequencies):
        return self.calibration_interpolator( frequencies )

    def Jones_ONLY(self, frequencies, zenith, azimuth, freq_fill=1.0):
        return self.AARTFAAC.Jones_Matrices( frequencies, zenith, azimuth, freq_fill )

    def Jones_Matrices(self, frequencies, zenith, azimuth, freq_fill=1.0):
        """return calibrated jones matrices"""
        JM = self.AARTFAAC.Jones_Matrices( frequencies, zenith, azimuth, freq_fill )
        C = self.get_calibrator( frequencies )
        JM[:, 0,0] *= C
        JM[:, 0,1] *= C
        JM[:, 1,0] *= C
        JM[:, 1,1] *= C
        return JM
        
    
    
            
def invert_2X2_matrix_list( matrices ):
    """ if matrices is an array of 2x2 matrices, then return the array of inverse matrices """
    num = len(matrices)
    out = np.zeros( (num, 2,2), dtype=matrices.dtype)
    
    out[:, 0,0] = matrices[:, 1,1]
    out[:, 0,1] = -matrices[:, 0,1]
    out[:, 1,0] = -matrices[:, 1,0]
    out[:, 1,1] = matrices[:, 0,0]
    
    determinants = matrices[:, 0,0]*matrices[:, 1,1] - matrices[:, 0,1]*matrices[:, 1,0]
    
    out /= determinants[:, np.newaxis, np.newaxis]
    
    return out

def fourier_series( x, p):
    """Evaluates a partial Fourier series

        F(x) \\approx \\frac{a_{0}}{2} + \\sum_{n=1}^{\\mathrm{order}} a_{n} \\sin(nx) + b_{n} \\cos(nx)
    """

    r = p[0] / 2

    order = int( (len(p) - 1) / 2 )

    for i in range(order):

        n = i + 1

        r += p[2*i + 1] * np.sin(n * x) + p[2*i + 2] * np.cos(n * x)

    return r

def getGalaxyCalibrationData(antenna_noise_power, timestamp, antenna_type="outer"):
    """return factor to correct for amplitude shifts. Essenturally returns sqrt( P_{expected} / P_{measured} ). Where P is noise power. 
    for antenna_type outer it returns factor for Y/X dipoles, for "inner" returns "X/Y". antenna_noise_power is an array of measured powers for each antenna. 
    Even/odd indecies should be Y/X dipole for outer and oppisite for inner.
    timestamp should be posix timestamp"""
    
    
    longitude = 6.869837540/RTD
    
    ## this is in outer order:  Y,X
    # coefficients_lba = [ np.array([ 0.01489468, -0.00129305,  0.00089477, -0.00020722, -0.00046507]),   ## for Y antennas
    #                       np.array( [ 0.01347391, -0.00088765,  0.00059822,  0.00011678, -0.00039787] )  ] ## for X antennas
    coefficients_lba = [ np.array( [ 3.85712631e+01, -2.17182149e+00,  1.68114451e+00 , 4.24076969e-01,  -9.24289199e-01,  2.11372242e-01, 1.09281884e-01, -1.74674795e-01, 3.70793388e-03] ),   ## for Y antennas
                         np.array( [38.13314007, -3.02861767 , 2.11558435, -0.30123627, -0.94641864, -0.14297615,  0.05037442, -0.04133833, -0.11443689])  ] ## for X antennas

    
    
    # Convert timestamp to datetime object
    t = datetime.datetime.utcfromtimestamp(timestamp)
    # Calculate JD(UT1)
    ut = tmf.gregorian2jd(t.year, t.month, float(t.day) + ((float(t.hour) + float(t.minute) / 60. + float(t.second) / 3600.) / 24.))
    # Calculate JD(TT)
    dtt = tmf.delta_tt_utc(tmf.date2jd(t.year, t.month, float(t.day) + ((float(t.hour) + float(t.minute) / 60. + float(t.second) / 3600.) / 24.)))
    tt = tmf.gregorian2jd(t.year, t.month, float(t.day) + ((float(t.hour) + float(t.minute) / 60. + (float(t.second) + dtt / 3600.)) / 24.))
    # Calculate Local Apparant Sidereal Time
    last = tmf.rad2circle(tmf.last(ut, tt, longitude))


    galactic_noise_power =[  fourier_series(last, coefficients_lba[0]),  fourier_series(last, coefficients_lba[1])  ]
    
    antenna_noise_power[ antenna_noise_power==0 ] = np.nan
    
    if antenna_type == 'outer':
        Y_measured_powers = antenna_noise_power[0::2]
        X_measured_powers = antenna_noise_power[1::2]
        
        Y_expected_power = galactic_noise_power[0]
        X_expected_power = galactic_noise_power[1]
        
    else:
        X_measured_powers = antenna_noise_power[0::2]
        Y_measured_powers = antenna_noise_power[1::2]
        
        X_expected_power = galactic_noise_power[0]
        Y_expected_power = galactic_noise_power[1]
    
    
    ## note this should make new arrays
    Y_factors = Y_expected_power / Y_measured_powers
    X_factors = X_expected_power / X_measured_powers
    
    np.sqrt(Y_factors, out=Y_factors)
    np.sqrt(X_factors, out=X_factors)
    
    if antenna_type == 'outer':
        return Y_factors, X_factors
    else:
        return X_factors, Y_factors


