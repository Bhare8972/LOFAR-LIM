#!/usr/bin/env python

"""
A set of tools for modelling the antenna reponse and for callibrating the amplitude vs frequency of the antennas
originally inspired by pyCRtools

If you don't know what to do, call the function load_default_antmodel. Which returns a SphHarm_antModel object. Then call the Jones_Matrices method

Note, this module should use degrees on all input/outputs.

author: Brian Hare
"""

##internal
import glob
from pickle import load
import datetime
from os.path import  dirname, abspath

##external 
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator, pchip_interpolate, PchipInterpolator
from scipy.io import loadmat



##mine
from LoLIM.utilities import processed_data_dir, MetaData_directory, RTD, SId_to_Sname, v_air
import LoLIM.transmogrify as tmf
import LoLIM.IO.metadata as md
from LoLIM import sph_harm_fit



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
        
        self.antenna_Z = np.empty([num_ants, num_ants, num_freqs ], dtype=complex)
        
        self.antenna_Z[:,:, :self.course_frequencyRange_low[1]] = course_Zant[:,:,  :self.course_frequencyRange_low[1]]
        self.antenna_Z[:,:, self.fine_frequencyRange[0]:self.fine_frequencyRange[1]] = fine_Zant[:,:,  :]
        self.antenna_Z[:,:, self.fine_frequencyRange[1]:] = course_Zant[:,:, self.fine_frequencyRange[1]-len(self.all_frequencies):]
        
        
        self.total_impedence = np.empty((num_ants, num_ants, num_freqs), dtype=complex)
        self.set_RC( R, C ) ## this sets self.total_impedence
        
        
        
        
        
        ### now we combine total G_z, which is voltage on antenna
        self.total_Gz = np.empty( (num_ants,2,num_thetas,num_phis,num_freqs), dtype=complex )
        
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
        
        
        
        tmp_mat = np.empty((num_ants, num_ants), dtype=complex)
        ZLMA_matrix =  np.zeros((num_ants, num_ants), dtype=complex)
        
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

    def get_antenna_model(self, ant_i):
        """
        a helper function that calls get_antenna_model_grid, then make_AntModel_SphHarm_fit, and returns a  SphHarm_antModel.
        Uses default settins. Call get_antenna_model_grid and make_AntModel_SphHarm_fit yourself for finer control.
        This function is only kept for historical purposes
            """

        grid = self.get_antenna_model_grid(ant_i)
        return make_AntModel_SphHarm_fit(grid, self.all_frequencies, self.AART_thetas, self.AART_Phis)

    def get_average_antenna_model(self):
        """same as get_antenna_model, but average over all active antennas"""
        grid = self.get_average_antenna_model_grid()
        return make_AntModel_SphHarm_fit(grid, self.all_frequencies, self.AART_thetas, self.AART_Phis)

    def get_antenna_model_grid(self, ant_i, grid_out=None, memoryOne=None, memoryTwo=None, memoryThree=None):
        """return model for ant_i (is internal index!!).
        grid_out, if given, should have shape [2,2,num_thetas, num_phis, num_freqs], dtype=complex
            note, this is the output model indecies are [ antenna(0=X,1=Y), e-field(0=zenithal,1=azimuthal), zenith_angle, azimuth_angle, frequency ]
            NOTE: if grid_out is not None, it should be filled with zeros. as this output is actually ADDED to grid_out (for use elsewhere)
        memoryOne, if given, should have lenth num_frequencies, dtype=complex
        memoryTwo, if given, should have shape (num_antennas, num_frequencies), dtype=complex
        memoryThree, if given, should have shape num_frequencies, dtype=complex
        """

        X_ant_i = int(ant_i / 2) * 2

        KX, KY = self.AART_ant_positions[:, X_ant_i]

        num_thetas =      len(self.AART_thetas)
        num_phis =        len(self.AART_Phis)
        num_frequencies = len(self.all_frequencies)
        num_antennas =    len(self.total_Gz)




        ## phase shifter to remove geometric delay
        # if memoryOne is None:
        #     shifter = np.empty([num_thetas, num_phis, num_frequencies], dtype=complex)
        # else:
        #     shifter = memoryOne
        # for zi in range(num_thetas):
        #     for ai in range(num_phis):
        #         Zenith = self.AART_thetas[zi] / RTD
        #         Azimuth = self.AART_Phis[ai] / RTD
        #         dt = (KX * np.sin(Zenith) * np.cos(Azimuth) + KY * np.sin(Zenith) * np.sin(Azimuth)) / v_air
        #         np.exp(self.all_frequencies * (1j * 2 * np.pi * (-dt)), out=shifter[zi, ai])




        if memoryOne is None:
            shifter = np.empty(num_frequencies, dtype=complex)
        else:
            shifter = memoryOne

        ## memory and convinience function
        if memoryTwo is None:
            tmp1 = np.empty((num_antennas, num_frequencies), dtype=complex)
        else:
            tmp1 = memoryTwo

        if memoryThree is None:
            tmp2 = np.empty(num_frequencies, dtype=complex)
        else:
            tmp2 = memoryThree

        if grid_out is None:
            grid_out = np.zeros([2, 2, num_thetas, num_phis, num_frequencies], dtype=complex)


        ## GO!
        for theta_i in range(num_thetas):
            for phi_i in range(num_phis):

                Zenith = self.AART_thetas[theta_i] / RTD
                Azimuth = self.AART_Phis[phi_i] / RTD
                dt = (KX * np.sin(Zenith) * np.cos(Azimuth) + KY * np.sin(Zenith) * np.sin(Azimuth)) / v_air
                np.exp(self.all_frequencies * (1j * 2 * np.pi * (-dt)), out=shifter)

                for ai in [0,1]:
                    antenna = ai + X_ant_i

                    for ei in [0,1]:

                        ## dot product between voltages and impedences
                        tmp1[:, :] = self.total_Gz[:, ei, theta_i, phi_i, :]
                        tmp1 *= self.total_impedence[antenna, :, :]

                        np.sum(tmp1, axis=0, out=tmp2)

                        ## shift phase due to arival direction
                        tmp2 *= shifter

                        ## correct for different in definition
                        if ei == 0:  ## zenith points in oppisite directions in two models??
                            tmp2 *= -1

                        grid_out[ai,ei,theta_i,phi_i,:] += tmp2

        return grid_out



    def get_average_antenna_model_grid(self, grid_out=None, memoryOne=None, memoryTwo=None, memoryThree=None):
        """same as get_antenna_model_grid, but averages over all antennas"""
        
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


        if grid_out is None:
            grid_out = np.zeros([2,2,num_zeniths, num_azimuths, num_frequencies], dtype=complex)

        if memoryOne is None:
            memoryOne = np.empty(num_frequencies, dtype=complex)

        if memoryTwo is None:
            memoryTwo = np.empty((total_num_antennas, num_frequencies), dtype=complex)

        if memoryThree is None:
            memoryThree = np.empty(num_frequencies, dtype=complex)

        Npairs = int((end_i - start_i) / 2)
        for pair_i in range(Npairs):
            ant_i = 2 * pair_i + start_i

            self.get_antenna_model_grid(ant_i, grid_out=grid_out, memoryOne=memoryOne, memoryTwo=memoryTwo, memoryThree=memoryThree)

        grid_out *= 1.0/Npairs
    
        return grid_out
                    
                    
    
    def get_antenna_locs(self):
        """return two arrays. First is X locations of all antennas. Second is Y locations. Each location is in pairs, the first is the X dipole, second is Y dipole"""
        return self.AART_ant_positions[0], self.AART_ant_positions[1]
        
    def get_grid_info(self):
        """return info about the internal grid. returns three real-valued numpy arrays: frequencies [Hz], zenithal [degrees], and azimuthal [degrees]"""
        return self.all_frequencies, self.AART_thetas, self.AART_Phis
    
    def get_response_grid(self, antenna_i, frequency_i, ant_pol_i, field_pol_i):
        """
        given an antenna_i, and frequency_i, return response of ant_pol_i (0 for X 1 for Y dipole) to field_pol_i (0 is zenithal, 1 is azimuthal).
        Response is a 2D matrix, first index is zenith angle, second is azimuthal
        """
        
        total_antenna_i = int(antenna_i/2)*2 + ant_pol_i
        
        
        #### some setup
        KX, KY = self.AART_ant_positions[:, total_antenna_i]
        
        num_thetas = len(self.AART_thetas)
        num_phis = len(self.AART_Phis)
        
        frequency = self.all_frequencies[ frequency_i ]
        
        ret = np.empty([num_thetas, num_phis], dtype=complex)
        
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


def make_AntModel_SphHarm_fit(antenna_model_grid, frequencies, zeniths, azimuths, min_freq_interp=0.5e6, numSphHarm_orders=10):
    """takes a antenna_model grid of shape [2,2,num_thetas, num_phis, num_freqs], dtype=complex (as output by aartfaac_LBA_model)
    Fits spherical harmonics, and returns a SphHarm_antModel object.
    frequences should be Hz. zeniths and azimuths should be degrees. See get_grid_info from SphHarm_antModel.
    Since SphHarm_antModel interps frequencies with a linear interpolation, this function has an in-between step where it
    interpolates frequences with a pchip interpolator to a fine grid, whose fine-ness is controlled by min_freq_interp"""


    num_zeniths =  len(zeniths)
    num_azimuths = len(azimuths)

    ## Preprocessing
    num_new_freqs = int(( (frequencies[-1]-frequencies[0])/min_freq_interp )) + 1
    new_freqs, new_freq_step = np.linspace(frequencies[0], frequencies[-1], num=num_new_freqs, endpoint=True, retstep=True)
    new_model_grid = np.empty([2,2, num_zeniths, num_azimuths, num_new_freqs], dtype=complex)

    memory_tmp = np.empty( len(frequencies), dtype=np.double )
    memory_tmp_Abasis = np.empty( (2, num_zeniths, num_new_freqs), dtype=complex )
    memory_tmp_Bbasis = np.empty( (2, num_zeniths, num_new_freqs), dtype=complex )
    memory_tmp_four = np.empty( (2, num_zeniths, num_new_freqs), dtype=complex )

    for azi in range(num_azimuths):
        ## first we interpolate to finer frequency bin
        for ai in [0,1]:
            for ei in [0,1]:
                for zi in range(num_zeniths):
                    F_data = antenna_model_grid[ai,ei,zi,azi,:]

                    ## interpolate amplitude and phase
                    np.abs(F_data, out=memory_tmp)
                    interp_ampltude = pchip_interpolate(frequencies, memory_tmp, new_freqs)

                    # angles = np.angle(F_data)
                    np.arctan2(F_data.imag, F_data.real, out=memory_tmp)
                    angles = np.unwrap(memory_tmp)

                    interp_angle = pchip_interpolate(frequencies, angles, new_freqs)


                    ## now convert back to real and imag
                    interp_angle = interp_angle * 1j
                    np.exp(interp_angle,   out=new_model_grid[ai,ei,zi,azi,:])
                    new_model_grid[ai,ei,zi,azi,:] *= interp_ampltude


        ## used for converting e-field basis
        azimuth_angle_radians = azimuths[azi]/RTD
        sin_az = np.sin( azimuth_angle_radians )
        cos_az = np.cos( azimuth_angle_radians )

        ze_to_A = sin_az
        az_to_A = cos_az
        ze_to_B = cos_az
        az_to_B = -sin_az

        ## next we shift electric field angles to a different basis
        memory_tmp_Abasis[:,:,:] = new_model_grid[:,0,:,azi,:] ## zenithal component
        memory_tmp_Abasis *= ze_to_A
        memory_tmp_four[:,:,:] = new_model_grid[:,1,:,azi,:] ## azimuthal component
        memory_tmp_four *= az_to_A
        memory_tmp_Abasis += memory_tmp_four

        memory_tmp_Bbasis[:,:,:] = new_model_grid[:,0,:,azi,:] ## zenithal component
        memory_tmp_Bbasis *= ze_to_B
        memory_tmp_four[:,:,:] = new_model_grid[:,1,:,azi,:] ## azimuthal component
        memory_tmp_four *= az_to_B
        memory_tmp_Bbasis += memory_tmp_four

        new_model_grid[:, 0, :, azi, :] = memory_tmp_Abasis
        new_model_grid[:, 1, :, azi, :] = memory_tmp_Bbasis



    ## now fit with sph harms
    SphH_fitter = sph_harm_fit.spherical_harmonics_fitter( numSphHarm_orders, zeniths, azimuths, angle_units=1.0/RTD )
    num_SphH_params = SphH_fitter.get_num_params()
    num_functions = 2*2*num_new_freqs  ## need a function per antenna, per e-field, and per frequency
    out_array = np.empty( (num_SphH_params, 2, 2, num_new_freqs), dtype=complex )
    for ai in [0,1]:
        for ei in [0,1]:
            for fi in range(num_new_freqs):
                out_array[:, ai,ei,fi] = SphH_fitter.solve_numeric(  new_model_grid[ai, ei, :, :, fi]  )


    return SphHarm_antModel(out_array, new_freqs[0], new_freq_step)

def load_default_antmodel():
    return SphHarm_antModel.load_from_numpyArray( MetaData_directory+"/lofar/antenna_response_model/AARTFAAC_AVE_LBAOUTER_R700C17pF_SphHarm.npz" )
aartfaac_average_LBA_model = load_default_antmodel ## for backwards compatibility


class SphHarm_antModel:
    def __init__(self, weight_array, freq_grid_start, freq_grid_spacing):
        """
        Note this constructor is typically called by internals and not by users.
        end users should use make_AntModel_SphHarm_fit or load_default_antModel
        weight array should be a complex array of weights for spherical harmonics. See make_AntModel_SphHarm_fit internals
            weight_array should be complex with shape (N, 2,2, M) where N is number of Sph. Harm weights. M is number of frequencies
        freq_grid_start and freq_grid_spacing should describe the frequency sampleing in Hz
         """

        self.weight_array = weight_array
        self.num_freqs = weight_array.shape[3]

        self.freq_grid_start = freq_grid_start
        self.freq_grid_spacing = freq_grid_spacing

        self.temp_memory_a = np.empty((2,2,self.num_freqs), dtype=complex)
        self.temp_memory_b = np.empty((2,2,self.num_freqs), dtype=complex)
        self.temp_matrix = np.empty((2,2), dtype=float)


        self.weights_reshaped = self.weight_array.reshape( (weight_array.shape[0], -1) )
        self.temp_memory_a_reshaped = self.temp_memory_a.reshape( (-1) )
        self.temp_memory_b_reshaped = self.temp_memory_b.reshape( (-1) )


    def Jones_Matrices(self, frequencies, zenith, azimuth, invert=False, freq_fill=0.0, out=None):
        """
        if frequencies is numpy array in Hz, zenith and azimuth in degrees, than return numpy array of jones matrices
        if frequencies is numpy array in Hz, zenith and azimuth in degrees, than return numpy array of jones matrices
            return has shape (len(frequencies),2,2), dtype=complex. Is places in out if out is not None
        that when doted with [zenithal,azimuthal] component of incidident E-field, then will give [X,Y] voltages on dipoles.
        freq_fill is placed in J diagonals when frequencies is outside bounds.
        if invert is true, the jones_matrices are inverted before returning.
        """


        if zenith < 0:
            zenith = 0
        elif zenith > 90:
            zenith = 90

        while azimuth < 0:
            azimuth += 360
        while azimuth > 360:
            azimuth -= 360

        ## evaluate spherical harmonics
        sph_harm_fit.evaluate_SphHarms_oneAngle(self.weights_reshaped, zenith/RTD, azimuth/RTD, out=self.temp_memory_a_reshaped, memory=self.temp_memory_b_reshaped)


        ## now convert back to ze/az basis
        sin_az = np.sin( azimuth/RTD )
        cos_az = np.cos( azimuth/RTD )
        #
        self.temp_matrix[0,0] = sin_az
        self.temp_matrix[0,1] = cos_az
        self.temp_matrix[1,0] = cos_az
        self.temp_matrix[1,1] = -sin_az

        np.dot(self.temp_matrix, self.temp_memory_a[0,:,:], out=self.temp_memory_a[0,:,:])
        np.dot(self.temp_matrix, self.temp_memory_a[1,:,:], out=self.temp_memory_a[1,:,:])




        ## interpolate frequencies
        if out is not None:
            return_matrices = out
        else:
            return_matrices = np.empty((len(frequencies), 2, 2), dtype=complex)


        minF = self.freq_grid_start
        maxF = self.freq_grid_start + (self.num_freqs-1)*self.freq_grid_spacing
        for newF_i, newF in enumerate(frequencies):
            if (newF<minF) or (newF>maxF):
                return_matrices[newF_i,0,0] = freq_fill
                return_matrices[newF_i,0,1] = 0.0
                return_matrices[newF_i,1,0] = 0.0
                return_matrices[newF_i,1,1] = freq_fill
                continue

            T = (newF-self.freq_grid_start)/self.freq_grid_spacing
            I = int( T )
            T -= I

            J00_l = self.temp_memory_a[0,0,I]
            J00_h = self.temp_memory_a[0,0,I+1]
            J01_l = self.temp_memory_a[0,1,I]
            J01_h = self.temp_memory_a[0,1,I+1]
            J10_l = self.temp_memory_a[1,0,I]
            J10_h = self.temp_memory_a[1,0,I+1]
            J11_l = self.temp_memory_a[1,1,I]
            J11_h = self.temp_memory_a[1,1,I+1]

            J00 = J00_l + (J00_h-J00_l)*T
            J01 = J01_l + (J01_h-J01_l)*T
            J10 = J10_l + (J10_h-J10_l)*T
            J11 = J11_l + (J11_h-J11_l)*T

            if invert:
                det = J00*J11 - J01*J10

                t = J11
                J11 = J00/det
                J00 = J11/det

                J01 *= -1/det
                J10 *= -1/det

            return_matrices[newF_i,0,0] = J00
            return_matrices[newF_i,0,1] = J01
            return_matrices[newF_i,1,0] = J10
            return_matrices[newF_i,1,1] = J11

        return return_matrices

    def save_as_numpyArray(self, fname):
        np.savez_compressed(fname, weight_array=self.weight_array, freq_grid_start=self.freq_grid_start, freq_grid_spacing=self.freq_grid_spacing)

    @classmethod
    def load_from_numpyArray(cls, fname):
        input = np.load(fname)
        return SphHarm_antModel( input['weight_array'], input['freq_grid_start'], input['freq_grid_spacing'] )

    def save_to_OlafText(self):

        freq_grid = np.linspace(10, 90, num=int((90 - 10) / 1) + 1)
        zenith_grid = np.linspace(0, 90, num=int(90 / 5) + 1)
        azimuth_grid = np.linspace(0, 360, num=int(360 / 10) + 1)

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
                    jonesy = self.Jones_Matrices([f * 1e6], z, a + 45)
                    start = str(f) + ' ' + str(z) + ' ' + str(a) + ' '
                    J00_out.write(start + str(np.real(jonesy[0, 0, 0])) + ' ' + str(np.imag(jonesy[0, 0, 0])) + '\n')
                    J01_out.write(start + str(np.real(jonesy[0, 0, 1])) + ' ' + str(np.imag(jonesy[0, 0, 1])) + '\n')
                    J10_out.write(start + str(np.real(jonesy[0, 1, 0])) + ' ' + str(np.imag(jonesy[0, 1, 0])) + '\n')
                    J11_out.write(start + str(np.real(jonesy[0, 1, 1])) + ' ' + str(np.imag(jonesy[0, 1, 1])) + '\n')


# class calibrated_AARTFAAC_model:
#     """returns the AARTFAAC model multiplied by katies cal."""
#
#     def __init__(self):
#         self.AARTFAAC = aartfaac_average_LBA_model()
#
#         calibration = [42484.88872879, 41519.47733373, 39694.16854372, 38435.36963118,
#                        36974.13596039, 35819.34454985, 35072.53901876, 33960.74197721,
#                        32944.65142405, 32112.44688046, 31590.52174516, 30433.52586868,
#                        29756.92041985, 28880.95581826, 28204.37711364, 27646.83132496,
#                        27058.11219529, 26693.43614747, 25952.89684011, 25517.72346825,
#                        25220.5602153 , 24782.62398452, 24349.88724441, 23755.78027908,
#                        23289.45439997, 22800.30039329, 22176.17569116, 21619.22598549,
#                        21341.84386379, 21368.96227069, 22032.38713345, 22127.96304476,
#                        22678.25184787, 24670.14420208, 25703.00236409, 24792.9736945,
#                        23817.8752368 , 22912.1048524 , 22393.11267467, 21844.38535201,
#                        20831.10515113, 20220.27315072, 19223.64518551, 18099.23669312,
#                        17569.76228552, 16298.99230863, 14845.6897604 , 13488.79405579,
#                        11966.53709451, 10895.66530218,  9703.34502309]
#
#         cal_frequencies = np.arange(30e6, 80.5e6, 1e6)
#         self.calibration_interpolator = PchipInterpolator(cal_frequencies, calibration, extrapolate=True  )
#
#     def get_calibrator(self, frequencies):
#         return self.calibration_interpolator( frequencies )
#
#     def Jones_ONLY(self, frequencies, zenith, azimuth, freq_fill=1.0):
#         return self.AARTFAAC.Jones_Matrices( frequencies, zenith, azimuth, freq_fill )
#
#     def Jones_Matrices(self, frequencies, zenith, azimuth, freq_fill=1.0):
#         """return calibrated jones matrices"""
#         JM = self.AARTFAAC.Jones_Matrices( frequencies, zenith, azimuth, freq_fill )
#         C = self.get_calibrator( frequencies )
#         JM[:, 0,0] *= C
#         JM[:, 0,1] *= C
#         JM[:, 1,0] *= C
#         JM[:, 1,1] *= C
#         return JM
        
    
    
            
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



if __name__ == '__main__':
    ## this generates the default antenna function tables
    print('loading AARTFAAC model')
    aartfaac_model = aartfaac_LBA_model(model_loc=None, R=700, C=17e-12, mode="LBA_OUTER")
    print('generating model, averaging, and fitting sph. harm')
    sph_fit = aartfaac_model.get_average_antenna_model()
    print('saving')
    sph_fit.save_as_numpyArray(  MetaData_directory + "/lofar/antenna_response_model/AARTFAAC_AVE_LBAOUTER_R700C17pF_SphHarm.npz" )
    print('donesies!')

