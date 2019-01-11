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

##external 
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator



##mine
from LoLIM.utilities import processed_data_dir, MetaData_directory, RTD
import LoLIM.transmogrify as tmf



## psuedo-code example use of calibrator
# AC = ant_calibrator( "D20160712T173455.100Z" )
#
# AC.FFT_prep( antenna_name, even_antenna_data,  odd_antnna_data ) ## note that ant_calibrator can be re-used for multiple data sets
# AC.apply_GalaxyCal()  ## this applies the relative callibration, and corrects for defficencies in the antenna model
# AC.unravelAntennaResponce(azimuth_degrees, elivation_degrees) # apply antenna model. Is better to NOT do this, and to apply anntenna model to E-field model instead
# out_1, out_2 = AC.get_result() ## out_1 is zenith component, out_2 is azimuthal component (if unravelAntennaResponce was called,  else is still even and odd antennas)
# out_1_hilbertEnvelope = np.abs( out_1 ) ## apply_GalaxyCal automatically applies a hilbert transform
# out_1_reals = np.real( out_1 )



class LBA_antenna_model:
    """a class encapsulating the antenna model for the Low Band Antennas."""
    
    def __init__(self):
        voltage_theta = np.loadtxt(MetaData_directory+"/lofar/antenna_response_model/LBA_Vout_theta.txt", skiprows=1)
        voltage_phi   = np.loadtxt(MetaData_directory+"/lofar/antenna_response_model/LBA_Vout_phi.txt", skiprows=1)

        voltage_theta_responce = voltage_theta[:, 3] + 1j*voltage_theta[:, 4]
        voltage_phi_responce = voltage_phi[:, 3] + 1j*voltage_phi[:, 4]
        
        freq_start = 10.0 * 1.e6
        freq_step = 1.0 * 1.e6
        num_freq = 101
        
        theta_start = 0.0
        theta_step = 5.0
        num_theta = 19
        
        phi_start = 0.0
        phi_step = 10.0
        num_phi = 37
        
        frequency_samples = np.arange(num_freq)*freq_step + freq_start
        theta_samples = np.arange(num_theta)*theta_step + theta_start
        phi_samples = np.arange(num_phi)*phi_step + phi_start
        
        voltage_theta_responce = voltage_theta_responce.reshape( (num_freq, num_theta, num_phi) )
        voltage_phi_responce = voltage_phi_responce.reshape( (num_freq, num_theta, num_phi) )
        
        self.theta_responce_interpolant = RegularGridInterpolator((frequency_samples, theta_samples, phi_samples),   voltage_theta_responce)
        self.phi_responce_interpolant =   RegularGridInterpolator((frequency_samples, theta_samples, phi_samples),   voltage_phi_responce)
        
    def JonesMatrix(self, frequency, zenith, azimuth):
        """return the Jones Matrix for a single frequency (in Hz), for a wave with a zenith and azimuth angle in degrees. Dot the jones matrix with the electric field vector, first component of vector is Zenith component of 
        electric field and second component is azimuthal electric field, then the first component of the resulting vector will be voltage on odd  (X) antenna and second component will be voltage on even (Y) antenna.
        Returns identity matrix where frequency is outside of 10 to 100 MHz"""
    
        jones_matrix = np.zeros( (2,2), dtype=complex )
        
        if frequency < 10.0E6 or frequency>100.0E6: ##if frequency is outside of range, then return some invertable nonsense
            jones_matrix[0,0] = 1.0
            jones_matrix[1,1] = 1.0
            return jones_matrix
        
        ## calculate for X dipole
        azimuth += 135 # put azimuth in coordinates of the X antenna
        while azimuth > 360: ## normalize the azimuthal angle
            azimuth -= 360
        while azimuth < 0:
            azimuth += 360
        jones_matrix[0, 0] = self.theta_responce_interpolant(  [frequency, zenith, azimuth] )
        jones_matrix[0, 1] = -1*self.phi_responce_interpolant( [frequency, zenith, azimuth] ) ## I don't really know why this -1 must be here
        
        ## calculate for Y dipole
        azimuth += 90.0 # put azimuth in coordinates of the Y antenna
        while azimuth > 360: ## normalize the azimuthal angle
            azimuth -= 360
        while azimuth < 0:
            azimuth += 360
        jones_matrix[1, 0] = -1*self.theta_responce_interpolant(  [frequency, zenith, azimuth] ) ## I don't really know why this -1 must be here
        jones_matrix[1, 1] = self.phi_responce_interpolant( [frequency, zenith, azimuth] )
        
        return jones_matrix
    
    def JonesMatrix_MultiFreq(self, frequencies, zenith, azimuth):
        """same as JonesMatrix, except that frequencies is expected to be an array. Returns an array of jones matrices"""
        
        out_JM = np.zeros( (len(frequencies), 2,2), dtype=complex )
        
        good_frequencies = np.logical_and( frequencies>10.0E6, frequencies<100E6)
        num_freqs = np.sum( good_frequencies )
        
        points = np.zeros( (num_freqs, 3) )
        points[:, 0] = frequencies[ good_frequencies ]
        points[:, 1] = zenith
        
        ## calculate for X dipole
        points[:, 2] = azimuth + 135 # put azimuth in coordinates of the X antenna
        while np.any( points[:, 2] > 360 ): ## normalize the azimuthal angle
            points[:, 2] [ points[:, 2]>360 ] -= 360
        while np.any( points[:, 2] <0 ): 
            points[:, 2] [ points[:, 2]<0 ] += 360
            
        out_JM[good_frequencies, 0, 0] = self.theta_responce_interpolant( points )
        out_JM[good_frequencies, 0, 1] = -1*self.phi_responce_interpolant( points )
        
        ## calculate for Y dipole
        points[:, 2] += 90.0 # put azimuth in coordinates of the Y antenna
        while np.any( points[:, 2] > 360 ): ## normalize the azimuthal angle
            points[:, 2] [ points[:, 2]>360 ] -= 360
        while np.any( points[:, 2] <0 ): 
            points[:, 2] [ points[:, 2]<0 ] += 360
            
        out_JM[good_frequencies, 1, 0] = -1*self.theta_responce_interpolant( points )
        out_JM[good_frequencies, 1, 1] = self.phi_responce_interpolant( points )
        
        ## set the frequencies outide 10 to 100 MHz to just identity matix
        out_JM[ np.logical_not(good_frequencies), 0, 0] = 1.0
        out_JM[ np.logical_not(good_frequencies), 1, 1] = 1.0
        
        fi = np.argmin( np.abs(frequencies-60.0E6) )
        
        return out_JM
    
def get_LBA_frequency_calibration(frequencies):
    """ given a set of frequencies, in units of Hz, return the antenna callibration"""
    
    Calibration_curve = np.zeros(101)
    
#    Calibration_curve[29:82] = np.array([0,  1.09124663e-06,   1.11049910e-06,   1.11101995e-06,
#                 1.14234774e-06,   1.15149299e-06,   1.17121699e-06,
#                 1.18578121e-06,   1.19696124e-06,   1.20458122e-06,
#                 1.24675978e-06,   1.27966600e-06,   1.32418333e-06,
#                 1.32115453e-06,   1.33871075e-06,   1.34295545e-06,
#                 1.34157430e-06,   1.37660390e-06,   1.39226359e-06,
#                 1.39827006e-06,   1.51409426e-06,   1.61610247e-06,
#                 1.74643510e-06,   1.74588169e-06,   1.73061463e-06,
#                 1.69229172e-06,   1.64633321e-06,   1.60982965e-06,
#                 1.59572009e-06,   1.64618678e-06,   1.81628916e-06,
#                 2.09520281e-06,   2.17610590e-06,   2.20907337e-06,
#                 2.12050148e-06,   2.04923844e-06,   2.06549879e-06,
#                 2.24906987e-06,   2.40356459e-06,   2.52199062e-06,
#                 2.48380048e-06,   2.40835417e-06,   2.38248922e-06,
#                 2.48599834e-06,   2.60617662e-06,   2.66466169e-06,
#                 2.78010597e-06,   2.90548503e-06,   3.08686745e-06,
#                 3.26101312e-06,   3.50261561e-06,   3.74739666e-06, 0])  
    
    Calibration_curve[29:82] = np.array([0,  
         5.17441583458e-07,  5.26112551988e-07,   5.54435685779e-07,
         5.85016839251e-07,  6.10870583894e-07,   6.47309035485e-07,
         6.5091456311e-07,   6.92446769829e-07,   7.24758841756e-07,
         7.68263947385e-07,  7.91148165708e-07,   8.40734630391e-07,
         8.43296509798e-07,  8.9313176062e-07,    9.2381731899e-07,
         9.49238145305e-07,  9.75552986024e-07,   9.90712789075e-07,
         1.05134677402e-06,  1.07555421962e-06,   1.09540733982e-06,
         1.11101225721e-06,  1.13901998747e-06,   1.20840289547e-06,
         1.25099501879e-06,  1.28820016316e-06,   1.42132795525e-06,
         1.58730515709e-06,  1.69274022328e-06,   1.79771448679e-06,
         1.69009894816e-06,  1.59916753151e-06,   1.52598688512e-06,
         1.35488689628e-06,  1.25281147938e-06,   1.25843707081e-06,
         1.28115364631e-06,  1.29327138907e-06,   1.30804348155e-06,
         1.30108509354e-06,  1.30377069039e-06,   1.30654052835e-06,
         1.30565358077e-06,  1.31712312807e-06,   1.31343342371e-06,
         1.32267337872e-06,  1.33943924503e-06,   1.36389183621e-06,
         1.4202852471e-06,   1.45672364953e-06,   1.5184892913e-06, 0])
    
    Calibration_curve_interp = interp1d(np.linspace(0.e6,100e6,101), Calibration_curve, kind='linear', bounds_error=False, fill_value=0.0)
    return Calibration_curve_interp( frequencies )
    
            
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

def getGalaxyCalibrationData(antenna_noise_power, timestamp, channel_width):
    """get the data, from galaxy-dominated noise, that is necisary to do a Galaxy calibration on data, at a later state"""
    
    longitude = 6.869837540/RTD
    
    
    coefficients_lba = [ np.array([ 0.01489468, -0.00129305,  0.00089477, -0.00020722, -0.00046507]),
                         np.array([ 0.01347391, -0.00088765,  0.00059822,  0.00011678, -0.00039787])  ]

    
    
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
    
    even_factors = antenna_noise_power[0::2]
    odd_factors = antenna_noise_power[1::2]
    
    even_factors = galactic_noise_power[0] * channel_width /even_factors
    odd_factors = galactic_noise_power[1] * channel_width /odd_factors
    
    np.sqrt(even_factors, out=even_factors)
    np.sqrt(odd_factors, out=odd_factors)
    
    
    return even_factors, odd_factors


class LBA_ant_calibrator:
    """ This is a class for callibrating the LBA antennas and removing the antenna responce function. Only valid between 30 to 80 MHz. NOTE: removing the antenna responce function is ill-conditioned. A better approach is to callibrate the data, 
    then filter a model using the antenna responce. Using this class will inherently do a hilbert transform (negative frequencies will be set to zero)"""
    
    def __init__(self, timeID=None, findRFI=None):
        
        if findRFI is None:
            findRFI = "/findRFI/findRFI_results"
            
        if isinstance(findRFI, str): ## load findRFI data from file
            galcal_data_loc = processed_data_dir(timeID) + findRFI
            with open( galcal_data_loc, 'rb' ) as fin:
                findRFI = load(fin)
                
        
        self.calibration_factors = {}
        for findRFI_info in findRFI.values():
            antenna_names = findRFI_info["antenna_names"]
            num_antennas = len( antenna_names )
            cleaned_power = findRFI_info["cleaned_power"]
            timestamp = findRFI_info["timestamp"]
            analyzed_blocksize = findRFI_info["blocksize"]
            
            even_cal_factors, odd_cal_factors = getGalaxyCalibrationData(cleaned_power,  timestamp, 5.0E-9/analyzed_blocksize  )
            
            for ant_i in range(0, int(num_antennas/2)):
                ant_name = antenna_names[ ant_i*2 ]
                self.calibration_factors[ ant_name ] = (even_cal_factors[ant_i], odd_cal_factors[ant_i])
            
        
        ### Galaxy callibration data ###
#        self.calibration_factors = {}
#        galcal_data_loc = processed_data_dir(timeID) + '/cal_tables/galaxy_cal'
#        galcal_fpaths = glob.glob(galcal_data_loc + '/*.gcal')
#        for fpath in galcal_fpaths:
#            with open(fpath, 'rb') as fin:
#                data = np.load(fin)
#                ant_names = data["arr_0"].astype(str, copy=False)
#                factors = data["arr_1"]
#            ant_i = 0
#            while ant_i<len(ant_names):
#                self.calibration_factors[ ant_names[ant_i] ] = [factors[ant_i], factors[ant_i+1]]
#                ant_i += 2
                
        self.antenna_model = LBA_antenna_model()
        
    def FFT_prep(self, even_ant_name, even_pol_data, odd_pol_data):
        """ prepare to apply callibrations to a pair of dipoles. Essentually just takes FFT. Assume a han window is not needed."""
        
        if len(even_pol_data) != len(odd_pol_data):
            raise ValueError('even and odd polarization data need to be same length')
            
        self.even_ant_name = even_ant_name
        self.N_points = len(even_pol_data)
        
        ##FFT
        self.even_pol_FFT = np.fft.fft(even_pol_data)
        self.odd_pol_FFT =  np.fft.fft(odd_pol_data)
        
        ##Frequencies
        self.frequencies = np.fft.fftfreq(self.N_points, 5.0E-9)
#    
    def apply_time_shift(self, even_time_shift, odd_time_shift):
        """apply phase shifts to the data, corresponding to an amount of time (in seconds)"""
        self.even_pol_FFT *= np.exp( self.frequencies*(-1j*2*np.pi*even_time_shift) )
        self.odd_pol_FFT *= np.exp( self.frequencies*(-1j*2*np.pi*odd_time_shift) )
        
    def apply_GalaxyCal(self):
        """applies the galaxy calibration to this data. Can be applied independantly of unravelAntennaResponce. Negative frequencies are set to zero."""
        
        ### first we apply the correction factors ###
        PolE_factor, PolO_factor = self.calibration_factors[self.even_ant_name]

        self.even_pol_FFT *= PolE_factor
        self.odd_pol_FFT  *= PolO_factor
        
        ## callibration curve to correct for frequency dependent model defeciencies 
        Calibration_curve_interp = get_LBA_frequency_calibration( self.frequencies )
        
        self.even_pol_FFT *= Calibration_curve_interp
        self.odd_pol_FFT  *= Calibration_curve_interp

        return [PolE_factor, PolO_factor]
        
    def unravelAntennaResponce(self, zenith, azimuth):
        """given a direction to source (azimuth off X and zenith from Z, in degrees ), if call this function, then apply_GalaxyCal MUST also be applied to the data
        Note that this function assumes the data is LBA_outer, which has flipped polarizations compared to LBA inner"""
        
        jones_matrices = self.antenna_model.JonesMatrix_MultiFreq(self.frequencies, zenith, azimuth)
        
        inverse_jones_matrix = invert_2X2_matrix_list( jones_matrices )
        
        ### apply the Jones matrix.  Note that the polarities (even and odd) are flipped)
        zenith_component = self.odd_pol_FFT*inverse_jones_matrix[:, 0,0] +  self.even_pol_FFT*inverse_jones_matrix[:, 0,1]
        azimuth_component = self.odd_pol_FFT*inverse_jones_matrix[:, 1,0] +  self.even_pol_FFT*inverse_jones_matrix[:, 1,1]
        
        self.even_pol_FFT = zenith_component
        self.odd_pol_FFT = azimuth_component
        
    def getResult(self):
        """get the results of analysis. Essentially preforms inverse FFT. 
        If unravelAntennaResponce was called, first return is zenith component, second is azimuthal. Else first return is even polarization, second is odd"""
        
        return np.fft.ifft(self.even_pol_FFT),    np.fft.ifft(self.odd_pol_FFT)
        
    
    
    
        
def plot_responce():
    N_azimuth = 100
    N_zenith = 500
    frequency = 58.0E6
    
    zeniths = np.linspace( 0, 90, N_zenith)
    azimuths = np.linspace( 0, 360, N_azimuth )
    
    resulting_grid = np.zeros((N_zenith, N_azimuth))
    AM = LBA_antenna_model()
#    AM = pycrtools_antenna_model()
    
    for ze_i in range(N_zenith):
        print(ze_i, '/', N_zenith)
        for az_i in range(N_azimuth):
            
            JM = AM.JonesMatrix(frequency, zeniths[ze_i], azimuths[az_i])
            resulting_grid[ze_i, az_i] = np.abs( JM[0,0] )
            
    plt.imshow(resulting_grid)
    plt.show()
        
        
if __name__ == "__main__":
    frequencies_MHz = np.linspace(0.0, 100.0, 10000)
    
    AM = LBA_antenna_model()
    matrix = AM.JonesMatrix_MultiFreq(frequencies_MHz*1E6, 0.0, 0.0)
    
    plt.semilogy(frequencies_MHz, np.abs(matrix[:, 0,0]), linewidth=7 )
    plt.show()
    
    
    
    
        

