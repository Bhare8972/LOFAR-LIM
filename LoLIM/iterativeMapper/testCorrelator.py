#!/usr/bin/env python3

import numpy as np  
from matplotlib import pyplot as plt 

from cython_utils import parabolic_fitter, autoadjusting_upsample_and_correlate

class aPulse_cls:
    def __init__(self, t0):
        self.t0 = t0 
        self.amp = 1.0
        self.gauss_width = 10e-9

        self.center_Freq = 60e6


    def sample(self, T):
        tmp = T-self.t0
        S = np.exp( tmp*(2j*np.pi*self.center_Freq ) )
        tmp *= tmp
        tmp *= -1/(2*self.gauss_width*self.gauss_width)
        np.exp(tmp, out=tmp)

        return tmp*S


if __name__ == "__main__":

    pulse1 = aPulse_cls( t0=0e-9 )
    pulse2 = aPulse_cls( t0=2e-9 )

    sample_time = 5.0e-9
    time_simulate = 100e-9
    num_samples = int(time_simulate/sample_time)

    pulse1_startT =  int( (pulse1.t0 - 50e-9)/sample_time)*sample_time
    pulse1_T = np.arange(  num_samples  )*sample_time + pulse1_startT
    pulse1_data = pulse1.sample( pulse1_T )

    pulse2_startT = int( (pulse2.t0 - 50e-9)/sample_time)*sample_time - (5*sample_time)
    pulse2_T = np.arange(  num_samples+10  )*sample_time + pulse2_startT
    pulse2_data = pulse2.sample( pulse2_T )

    plt.plot(pulse1_T/(1e-9), np.abs(pulse1_data), c='r')
    plt.plot(pulse2_T/(1e-9), np.abs(pulse2_data), c='g')
    plt.show()


    upsample_factor = 4
    correlator = autoadjusting_upsample_and_correlate( min_data_length=1, upsample_factor=4 )
    correlator.set_referance( pulse1_data )
    cross_corelation = correlator.correlate( pulse2_data )
    abs_CC = np.abs(cross_corelation)


    peak_fitter = parabolic_fitter()
    CC_peak = peak_fitter.fit( abs_CC )

    plotable_CC_peak = CC_peak
    if CC_peak > len(abs_CC)/2:
        CC_peak -= len(abs_CC)
    
    peak_location = CC_peak*sample_time/upsample_factor + pulse2_startT - pulse1_startT

    print()
    print()
    print('CC-peak sample:', plotable_CC_peak, '(',CC_peak/upsample_factor,')' , 'T[ns]:', (CC_peak*sample_time/upsample_factor)/1e-9 )
    print('measured CC time offset [ns]:', peak_location/(1e-9))
    print('    correct [ns]:', (pulse2.t0 - pulse1.t0)/(1e-9) )
    print()
    print()


    plt.plot( abs_CC )
    plt.plot( np.real(cross_corelation) )
    plt.axvline( plotable_CC_peak )
    plt.show()



