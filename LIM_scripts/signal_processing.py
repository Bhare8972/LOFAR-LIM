#!/usr/bin/env python3

import numpy as np


from matplotlib import pyplot as plt

### need to move these functions here
from LoLIM.utilities import upsample_N_envelope, parabolic_fit, half_hann_window, num_double_zeros
    
    
def remove_saturation(data, half_hann_percent=0.1):
    """given some data, as a 1-D numpy array, remove areas where the signal saturates by multiplying with a half-hann filter. 
    Operates on input data"""
    
    print("remove saturation is not operational")
    
    plt.plot(data)
    plt.show()
