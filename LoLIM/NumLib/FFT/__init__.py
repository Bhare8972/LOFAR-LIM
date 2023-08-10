#!/usr/bin/env python3
""" Numpy/scipy FFT sucks. The essential problem is that they allocate new memory every time. Thus, this code wraps the GSL FFT functions for use in Python.
The cython code will eventually be adjusted to written to be imported into other cython applications. The LoLIM.FFT.complex_fft_obj wraps this, can automatically compile GSL,
and falls back on numpy if there is a problem. TEST.py can run some tests, and show how to use the complex_fft_obj object"""

import numpy as np

use_numpy = False
try:
    from .GSL_FFT import GSL_complex_FFT
except:
    print("cannot import GSL FFT. Trying to compile")

    ## ths is a dumb way to this, but I don't know a better way
    import subprocess
    R1 = ['cython', '-a', 'GSL_FFT.pyx']
    R2 = ['python3', 'setup_utils.py', 'build_ext', '--inplace'] ## this one is extra stupid

    try:
        subprocess.run(R1, capture_output=False, check=True)
        subprocess.run(R2, capture_output=False, check=True)
        from .GSL_FFT import GSL_complex_FFT
    except:
        print('cannot compile GSL FFT. Using Numpy instead')
        use_numpy = True


class complex_fft_obj:
    def __init__(self, N):
        self.N = N
        self.using_numpy = use_numpy

        if not self.using_numpy:
            self.GSL_OBJ = GSL_complex_FFT(N)

    def freqs(self, D):
        """WHere D is the 'time' between samples, return the frequencies of each FFT sample. Note this is the same for numpy and GSL."""
        return np.fft.fftfreq(self.N, D)

    def fft(self, DATA):
        if self.using_numpy:
            A = np.fft.fft(DATA)
            DATA[:] = A
        else:
            self.GSL_OBJ.do_FFT(DATA)

    def ifft(self, DATA):
        if self.using_numpy:
            A = np.fft.ifft(DATA)
            DATA[:] = A
        else:
            self.GSL_OBJ.do_IFFT(DATA)

