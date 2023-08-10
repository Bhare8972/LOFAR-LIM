#!/usr/bin/env python3
""" Numpy/scipy FFT sucks. The essential problem is that they allocate new memory every time. Thus, this code wraps the GSL FFT functions for use in Python.
The cython code will eventually be adjusted to written to be imported into other cython applications. The LoLIM.FFT.complex_fft_obj wraps this, can automatically compile GSL,
and falls back on numpy if there is a problem. TEST.py can run some tests, and show how to use the complex_fft_obj object"""

import numpy as np

## TODO: this should be expanded to allow for more options
##   different fitter options
##   finite difference jacobian options
##   maybe a callback?

use_numpy = False
try:
    from .GSL_LeastSquares import GSL_LeastSquares
except:
    print("cannot import GSL LeastSquares. Trying to compile")

    ## ths is a dumb way to this, but I don't know a better way
    import subprocess
    R1 = ['cython', '-a', 'GSL_LeastSquares.pyx']
    R2 = ['python3', 'setup_utils.py', 'build_ext', '--inplace'] ## this one is extra stupid

    try:
        subprocess.run(R1, capture_output=False, check=True)
        subprocess.run(R2, capture_output=False, check=True)
        from .GSL_LeastSquares import GSL_LeastSquares
    except:
        print('cannot compile GSL LeastSquares.')
