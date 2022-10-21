#!/bin/sh
cython -a GSL_FFT.pyx
python3 setup_utils.py build_ext --inplace
