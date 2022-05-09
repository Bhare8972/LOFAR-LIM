#!/bin/sh
cython -a cython_beamforming_tools.pyx
python3 setup_utils.py build_ext --inplace
