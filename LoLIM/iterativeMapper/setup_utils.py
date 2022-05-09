#!/usr/bin/env python3
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import numpy as np

from LoLIM.utilities import GSL_include, GSL_library_dir


ext = Extension("cython_utils", ["cython_utils.pyx"],
    include_dirs=[np.get_include(), 
                  GSL_include()],
    library_dirs=[GSL_library_dir()],
    libraries=["gsl", 'blas']
)
 
setup(ext_modules=[ext],
    cmdclass = {'build_ext': build_ext})