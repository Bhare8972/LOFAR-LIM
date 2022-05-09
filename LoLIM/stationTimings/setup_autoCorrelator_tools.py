#!/usr/bin/env python3
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import numpy as np

from LoLIM.utilities import GSL_include, GSL_library_dir

ext = Extension("autoCorrelator_tools", ["autoCorrelator_tools.pyx"],
    include_dirs=[np.get_include(), 
                  GSL_include()],
    library_dirs=[GSL_library_dir()],
    libraries=["gsl", 'blas']
)
 
setup(ext_modules=[ext],
    cmdclass = {'build_ext': build_ext})

#setup(
#    name = "impulsive_imager_tools",
#    ext_modules = cythonize("impulsive_imager_tools.pyx", include_path = [np.get_include(), ]),
#    library_dirs=["/usr/local/lib/"],
#    libraries=["gsl"]
#    )
