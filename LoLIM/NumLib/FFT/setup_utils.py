#!/usr/bin/env python3
from distutils.core import setup
from distutils.extension import Extension
from Cython import __version__
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import numpy as np

from LoLIM.utilities import GSL_include, GSL_library_dir


from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()

CT=[]

# CT =[('CYTHON_TRACE', '1')]
# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True
# directive_defaults['profile'] = True

print('cython version', __version__ )

# ext = Extension("cython_beamforming_tools", ["cython_beamforming_tools.pyx"],
#     include_dirs=[np.get_include(),
#                   GSL_include()],
#     library_dirs=[GSL_library_dir()],
#     libraries=["gsl", 'blas'],
#     define_macros=CT
# )
#
# setup(ext_modules=[ext],
#     cmdclass = {'build_ext': build_ext})

print()
print()

ext_CT = Extension("GSL_FFT", ["GSL_FFT.pyx"],
    include_dirs=[np.get_include(),
                  GSL_include()],
    library_dirs=[GSL_library_dir()],
    libraries=["gsl", "blas"],
    define_macros=CT
)

setup(ext_modules=[ext_CT],
    cmdclass = {'build_ext': build_ext})
 

