#!/usr/bin/env python3
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import numpy as np


ext = Extension("impulsive_imager_tools", ["impulsive_imager_tools.pyx"],
    include_dirs=[np.get_include(), 
                  "/usr/local/include/"],
    library_dirs=["/usr/local/lib/"],
    libraries=["gsl"]
)
 
setup(ext_modules=[ext],
    cmdclass = {'build_ext': build_ext})

ext2 = Extension("impulsive_imager_tools_2", ["impulsive_imager_tools_2.pyx"],
    include_dirs=[np.get_include(), 
                  "/usr/local/include/"],
    library_dirs=["/usr/local/lib/"],
    libraries=["gsl"]
)



#setup(
#    name = "impulsive_imager_tools",
#    ext_modules = cythonize("impulsive_imager_tools.pyx", include_path = [np.get_include(), ]),
#    library_dirs=["/usr/local/lib/"],
#    libraries=["gsl"]
#    )