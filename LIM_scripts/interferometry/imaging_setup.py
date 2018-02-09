#!/usr/bin/env python3

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
#    Extension("primes", ["primes.pyx"],
#        include_dirs = [...],
#        libraries = [...],
#        library_dirs = [...]),
#    # Everything but primes.pyx is included here.
    Extension("*", ["*.pyx"],
    define_macros=[('CYTHON_TRACE', '1')],
              ),
#        include_dirs = [...],
#        libraries = [...],
#        library_dirs = [...]),
    
]

setup(
    name = "Interferometric Imager",
    ext_modules = cythonize(extensions, compiler_directives={'linetrace': True}),
    )