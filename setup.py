#!/usr/bin/env python3

from setuptools import setup
from setuptools.command.install import install
import subprocess


# class InstallLocalPackage(install):
#      def run(self):
#          install.run(self)
#          subprocess.call(
# #             "python3 setup_utils.py build_ext --inplace", shell=True
#              "bash ./LoLIM/build_allCython.sh", shell=True
#          )

#exec(open('version.py').read())
setup(
   name='LoLIM',
   version='1.0',
   description='LOFAR Lightning Imaging',
   author='Brian Hare',
   author_email='',
   packages=['LoLIM', 'LoLIM.IO', 'LoLIM.GLP', 'LoLIM.iterativeMapper', 'LoLIM.NumLib', 'LoLIM.NumLib.FFT', 'LoLIM.NumLib.LeastSquares', 'LoLIM.pol_beamforming', 'LoLIM.stationTimings','LoLIM.pipeline',
              'LoLIM.pipeline.cythonTools'],
   install_requires=['numpy', 'scipy', 'matplotlib', 'h5py', 'cython'], ##pyqt, GSL, pandas
   package_data={"LoLIM": ["data/*", "data/lofar/*", 
                           "data/lofar/antenna_response_model/*",
                           "data/lofar/StaticMetaData/*", "data/lofar/StaticMetaData/AntennaArrays/*", "data/lofar/StaticMetaData/CableDelays/*", "data/lofar/StaticMetaData/AntennaFields/*",
                           "data/lofar/station_clock_offsets/*"],

                  "LoLIM.NumLib.FFT" :["GSL_FFT.pyx"],
                  "LoLIM.NumLib.LeastSquares" :["GSL_LeastSquares.pyx"],
                  "LoLIM.iterativeMapper" :["cython_utils.pyx"],
                  "LoLIM.stationTimings" :["autoCorrelator_tools.pyx"],
                  "LoLIM.pipeline.cythonTools" :["cythonPipelineTools.pyx"],
               },
   include_package_data=True,

   #namespace_packages=['NumLib/FFT']
   #cmdclass={ 'install': InstallLocalPackage },  ## this is to force cython packages to build.... maybe a better way
)

