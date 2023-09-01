
RANDOM NOTE:
to get matplotlib and pyqt5 to work together, sometimes you must install:
apt install libgl1-mesa-glx

a conda install enviroment can be:
conda create -n MyEnv --strict-channel-priority -c conda-forge -c anaconda python numpy matplotlib scipy h5py cython qt pyqt gsl


Here is a short description of some of the scripts:

utilities
	Includes a few constants, such as C and index of refraction of air
	also includes a few data processing utilities
	folder opening utilities (for finding the correct folder corresponding to a timeID)
	and an array to convert from station ID to station Name (and a dictionary to go back)	

metadata
	Includes a number of utilities to read antenna locations, phase callibrations, and other stuff from files

raw_tbb_IO.py
	tools for reading from raw data files. See the script for details.

get_phase_callibration
	Tool for downloading the phase callibration for files that don't have it. Shouldn't be often needed, as I (Brian 
	Hare) will generally do this when I make the data available

findRFI
	A tool to find human Radio interference based on phase stability.

run_findRFI
	runs findRFI on many stations, and saves the result to a python pickle file (as well as the plots)

plot_max_over_blocks
	plots the maximum of each block of data for each antenna over the full data file
	useful for finding the lightning flash and assesing the general data health
	also a good example on how to use the file IO

antenna_response
	Contains a tool for modeling the antenna response function, and for callibrating the antennas. This tool for 
	callibrating the antennas, however, still needs work.

