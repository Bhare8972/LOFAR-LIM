# LOFAR-LIM

This is a set of scripts used to analyze the data for the LOFAR-LIM (LOFAR for Lightning IMaging) project. At the moment it is poorly documented. Please email me when (not if) you have questions.

Basic usage:
  the contents of this module can be placed anywhere. However, the data is expected to be in a particular file structure. Since the data used by these scripts is mostly produced by other scripts, then the data are split between analysis, which is the output of some analysis. The only exception is that for raw data, but that isn't handled by any of the scripts in this package, yet. The structure of the data should be: "root_folder"/"year"/"time id"/"analysis". Where "analysis" is the name of the analysis folder, "time id" is a unique identifier for each LOFAR trigger (lightning flash), D20160712T173455.100Z is an example of a time ID. Year is the year the data was taken (2016 in the previous example), and "root_folder" is the folder that all this is placed in.
  
  In order for the scripts to know where the data is, the variable name "default_processed_data_loc" in utilities.py must be set to "root_folder", as discussed above. "default_raw_data_loc" is presently not used (no longer true, need to update this). 
  Furthermore, the file "data.tar.gz" should be unpacked, as it contains useful metadata used by the code. (such as antenna positions and such)
  
  After processing the raw data (using a script not yet in this package), the found RF pulses are saved to pulse files (which is a type of analysis). These pulse files can be read by the "read_pulse_data.py" script. The pulses can be analyzed for single-station plane waves (SSPW) by FindNSave_SSPW.py. The resulting planewave analysis data can be read by planewave_functions.py. The pulses (or planewaves depending on method) can be grouped into point sources. The script read_PSE.py reads the resulting PSE analysis. 
  
  main_plotter.py is a plotting utility that can read these point sources and plot them LMA style. HOwever, it seems to have problems on different systems (I use python 3 on ubuntu). The bottom of the main_plotter.py script is a good example of how to use the read_PSE.py functions. Finally, antenna_responce.py includes useful tools for calculating and correcting for the frequency responce of the antennas. antenna_responce.py, however, requires aditional calibration data to work, this is included under the analysis name "cal_tables", and is provided by the initial raw data analysis, not included here (some this is also no longer true and needs to be updated...).
  
 
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
