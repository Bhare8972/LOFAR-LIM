# LOFAR-LIM

This is a set of scripts used to analyze the data for the LOFAR-LIM (LOFAR for Lightning IMaging) project. At the moment it is poorly documented. Please email me when (not if) you have questions.

Basic usage:
  the contents of this module can be placed anywhere. However, the data is expected to be in a particular file structure. Since the data used by these scripts is mostly produced by other scripts, then the data are split between analysis, which is the output of some analysis. The only exception is that for raw data, but that isn't handled by any of the scripts in this package, yet. The structure of the data should be: "root_folder"/"year"/"time id"/"analysis". Where "analysis" is the name of the analysis folder, "time id" is a unique identifier for each LOFAR trigger (lightning flash), D20160712T173455.100Z is an example of a time ID. Year is the year the data was taken (2016 in the previous example), and "root_folder" is the folder that all this is placed in.
  
  In order for the scripts to know where the data is, the variable name "default_processed_data_loc" in utilities.py must be set to "root_folder", as discussed above. "default_raw_data_loc" is presently not used. 
  
  After processing the raw data (using a script not yet in this package), the found RF pulses are saved to pulse files (which is a type of analysis). These pulse files can be read by the "read_pulse_data.py" script. The pulses can be analyzed for single-station plane waves (SSPW) by FindNSave_SSPW.py. The resulting planewave analysis data can be read by planewave_functions.py. The pulses (or planewaves depending on method) can be grouped into point sources. The script read_PSE.py reads the resulting PSE analysis. 
  
  main_plotter.py is a plotting utility that can read these point sources and plot them LMA style. HOwever, it seems to have problems on different systems (I use python 3 on ubuntu). The bottom of the main_plotter.py script is a good example of how to use the read_PSE.py functions. Finally, antenna_responce.py includes useful tools for calculating and correcting for the frequency responce of the antennas. antenna_responce.py, however, requires aditional calibration data to work, this is included under the analysis name "cal_tables", and is provided by the initial raw data analysis, not included here.
  
 
