#!/usr/bin/env python3

"""This is code to plot the traces used in a IPSE. Note that it is not really *correct*. This code should be moved into the IPSE class (under interferometry), and then called as a method."""

import numpy as np
import matplotlib.pyplot as plt

import h5py

from LoLIM.utilities import log, processed_data_dir, v_air, SId_to_Sname, Sname_to_SId_dict, RTD, even_antName_to_odd
from LoLIM.interferometry import read_interferometric_PSE as R_IPSE


timeID = "D20170929T202255.000Z"
input_folder = "interferometry_out4"#4_tstNORMAL"interferometry_out4_sumLog

IPSE_index = 55700
block = int(IPSE_index/100)

processed_data_folder = processed_data_dir(timeID)
data_dir = processed_data_folder + "/" + input_folder
interferometry_header, IPSE_list = R_IPSE.load_interferometric_PSE( data_dir, blocks_to_open=[block] )

IPSE = R_IPSE.get_IPSE(IPSE_list, IPSE_index)

print(IPSE.unique_index)
print("XYZT:", IPSE.XYZT)
print("intensity:", IPSE.intensity, "S1 S2 distance:", IPSE.S1_S2_distance)
print("amplitude:", IPSE.amplitude)

IPSE.plot()