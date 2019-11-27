#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


from LoLIM.utilities import processed_data_dir
from LoLIM.interferometry import read_interferometric_PSE as R_IPSE

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_processed_data_loc = "/home/brian/processed_files"

#timeID = "D20180813T153001.413Z"
#input_folder = "interferometry_out2"

timeID = "D20170929T202255.000Z"
input_folder = "interferometry_out4"

#IPSE_index = 240253 ## 0.974 note hard zeros on both sides of pulse. Image is very wide

#IPSE_index = 241195 # 0.980 Looks like noise. Image is very wide

#IPSE_index = 239901 # 0.975 Clean pulses that line up-okay. Image is more reasonable.

#IPSE_index = 240468 # 0.972 Event is particularly noisy...Sometimes peaks appear to line-up. 

#IPSE_index = 239865 # 0.887 Trace looks like someone threw-up on the antenna. Image is not sharp, seems to be at a local, not global minima

#IPSE_index = 240783 # 0.854 nonsencical. Image is not great


#IPSE_index = 20606
#IPSE_index = 240700
IPSE_index = 33708

block = int(IPSE_index/100)

processed_data_folder = processed_data_dir(timeID)
data_dir = processed_data_folder + "/" + input_folder
interferometry_header, IPSE_list = R_IPSE.load_interferometric_PSE( data_dir, blocks_to_open=[block] )

IPSE = R_IPSE.get_IPSE(IPSE_list, IPSE_index)

print(IPSE.unique_index)
print("XYZT:", IPSE.XYZT)
print("intensity:", IPSE.intensity, "S1 S2 distance:", IPSE.S1_S2_distance)
print("amplitude:", IPSE.amplitude)
print("peak index:", IPSE.peak_index + IPSE.block_start_index)

IPSE.plot()