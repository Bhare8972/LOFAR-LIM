#!/usr/bin/env python3

print("WARNING: read_pulse_data is being depreciated. Please import the correction file read functions from raw_tbb_IO in the future")

from LoLIM.IO.raw_tbb_IO import read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays