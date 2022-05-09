#!/usr/bin/env python3

from LoLIM.stationTimings.reCallibrate_pulses import recalibrate_pulse

input_folder = 'pulse_finding'
output_folder = 'pulse_finding_recal'
pulse = 'potSource_6.h5'

recalibrate_pulse(timeID="D20180809T141413.250Z", 
                  input_fname = input_folder+'/'+pulse, 
                  output_fname = output_folder+'/'+pulse,
                  polarization_flips="polarization_flips.txt")