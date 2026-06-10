#!/usr/bin/env python3


import os
import datetime
import json

import numpy as np
import h5py

import LoLIM.IO.metadata as md
import LoLIM.utilities as util
import LoLIM.atmosphere as atmo


########  The following four functions read what I call "correction files" these are corrections made to improve the data ##########
## THESE are the "old" way of storing calibrations.  New best-practice is total-cal below ##

def read_antenna_pol_flips(fname):
    antennas_to_flip = []
    with open(fname) as fin:
        for line in fin:
            ant_name = line.split()[0]
            antennas_to_flip.append( ant_name )
    return antennas_to_flip

def read_bad_antennas(fname):
    bad_antenna_data = []
    
    def parse_line_v1(line):
        ant_name, pol = line.split()[0:2]
        # bad_antenna_data.append((ant_name,int(pol)))
        if pol: 
            bad_antenna_data.append(  util.even_antName_to_odd( ant_name )  )
        else:
            bad_antenna_data.append( ant_name )
    
    def parse_line_v2(line):
        ant_name = line.split()[0]
        bad_antenna_data.append(  ant_name  )
        # pol = 0
        # if not util.antName_is_even(ant_name):
        #     ant_name = util.even_antName_to_odd(ant_name)
        #     pol = 1
        # bad_antenna_data.append((ant_name,pol))
        
    version = 1
    with open(fname) as fin:
        is_line_0 = True
        for line in fin:
            if is_line_0 and line[:2] == 'v2':
                version = 2
            else:
                if version == 1:
                    parse_line_v1( line )
                elif version == 2:
                    parse_line_v2( line )
                
            if is_line_0:
                is_line_0 = False
                
                
            
    return bad_antenna_data

def read_antenna_delays(fname):
    additional_ant_delays = {}
    
    def parse_line_v1(line):
        ant_name, pol_E_delay, pol_O_delay = line.split()[0:3]
        additional_ant_delays[ant_name] = [float(pol_E_delay), float(pol_O_delay)]
        
    def parse_line_v2(line):
        ant_name, delay = line.split()[0:2]
        pol = 0
        if not util.antName_is_even(ant_name):
            ant_name = util.even_antName_to_odd(ant_name)
            pol = 1
            
        if ant_name not in additional_ant_delays:
            additional_ant_delays[ant_name] = [0.0, 0.0]
            
        additional_ant_delays[ant_name][pol] = float(delay)
    
    parse_function = parse_line_v1
    with open(fname) as fin:
        is_line_0=True
        for line in fin:
            if is_line_0 and line[0] == 'v':
                if line[:2]=='v1':
                    pass
                elif line[:2]=='v2':
                    parse_function = parse_line_v2
            else:
                parse_function(line)
                
            if is_line_0:
                is_line_0 = False
            
    return additional_ant_delays

def read_station_delays(fname):
    station_delays = {}
    with open(fname) as fin:
        for line in fin:
            sname, delay = line.split()[0:2]
            station_delays[sname] = float(delay)
    return station_delays







### this replaces the above four, and reads from one "ultimate" cal file
#def read_cal_file(fname, pol_flips_are_bad, timeID=None):

def read_cal_file(fname, pol_flips_are_bad=True, timeID=None):

    # station_delays = {}
    # bad_antenna_data = []
    # ant_delays = {}
    # polarization_flips = []
    # sign_flips = []

    if timeID is not None:
        dir = util.processed_data_dir( timeID )
        fname = os.path.join( dir, fname )



    RET = total_cal_object()
    
    with open(fname) as fin:
        version = fin.readline() ## version of type of file. Presently unused as we only have one version

        if version[0]=='O':
            return read_olaf_file(fname, pol_flips_are_bad, unCalAnts_are_bad=False)
        elif version[0]=='J':
            return total_cal_object.from_JSONable_object( json.load( fin ))


        mode = 1 ## 1 = bad_antennas 2 = pol_flips 3 = station_delays 4 = antenna_delays
        
        for line in fin:

            line_data = line.split()
            
            if len(line_data) == 0: # empyt line
                continue
            
            if line_data[0][0] == '#':
                ## comment!
                continue
        
            ## first check mode
            
            if line_data[0] == "MDA": ## metadata adjusts
                if line_data[1] == 'ANTENNA_SET':
                    RET.metadata_adjusts['ANTENNA_SET'] = line_data[2]
                    
            elif line_data[0] == "bad_antennas":
                mode = 1
            elif line_data[0] == "pol_flips":
                mode = 2
            elif line_data[0] == "station_delays":
                mode = 3
            elif line_data[0] == "antenna_delays":  ## these should be AFTER pol_flips. I.E., do not flip with pol_flips
                mode = 4
            elif line_data[0] == "sign_flips":
                mode = 5
            elif line_data[0] == "vair":
                v = float(line_data[1])
                util.set_vair( v )
                RET.atmosphere = atmo.simple_atmosphere( v )
                
            ### now we parse
            elif mode == 1 : ## bad antennas
                RET.bad_antenna_data.append( line_data[0] )
            elif mode == 2:
                if pol_flips_are_bad:
                    RET.bad_antenna_data.append( util.antName_to_even( line_data[0] ) )
                    RET.bad_antenna_data.append( util.antName_to_odd( line_data[0] ) )
                else:
                    RET.polarization_flips.append( util.antName_to_even( line_data[0] ) )
            elif mode == 3:
                RET.station_delays[ line_data[0] ] = float( line_data[1] )
            elif mode == 4:
                RET.ant_delays[ line_data[0] ] = float( line_data[1] )
            elif mode == 5:
                RET.sign_flips.append( line_data[0] )

            ## err
            else:
                print('reading cal file error! in mode:', mode, 'line:', line)
        
    # return bad_antenna_data, polarization_flips, station_delays, ant_delays
    return RET

class total_cal_object:
    def __init__(self):
        self.atmosphere = atmo.default_atmosphere
        self.bad_antenna_data = [] ## this is a list of stings, where each string is the name of a "bad" antenna
        self.polarization_flips = [] ## list of strings. Each string is an antenna that is poarlization flipeed with its polarization partner
        self.station_delays = {}     ## a dictionary with key of station name in string, value is float number. Is number of seconds of delay
        self.ant_delays = {}         ## a dictionary with key of antenna name in string, value is float number. Is number of seconds of delay
        self.sign_flips = []         ## list of strings. Each is an antenna that should be multiplied by a -1

        self.metadata_adjusts = {}   ## dictionary of strings and values of strings. 

    def to_JSONable_object(self):

        out_dict = {
        'atmo':self.atmosphere.to_JSONable_object(),
        'bad_antenna_data':self.bad_antenna_data,
        'polarization_flips':self.polarization_flips,
        'station_delays':self.station_delays,
        'ant_delays':self.ant_delays,
        'sign_flips':self.sign_flips,
        'metadata_adjusts':self.metadata_adjusts,
        }

        return out_dict

    @staticmethod 
    def from_JSONable_object( jsonable_dictionary ):
        ret_cal = total_cal_object()
        ret_cal.atmosphere = atmo.base_atmosphere.from_JSONable_object( jsonable_dictionary['atmo'] )
        ret_cal.bad_antenna_data = jsonable_dictionary['bad_antenna_data']
        ret_cal.polarization_flips = jsonable_dictionary['polarization_flips']
        ret_cal.station_delays = jsonable_dictionary['station_delays']
        ret_cal.ant_delays = jsonable_dictionary['ant_delays']
        ret_cal.sign_flips = jsonable_dictionary['sign_flips']
        ret_cal.metadata_adjusts = jsonable_dictionary['metadata_adjusts']

        return ret_cal

#### NOTE:  metadata_adjusts is a dictionary of fixes to the metadata.
  ## key is string, which is type of metadata to adjust
  ## value is the new metadata info. Depends of type.
  ## types for LOFAR1.0:
    ## ANTENNA_SET:   value is string ("LBA_OUTER", "LBA_SPARSE", etc.)  use this value for antenna set instead of that in file
  ## LOFAR2.0 currently has not metadata_adjusts





def read_olaf_file(fname, pol_flips_are_bad, unCalAnts_are_bad=True):
    """read Olaf's cal fill. HOWEVER: the cal file needs to be hand-modified to be read here. Ask Brian for details and hope he remembers"""

    def antSAI_to_antname( SAI ):
        if len(SAI)==9:
            return SAI

        ant_num = SAI[-3:]
        sname = SAI[:-3]
        sname = sname.zfill(3)
        RCU = str( int(int(ant_num)/8) ).zfill(3)
        return sname + RCU + ant_num

        
    
    RET = total_cal_object()
    RET.atmosphere = atmo.olaf_constant_atmosphere
    
    with open(fname) as fin:
        firstline = fin.readline() ## version of type of file. Presently unused as we only have one version
        firstline_data = firstline.split() ## first item is version
        if len(firstline_data) > 1:
            atmo_info = firstline_data[0]
            if atmo_info == 'HeightCorrectIndxRef':
                RET.atmosphere = atmo.olaf_varying_atmosphere
            elif atmo_info == 'ConstIndxRef':
                RET.atmosphere = atmo.olaf_constant_atmosphere ## may be redundeant




        mode = 1 ## 1 = antenna delays, 2 = station delays, 3 = bad antennas, 4 = sign flips 
        
        for line in fin:
            if len(line) == 0: # emptu line
                continue

            line_data = line.split()
            
          #  if line_data[0][0] == '#':
                ## comment!
          #      continue
        
            ## first check mode
            
            if line_data[0][0] == "=":
                mode = 2
            elif line_data[0] == "bad_antennas":
                mode = 3
            elif line_data[0] == "sign_flips":
                mode = 4
            #elif line_data[0] == "pol_flips":
            #    mode = 5
                
            ### now we parse
            elif mode == 1:
                antSAI, delay_samples, do = line_data
                antname = antSAI_to_antname( antSAI )

                if (not int(do)) and unCalAnts_are_bad:
                    RET.bad_antenna_data.append( antname )
                else:
                    RET.ant_delays[ antname ] = float( delay_samples )*5.0e-9


            elif mode == 2:
                RET.station_delays[ line_data[0] ] = float( line_data[1] )*5.0e-9


            elif mode == 3:
                antname = antSAI_to_antname( line_data[0] )
                RET.bad_antenna_data.append( antname )

            elif mode == 4:
                antname = antSAI_to_antname( line_data[0] )
                RET.sign_flips.append( antname )

            else:
                print('reading cal file error! in mode:', mode, 'line:', line)
        
    # return bad_antenna_data, polarization_flips, station_delays, ant_delays
    return RET


