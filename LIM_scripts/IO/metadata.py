#!/usr/bin/env python3

"""This module reads in calibration metadata from file in the early fases of LOFAR. In the future this should be replaced by reading the metadata from the files.

.. moduleauthor:: Sander ter Veen <s.terveen@astro.ru.nl>

Modified by Brian Hare for use with LOFAR for Lightning Imaging
"""

## Imports
import numpy as np
import struct

from LoLIM.utilities import SId_to_Sname, latlonCS002, RTD, MetaData_directory

#### first some simple utility functions ###

def mapAntennasetKeyword(antennaset):
    """Ugly fix to map correct antenna names in input to wrong antenna names
    for metadata module.
    """

    # Strip whitespace
    antennaset = antennaset.strip()
    
    allowed = ["LBA_OUTER", "LBA_INNER", "LBA_X", "LBA_Y", "HBA", "HBA_0", "HBA_1"]
    incorrect = {'LBA_INNER': 'LBA_INNER',
                  'LBA_OUTER': 'LBA_OUTER',
                  'HBA_ZERO': 'HBA_0',
                  'HBA_ONE': 'HBA_1',
                  'HBA_DUAL': 'HBA',
                  'HBA_JOINED': 'HBA',
				  'HBA_ZERO_INNER': 'HBA_0', # Only true for core stations
				  'HBA_ONE_INNER': 'HBA_1',  # Only true for core stations
				  'HBA_DUAL_INNER': 'HBA',   # Only true for core stations
				  'HBA_JOINED_INNER': 'HBA'} # Only true for core stations

    if antennaset in incorrect:
        antennaset = incorrect[antennaset]
    elif antennaset == "HBA_BOTH":
        # This keyword is also wrong but present in file headers
        print( "Keyword " + antennaset + " does not comply with ICD, mapping...")
        antennaset = "HBA"
    
    assert antennaset in allowed

    return antennaset
    
def make_antennaID_filter(antennaIDs):
    """For a list of antennaIDs, return a filter to filter data by antenna.
    example use:
        getStationPhaseCalibration("CS001", "LBA_OUTER")[ make_antennaID_filter(["002000001"]) ]
        
    note: Only works for one station at a time. Assumes that the array you want to filter includes ALL antennas in the appropriate antenna set"""
        
    RCU_id = np.array([int(ID[-3:]) for ID in antennaIDs])
    return RCU_id
    





#### read callibration data ###

#def getStationPhaseCalibration(station, antennaset, file_location=None):
#    """Read phase calibration data for a station.
#
#    Required arguments:
#
#    ================== ====================================================
#    Parameter          Description
#    ================== ====================================================
#    *station*          station name (as str) or ID.
#    *mode*             observation mode.
#    ================== ====================================================
#    
#    returns weights for 512 subbands.
#
#    Examples::
#
#        >>> metadata.getStationPhaseCalibration("CS002","LBA_OUTER")
#        array([[ 1.14260161 -6.07397622e-18j,  1.14260161 -6.05283530e-18j,
#             1.14260161 -6.03169438e-18j, ...,  1.14260161 +4.68675289e-18j,
#             1.14260161 +4.70789381e-18j,  1.14260161 +4.72903474e-18j],
#           [ 0.95669876 +2.41800591e-18j,  0.95669876 +2.41278190e-18j,
#             0.95669876 +2.40755789e-18j, ...,  0.95669876 -2.41017232e-19j,
#             0.95669876 -2.46241246e-19j,  0.95669876 -2.51465260e-19j],
#           [ 0.98463207 +6.80081617e-03j,  0.98463138 +6.89975906e-03j,
#             0.98463069 +6.99870187e-03j, ...,  0.98299670 +5.71319125e-02j,
#             0.98299096 +5.72306908e-02j,  0.98298520 +5.73294686e-02j],
#           ...,
#           [ 1.03201290 +7.39535744e-02j,  1.03144532 +8.14880844e-02j,
#             1.03082273 +8.90182487e-02j, ..., -0.82551740 -6.23731331e-01j,
#            -0.82094046 -6.29743206e-01j, -0.81631975 -6.35721497e-01j],
#           [ 1.12370332 -1.15296909e-01j,  1.12428451 -1.09484545e-01j,
#             1.12483564 -1.03669252e-01j, ..., -0.92476286 +6.48703460e-01j,
#            -0.92810503 +6.43912711e-01j, -0.93142239 +6.39104744e-01j],
#           [ 1.10043006 -6.18995646e-02j,  1.10075250 -5.58731668e-02j,
#             1.10104193 -4.98450938e-02j, ..., -1.01051042 +4.40052904e-01j,
#            -1.01290481 +4.34513198e-01j, -1.01526883 +4.28960464e-01j]])
#
#        >>> metadata.getStationPhaseCalibration(122,"LBA_OUTER")
#        Calibration data not yet available. Returning 1
#        array([[ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
#           [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
#           [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
#           ...,
#           [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
#           [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
#           [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j]])
#
#    """
#
#    # Return mode nr depending on observation mode
#    antennasetToMode = {"LBA_OUTER": "1",
#                         "LBA_INNER": "3",
#                         "HBA": "5",
#                         "HBA_0": "5",
#                         "HBA_1": "5",
#                         }
#
#    antennaset = mapAntennasetKeyword( antennaset )
#
#    if antennaset not in antennasetToMode.keys():
#        raise KeyError("Not a valid antennaset " + antennaset)
#
#    modenr = antennasetToMode[antennaset]
#    if not isinstance(station, str):
#        # Convert a station id to a station name
#        station = SId_to_Sname[station]
#
#    stationNr = station[2:]
#
#    # filename
#    if file_location is None:
#        file_location = MetaData_directory + '/lofar/StaticMetaData/CalTables'
#
#    filename = file_location + '/CalTable_' + stationNr + '_mode' + modenr + '.dat'
#    with open(filename, 'rb') as fin:
#        # Test for header record above raw data - present in newer caltables (starting 2012)
#        line = fin.readline().decode()
#        if 'HeaderStart' in line:
#            while not 'HeaderStop' in line:
#                line = fin.readline().decode()
#        else:  # no header present, seek to starting position
#            fin.seek(0)
#            
#        data = np.fromfile(fin, dtype=np.double)
#        
#    data.resize(512, 96, 2)
#
#    complexdata = np.empty(shape=(512, 96), dtype=complex)
#    complexdata.real = data[:, :, 0]
#    complexdata.imag = data[:, :, 1]
#
#    return complexdata.transpose()

def getStationPhaseCalibration(station, antennaset, file_location=None):
    """Read phase calibration data for a station.

    Required arguments:

    ================== ====================================================
    Parameter          Description
    ================== ====================================================
    *station*          station name (as str) or ID.
    *mode*             observation mode.
    ================== ====================================================
    
    returns weights for 512 subbands.

    Examples::

        >>> metadata.getStationPhaseCalibration("CS002","LBA_OUTER")
        array([[ 1.14260161 -6.07397622e-18j,  1.14260161 -6.05283530e-18j,
             1.14260161 -6.03169438e-18j, ...,  1.14260161 +4.68675289e-18j,
             1.14260161 +4.70789381e-18j,  1.14260161 +4.72903474e-18j],
           [ 0.95669876 +2.41800591e-18j,  0.95669876 +2.41278190e-18j,
             0.95669876 +2.40755789e-18j, ...,  0.95669876 -2.41017232e-19j,
             0.95669876 -2.46241246e-19j,  0.95669876 -2.51465260e-19j],
           [ 0.98463207 +6.80081617e-03j,  0.98463138 +6.89975906e-03j,
             0.98463069 +6.99870187e-03j, ...,  0.98299670 +5.71319125e-02j,
             0.98299096 +5.72306908e-02j,  0.98298520 +5.73294686e-02j],
           ...,
           [ 1.03201290 +7.39535744e-02j,  1.03144532 +8.14880844e-02j,
             1.03082273 +8.90182487e-02j, ..., -0.82551740 -6.23731331e-01j,
            -0.82094046 -6.29743206e-01j, -0.81631975 -6.35721497e-01j],
           [ 1.12370332 -1.15296909e-01j,  1.12428451 -1.09484545e-01j,
             1.12483564 -1.03669252e-01j, ..., -0.92476286 +6.48703460e-01j,
            -0.92810503 +6.43912711e-01j, -0.93142239 +6.39104744e-01j],
           [ 1.10043006 -6.18995646e-02j,  1.10075250 -5.58731668e-02j,
             1.10104193 -4.98450938e-02j, ..., -1.01051042 +4.40052904e-01j,
            -1.01290481 +4.34513198e-01j, -1.01526883 +4.28960464e-01j]])

        >>> metadata.getStationPhaseCalibration(122,"LBA_OUTER")
        Calibration data not yet available. Returning 1
        array([[ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
           [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
           [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
           ...,
           [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
           [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j],
           [ 1.+0.j,  1.+0.j,  1.+0.j, ...,  1.+0.j,  1.+0.j,  1.+0.j]])

    """

    # Return mode nr depending on observation mode
    antennasetToMode = {"LBA_OUTER": "LBA_OUTER-10_90",
                         "LBA_INNER": "LBA_INNER-10_90",
                         "HBA": "HBA-110_190",
                         "HBA_0": "HBA-110_190",
                         "HBA_1": "HBA-110_190",
                         }

    antennaset = mapAntennasetKeyword( antennaset )

    if antennaset not in antennasetToMode.keys():
        raise KeyError("Not a valid antennaset " + antennaset)

    mode_name = antennasetToMode[antennaset]
    if not isinstance(station, str):
        # Convert a station id to a station name
        station = SId_to_Sname[station]

    stationNr = station[2:]

    # filename
    if file_location is None:
        file_location = MetaData_directory + '/lofar/StaticMetaData/CalTables'

    filename = file_location + '/CalTable-' + stationNr + '-' + mode_name + '.dat'
    with open(filename, 'rb') as fin:
        # Test for header record above raw data - present in newer caltables (starting 2012)
        line = fin.readline().decode()
        if 'HeaderStart' in line:
            while not 'HeaderStop' in line:
                line = fin.readline().decode()
        else:  # no header present, seek to starting position
            fin.seek(0)
            
        data = np.fromfile(fin, dtype=np.double)
        
    data.resize(512, 96, 2)

    complexdata = np.empty(shape=(512, 96), dtype=complex)
    complexdata.real = data[:, :, 0]
    complexdata.imag = data[:, :, 1]

    return complexdata.transpose()

def convertPhase_to_Timing(phase_calibration, sample_time=5.0e-9):
    """Given the phase calibration of the 512 LOFAR subbands, such as the output of getStationPhaseCalibration, return the timing callibration of each antenna. 
    Not sure how well this works with HBA antennas. Sample time should be seconds per sample. Default is 5 ns"""
    phases = np.angle(phase_calibration)
    delays = (phases[:, 1] - phases[:, 0]) * (1024 / (2*np.pi)) * sample_time ## this just finds the slope from the first two points. Are there better methods?
    ### TODO: add a conditional that takes different points if the slope is too large
    return delays
    


#def getStationGainCalibration(station, antennaset, file_location=None):
#    """Read phase calibration data for a station.
#
#    Required arguments:
#
#    ================== ====================================================
#    Parameter          Description
#    ================== ====================================================
#    *station*          station name or ID.
#    *mode*             observation mode.
#    ================== ====================================================
#
#    Optional arguments:
#
#    ================== ====================================================
#    Parameter          Description
#    ================== ====================================================
#    *return_as_hArray* Default False
#    ================== ====================================================
#
#    returns one gain per RCU. This gain is calculated using the absolute
#    value from the CalTables assuming these are not frequency dependent.
#    This seems to be true in current (2013-08) tables.
#    """
#
#    cal = getStationPhaseCalibration(station, antennaset, file_location)
#
#    gain = np.abs(cal[:,0])
#    return gain



#### information about cable lengths and delays ####

def getCableDelays(station, antennaset):
    """ Get cable delays in seconds.

    Required arguments:

    ================== ====================================================
    Parameter          Description
    ================== ====================================================
    *station*          Station name or ID e.g. "CS302", 142
    *antennaset*       Antennaset used for this station. Options:

                       * LBA_INNER
                       * LBA_OUTER
                       * LBA_X
                       * LBA_Y
                       * LBA_SPARSE0
                       * LBA_SPARSE1
                       * HBA_0
                       * HBA_1
                       * HBA

    ================== ====================================================

    returns "array of (rcus * cable delays ) for all dipoles in a station"

    """

    # Check station id type
    if not isinstance(station, str):
        # Convert a station id to a station name
        station = SId_to_Sname[station]
        
    antennaset = mapAntennasetKeyword( antennaset ) 
        
    if "LBA_OUTER" == antennaset:
        rcu_connection = "LBL"
    elif "LBA_INNER" == antennaset:
        rcu_connection = "LBH"
    elif antennaset in ['HBA', "HBA_1", "HBA_0"]:
        rcu_connection = "HBA"
    else:
        raise KeyError("Not a valid antennaset " + antennaset)

    cabfilename = MetaData_directory + '/lofar/StaticMetaData/CableDelays/' + station + '-CableDelays.conf'
    cabfile = open(cabfilename)

    cable_delays = np.zeros(96)

    str_line = ''
    while "RCUnr" not in str_line:
        str_line = cabfile.readline()
        if len(str_line) == 0:
            # end of file reached, no data available
            assert False

    str_line = cabfile.readline()
    for i in range(96):
        str_line = cabfile.readline()
        sep_line = str_line.split()
        if rcu_connection == "LBL":
            cable_delays[int(sep_line[0])] = float(sep_line[2]) * 1e-9
        elif rcu_connection == "LBH":
            cable_delays[int(sep_line[0])] = float(sep_line[4]) * 1e-9
        elif rcu_connection == "HBA":
            cable_delays[int(sep_line[0])] = float(sep_line[6]) * 1e-9

    return cable_delays

def getCableLength(station,antennaset):
    
    # Check station id type
    if not isinstance(station, str):
        # Convert a station id to a station name
        station = SId_to_Sname[station]
        
    antennaset = mapAntennasetKeyword( antennaset ) 
        
    if "LBA_OUTER" == antennaset:
        rcu_connection = "LBL"
    elif "LBA_INNER" == antennaset:
        rcu_connection = "LBH"
    elif antennaset in ['HBA', "HBA_1", "HBA_0"]:
        rcu_connection = "HBA"
    else:
        raise KeyError("Not a valid antennaset " + antennaset)

    cabfilename = MetaData_directory + '/lofar/StaticMetaData/CableDelays/' + station + '-CableDelays.conf'
    cabfile = open(cabfilename)

    cable_length = np.zeros(96)

    str_line = ''
    while "RCUnr" not in str_line:
        str_line = cabfile.readline()
        if len(str_line) == 0:
            # end of file reached, no data available
            assert False

    str_line = cabfile.readline()
    for i in range(96):
        str_line = cabfile.readline()
        sep_line = str_line.split()
        if rcu_connection == "LBL":
            cable_length[int(sep_line[0])] = float(sep_line[1])
        elif rcu_connection == "LBH":
            cable_length[int(sep_line[0])] = float(sep_line[3])
        elif rcu_connection == "HBA":
            cable_length[int(sep_line[0])] = float(sep_line[5])

    return cable_length

def antennaset2rcumode(antennaset,filter):
    antennaset = mapAntennasetKeyword( antennaset ) 
    rcumode=dict()
    rcumode[('LBA_INNER','LBA_10_90')]=1
    rcumode[('LBA_OUTER','LBA_10_90')]=2
    rcumode[('LBA_INNER','LBA_30_90')]=3
    rcumode[('LBA_OUTER','LBA_30_90')]=4
    rcumode[('HBA','HBA_110_190')]=5
    rcumode[('HBA','HBA_170_230')]=6
    rcumode[('HBA','HBA_210_250')]=7
    if 'HBA' in antennaset:
        antennaset='HBA'
    return rcumode[(antennaset,filter)]

def getCableAttenuation(station,antennaset,filter=None):

    cable_length = getCableLength(station, antennaset)

    attenuationFactor=dict()
    attenuationFactor[1]=-0.0414#{50:-2.05,80:-3.32,85:-3.53,115:-4.74,130:-5.40}
    attenuationFactor[2]=attenuationFactor[1]
    attenuationFactor[3]=attenuationFactor[1]
    attenuationFactor[4]=attenuationFactor[1]
    attenuationFactor[5]=-0.0734#{50:-3.64,80:-5.87,85:-6.22,115:-8.53,130:-9.52}
    attenuationFactor[6]=-0.0848#{50:-4.24,80:-6.82,85:-7.21,115:-9.70,130:-11.06}
    attenuationFactor[7]=-0.0892#{50:-4.46,80:-7.19,85:-7.58,115:-10.18,130:-11.61}
    if filter==None:
        if 'LBA' in antennaset:
            filter='LBA_30_90'
        else:
            print( "Please specify the filter!")
            filter='HBA_110_190'
    rcumode=antennaset2rcumode(antennaset,filter)
    att=attenuationFactor[rcumode]
    return cable_length*att











#### functions for antenna and station location #####

def getItrfAntennaPosition(station, antennaset):
    """Returns the antenna positions of all the antennas in the station
    in ITRF coordinates for the specified antennaset.
    station can be the name or id of the station. 
    
    Required arguments:

    =================== ==============================================
    Parameter           Description
    =================== ==============================================
    *station*           Name or id of the station. e.g. "CS302" or 142
    *antennaset*        Antennaset used for this station. Options:

                        * LBA_INNER
                        * LBA_OUTER
                        * LBA_X
                        * LBA_Y
                        * LBA_SPARSE0
                        * LBA_SPARSE1
                        * HBA_0
                        * HBA_1
                        * HBA

    =================== ==============================================

    """
    # Check station id type
    if isinstance(station, int):
        # Convert a station id to a station name
        station = SId_to_Sname[station]
        
    antennaset = mapAntennasetKeyword( antennaset ) 
    
    if "LBA" in antennaset:
        antennatype = "LBA"
    elif "HBA" in antennaset:
        antennatype = "HBA"
    

    # Obtain filename of antenna positions
    filename = MetaData_directory + "/lofar/StaticMetaData/AntennaFields/" + station + "-AntennaField.conf"

    # Open file
    f = open(filename, 'r')

    if station[0:2] != "CS":
        if "HBA" in antennaset:
            antennaset = "HBA"

    # Find position of antennaset in file
    str_line = ''
    while antennatype != str_line.strip():
        str_line = f.readline()
        if len(str_line) == 0:
            # end of file reached, no data available
            assert False

    # Find the location of the station. Antenna locations are relative to this
    str_line = f.readline()
    str_split = str_line.split()
    stationX = float(str_split[2])
    stationY = float(str_split[3])
    stationZ = float(str_split[4])

    str_line = f.readline()

    # Get number of antennas and the number of directions
    nrantennas = int(str_line.split()[0])
    nrdir = int(str_line.split()[4])

    antenna_positions = np.empty((2*nrantennas, nrdir), dtype=np.double)
    for i in range(nrantennas):
        line = f.readline().split()
        
        antenna_positions[2*i, 0] = float(line[0]) + stationX
        antenna_positions[2*i, 1] = float(line[1]) + stationY
        antenna_positions[2*i, 2] = float(line[2]) + stationZ
        
        antenna_positions[2*i+1, 0] = float(line[3]) + stationX
        antenna_positions[2*i+1, 1] = float(line[4]) + stationY
        antenna_positions[2*i+1, 2] = float(line[5]) + stationZ

    if antennatype == "LBA":
        # There are three types of feed
        # H for HBA
        # h for lbh
        # l for lbl
        feed = {}
        feed["CS"] = {}
        feed["RS"] = {}
        feed["DE"] = {}
        feed["CS"]["LBA_SPARSE_EVEN"] = "24llhh"
        feed["CS"]["LBA_SPARSE_ODD"] = "24hhll"
        feed["CS"]["LBA_X"] = "48hl"
        feed["CS"]["LBA_Y"] = "48lh"
        feed["CS"]["LBA_INNER"] = "96h"
        feed["CS"]["LBA_OUTER"] = "96l"
        feed["RS"]["LBA_SPARSE_EVEN"] = "24llhh"
        feed["RS"]["LBA_SPARSE_ODD"] = "24hhll"
        feed["RS"]["LBA_X"] = "48hl"
        feed["RS"]["LBA_Y"] = "48lh"
        feed["RS"]["LBA_INNER"] = "96h"
        feed["RS"]["LBA_OUTER"] = "96l"
        feed["DE"]["LBA"] = "192h"
        if station[0:2] == "CS" or "RS":
            feedsel = feed[station[0:2]][antennaset]
            nrset = int(feedsel.split('l')[0].split('h')[0].split('H')[0])
            feeds = ''
            feedsel = feedsel[len(str(nrset)):]
            for i in range(nrset):
                feeds += feedsel

        indexselection = []
        for i in range(len(feeds)):
            if feeds[i] == 'l':
                # The 'l' feeds are the last 96 numbers of the total list
                indexselection.append(i + 96)
            elif feeds[i] == 'h':
                # The 'h' feeds are the first 96 numbers of the total list
                indexselection.append(i)
            else:
                # This selection is not yet supported
                assert False
        antenna_positions = antenna_positions[indexselection]

    return antenna_positions

def getStationPositions(station, antennaset, coordinatesystem):
    """Returns the antenna positions of all the antennas in the station
    relative to the station center for the specified antennaset.
    station can be the name or id of the station. 

    Required arguments:

    ================== ==============================================
    Argument           Description
    ================== ==============================================
    *station*          Name or id of the station. e.g. "CS302" or 142
    *antennaset*       Antennaset used for this station. Options:

                       * LBA_INNER
                       * LBA_OUTER
                       * LBA_X
                       * LBA_Y
                       * LBA_SPARSE0
                       * LBA_SPARSE1
                       * HBA_0
                       * HBA_1
                       * HBA

    *coordinatesystem* WGS84 or ITRF

    output: 
        if coordinatesystem == "WGS84", then return [lat, lon, alt] as a numpy array
        else if coordinatesystem=="ITRF", then return [X, Y, Z] as a numpy array

    """

    # Check if requested antennaset is known
    assert coordinatesystem in ["WGS84", 'ITRF']
    
    # Check station id type
    if isinstance(station, int):
        # Convert a station id to a station name
        station = SId_to_Sname[station]
        
    antennaset = mapAntennasetKeyword( antennaset ) 

    # Obtain filename of antenna positions
    if 'WGS84' in coordinatesystem:
        filename = MetaData_directory + "/lofar/StaticMetaData/AntennaArrays/" + station + "-AntennaArrays.conf"
    else:
        filename = MetaData_directory + "/lofar/StaticMetaData/AntennaFields/" + station + "-AntennaField.conf"

    # Open file
    f = open(filename, 'r')
    
    if "LBA" in antennaset:
        antennaset = "LBA"

    if station[0:2] != "CS":
        if "HBA" in antennaset:
            antennaset = "HBA"

    # Find position of antennaset in file
    str_line = ''
    while antennaset != str_line.strip():
        str_line = f.readline()
        if len(str_line) == 0:
            # end of file reached, no data available
            print( "Antenna set not found in calibration file", filename)
            return None

    # Skip name and station reference position
    str_line = f.readline().split()
    
    A = float(str_line[2]) ## lon in WGS84, X in ITRF
    B = float(str_line[3]) ## lat in WGS84, Y in ITRF
    C = float(str_line[4]) ## alt in WGS84, Z in ITRF

    return np.array([A,B,C])

ITRFCS002 = getStationPositions('CS002', 'LBA_OUTER', coordinatesystem='ITRF')  # ($LOFARSOFT/data/lofar/StaticMetaData/AntennaFields/CS002-AntennaField.conf)

def convertITRFToLocal(itrfpos, phase_center=ITRFCS002, reflatlon=latlonCS002, out=None):
    """
    ================== ==============================================
    Argument           Description
    ================== ==============================================
    *itrfpos*          an ITRF position as 1D numpy array, or list of positions as a 2D array
    *phase_center*     the origin of the coordinate system, in ITRF. Default is CS002.
    *reflatlon*        the rotation of the coordinate system. Is the [lat, lon] (in degrees) on the Earth which defines "UP"
    
    Function returns a 2D numpy array (even if input is 1D).
    Out cannot be same array as itrfpos
    """
    if out is itrfpos:
        print("out cannot be same as itrfpos in convertITRFToLocal. TODO: make this a real error")
        quit()
    
    lat = reflatlon[0]/RTD
    lon = reflatlon[1]/RTD
    arg0 = np.array([-np.sin(lon),   -np.sin(lat) * np.cos(lon),   np.cos(lat) * np.cos(lon)])
    arg1 = np.array([np.cos(lon) ,   -np.sin(lat) * np.sin(lon),   np.cos(lat) * np.sin(lon)])
    arg2 = np.array([         0.0,    np.cos(lat),                 np.sin(lat)])
    
    if out is None:
        ret = np.empty(itrfpos.shape, dtype=np.double )
    else:
        ret = out
    
    ret[:]  = np.outer(itrfpos[...,0]-phase_center[0], arg0 )
    ret += np.outer(itrfpos[...,1]-phase_center[1], arg1 )
    ret += np.outer(itrfpos[...,2]-phase_center[2], arg2 )
    
    return ret
    
def geoditic_to_ITRF(latLonAlt):
    """for a latLonAlt in degrees, (can be list of three numpy arrays), convert to ITRF coordinates. Using information at: https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Geodetic_to/from_ENU_coordinates and 
    http://itrf.ensg.ign.fr/faq.php?type=answer"""
    
    a = 6378137.0  #### semi-major axis, m
    e2 = 0.00669438002290 ## eccentricity squared
    
    def N(lat):
        return a/np.sqrt( 1 - e2*(np.sin(lat)**2) )
    
    lat = latLonAlt[0]/RTD
    lon = latLonAlt[1]/RTD
    
    X = ( N(lat) + latLonAlt[2] ) *np.cos(lat) *np.cos(lon)
    Y = ( N(lat) + latLonAlt[2] ) *np.cos(lat) *np.sin(lon)

    b2_a2 =  1-e2
    Z = ( b2_a2*N(lat) + latLonAlt[2] ) *np.sin(lat)
    
    return np.array( [X,Y,Z] )

from scipy.optimize import least_squares
def ITRF_to_geoditic(ITRF_XYZ):
    """given an ITRF XYZ  in meters, return lat, lon, and altitude in degrees and meters. Uses geoditic_to_ITRF with least_squares minimization. Only works with one set of coordinates at a time (no numpy arrays) """
    
#    a = 6378137.0  #### semi-major axis, m
#    e2 = 0.00669438002290 ## eccentricity squared
#    me2 = 1-e2
#    b2 = a*a*me2 ## semi-minor axis squared
#    
#    r2 = ITRF_XYZ[0]*ITRF_XYZ[0]
#    r2 += ITRF_XYZ[1]*ITRF_XYZ[1]
#    r = np.sqrt(r2)
#    Z2 = ITRF_XYZ[2]*ITRF_XYZ[2]
#    
#    F = Z2*(54*b2)
#    G = r2 + me2*Z2 - e2*(a*a - b2)
#    c = e2*e2*F*r2/(G*G*G)
#    s = np.cbrt( 1 + c + np.sqrt(c*c + 2*c) )
#    P = G*(1+s+1/s)
#    P = F/(3*P*P)
#    Q = np.sqrt( 1 + 2*e2*e2*P )
#    
#    Ro = -P*e2*r/(1+Q) + np.sqrt( a*a*(1+1/Q)*0.5 - P*me2*Z2/(Q*(1+Q)) - P*r2*0.5 )
#    N = r-e2*Ro
#    U = np.sqrt( N*N + Z2 )
#    V = np.sqrt( N*N + me2*Z2 )
#    zo = b2*ITRF_XYZ[2]/(a*V)
#    
#    h = U*(1-b2/(a*V))
#    lat = np.arctan( (ITRF_XYZ[2]+e2*zo)/r )*RTD
#    lon = np.arctan2(ITRF_XYZ[1], ITRF_XYZ[0])*RTD
#    return lat, lon, h
    
    class minimizer:
        def __init__(self, X,Y,Z):
            self.X = X
            self.Y = Y
            self.Z = Z
        def __call__(self, lat_lon_h):
            ret =  geoditic_to_ITRF( lat_lon_h )
            ret[0] -= self.X
            ret[1] -= self.Y
            ret[2] -= self.Z
            return ret
    
    guess = np.append( latlonCS002, [0.0] )
    ret = least_squares(minimizer(ITRF_XYZ[0], ITRF_XYZ[1], ITRF_XYZ[2]), guess, x_scale='jac' )
    return ret.x
    
    


#### previously known clock offsets. Only used for compatibility with past data ####
def getClockCorrectionFromParsetAddition():
    parsetFilename = MetaData_directory + '/lofar/station_clock_offsets/StationCalibration.parset'
    
    offsetDictX = {}
    offsetDictY = {}
    
    infile = open(parsetFilename, 'r')
    for line in infile:
        s = line.split('=')
        value = s[1]
        params = s[0].split('.')
        thisStation = params[2][0:5]
        thisAntennaSet = params[3]
        thisFilter = params[4]
        thisValueType = params[5]
        thisPolarization = params[6][0]

        if thisAntennaSet == 'LBA_OUTER' and thisFilter == 'LBA_30_90' and thisValueType == 'delay':
            if thisPolarization == 'X':
                offsetDictX[thisStation] = float(value)
            elif thisPolarization == 'Y':
                offsetDictY[thisStation] = float(value)
            else:
                raise ValueError('Wrong!')
    infile.close()
    
    offsetDictCombined = {}
    
    for key in offsetDictX.keys():
        combined = 0.5 * (offsetDictX[key] + offsetDictY[key])
        offsetDictCombined[key] = combined
    
    return offsetDictCombined


def getClockCorrections( antennaset="LBA", time=1383264000+1000):
    """Get clock correction for superterp stations in seconds. Currently static values.

    *station* Station name or number for which to get the correction.
    *time* Optional. Linux time of observation. As clocks drift the value from the correct time should be given. Not yet implemented.
    """

    clockcorrection = {}   
    if "LBA" in antennaset:
        if time < (1383264000):
            # Values before 1 Nov 2013, eventID-time 120960000, Unix time: add 1262304000.
            clockcorrection["CS002"] = 8.32233e-06 # definition, global offset
            # Addition is the finetuning using Smilde from 1 or 2 random events, to about +/- 0.2 ns. Need to check constancy over time.
            clockcorrection["CS003"] = 6.921444e-06 + 0.35e-9
            clockcorrection["CS004"] = 7.884847e-06 + 1.0e-9
            clockcorrection["CS005"] = 8.537828e-06 + 0.14e-9
            clockcorrection["CS006"] = 7.880705e-06 - 0.24e-9
            clockcorrection["CS007"] = 7.916458e-06 - 0.22e-9

            clockcorrection["CS001"] = 4.755947e-06
            clockcorrection["CS011"] = 7.55500e-06 - 0.3e-9
            clockcorrection["CS013"] = 9.47910e-06
            clockcorrection["CS017"] = 1.540812e-05 - 0.87e-9
            clockcorrection["CS021"] = 6.044335e-06 + 1.12e-9
            clockcorrection["CS024"] = 4.66335e-06 - 1.24e-9
            clockcorrection["CS026"] = 1.620482e-05 - 1.88e-9
            clockcorrection["CS028"] = 1.6967048e-05 + 1.28e-9
            clockcorrection["CS030"] = 9.7110576e-06 + 3.9e-9
            clockcorrection["CS031"] = 6.375533e-06 + 1.87e-9
            clockcorrection["CS032"] = 8.541675e-06 + 1.1e-9
            clockcorrection["CS101"] = 1.5155471e-05
            clockcorrection["CS103"] = 3.5503206e-05
            clockcorrection["CS201"] = 1.745439e-05
            clockcorrection["CS301"] = 7.685249e-06
            clockcorrection["CS302"] = 1.2317004e-05
            clockcorrection["CS401"] = 8.052200e-06
            clockcorrection["CS501"] = 1.65797e-05
        else:
            clockcorrection = getClockCorrectionFromParsetAddition()
            clockcorrection["CS003"] = clockcorrection["CS003"] - 1.7e-9 + 2.0e-9
            clockcorrection["CS004"] = clockcorrection["CS004"] - 9.5e-9 + 4.2e-9
            clockcorrection["CS005"] = clockcorrection["CS005"] - 6.9e-9 + 0.4e-9
            clockcorrection["CS006"] = clockcorrection["CS006"] - 8.3e-9 + 3.8e-9
            clockcorrection["CS007"] = clockcorrection["CS007"] - 3.6e-9 + 3.4e-9
            clockcorrection["CS011"] = clockcorrection["CS011"] - 18.7e-9 + 0.6e-9
            
# Old values were
    elif "HBA" in antennaset:
        # Correct to 2013-03-26 values from parset L111421
        clockcorrection["CS001"] = 4.759754e-06
        clockcorrection["CS002"] = 8.318834e-06
        clockcorrection["CS003"] = 6.917926e-06
        clockcorrection["CS004"] = 7.889961e-06
        clockcorrection["CS005"] = 8.542093e-06
        clockcorrection["CS006"] = 7.882892e-06
        clockcorrection["CS007"] = 7.913020e-06
        clockcorrection["CS011"] = 7.55852e-06
        clockcorrection["CS013"] = 9.47910e-06
        clockcorrection["CS017"] = 1.541095e-05
        clockcorrection["CS021"] = 6.04963e-06
        clockcorrection["CS024"] = 4.65857e-06
        clockcorrection["CS026"] = 1.619948e-05
        clockcorrection["CS028"] = 1.6962571e-05
        clockcorrection["CS030"] = 9.7160576e-06
        clockcorrection["CS031"] = 6.370090e-06
        clockcorrection["CS032"] = 8.546255e-06
        clockcorrection["CS101"] = 1.5157971e-05
        clockcorrection["CS103"] = 3.5500922e-05
        clockcorrection["CS201"] = 1.744924e-05
        clockcorrection["CS301"] = 7.690431e-06
        clockcorrection["CS302"] = 1.2321604e-05
        clockcorrection["CS401"] = 8.057504e-06
        clockcorrection["CS501"] = 1.65842e-05

    else:
        print( "ERROR: no clock offsets available for this antennaset: ", antennaset)
        return 0

    return clockcorrection
    