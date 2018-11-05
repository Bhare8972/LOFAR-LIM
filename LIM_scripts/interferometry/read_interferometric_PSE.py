#!/usr/bin/env python3
import os
import h5py

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from LoLIM.utilities import processed_data_dir

class interferometry_header:
    class antenna_data_object:
        def __init__(self, ant_i, h5_group):
            self.antenna_index = int(ant_i)
            self.name = h5_group.attrs["antenna_name"]
            self.location = h5_group.attrs["location"]
            self.timing_delay = h5_group.attrs["timing_delay"]
            self.data_offset = h5_group.attrs["data_offset"]
            self.half_window_length = h5_group.attrs["half_window_length"]
            self.station_antenna_i = h5_group.attrs["station_antenna_i"]
            self.station = h5_group.attrs["station"]
    
    def __init__(self, h5_header_group):
        self.h5_group = h5_header_group
        
        self.bounding_box = self.h5_group.attrs["bounding_box"]
        self.pulse_length = self.h5_group.attrs["pulse_length"]
        self.num_antennas_per_station = self.h5_group.attrs["num_antennas_per_station"]
        self.stations_to_exclude = self.h5_group.attrs["stations_to_exclude"]
        self.stations_to_exclude = [stat.decode() for stat in self.stations_to_exclude] ## convert binary to strings
        
        self.do_RFI_filtering = self.h5_group.attrs["do_RFI_filtering"]
        self.block_size = self.h5_group.attrs["block_size"]
        self.upsample_factor = self.h5_group.attrs["upsample_factor"]
        self.max_events_perBlock = self.h5_group.attrs["max_events_perBlock"]
        self.hann_window_fraction = self.h5_group.attrs["hann_window_fraction"]
        
        self.bad_antennas = self.h5_group.attrs["bad_antennas"] 
        self.bad_antennas = [[name.decode(),int(pol.decode())] for name,pol in self.bad_antennas]
        self.pol_flips = self.h5_group.attrs["polarization_flips"]
        self.pol_flips = [name.decode() for name in self.pol_flips]
        
        self.antenna_data = [ None ]*len(self.h5_group)
        for ant_i, ant_group in self.h5_group.items():
            new_ant_data = self.antenna_data_object(ant_i, ant_group)
            self.antenna_data[new_ant_data.antenna_index] = new_ant_data
            
            
        self.prefered_station_name = self.h5_group.attrs["prefered_station_name"]
        self.trace_length_stage2 = self.h5_group.attrs["trace_length_stage2"]
            
    def get_station_names(self):
        station_names = []
        for ant in self.antenna_data:
            if ant.station not in station_names:
                station_names.append( ant.station )
                
        return station_names
    
    def antenna_info_by_station(self):
        station_ant_info = {}
        for ant in self.antenna_data:
            if ant.station not in station_ant_info:
                station_ant_info[ant.station] = []
            station_ant_info[ant.station].append( ant )
            
        return station_ant_info
    
        
    
class Ipse:
    def __init__(self, block_index, ID, h5_PSE_dataset, block_group):
        self.block_index = block_index
        self.IPSE_index = ID
        self.file_dataset = h5_PSE_dataset
        
        self.unique_index = h5_PSE_dataset.attrs["unique_index"]
        
        self.loc = h5_PSE_dataset.attrs["loc"]
        self.T = h5_PSE_dataset.attrs["T"]
        self.peak_index = h5_PSE_dataset.attrs["peak_index"]
        self.intensity = h5_PSE_dataset.attrs["intensity"]
        self.stage_1_success = h5_PSE_dataset.attrs["stage_1_success"]
        
        self.stage_1_num_itters = h5_PSE_dataset.attrs["stage_1_num_itters"]
        self.amplitude = h5_PSE_dataset.attrs["amplitude"]
        self.S1_S2_distance = h5_PSE_dataset.attrs["S1_S2_distance"]
        
        self.block_start_index = block_group.attrs['start_index']
        self.prefered_antenna_index = block_group.attrs['prefered_ant_i']
        
        if "converged" in h5_PSE_dataset.attrs:
            self.converged = h5_PSE_dataset.attrs["converged"]
        else:
            self.converged = True
            
        if "RMS" in h5_PSE_dataset.attrs:
            self.RMS = h5_PSE_dataset.attrs["RMS"]
        else:
            self.RMS = 0.0
            
            
        if "XYZT_s1" in h5_PSE_dataset.attrs:
            self.XYZT_s1 = h5_PSE_dataset.attrs["XYZT_s1"]
            
        self.XYZT = np.append(self.loc, [self.T])
        
    def set_header(self, header):
        self.header = header
        
    def plot(self, station_set="all"):
        """station_set should be "all", "CS", "RS" """
        
        station_ant_info = self.header.antenna_info_by_station()
        for stat_i, (sname, ant_list) in enumerate(station_ant_info.items()):
            if station_set=="all" or station_set==sname[:2]:
                traces = []
                max = 0.0
                for ant in ant_list:
                    traces.append( np.abs(self.file_dataset[ ant.antenna_index ] ) )
                    tmax = np.max( traces[-1] )
                    if tmax > max:
                        max = tmax
                    
                for t in traces:
                    plt.plot( np.abs(t)/max+stat_i*1.1, linewidth=3 )
                plt.annotate(sname, (0.0,stat_i*1.1+0.5))
                
        plt.axvline(x=25)
        plt.show()

        
def load_interferometric_PSE(folder, blocks_to_open=None):
    print("reading PSE")
    
    file_names = [fname for fname in os.listdir(folder) if fname.endswith(".h5") and not fname.startswith("tmp")]
    
    header = None
    IPSE_list = []
    for fname in file_names:
        print("  ", fname)
        
        try:
            infile = h5py.File(folder+'/'+fname, "r+")
        except:
            continue
        
        for groupname, group in infile.items():
            if groupname == "header":
                header = interferometry_header( group )
                
            elif (blocks_to_open is None) or (int(groupname) in blocks_to_open): ## is  a block of data
                block_index = int(groupname)
                for ID, IPSE_dataset in group.items():
                    IPSE_list.append( Ipse(block_index, int(ID), IPSE_dataset, group) )
                    
    for IPSE in IPSE_list:
        IPSE.set_header( header )
        
    return header, IPSE_list

class block_info:
    def __init__(self, block_group):
        self.block_group = block_group
        
        self.block_index = block_group.attrs['block_index']
        self.start_index = block_group.attrs['start_index']
        self.prefered_ant_i = block_group.attrs['prefered_ant_i']
        
        self.IPSE_list = []
        for ID, IPSE_dataset in block_group.items():
            self.IPSE_list.append( Ipse(self.block_index, int(ID), IPSE_dataset, block_group) )
            
    def set_header(self, header):
        self.header = header
        for IPSE in self.IPSE_list:
            IPSE.set_header(header)

def load_interferometric_PSE_ByBlock(folder, blocks_to_open=None):
    print("reading PSE")
    
    file_names = [fname for fname in os.listdir(folder) if fname.endswith(".h5") and not fname.startswith("tmp")]
    
    header = None
    block_dict = {}
    for fname in file_names:
        print("  ", fname)
        
        try:
            infile = h5py.File(folder+'/'+fname, "r+")
        except:
            continue
        
        for groupname, group in infile.items():
            if groupname == "header":
                header = interferometry_header( group )
                
            elif (blocks_to_open is None) or (int(groupname) in blocks_to_open): ## is  a block of data
                new_block = block_info( group )
                block_dict[new_block.block_index] = new_block
                    
    for block in block_dict.values():
        block.set_header( header )
        
    return header, block_dict

def filter_IPSE( IPSE_list, bounds ):
    """filter IPSE by XYZT bounds."""
    return [IPSE for IPSE in IPSE_list if bounds[0][0]<IPSE.loc[0]<bounds[0][1] and bounds[1][0]<IPSE.loc[1]<bounds[1][1] \
            and bounds[2][0]<IPSE.loc[2]<bounds[2][1] and bounds[3][0]<IPSE.T<bounds[3][1] ]
    
def get_IPSE( IPSE_list, unique_index):
    for IPSE in IPSE_list:
        if IPSE.unique_index == unique_index:
            return IPSE
    return None
    

def IPSE_to_txt(IPSE_list, out_file):
    for ipse in IPSE_list:
        ID = ipse.unique_index
        X = ipse.loc[0]
        Y = ipse.loc[1]
        Z = ipse.loc[2]
        T = ipse.T
        I = np.log( ipse.intensity )
        
        out_file.write(str(ID)+' E '+str(X)+" "+str(Y)+" "+str(Z)+" "+str(T)+" " + str(I)+'\n' )
                    
    
if __name__ == "__main__":
    timeID = "D20170929T202255.000Z"
    
    input_folder = "interferometry_out4_no_erase"
    max_S1S2_distance = 50.0
    min_intensity = 0.85

    print("opening data")
    processed_data_folder = processed_data_dir(timeID)
    data_dir = processed_data_folder + "/" + input_folder
    print()
    header_data, IPSE_list = load_interferometric_PSE( data_dir )
    
    filtered_intensities = [IPSE.intensity for IPSE in IPSE_list if IPSE.converged and IPSE.S1_S2_distance<max_S1S2_distance]
    num_to_plot = len([i for i in filtered_intensities if i>min_intensity])
    
    print(len(IPSE_list), "sources total", num_to_plot, "good")
    
    plt.hist(filtered_intensities, bins=20)
    plt.xlabel("image intensity", fontsize=30)
    plt.ylabel("number of sources", fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()


#    min_intensity = 30.0
#    and_stage1_success = False
#    max_S1S2_distance = 50.0
#    
#    Xlims = [-20000.0, -14000.0]
#    Ylims = [7000.0, 12000.0]
#    Zlims = [0.0, 7500.0]#[1000,8000]
#    T0 = 0.0
#    Tlims = [1140.,1320.]
#    
#    print("opening data")
#    processed_data_folder = processed_data_dir(timeID)
#    data_dir = processed_data_folder + "/" + input_folder
#    print()
#    header_data, IPSE_list = load_interferometric_PSE( data_dir )
#    print("opened", len(IPSE_list), "IPSE")
#    IPSE_filtered = [IPSE for IPSE in IPSE_list if IPSE.intensity>min_intensity and (IPSE.stage_1_success or not and_stage1_success) and IPSE.S1_S2_distance<max_S1S2_distance]
#    print(len(IPSE_filtered), "filtered IPSE")
#    IPSE_filtered = [IPSE for IPSE in IPSE_filtered if Xlims[0]<=IPSE.loc[0]<=Xlims[1] and Ylims[0]<=IPSE.loc[1]<=Ylims[1] and Zlims[0]<=IPSE.loc[2]<=Zlims[1]]
#    IPSE_filtered = [IPSE for IPSE in IPSE_filtered if Tlims[0] <= (IPSE.T - T0)*1000.0 <= Tlims[1]]
#    print(len(IPSE_filtered), "lvl 2 filter")
#    
#    X = [IPSE.loc[0] for IPSE in IPSE_filtered]
#    Y = [IPSE.loc[1] for IPSE in IPSE_filtered]
#    Z = [IPSE.loc[2] for IPSE in IPSE_filtered]
#    T = [(IPSE.T - T0)*1000.0 for IPSE in IPSE_filtered]
#    I = [IPSE.intensity for IPSE in IPSE_filtered]
#    
#    plt.scatter(T,Z,c=T, s=30 )#np.log10(I), s=30 ) 
#    plt.tick_params(axis='both', which='major', labelsize=20)
#    plt.colorbar()
#    plt.show()
#    
#    plt.scatter(X,Y,c=T, s=30)
#    plt.tick_params(axis='both', which='major', labelsize=20)
#    plt.colorbar()
#    plt.show()
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(X, Y, Z, c=np.log10(I), s=3)
#    plt.show()
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(X, Y, Z, c=T, s=3)
#    plt.show()
#    
#    IPSE_to_txt(IPSE_filtered, open("IPSE_out.txt", 'w'))
#    
#    D = [IPSE.S1_S2_distance for IPSE in IPSE_list]
##    plt.hist(D, bins=50, range=(0,100))
##    plt.show()
##    
##    
#    all_I = [IPSE.intensity for IPSE in IPSE_list]
##    plt.hist(D, bins=50, range=(0,10))
##    plt.show()
#    
#    plt.hist2d(D, all_I, bins=50, range=[[0,100],[0,100]])
#    plt.colorbar()
#    plt.show()
#    
#    ### probability vs ID
#    bins_plotted = np.zeros(100, dtype=float)
#    bins_numIPSE = np.zeros(100, dtype=int)
#    for IPSE in IPSE_list:
#        good = IPSE.intensity>min_intensity and IPSE.S1_S2_distance<max_S1S2_distance 
#        good = good and Xlims[0]<=IPSE.loc[0]<=Xlims[1] and Ylims[0]<=IPSE.loc[1]<=Ylims[1] and Zlims[0]<=IPSE.loc[2]<=Zlims[1]
#    
#        if good:
#            bins_plotted[ IPSE.IPSE_index ] += 1.0
#        bins_numIPSE[ IPSE.IPSE_index ] += 1.0
#            
#    bins_plotted /= bins_numIPSE
#    plt.bar(np.arange(100)-0.5, bins_plotted, 1)
#    plt.show()
    
    