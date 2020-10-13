#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

from LoLIM.utilities import v_air
from LoLIM.findRFI import window_and_filter
from LoLIM.antenna_response import LBA_antenna_model
from LoLIM.signal_processing import num_double_zeros
from LoLIM.getTrace_fromLoc import getTrace_fromLoc

def model_EnergyAmp_ratio(dt=5.0E-9, N=1000):
    antenna_model = LBA_antenna_model()
    
    frequencies = np.fft.fftfreq(N, dt)

    jones_matrices = antenna_model.JonesMatrix_MultiFreq(frequencies=frequencies, zenith=80.0, azimuth=0.0)
    Ze_to_ant1 = jones_matrices[:, 0,0]
    
    filter = window_and_filter(blocksize=N, time_per_sample=dt)
    
    timeseries_signal =  np.fft.ifft( Ze_to_ant1*filter.bandpass_filter, axis=-1)
    
    # plt.plot(np.real(timeseries_signal))
    # plt.plot(np.abs(timeseries_signal))
    # plt.show()
    
    HE = np.abs(timeseries_signal)
    power = HE*HE
    
    ratio = simps( power, dx=dt )/np.max( power )


    return ratio
    
def get_noise_power(TBB_datafile, RFI_filter, ant_i, block_range, max_fraction_doubleZeros=0.0005):
    
    ret = None
    
    for block in range(block_range[0], block_range[1]):
        data = TBB_datafile.get_data(block*RFI_filter.blocksize, RFI_filter.blocksize, antenna_index=ant_i  )
        DBZ_fraction = num_double_zeros(data)/RFI_filter.blocksize
        
        if DBZ_fraction > max_fraction_doubleZeros:
            continue
    
        data = RFI_filter.filter( np.array(data,dtype=np.double) )
        
        edge_width = int( RFI_filter.blocksize*RFI_filter.half_window_percent )
        HE = np.abs( data[edge_width:-edge_width] )
        noise_power = HE*HE
        ave_noise_power = np.average(noise_power)
        
        ret = ave_noise_power
        break
    
    return ret




        
        
        
def corners_of_bounds(bounds):
    ret = []
    for xi in [0,1]:
        for yi in [0,1]:
            for zi in [0,1]:
                ret.append([  bounds[0,xi], bounds[1,yi], bounds[2,zi]  ])
    return np.array(ret)
                
                
class ImagedEnergyHelper:
    """Class to help find the fraction of imaged energy. At least, sources need XYZT. Doesn't really handle polarization well??"""
    
    def __init__(self, TBB_datafile, RFI_filter, good_sources, noise_block_range, pulse_width=31E-9, max_fraction_doubleZeros=0.0005):
        self.TBB_datafile = TBB_datafile
        self.RFI_filter = RFI_filter
        self.blocksize = self.RFI_filter.blocksize
        self.pulse_width = pulse_width
        self.edge_width = int( RFI_filter.blocksize*RFI_filter.half_window_percent )
        self.good_sources = good_sources
        self.max_fraction_doubleZeros = max_fraction_doubleZeros
        self.antenna_names = TBB_datafile.get_antenna_names()
        self.num_antennas = len( self.antenna_names )
        self.num_pairs = int( len( self.antenna_names )/2 )
        self.station_name = TBB_datafile.StationName
        
        self.TraceLocator = getTrace_fromLoc( {self.station_name:TBB_datafile}, {self.station_name:RFI_filter} )
        
        self.antenna_noise_power = [ get_noise_power(TBB_datafile, RFI_filter, anti, noise_block_range, max_fraction_doubleZeros) for anti in range( self.num_antennas ) ]
        
        self.event_samples = [ [self.TraceLocator.source_recieved_index(source.XYZT, ant_name ) for ant_name in self.antenna_names] for source in self.good_sources  ]
        
    def sources_in_bounds(self, bounds, Tcut=True):
        def in_bounds(XYZT):
            inX = bounds[0,0] <= XYZT[0] <= bounds[0,1]
            inY = bounds[1,0] <= XYZT[1] <= bounds[1,1]
            inZ = bounds[2,0] <= XYZT[2] <= bounds[2,1]
            inT = bounds[3,0] <= XYZT[3] <= bounds[3,1]
            
            return inX and inY and inZ and (inT or (not Tcut))
        
        # TST = [s for s in self.good_sources if s.in_BoundingBox(bounds)]
        
        R = [[source,indeces] for source,indeces in zip(self.good_sources,self.event_samples) if in_bounds(source.XYZT)]
        
        ret_sources = []
        ret_indeces = []
        for s,i in R:
            ret_sources.append(s)
            ret_indeces.append(i)
        
        
        return ret_sources, ret_indeces
        
#    def find_recieved_energy(self, source_bounds):
#        """given a 2D numpy array of bounds, returns total energy in the data, subtracting off noise. Averages available antennas, corresponding to center of bounds."""
#        
#        locX = (source_bounds[0,0]+source_bounds[0,1])/2
#        locY = (source_bounds[1,0]+source_bounds[1,1])/2
#        locZ = (source_bounds[2,0]+source_bounds[2,1])/2
#        XYZT_start = np.array([ locX,locY,locZ, source_bounds[3,0]  ])
#        
#        samples_left = int( (source_bounds[3,1] - source_bounds[3,0])/5.0e-9 )
#        current_sample = [self.TraceLocator.source_recieved_index(XYZT_start, ant_name ) for ant_name in self.antenna_names ]
#        
#        edge = self.edge_width + 20 #  a little extra for pulse width
#        usable_blockwidth = self.blocksize - 2*edge 
#        total_energy = 0.0
#        while samples_left > 0:
#            total_ant_energy = 0.0
#            num_ants = 0
#            
#            width = usable_blockwidth
#            if samples_left < width:
#                width = samples_left
#            samples_left -= width
#            
#            for ant_i in range( self.num_antennas ):
#                noise_power = self.antenna_noise_power[ ant_i ]
#                if noise_power is None:
#                    continue
#                
#                start_sample = current_sample[ ant_i ]
#                current_sample[ ant_i ] += width
#                
#                data = self.TBB_datafile.get_data(start_sample-edge, self.RFI_filter.blocksize, antenna_index=ant_i  )
#                DBZ_fraction = num_double_zeros(data)/self.RFI_filter.blocksize
#                
#                if DBZ_fraction > self.max_fraction_doubleZeros:
#                    continue
#                
#                data = self.RFI_filter.filter( np.array(data,dtype=np.double) )
#                HE = np.abs( data[edge:edge+width] )
#                power = HE*HE
#                A = simps( power, dx=5.0E-9 )
#                N = noise_power*width*5.0E-9
#                energy =  A - N
#                
#                if energy < 0:
#                    energy = 0.001*N
#                
#                total_ant_energy += energy
#                num_ants += 1
#                
#                
#            total_energy += total_ant_energy/num_ants
#            
#        return total_energy
    
#    def find_imaged_energy_full(self, source_bounds):
#        """for each source, find the peak power averaged across availabel antennas, sum all sources and multiply by width"""
#        
#        
#        edge = self.edge_width + 20 #  a little extra for pulse width
#        
#        # def in_bounds(XYZT):
#        #     inX = source_bounds[0,0] <= XYZT[0] <= source_bounds[0,1]
#        #     inY = source_bounds[1,0] <= XYZT[1] <= source_bounds[1,1]
#        #     inZ = source_bounds[2,0] <= XYZT[2] <= source_bounds[2,1]
#        #     inT = source_bounds[3,0] <= XYZT[3] <= source_bounds[3,1]
#            
#        #     return inX and inY and inZ and inT
#        
#        source_XYZTs_in_bounds = np.array( [source.XYZT for source in self.sources_in_bounds(source_bounds) ] )
#        
#        if len(source_XYZTs_in_bounds) == 0:
#            return 0.0
#        
#        sorter = np.argsort( source_XYZTs_in_bounds[:, 3]  )
#        source_XYZTs_in_bounds = source_XYZTs_in_bounds[ sorter ]
#        
#        source_powers_antennaSummed = np.zeros(len(source_XYZTs_in_bounds), dtype=np.double )
#        source_participating_antennas = np.zeros(len(source_XYZTs_in_bounds), dtype=np.int )
#        
#        for ant_i in range( self.num_antennas ):
#            noise_power = self.antenna_noise_power[ ant_i ]
#            if noise_power is None:
#                continue
#            
#            ant_name = self.antenna_names[ ant_i ]
#            source_arrival_indeces = np.array([ self.TraceLocator.source_recieved_index(XYZT, ant_name ) for XYZT in  source_XYZTs_in_bounds ])
#            source_arrival_indeces = np.sort( source_arrival_indeces )
#            
#            current_source_index = 0
#            
#            while True:
#                block_arrival_index = source_arrival_indeces[current_source_index] - edge
#                
#                data = self.TBB_datafile.get_data(block_arrival_index, self.RFI_filter.blocksize, antenna_index=ant_i  )
#                if len(data) < self.RFI_filter.blocksize:
#                    print('error!!')
#                
#                DBZ_fraction = num_double_zeros(data)/self.RFI_filter.blocksize
#                
#                data = self.RFI_filter.filter( np.array(data,dtype=np.double) )
#                HE = np.abs( data )
#                did_done_broke = False
#        
#                for source_i in range( current_source_index, len(source_arrival_indeces) ):
#                    current_source_index = source_i
#                    local_source_arrival_index = source_arrival_indeces[ current_source_index ] - block_arrival_index
#                    
#                    if local_source_arrival_index > (self.RFI_filter.blocksize-edge):
#                        did_done_broke = True
#                        break
#                    elif DBZ_fraction < self.max_fraction_doubleZeros: ## good
#                        S = HE[local_source_arrival_index-3:local_source_arrival_index+3]
#                        if len(S)<3:
#                            print("E:", len(S), len(HE), local_source_arrival_index, self.RFI_filter.blocksize-2*edge)
#                        amp = np.max( S )
#                        source_powers_antennaSummed[ source_i ] += amp*amp
#                        source_participating_antennas[ source_i ] += 1
#                        
#                if not did_done_broke: ## all sources measured!
#                    break
#                
#        ave_powers = source_powers_antennaSummed/source_participating_antennas
#        return np.sum( ave_powers )*self.pulse_width
    
    def FindEnergy_BothPolarizations(self, source_bounds, min_amp=None, ret_amps=False, plot=False):
        
        edge = self.edge_width + 20 #  a little extra for pulse width
        usable_blockwidth = self.blocksize - 2*edge 
        
        bounds_corners = corners_of_bounds( source_bounds )
        largest_edge = 0
        for C1 in bounds_corners:
            for C2 in bounds_corners:
                L = np.linalg.norm( C2-C1 )
#                print(L, C1, C2)
                if L > largest_edge:
                    largest_edge = L
        new_XYZT_bounds = np.array( source_bounds )
        new_XYZT_bounds[3,0] -= largest_edge/v_air
        new_XYZT_bounds[3,1] += largest_edge/v_air
                
        
        sources, indeces_by_source = self.sources_in_bounds(new_XYZT_bounds)
        sources = list(sources)
        
        indeces_by_antenna = [ [] for i in range(self.num_antennas) ]
        for source_ilist in indeces_by_source:
            for anti, i in enumerate(source_ilist):
                indeces_by_antenna[anti].append(i)
            
        
#        source_XYZTs_in_bounds = np.array( [source.XYZT for source in sources ] )
        
        locX = (source_bounds[0,0]+source_bounds[0,1])/2
        locY = (source_bounds[1,0]+source_bounds[1,1])/2
        locZ = (source_bounds[2,0]+source_bounds[2,1])/2
        XYZT_start = np.array([ locX,locY,locZ, source_bounds[3,0]  ])
        
        samples_left = int( (source_bounds[3,1] - source_bounds[3,0])/5.0e-9 )
        
        current_sample = np.array(  [self.TraceLocator.source_recieved_index(XYZT_start, ant_name ) for ant_name in self.antenna_names ]  )
        
#        event_samples = [ np.sort([self.TraceLocator.source_recieved_index(XYZT, ant_name ) for XYZT in source_XYZTs_in_bounds]) for ant_name in self.antenna_names ]
        event_samples = [ np.sort(I) for I in indeces_by_antenna ]
        
        current_event_index = [0]*self.num_antennas
        
        print('num total source:', len( sources) )
        
        total_energy = 0.0
        imaged_energy = 0.0
        amps = []    
        
        while samples_left > 0:
            
            width = usable_blockwidth
            if samples_left < width:
                width = samples_left
            samples_left -= width
            
            ### FIRST FOR EVENS
            even_ant_found = False
            for ant_pair_i in range(self.num_pairs ):
                ant_i = ant_pair_i*2
                
                noise_power = self.antenna_noise_power[ ant_i ]
                if noise_power is None:
                    continue
                
                start_sample = current_sample[ ant_i ]
                
                data = self.TBB_datafile.get_data(start_sample-edge, self.RFI_filter.blocksize, antenna_index=ant_i  )
                DBZ_fraction = num_double_zeros(data)/self.RFI_filter.blocksize
                
                if DBZ_fraction > self.max_fraction_doubleZeros:
                    continue
                
                even_ant_found = True
                
                data = self.RFI_filter.filter( np.array(data,dtype=np.double) )
                HE = np.abs( data )
                if plot:
                    T = np.arange(self.RFI_filter.blocksize) + (start_sample-edge)
                    plt.plot(T, HE)
                power = HE*HE
                
                N = noise_power*width*5.0E-9
                if min_amp is None:
                    power_to_integrate = power
                else:
                    power_to_integrate = np.array(power)
                    power_to_integrate[ HE<min_amp ] = 0.0
#                    print('ARG:', np.sum(HE<min_amp))
                    N *= 0.0000000001 
                
                # recieved energy
                A = simps( power_to_integrate[edge:edge+width] , dx=5.0E-9 )
                energy =  A - N
                
                if energy < 0:
                    energy = 0.001*N
                
                total_energy += energy
                
                # imaged energy
                N = 0
#                used_samples = []
                
                event_sample_numbers = event_samples[ant_i]
                last_sample_number = 0
                for source_i in range(current_event_index[ant_i], len(event_sample_numbers)):
                    source_sample_number = event_sample_numbers[ source_i ]
                    
                    if source_sample_number < start_sample:
                        last_sample_number = source_i
                        
                    elif start_sample <= source_sample_number < start_sample+width:
#                        used_samples.append( source_sample_number )
                        last_sample_number = source_i
                        local_sample_number = source_sample_number - (start_sample-edge)
                        A = HE[local_sample_number-3 : local_sample_number+3]
                        E = np.max(A)
                        if E>= min_amp:
                            imaged_energy += E*E
                            N += 1
                        if ret_amps :
                            amps.append(E)
                        
                    else:
                        break
                current_event_index[ant_i] = last_sample_number
                
#                print('E:', ant_i, 'samples:', start_sample, start_sample+width)
#                print('  NS:', N)
#                print('  used sources locs', used_samples)
#                if N > 0:
#                    print('  final source loc:', event_sample_numbers[last_sample_number] )
#                print( event_sample_numbers )
                break
                
            if not even_ant_found:
                print("ERROR! too much data loss on even antenna!")
                quit()
            
            ### Odd antennas
            odd_ant_found = False
            for ant_pair_i in range(self.num_pairs ):
                ant_i = ant_pair_i*2 + 1
                
                noise_power = self.antenna_noise_power[ ant_i ]
                if noise_power is None:
                    continue
                
                start_sample = current_sample[ ant_i ]
                
                data = self.TBB_datafile.get_data(start_sample-edge, self.RFI_filter.blocksize, antenna_index=ant_i  )
                DBZ_fraction = num_double_zeros(data)/self.RFI_filter.blocksize
                
                if DBZ_fraction > self.max_fraction_doubleZeros:
                    continue
                
                odd_ant_found = True
                
                data = self.RFI_filter.filter( np.array(data,dtype=np.double) )
                HE = np.abs( data )
                if plot:
                    T = np.arange(self.RFI_filter.blocksize) + (start_sample-edge)
                    plt.plot(T, HE)
                    # print("odd num GE:", np.sum())
                power = HE*HE
                
                
                N = noise_power*width*5.0E-9
                if min_amp is None:
                    power_to_integrate = power
                else:
                    power_to_integrate = np.array(power)
                    power_to_integrate[ HE<min_amp ] = 0.0
#                    print('ARG:', np.sum(HE<min_amp))
                    N *= 0.0000000001 
                
                # recieved energy
                A = simps( power_to_integrate[edge:edge+width] , dx=5.0E-9 )
                energy =  A - N
                
                if energy < 0:
                    energy = 0.001*N
                    
                
                total_energy += energy
                
                # imaged energy
                N = 0
                event_sample_numbers = event_samples[ant_i]
                last_sample_number = 0
                for source_i in range(current_event_index[ant_i], len(event_sample_numbers)):
                    source_sample_number = event_sample_numbers[ source_i ]
                    
                    if source_sample_number < start_sample:
                        last_sample_number = source_i
                        
                    elif start_sample <= source_sample_number < start_sample+width:
                        last_sample_number = source_i
                        local_sample_number = source_sample_number - (start_sample-edge)
                        A = HE[local_sample_number-3 : local_sample_number+3]
                        E = np.max(A)
                        if E>= min_amp:
                            imaged_energy += E*E
                            N += 1
#                        if ret_amps :
#                            amps.append(E)
                        
                    else:
                        break
                current_event_index[ant_i] = last_sample_number
                break
                
            if not odd_ant_found:
                print("ERROR! too much data loss on even antenna!")
                quit()
                
            ## update
            current_sample += width
                
        
        if plot:
            print(total_energy, imaged_energy*self.pulse_width)
            plt.show()
                
        if ret_amps:
            return total_energy, imaged_energy*self.pulse_width, amps
        else:
            return total_energy, imaged_energy*self.pulse_width
                    
        
        
        
        
        
                
        