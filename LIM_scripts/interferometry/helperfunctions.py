#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#### some algorithms for choosing pairs of antennas ####
## need to exclude bad antennas, and consider pol-flips!
def pairData_1antPerStat(input_files):
    """return the antenna-pair data for choosing one antenna per station"""
    num_stations = len(input_files)
    
    antennas = [ [file_index,0] for file_index in range(num_stations) ]
    
    num_pairs = int( num_stations*(num_stations-1)*0.5 )
    pairs = np.zeros((num_pairs,2), dtype=np.int64)
    
    pair_i = 0
    for i in range(num_stations):
        for j in range(num_stations):
            if j<=i:
                continue
            
            pairs[pair_i,0] = i
            pairs[pair_i,1] = j
            
            pair_i += 1
            
    return np.array(antennas, dtype=int), pairs

def pairData_closest_antennas(input_files, ant_range):
    """return the antenna-pair data using only the antennas that are within a certain range"""
    
    ## we use all antennas
    antennas = []
    for file_index, in_file in enumerate(input_files):
        num_ant = len(in_file.get_antenna_names())
        for ant_i in range(num_ant):
            if (ant_i%2)==0:
                antennas.append([file_index, ant_i])
                
                
    pairs = []
    for ant_i, (file_i, ant_sub_i) in enumerate(antennas):
        anti_location = input_files[file_i].get_LOFAR_centered_positions()[ant_sub_i]
        
        for ant_j, (file_j, ant_sub_j) in enumerate(antennas):
            if ant_j <= ant_i:
                continue
            
            antj_location = input_files[file_i].get_LOFAR_centered_positions()[ant_sub_i]
            
            if np.linalg.norm(anti_location-antj_location) < ant_range:
                pairs.append([ant_i, ant_j])
                
    return np.array(antennas, dtype=int), np.array(pairs, dtype=np.int64)
                
    
def pairData_everyEvenAntennaOnce(input_files, image_location):
    """return the antenna-pair data using every antenna once (and the same antenna in every pair)"""
    
    ## we use all antennas
    antennas = []
    for file_index, in_file in enumerate(input_files):
        num_ant = len(in_file.get_antenna_names())
        for ant_i in range(num_ant):
            if (ant_i%2)==0:
                antennas.append([file_index, ant_i])
            
    ##find the closest antenna
    distance = np.inf
    ant_index = None
    for ant_i, (file_i, ant_sub_i) in enumerate(antennas):
        ant_location = input_files[file_i].get_LOFAR_centered_positions()[ant_sub_i]
        D = np.linalg.norm(ant_location-image_location)
        if D<distance:
            distance = D
            ant_index = ant_i
            
    ##all antenna pairs
    pairs = []
    for ant_i in range(len(antennas)):
        pairs.append([ant_index, ant_i])
        
    return np.array(antennas, dtype=int), np.array(pairs, dtype=np.int64)

def pairData_OptimizedEvenAntennas(input_files, image_location, num_pairs):
    def make_gradient(ant_i_loc, ant_j_loc):
        
        delta_anti = image_location - ant_i_loc
        delta_antj = image_location - ant_j_loc
        
        delta_anti /= np.linalg.norm(delta_anti)
        delta_antj /= np.linalg.norm(delta_antj)
        
        return delta_anti-delta_antj
        
    ## we use all antennas
    antennas = []
    for file_index, in_file in enumerate(input_files):
        num_ant = len(in_file.get_antenna_names())
        for ant_i in range(num_ant):
            if (ant_i%2)==0:
                antennas.append([file_index, ant_i])
    antennas = np.array(antennas, dtype=int)
                
    ## every possible pair
    pairs = []
    for ant_i, (file_i, ant_sub_i) in enumerate(antennas):
        for ant_j, (file_j, ant_sub_j) in enumerate(antennas):
            if ant_j <= ant_i:
                continue
            pairs.append([ant_i, ant_j])
    pairs = np.array(pairs, dtype=int)
    
    
    
    
    
    ## calculate covarience matrix and find first pair (largest gradient)
    covarience_matrices = np.zeros((len(pairs),3,3), dtype=np.double)
    max_gradient = 0
    max_gradient_pair_k = None
    for pair_k, (ant_i, ant_j) in enumerate(pairs):
        station_i, station_ant_i = antennas[ant_i]
        station_j, station_ant_j = antennas[ant_j]
        
        ant_i_loc = input_files[station_i].get_LOFAR_centered_positions()[station_ant_i]
        ant_j_loc = input_files[station_j].get_LOFAR_centered_positions()[station_ant_j]

        gradient = make_gradient(ant_i_loc, ant_j_loc)
        magnitude = np.linalg.norm( gradient )
        
        if magnitude > max_gradient:
            max_gradient = magnitude
            max_gradient_pair_k = pair_k
        
        grad_norm = gradient/magnitude
        
        covarience_matrices[pair_k] = np.outer(grad_norm, grad_norm)/(magnitude*magnitude)
        
        
        
    ##initiallize the bits we need
    pairs_used = np.zeros(len(pairs), dtype=bool )
    antennas_used = np.zeros(len(antennas), dtype=bool )
    
    current_matrix = covarience_matrices[max_gradient_pair_k]
    
    pairs_used[max_gradient_pair_k] = True
    antennas_used[ pairs[max_gradient_pair_k][0] ] = True
    antennas_used[ pairs[max_gradient_pair_k][1] ] = True
    
    
    
    ## now the main loop. Find the best set of antennas!
    for i in range(1,num_pairs):
        
        ##search through all available pairs
        min_parameter = np.inf
        best_pair_k = None
        for pair_k in range(len(pairs)):
            if pairs_used[pair_k]:
                continue
            
            new_matrix = current_matrix + covarience_matrices[pair_k]
            current_param = np.max( np.abs( np.linalg.eigvals(new_matrix) ) )
#            current_param = np.linalg.det(new_matrix)
            if current_param < min_parameter:
                min_parameter = current_param
                best_pair_k = pair_k
        print(i, num_pairs, min_parameter/(i+1))
        ## update covarience matrix
        current_matrix += covarience_matrices[best_pair_k]
    
        pairs_used[best_pair_k] = True
        antennas_used[ pairs[best_pair_k][0] ] = True
        antennas_used[ pairs[best_pair_k][1] ] = True
            
        
        
    #### cleanup ####
    # antennas
    used_antennas = []
    antenna_map = {}
    for ant_i, is_used in enumerate(antennas_used):
        if is_used:
            antenna_map[ant_i] = len(used_antennas)
            used_antennas.append( antennas[ant_i] )
        
    #paris
    used_pairs = []
    for pair, is_used in zip(pairs, pairs_used):
        if is_used:
            ant_i, ant_j = pair
            used_pairs.append( [antenna_map[ant_i], antenna_map[ant_j]] )
        
    return np.array(used_antennas, dtype=int), np.array(used_pairs, dtype=int)
        

def pairData_OptimizedEvenAntennas_two(input_files, image_location, num_pairs):
    def make_gradient(ant_i_loc, ant_j_loc):
        
        delta_anti = image_location - ant_i_loc
        delta_antj = image_location - ant_j_loc
        
        delta_anti /= np.linalg.norm(delta_anti)
        delta_antj /= np.linalg.norm(delta_antj)
        
        return delta_anti-delta_antj
        
    ## we use all antennas
    antennas = []
    for file_index, in_file in enumerate(input_files):
        num_ant = len(in_file.get_antenna_names())
        for ant_i in range(num_ant):
            if (ant_i%2)==0:
                antennas.append([file_index, ant_i])
    antennas = np.array(antennas, dtype=int)
                
    ## every possible pair
    pairs = []
    for ant_i, (file_i, ant_sub_i) in enumerate(antennas):
        for ant_j, (file_j, ant_sub_j) in enumerate(antennas):
            if ant_j <= ant_i:
                continue
            pairs.append([ant_i, ant_j])
    pairs = np.array(pairs, dtype=int)
    
    
    
    
    
    ## calculate covarience matrix and find first pair (largest gradient)
    covarience_matrices = np.zeros((len(pairs),3,3), dtype=np.double)
    gradients = np.zeros((len(pairs),3), dtype=np.double)
    max_gradient = 0
    max_gradient_pair_k = None
    for pair_k, (ant_i, ant_j) in enumerate(pairs):
        station_i, station_ant_i = antennas[ant_i]
        station_j, station_ant_j = antennas[ant_j]
        
        ant_i_loc = input_files[station_i].get_LOFAR_centered_positions()[station_ant_i]
        ant_j_loc = input_files[station_j].get_LOFAR_centered_positions()[station_ant_j]

        gradient = make_gradient(ant_i_loc, ant_j_loc)
        magnitude = np.linalg.norm( gradient )
        
        if magnitude > max_gradient:
            max_gradient = magnitude
            max_gradient_pair_k = pair_k
        
        grad_norm = gradient/magnitude
        
        covarience_matrices[pair_k] = np.outer(grad_norm, grad_norm)/(magnitude*magnitude)
        gradients[pair_k] = gradient
        
        
        
    ##initiallize the bits we need
    pairs_used = np.zeros(len(pairs), dtype=bool )
    antennas_used = np.zeros(len(antennas), dtype=bool )
    
    current_matrix = covarience_matrices[max_gradient_pair_k]
    
    pairs_used[max_gradient_pair_k] = True
    antennas_used[ pairs[max_gradient_pair_k][0] ] = True
    antennas_used[ pairs[max_gradient_pair_k][1] ] = True
    
    
    
    ## now the main loop. Find the best set of antennas!
    for i in range(1,num_pairs):
        
        ## get current eigenstuff
        eigvals, eigvectors = np.linalg.eig( current_matrix )
        eig_i = np.argmin( eigvals )
        min_eigvector = eigvectors[:,eig_i]
        
        ##search through all available pairs
        max_parameter = 0
        best_pair_k = None
        for pair_k in range(len(pairs)):
            if pairs_used[pair_k]:
                continue
            
            current_param = np.abs( np.dot(min_eigvector, gradients[pair_k] ) )
            
            if current_param > max_parameter:
                max_parameter = current_param
                best_pair_k = pair_k
        print(i, num_pairs, max_parameter, eigvectors[eig_i]/(i+1))
        ## update covarience matrix
        current_matrix += covarience_matrices[best_pair_k]
    
        pairs_used[best_pair_k] = True
        antennas_used[ pairs[best_pair_k][0] ] = True
        antennas_used[ pairs[best_pair_k][1] ] = True
            
        
        
    #### cleanup ####
    # antennas
    used_antennas = []
    antenna_map = {}
    for ant_i, is_used in enumerate(antennas_used):
        if is_used:
            antenna_map[ant_i] = len(used_antennas)
            used_antennas.append( antennas[ant_i] )
        
    #paris
    used_pairs = []
    for pair, is_used in zip(pairs, pairs_used):
        if is_used:
            ant_i, ant_j = pair
            used_pairs.append( [antenna_map[ant_i], antenna_map[ant_j]] )
        
    return np.array(used_antennas, dtype=int), np.array(used_pairs, dtype=int)


#### some bounding box functions ###
def closest_distance(ant_loc, bounding_box):
    """given an antenna location and a bounding box, and a bounding box, give closest distance from antenna to box"""

    states = [0, 0, 0]
    num_inside_bounds = 0
    for axis in range(3):
        if bounding_box[axis,0] <= ant_loc[axis] <= bounding_box[axis,1]:
            states[axis] = 0
            num_inside_bounds += 1
        elif ant_loc[axis]<bounding_box[axis,0]:
            states[axis] = -1
        else:
            states[axis] = 1
    
    
    if num_inside_bounds == 3:
        ## inside of the box! ##
        return 0.0
    
    elif num_inside_bounds == 0:
        ## one of the corners is the closest
        xi = 0 if states[0]==-1 else 1
        yi = 0 if states[1]==-1 else 1
        zi = 0 if states[2]==-1 else 1
        nearist_corner = np.array([ bounding_box[0,xi], bounding_box[1,yi], bounding_box[2,zi] ])
        return np.linalg.norm( nearist_corner-ant_loc )
    
    elif num_inside_bounds == 1:
        ## one of the lines is closest
        inclosed_axis = [i for i in [0,1,2] if states[i]==0][0]
        axis_1, axis_2 = [i for i in [0,1,2] if states[i]!=0]
        
        axis_1_i = 0 if states[axis_1]==-1 else 1
        axis_2_i = 0 if states[axis_2]==-1 else 1
        
        lower_point = np.zeros(3)
        upper_point = np.zeros(3)
        
        lower_point[axis_1] = bounding_box[axis_1, axis_1_i]
        lower_point[axis_2] = bounding_box[axis_2, axis_2_i]
        upper_point[axis_1] = bounding_box[axis_1, axis_1_i]
        upper_point[axis_2] = bounding_box[axis_2, axis_2_i]
        
        lower_point[inclosed_axis] = bounding_box[inclosed_axis,0]
        upper_point[inclosed_axis] = bounding_box[inclosed_axis,1]
        
        D1 = ant_loc-lower_point
        D2 = ant_loc-upper_point
        denom = np.linalg.norm( lower_point-upper_point )
        
        return np.linalg.norm( np.cross(D1,D2) )/denom
        
    elif num_inside_bounds == 2:
        ## a plane is the closest
        uninclosed_axis = [i for i in [0,1,2] if states[i]!=0][0]
        
        uninclosed_axis_i = 0 if states[uninclosed_axis]==-1 else 1
        
        return np.abs( ant_loc[uninclosed_axis] - bounding_box[uninclosed_axis, uninclosed_axis_i] )
    
def farthest_distance(ant_loc, bounding_box):
    """given an antenna location and a bounding box, and a bounding box, give farthest distance from antenna to a point on the box"""
    
    D = 0
    for X in bounding_box[0]:
        for Y in bounding_box[1]:
            for Z in bounding_box[2]:
                
                DX = ant_loc[0]-X
                DY = ant_loc[1]-Y
                DZ = ant_loc[2]-Z
                
                new_D = np.sqrt( DX*DX + DY*DY + DZ*DZ )
                if new_D > D:
                    D = new_D
    return D
            
    
        
        
#### functions to view data ####
class multi_slice_viewer:
    def __init__(self, image, axis, Xbounds, Ybounds, Zbounds):
        self.image = image
        self.axis = axis
        self.slice = [slice(None, None), slice(None, None), slice(None,None)]
        self.max_index = self.image.shape[ self.axis ]
        self.slice[axis] = 0
        
        self.max_image = np.max(self.image)
        self.min_image = np.min(self.image)
        self.fig, self.ax = plt.subplots(figsize=(20, 20))
        self.colorBar = None
        
        if axis == 0:
            self.A_bounds = Ybounds
            self.B_bounds = Zbounds
            self.C_bounds = Xbounds
            
        elif axis == 1:
            self.A_bounds = Xbounds
            self.B_bounds = Zbounds
            self.C_bounds = Ybounds
            
        elif axis == 2:
            self.A_bounds = Xbounds
            self.B_bounds = Ybounds
            self.C_bounds = Zbounds
            
        self.make_plot()
        self.button_press = self.fig.canvas.mpl_connect('scroll_event', self.onScroll)
        
    def make_plot(self, index=None):
        self.ax.clear()
#        if self.colorBar is not None:
#            self.colorBar.remove()
        
        if index is not None:
            self.slice[self.axis] = index
#        
        self.fig.suptitle(str((self.C_bounds[self.slice[self.axis]] + self.C_bounds[self.slice[self.axis]+1])*0.5))
            
        plot = self.ax.pcolormesh(self.A_bounds, self.B_bounds, self.image[self.slice].T, vmin=self.min_image,vmax=self.max_image)
        if self.colorBar is None:
            self.colorBar = self.fig.colorbar( plot )
        self.fig.canvas.draw()
        
    def onScroll(self, event):
        if event.button=='up':
           self.slice[self.axis] += 1
           if self.slice[self.axis] >= self.max_index:
               self.slice[self.axis] = 0
               
        elif event.button=='down':
           self.slice[self.axis] -= 1
           if self.slice[self.axis] <0:
               self.slice[self.axis] = self.max_index -1
               
        self.make_plot()
        
    def save_animation(self, fname, interval, dpi=80):
        anim = FuncAnimation(self.fig, self.make_plot, frames=np.arange(0, self.max_index), interval=interval)
        anim.save(fname, dpi=dpi, writer='imagemagick')














