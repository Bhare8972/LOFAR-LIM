#!/usr/bin/env python

## on APP machine

from __future__ import unicode_literals
import sys
import os

from time import sleep

import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
#matplotlib.use('svg')
from PyQt5 import QtCore, QtWidgets

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, RectangleSelector
import matplotlib.colors as colors

## mine
from LoLIM.utilities import v_air, processed_data_dir

def gen_cmap(cmap_name, minval,maxval, num=100):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, num)))
    return new_cmap

def List_to_Array(IN):
    if isinstance(IN, list):
        return np.array(IN)
    else:
        return IN

### TODO: make a data set that is a map of the stations!!!!



class coordinate_transform:
    """a coordinate transform defines how to go from LOFAR coordinates in MKS to plotted coordinates, and back again. This is the base transform. Which does nothing"""
    
    def __init__(self):
        self.x_label = "distance East/West (m)"
        self.y_label = "distance North/South (m)"
        self.z_label = "altitude (m)"
        self.t_label = "time (s)"
        
    def transform(self, X, Y, Z, T):
        return X, Y, Z, T
    
    def invert(self, X_bar, Y_bar, Z_bar, T_bar):
        return X_bar, Y_bar, Z_bar, T_bar
    
class typical_transform(coordinate_transform):
    """display in km and ms, with an arbitrary space zero and arbitrary time zero."""
    
    def __init__(self, space_center, time_center):
        """center should be in LOFAR (m,s) coordinates"""
        
        self.space_center = space_center
        self.time_center = time_center
        self.x_label = "distance east-west (km)"
        self.y_label = "distance north-south (km)"
        self.z_label = "altitude (km)"
        self.t_label = "time (ms)"
        
    def transform(self, X, Y, Z, T):
        X = List_to_Array(X)
        Y = List_to_Array(Y) 
        Z = List_to_Array(Z)
        T = List_to_Array(T)
        return (X-self.space_center[0])/1000.0, (Y-self.space_center[1])/1000.0, (Z-self.space_center[2])/1000.0, (T-self.time_center)*1000.0
    
    def invert(self, X_bar, Y_bar, Z_bar, T_bar):
        X_bar = List_to_Array(X_bar)
        Y_bar = List_to_Array(Y_bar)
        Z_bar = List_to_Array(Z_bar)
        T_bar = List_to_Array(T_bar)
        return X_bar*1000+self.space_center[0], Y_bar*1000+self.space_center[1], Z_bar*1000+self.space_center[2], T_bar/1000.0+self.time_center
    
## data sets work in lofar-centered mks. Except for plotting, where the coordinate system will be provided
class DataSet_Type:
    """All data plotted will be part of a data set. There can be different types of data sets, this class is a wrapper over all data sets"""
    
    def __init__(self, name):
        self.name = name
        self.display = True
        
    def set_show_all(self):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]
        Set all properties so that all points are shown"""
        return [[np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan]]
    
    def bounding_box(self):
        return [[np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan]]

    def set_T_lims(self, min, max):
        pass
    
    def set_X_lims(self, min, max):
        pass
    
    def set_Y_lims(self, min, max):
        pass
    
    def set_alt_lims(self, min, max):
        pass
    
    def get_T_lims(self):
        pass
    
    def get_X_lims(self):
        pass
    
    def get_Y_lims(self):
        pass
    
    def get_alt_lims(self):
        pass
    
    
    
    def get_all_properties(self):
        return {}
    
    def set_property(self, name, str_value):
        pass
    

    
    def plot(self, AltvsT_axes, AltvsEW_axes, NSvsEW_axes, NsvsAlt_axes, ancillary_axes, coordinate_system):
        pass
    
    
    
    def get_viewed_events(self):
        return []
    
    def clear(self):
        pass
    
    def use_ancillary_axes(self):
        return False

    def print_info(self):
        print( "not implemented" )
        
    def search(self, ID):
        print("not implemented")
        return None   
    
    def toggle_on(self):
        pass
    
    def toggle_off(self):
        pass
    
    def copy_view(self):
        print("not implemented")
        return None   
    
    def text_output(self):
        print("not implemented")
    
class DataSet_simplePointSources(DataSet_Type):
    """This represents a set of simple dual-polarized point sources"""
    
    def __init__(self, PSE_list, markers, marker_size, color_mode, name, cmap):
        self.markers = markers
        self.marker_size = marker_size
        self.color_mode = color_mode
        self.PSE_list = PSE_list
        self.cmap = cmap
        self.polarity = 0 ##0 is show both. 1 is show even, 2 is show odd
        
        self.t_lims = [None, None]
        self.x_lims = [None, None]
        self.y_lims = [None, None]
        self.z_lims = [None, None]
        self.max_RMS = None
        self.min_numAntennas = None
        
        ## probably should call previous constructor here
        self.name = name
        self.display = True
        
        
        #### get the data ####
        self.polE_loc_data = np.array([PSE.PolE_loc for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==2) ])
        self.polO_loc_data = np.array([PSE.PolO_loc for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==1) ])

        self.PolE_RMS_vals = np.array( [PSE.PolE_RMS for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==2)] )
        self.PolO_RMS_vals = np.array( [PSE.PolO_RMS for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==1)] )
        
        self.PolE_numAntennas= np.array( [PSE.num_even_antennas for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==2)])
        self.PolO_numAntennas= np.array( [PSE.num_odd_antennas for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==1)])
        
        
        #### make maskes
        bool_array = [True]*len(self.polE_loc_data)
        self.PolE_mask_on_alt = np.array(bool_array, dtype=bool)
        self.PolE_mask_on_X   = np.array(bool_array, dtype=bool)
        self.PolE_mask_on_Y   = np.array(bool_array, dtype=bool)
        self.PolE_mask_on_T   = np.array(bool_array, dtype=bool)
        self.PolE_mask_on_RMS = np.array(bool_array, dtype=bool)
        self.PolE_mask_on_min_numAntennas = np.array(bool_array, dtype=bool)
        
        self.PolE_total_mask = np.array(bool_array, dtype=bool)
        self.PolE_total_mask = np.stack([self.PolE_total_mask,self.PolE_total_mask,self.PolE_total_mask,self.PolE_total_mask], -1)
        
        
        bool_array = [True]*len(self.polO_loc_data)
        self.PolO_mask_on_alt = np.array(bool_array, dtype=bool)
        self.PolO_mask_on_X   = np.array(bool_array, dtype=bool)
        self.PolO_mask_on_Y   = np.array(bool_array, dtype=bool)
        self.PolO_mask_on_T   = np.array(bool_array, dtype=bool)
        self.PolO_mask_on_RMS = np.array(bool_array, dtype=bool)
        self.PolO_mask_on_min_numAntennas = np.array(bool_array, dtype=bool)
        
        self.PolO_total_mask = np.array(bool_array, dtype=bool)
        self.PolO_total_mask = np.stack([self.PolO_total_mask,self.PolO_total_mask,self.PolO_total_mask,self.PolO_total_mask], -1)
        
        
        self.PolE_masked_loc_data   = np.ma.masked_array(self.polE_loc_data, mask=self.PolE_total_mask, copy=False)
        self.PolE_masked_RMS_vals  = np.ma.masked_array(self.PolE_RMS_vals, mask=self.PolE_total_mask[:,0], copy=False)
        self.PolE_masked_numAntennas  = np.ma.masked_array(self.PolE_numAntennas, mask=self.PolE_total_mask[:,0], copy=False)
        
        self.PolO_masked_loc_data   = np.ma.masked_array(self.polO_loc_data, mask=self.PolO_total_mask, copy=False)
        self.PolO_masked_RMS_vals  = np.ma.masked_array(self.PolO_RMS_vals, mask=self.PolO_total_mask[:,0], copy=False)
        self.PolO_masked_numAntennas  = np.ma.masked_array(self.PolO_numAntennas, mask=self.PolO_total_mask[:,0], copy=False)
    
    
        
        self.set_show_all()
    
        #### some axis data ###
        self.PolE_AltVsT_paths = None
        self.PolE_AltVsEw_paths = None
        self.PolE_NsVsEw_paths = None
        self.PolE_NsVsAlt_paths = None
        
        self.PolO_AltVsT_paths = None
        self.PolO_AltVsEw_paths = None
        self.PolO_NsVsEw_paths = None
        self.PolO_NsVsAlt_paths = None
        
        
    def set_show_all(self):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
        
        max_RMS = max( np.max(self.PolE_RMS_vals), np.max(self.PolO_RMS_vals) )
        min_antennas = min( np.min(self.PolE_numAntennas), np.min(self.PolO_numAntennas) )
        
        self.set_max_RMS( max_RMS )
        self.set_min_numAntennas( min_antennas )
        
        min_X = min( np.min(self.polE_loc_data[:,0]), np.min(self.polO_loc_data[:,0]) )
        max_X = max( np.max(self.polE_loc_data[:,0]), np.max(self.polO_loc_data[:,0]) )
        
        min_Y = min( np.min(self.polE_loc_data[:,1]), np.min(self.polO_loc_data[:,1]) )
        max_Y = max( np.max(self.polE_loc_data[:,1]), np.max(self.polO_loc_data[:,1]) )
        
        min_Z = min( np.min(self.polE_loc_data[:,2]), np.min(self.polO_loc_data[:,2]) )
        max_Z = max( np.max(self.polE_loc_data[:,2]), np.max(self.polO_loc_data[:,2]) )
        
        min_T = min( np.min(self.polE_loc_data[:,3]), np.min(self.polO_loc_data[:,3]) )
        max_T = max( np.max(self.polE_loc_data[:,3]), np.max(self.polO_loc_data[:,3]) )
        
        self.set_T_lims(min_T, max_T)
        self.set_X_lims(min_X, max_X)
        self.set_Y_lims(min_Y, max_Y)
        self.set_alt_lims(min_Z, max_Z)
        
        return np.array([ [min_X, max_X], [min_Y, max_Y], [min_Z, max_Z], [min_T, max_T] ])
    
    def bounding_box(self):
        min_X = min( np.min(self.polE_loc_data[:,0]), np.min(self.polO_loc_data[:,0]) )
        max_X = max( np.max(self.polE_loc_data[:,0]), np.max(self.polO_loc_data[:,0]) )
        
        min_Y = min( np.min(self.polE_loc_data[:,1]), np.min(self.polO_loc_data[:,1]) )
        max_Y = max( np.max(self.polE_loc_data[:,1]), np.max(self.polO_loc_data[:,1]) )
        
        min_Z = min( np.min(self.polE_loc_data[:,2]), np.min(self.polO_loc_data[:,2]) )
        max_Z = max( np.max(self.polE_loc_data[:,2]), np.max(self.polO_loc_data[:,2]) )
        
        min_T = min( np.min(self.polE_loc_data[:,3]), np.min(self.polO_loc_data[:,3]) )
        max_T = max( np.max(self.polE_loc_data[:,3]), np.max(self.polO_loc_data[:,3]) )
        
        self.set_T_lims(min_T, max_T)
        self.set_X_lims(min_X, max_X)
        self.set_Y_lims(min_Y, max_Y)
        self.set_alt_lims(min_Z, max_Z)
        
        return np.array([ [min_X, max_X], [min_Y, max_Y], [min_Z, max_Z], [min_T, max_T] ])
    
    def set_T_lims(self, min, max):
        self.t_lims = [min, max]
        self.PolE_mask_on_T = np.logical_and( self.polE_loc_data[:,3]>min, self.polE_loc_data[:,3]<max )
        self.PolO_mask_on_T = np.logical_and( self.polO_loc_data[:,3]>min, self.polO_loc_data[:,3]<max )
    
    def set_X_lims(self, min, max):
        self.x_lims = [min, max]
        self.PolE_mask_on_X = np.logical_and( self.polE_loc_data[:,0]>min, self.polE_loc_data[:,0]<max )
        self.PolO_mask_on_X = np.logical_and( self.polO_loc_data[:,0]>min, self.polO_loc_data[:,0]<max )
    
    def set_Y_lims(self, min, max):
        self.y_lims = [min, max]
        self.PolE_mask_on_Y = np.logical_and( self.polE_loc_data[:,1]>min, self.polE_loc_data[:,1]<max )
        self.PolO_mask_on_Y = np.logical_and( self.polO_loc_data[:,1]>min, self.polO_loc_data[:,1]<max )
    
    def set_alt_lims(self, min, max):
        self.z_lims = [min, max]
        self.PolE_mask_on_alt = np.logical_and( self.polE_loc_data[:,2]>min, self.polE_loc_data[:,2]<max )
        self.PolO_mask_on_alt = np.logical_and( self.polO_loc_data[:,2]>min, self.polO_loc_data[:,2]<max )
    
    
    def get_T_lims(self):
        return list(self.t_lims)
    
    def get_X_lims(self):
        return list(self.x_lims)
    
    def get_Y_lims(self):
        return list(self.y_lims)
    
    def get_alt_lims(self):
        return list(self.z_lims)
    
    
    def get_all_properties(self):
        return {"marker size":str(self.marker_size),  "color mode":str(self.color_mode),  "max RMS (ns)":str(self.max_RMS*1.0E9),
                "min. num. ant.":str(self.min_numAntennas), "polarity":int(self.polarity)}
        ## need: marker type, color map
    
    def set_property(self, name, str_value):
        
        try:
            if name == "marker size":
                self.marker_size = int(str_value)
            elif name == "color mode":
                if str_value in ["time"] or str_value[0] =="*":
                    self.color_mode = str_value
            elif name == "max RMS (ns)":
                self.set_max_RMS( float(str_value)*1.0E-9 )
            elif name == "min. num. ant.":
                self.set_min_numAntennas( int(str_value) )
            elif name == "polarity":
                self.polarity = int(str_value)
            else:
                print("do not have property:", name)
        except:
            print("error in setting property", name, str_value)
            pass
    
    def set_max_RMS(self, max_RMS):
        self.max_RMS = max_RMS
        self.PolE_mask_on_RMS = self.PolE_RMS_vals<max_RMS
        self.PolO_mask_on_RMS = self.PolO_RMS_vals<max_RMS
    
    def set_min_numAntennas(self, min_numAntennas):
        self.min_numAntennas = min_numAntennas
        self.PolE_mask_on_min_numAntennas = self.PolE_masked_numAntennas>min_numAntennas
        self.PolO_mask_on_min_numAntennas = self.PolO_masked_numAntennas>min_numAntennas
    
    
    
    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system):

        ####  set total mask ####
        if self.polarity == 0 or self.polarity == 1:
            self.PolE_total_mask[:,0] = self.PolE_mask_on_alt
            np.logical_and(self.PolE_total_mask[:,0], self.PolE_mask_on_X, out=self.PolE_total_mask[:,0])
            np.logical_and(self.PolE_total_mask[:,0], self.PolE_mask_on_Y, out=self.PolE_total_mask[:,0])
            np.logical_and(self.PolE_total_mask[:,0], self.PolE_mask_on_T, out=self.PolE_total_mask[:,0])
            np.logical_and(self.PolE_total_mask[:,0], self.PolE_mask_on_RMS, out=self.PolE_total_mask[:,0])
            np.logical_and(self.PolE_total_mask[:,0], self.PolE_mask_on_min_numAntennas, out=self.PolE_total_mask[:,0])
            np.logical_not(self.PolE_total_mask[:,0], out=self.PolE_total_mask[:,0]) ##becouse the meaning of the masks is flipped
            self.PolE_total_mask[:,1] = self.PolE_total_mask[:,0]
            self.PolE_total_mask[:,2] = self.PolE_total_mask[:,0]
            self.PolE_total_mask[:,3] = self.PolE_total_mask[:,0]
            
        if self.polarity==0 or self.polarity==2:
            self.PolO_total_mask[:,0] = self.PolO_mask_on_alt
            np.logical_and(self.PolO_total_mask[:,0], self.PolO_mask_on_X, out=self.PolO_total_mask[:,0])
            np.logical_and(self.PolO_total_mask[:,0], self.PolO_mask_on_Y, out=self.PolO_total_mask[:,0])
            np.logical_and(self.PolO_total_mask[:,0], self.PolO_mask_on_T, out=self.PolO_total_mask[:,0])
            np.logical_and(self.PolO_total_mask[:,0], self.PolO_mask_on_RMS, out=self.PolO_total_mask[:,0])
            np.logical_and(self.PolO_total_mask[:,0], self.PolO_mask_on_min_numAntennas, out=self.PolO_total_mask[:,0])
            np.logical_not(self.PolO_total_mask[:,0], out=self.PolO_total_mask[:,0]) ##becouse the meaning of the masks is flipped
            self.PolO_total_mask[:,1] = self.PolO_total_mask[:,0]
            self.PolO_total_mask[:,2] = self.PolO_total_mask[:,0]
            self.PolO_total_mask[:,3] = self.PolO_total_mask[:,0]
    
        #### random book keeping ####
        self.clear()
        
        if not self.display:
            return
            
        if self.color_mode == "time":
            polE_color = self.PolE_masked_loc_data[:,3]
            polO_color = self.PolO_masked_loc_data[:,3]
            
        elif self.color_mode[0] == '*':
            polE_color = self.color_mode[1:]
            polO_color = self.color_mode[1:]
            
            
        #### plot ####
        if self.polarity == 0 or self.polarity == 1:
            X_bar, Y_bar, Z_bar, T_bar = coordinate_system.transform( X=self.PolE_masked_loc_data[:,0], Y=self.PolE_masked_loc_data[:,1],
                                                                     Z=self.PolE_masked_loc_data[:,2],  T=self.PolE_masked_loc_data[:,3])
            
            self.PolE_AltVsT_paths = AltVsT_axes.scatter(x=T_bar, y=Z_bar, c=polE_color, marker=self.markers[0], s=self.marker_size, cmap=self.cmap)
            self.PolE_AltVsEw_paths = AltVsEw_axes.scatter(x=X_bar, y=Z_bar, c=polE_color, marker=self.markers[0], s=self.marker_size, cmap=self.cmap)
            self.PolE_NsVsEw_paths = NsVsEw_axes.scatter(x=X_bar, y=Y_bar, c=polE_color, marker=self.markers[0], s=self.marker_size, cmap=self.cmap)
            self.PolE_NsVsAlt_paths = NsVsAlt_axes.scatter(x=Z_bar, y=Y_bar, c=polE_color, marker=self.markers[0], s=self.marker_size, cmap=self.cmap)
            
        if self.polarity==0 or self.polarity==2:
            X_bar, Y_bar, Z_bar, T_bar = coordinate_system.transform( X=self.PolO_masked_loc_data[:,0], Y=self.PolO_masked_loc_data[:,1],
                                                                     Z=self.PolO_masked_loc_data[:,2],  T=self.PolO_masked_loc_data[:,3])
            
            self.PolO_AltVsT_paths = AltVsT_axes.scatter(x=T_bar, y=Z_bar, c=polO_color, marker=self.markers[1], s=self.marker_size, cmap=self.cmap)
            self.PolO_AltVsEw_paths = AltVsEw_axes.scatter(x=X_bar, y=Z_bar, c=polO_color, marker=self.markers[1], s=self.marker_size, cmap=self.cmap)
            self.PolO_NsVsEw_paths = NsVsEw_axes.scatter(x=X_bar, y=Y_bar, c=polO_color, marker=self.markers[1], s=self.marker_size, cmap=self.cmap)
            self.PolO_NsVsAlt_paths = NsVsAlt_axes.scatter(x=Z_bar, y=Y_bar, c=polO_color, marker=self.markers[1], s=self.marker_size, cmap=self.cmap)
            
            
    def loc_filter(self, loc):
        if self.x_lims[0]<loc[0]<self.x_lims[1] and self.y_lims[0]<loc[1]<self.y_lims[1] and self.z_lims[0]<loc[2]<self.z_lims[1] and self.t_lims[0]<loc[3]<self.t_lims[1]:
            return True
        else:
            return False

    def get_viewed_events(self):
        return [PSE for PSE in self.PSE_list if 
                (self.loc_filter(PSE.PolE_loc) and PSE.PolE_RMS<self.max_RMS and PSE.num_even_antennas>self.min_numAntennas) 
                or (self.loc_filter(PSE.PolO_loc) and PSE.PolO_RMS<self.max_RMS and PSE.num_odd_antennas>self.min_numAntennas) ]

    
    def clear(self):
        if self.PolE_AltVsT_paths is not None:
            self.PolE_AltVsT_paths.remove()
            self.PolE_AltVsT_paths = None
        if self.PolE_AltVsEw_paths is not None:
            self.PolE_AltVsEw_paths.remove()
            self.PolE_AltVsEw_paths = None
        if self.PolE_NsVsEw_paths is not None:
            self.PolE_NsVsEw_paths.remove()
            self.PolE_NsVsEw_paths = None
        if self.PolE_NsVsAlt_paths is not None:
            self.PolE_NsVsAlt_paths.remove()
            self.PolE_NsVsAlt_paths = None
            
        if self.PolO_AltVsT_paths is not None:
            self.PolO_AltVsT_paths.remove()
            self.PolO_AltVsT_paths = None
        if self.PolO_AltVsEw_paths is not None:
            self.PolO_AltVsEw_paths.remove()
            self.PolO_AltVsEw_paths = None
        if self.PolO_NsVsEw_paths is not None:
            self.PolO_NsVsEw_paths.remove()
            self.PolO_NsVsEw_paths = None
        if self.PolO_NsVsAlt_paths is not None:
            self.PolO_NsVsAlt_paths.remove()
            self.PolO_NsVsAlt_paths = None
            
#    def search(self, index, marker, marker_size, color_mode, cmap=None):
#        searched_PSE = [PSE for PSE in self.PSE_list if PSE.unique_index==index]
#        return DataSet_simplePointSources( searched_PSE, [marker,marker], marker_size, color_mode, self.name+"_search", self.coordinate_system, cmap)

    def use_ancillary_axes(self):
        return False

    def print_info(self):
        print( "not implemented" )

class DataSet_interferometricPointSources(DataSet_Type):
    """This represents a set of simple dual-polarized point sources"""
    
    def __init__(self, IPSE_list, marker, marker_size, color_mode, name, cmap):
        self.marker = marker
        self.marker_size = marker_size
        self.color_mode = color_mode
        self.IPSE_list = IPSE_list
        self.cmap = cmap
        
        self.t_lims = [None, None]
        self.x_lims = [None, None]
        self.y_lims = [None, None]
        self.z_lims = [None, None]
        self.min_intensity = 0.0
        self.max_S1S2distance = 100.0
        self.min_amplitude = 0.0
        self.max_RMS = 1
        
        ## probably should call previous constructor here
        self.name = name
        self.display = True
        
        
        #### get the data ####
        self.locations = np.array([ np.append( IPSE.loc, [IPSE.T] ) for IPSE in IPSE_list if IPSE.converged])
        
        print(len(self.locations))

        self.intensities = np.array([ IPSE.intensity for IPSE in  IPSE_list if IPSE.converged])
        self.amplitudes = np.array([ IPSE.amplitude for IPSE in  IPSE_list if IPSE.converged])
        self.S1S2distances = np.array([ IPSE.S1_S2_distance for IPSE in  IPSE_list if IPSE.converged])
        self.RMSs = np.array([ IPSE.RMS for IPSE in  IPSE_list if IPSE.converged])
        
        
        #### make maskes
        bool_array = [True]*len(self.locations)
        self.mask_on_X   = np.array(bool_array, dtype=bool)
        self.mask_on_Y   = np.array(bool_array, dtype=bool)
        self.mask_on_alt = np.array(bool_array, dtype=bool)
        self.mask_on_T   = np.array(bool_array, dtype=bool)
        
        self.mask_on_intensity   = np.array(bool_array, dtype=bool)
        self.mask_on_S1S2distance = np.array(bool_array, dtype=bool)
        self.mask_on_amplitude = np.array(bool_array, dtype=bool)
        self.mask_on_RMS   = np.array(bool_array, dtype=bool)
        
        self.total_mask = np.array(bool_array, dtype=bool)
        self.total_mask = np.stack([self.total_mask,self.total_mask,self.total_mask,self.total_mask], -1)
        
        
        self.masked_loc_data   = np.ma.masked_array(self.locations, mask=self.total_mask, copy=False)
        
        self.set_show_all()
    
        #### some axis data ###
        self.AltVsT_paths = None
        self.AltVsEw_paths = None
        self.NsVsEw_paths = None
        self.NsVsAlt_paths = None
        
    def set_show_all(self):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
        
        self.set_min_intensity( np.min(self.intensities) )
        self.set_max_S1S2distance( np.max(self.S1S2distances) )
        self.set_min_amplitude( 0.0 )
        self.set_max_RMS( 1 )
        
        min_X = np.min(self.locations[:,0])
        max_X = np.max(self.locations[:,0])
        
        min_Y = np.min(self.locations[:,1])
        max_Y = np.max(self.locations[:,1])
        
        min_Z = np.min(self.locations[:,2])
        max_Z = np.max(self.locations[:,2])
        
        min_T = np.min(self.locations[:,3])
        max_T = np.max(self.locations[:,3])
        
        self.set_T_lims(min_T, max_T)
        self.set_X_lims(min_X, max_X)
        self.set_Y_lims(min_Y, max_Y)
        self.set_alt_lims(min_Z, max_Z)
        
        return np.array([ [min_X, max_X], [min_Y, max_Y], [min_Z, max_Z], [min_T, max_T] ])
    
    
    def bounding_box(self):
        mask = np.logical_and(self.mask_on_intensity, self.mask_on_S1S2distance)
        mask = np.logical_and(mask, self.mask_on_RMS, out=mask)
        mask = np.logical_and(mask, self.mask_on_amplitude, out=mask)
        
        min_X = np.min(self.locations[mask,0])
        max_X = np.max(self.locations[mask,0])
        
        min_Y = np.min(self.locations[mask,1])
        max_Y = np.max(self.locations[mask,1])
        
        min_Z = np.min(self.locations[mask,2])
        max_Z = np.max(self.locations[mask,2])
        
        min_T = np.min(self.locations[mask,3])
        max_T = np.max(self.locations[mask,3])
        
        self.set_T_lims(min_T, max_T)
        self.set_X_lims(min_X, max_X)
        self.set_Y_lims(min_Y, max_Y)
        self.set_alt_lims(min_Z, max_Z)
        
        return np.array([ [min_X, max_X], [min_Y, max_Y], [min_Z, max_Z], [min_T, max_T] ])
    
        
    def set_T_lims(self, min, max):
        self.t_lims = [min, max]
        self.mask_on_T = np.logical_and( self.locations[:,3]>min, self.locations[:,3]<max )
    
    def set_X_lims(self, min, max):
        self.x_lims = [min, max]
        self.mask_on_X = np.logical_and( self.locations[:,0]>min, self.locations[:,0]<max )
    
    def set_Y_lims(self, min, max):
        self.y_lims = [min, max]
        self.mask_on_Y = np.logical_and( self.locations[:,1]>min, self.locations[:,1]<max )
    
    def set_alt_lims(self, min, max):
        self.z_lims = [min, max]
        self.mask_on_alt = np.logical_and( self.locations[:,2]>min, self.locations[:,2]<max )
    
    
    def get_T_lims(self):
        return list(self.t_lims)
    
    def get_X_lims(self):
        return list(self.x_lims)
    
    def get_Y_lims(self):
        return list(self.y_lims)
    
    def get_alt_lims(self):
        return list(self.z_lims)
    
    
    
    def get_all_properties(self):
        return {"marker size":str(self.marker_size),  "color mode":str(self.color_mode),  "min intensity":str(self.min_intensity),
                "max S1S2 distance":str(self.max_S1S2distance), "min amplitude":str(self.min_amplitude), 'max RMS':str(self.max_RMS*1.0e9)}
        ## need: marker type, color map
    
    def set_property(self, name, str_value):
        
        try:
            if name == "marker size":
                self.marker_size = int(str_value)
            elif name == "color mode":
                if str_value in ["time", "amplitude", "intensity"] or str_value[0] =="*":
                    self.color_mode = str_value
            elif name == "max S1S2 distance":
                self.set_max_S1S2distance( float(str_value) )
            elif name == "min intensity":
                self.set_min_intensity( float(str_value) )
            elif name == "min amplitude":
                self.set_min_amplitude( float(str_value) )
            elif name == "max RMS":
                self.set_max_RMS( float(str_value)*1.0e-9 )
            else:
                print("do not have property:", name)
        except:
            print("error in setting property", name, str_value)
            pass
    
    def set_max_S1S2distance(self, max_S1S2distance):
        self.max_S1S2distance = max_S1S2distance
        self.mask_on_S1S2distance = self.S1S2distances < max_S1S2distance
    
    def set_min_intensity(self, min_intensity):
        self.min_intensity = min_intensity
        self.mask_on_intensity = self.intensities > min_intensity
        
    def set_min_amplitude(self, min_amplitude):
        self.min_amplitude = min_amplitude
        self.mask_on_amplitude = self.amplitudes > self.min_amplitude
        
    def set_max_RMS(self, max_RMS):
        self.max_RMS = max_RMS
        self.mask_on_RMS = self.RMSs < max_RMS
    
    
    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system):

        ####  set total mask ####
        self.total_mask[:,0] = self.mask_on_alt
        np.logical_and(self.total_mask[:,0], self.mask_on_X, out=self.total_mask[:,0])
        np.logical_and(self.total_mask[:,0], self.mask_on_Y, out=self.total_mask[:,0])
        np.logical_and(self.total_mask[:,0], self.mask_on_T, out=self.total_mask[:,0])
        np.logical_and(self.total_mask[:,0], self.mask_on_intensity, out=self.total_mask[:,0])
        np.logical_and(self.total_mask[:,0], self.mask_on_S1S2distance, out=self.total_mask[:,0])
        np.logical_and(self.total_mask[:,0], self.mask_on_amplitude, out=self.total_mask[:,0])
        np.logical_and(self.total_mask[:,0], self.mask_on_RMS, out=self.total_mask[:,0])
            
        np.logical_not(self.total_mask[:,0], out=self.total_mask[:,0]) ##becouse the meaning of the masks is flipped
        self.total_mask[:,1] = self.total_mask[:,0]
        self.total_mask[:,2] = self.total_mask[:,0]
        self.total_mask[:,3] = self.total_mask[:,0]
    
        #### random book keeping ####
        self.clear()
        
        if not self.display:
            return
        
        if self.color_mode == "time":
            color = self.locations[:,3]
            
        elif self.color_mode == "amplitude":
            color = np.log( self.amplitudes )
            
        elif self.color_mode == "intensity":
            color = self.intensities
            
        elif self.color_mode[0] == '*':
            color = self.color_mode[1:]
            
            
        #### plot ####
        X_bar, Y_bar, Z_bar, T_bar = coordinate_system.transform( X=self.masked_loc_data[:,0], Y=self.masked_loc_data[:,1],
                                                                 Z=self.masked_loc_data[:,2],  T=self.masked_loc_data[:,3])
        
        self.AltVsT_paths = AltVsT_axes.scatter(x=T_bar, y=Z_bar, c=color, marker=self.marker, s=self.marker_size, cmap=self.cmap)
        self.AltVsEw_paths = AltVsEw_axes.scatter(x=X_bar, y=Z_bar, c=color, marker=self.marker, s=self.marker_size, cmap=self.cmap)
        self.NsVsEw_paths = NsVsEw_axes.scatter(x=X_bar, y=Y_bar, c=color, marker=self.marker, s=self.marker_size, cmap=self.cmap)
        self.NsVsAlt_paths = NsVsAlt_axes.scatter(x=Z_bar, y=Y_bar, c=color, marker=self.marker, s=self.marker_size, cmap=self.cmap)
        

            
    def loc_filter(self, loc):
        if self.x_lims[0]<loc[0]<self.x_lims[1] and self.y_lims[0]<loc[1]<self.y_lims[1] and self.z_lims[0]<loc[2]<self.z_lims[1] and self.t_lims[0]<loc[3]<self.t_lims[1]:
            return True
        else:
            return False

    def get_viewed_events(self):
        print("get viewed events not implemented")
        return []
#        return [PSE for PSE in self.PSE_list if 
#                (self.loc_filter(PSE.PolE_loc) and PSE.PolE_RMS<self.max_RMS and PSE.num_even_antennas>self.min_numAntennas) 
#                or (self.loc_filter(PSE.PolO_loc) and PSE.PolO_RMS<self.max_RMS and PSE.num_odd_antennas>self.min_numAntennas) ]

    
    def clear(self):
        pass
#        if self.AltVsT_paths is not None:
#            self.AltVsT_paths.remove()
#            self.AltVsT_paths = None
#        if self.AltVsEw_paths is not None:
#            self.AltVsEw_paths.remove()
#            self.AltVsEw_paths = None
#        if self.NsVsEw_paths is not None:
#            self.NsVsEw_paths.remove()
#            self.NsVsEw_paths = None
#        if self.NsVsAlt_paths is not None:
#            self.NsVsAlt_paths.remove()
#            self.NsVsAlt_paths = None
        
    def use_ancillary_axes(self):
        return False
                

    def toggle_on(self):
        self.display = True
    
    def toggle_off(self):
        self.display = False
        self.clear()
        
    def search(self, ID):
        try:
            ID = int(ID)
        except:
            return None
        
        for IPSE in self.IPSE_list:
            if IPSE.unique_index == ID:
                break
            
        if IPSE.unique_index != ID:
            print("cannot find:", ID)
            return None
        
        if not IPSE.converged:
           print("event not converged")
           return None
        
        new_DS = DataSet_interferometricPointSources( [IPSE], self.marker, self.marker_size, "*k", self.name+"_S"+str(ID), self.cmap )
        new_DS.set_max_S1S2distance( int(new_DS.max_S1S2distance)+2)
        return new_DS

    def print_info(self):
        print()
        print()
        i=0
        for IPSE in self.IPSE_list:
            if self.loc_filter( np.append(IPSE.loc, [IPSE.T]) ) and IPSE.intensity>self.min_intensity and IPSE.S1_S2_distance<self.max_S1S2distance  \
            and IPSE.converged and IPSE.amplitude>self.min_amplitude and IPSE.RMS<self.max_RMS:
                print( "IPSE  block:", IPSE.block_index, "ID:", IPSE.IPSE_index, "unique index:", IPSE.unique_index )
                print( "    S1-S2 distance:", IPSE.S1_S2_distance, "intensity:", IPSE.intensity, "amplitude:", IPSE.amplitude)
                print( "    location:", IPSE.loc)
                print( "    T:", IPSE.T, "  RMS:", IPSE.RMS)
                i+=1
        print(i, "sources")
        
    
    def copy_view(self):
        new_list = []
        for IPSE in self.IPSE_list:
            if self.loc_filter( np.append(IPSE.loc, [IPSE.T]) ) and IPSE.intensity>self.min_intensity and IPSE.S1_S2_distance<self.max_S1S2distance  \
            and IPSE.converged and IPSE.amplitude>self.min_amplitude and IPSE.RMS<self.max_RMS:
                new_list.append( IPSE )
                
        if len(new_list) ==0:
            return None
                
        new_DS = DataSet_interferometricPointSources( new_list, self.marker, self.marker_size, "*k", self.name+"_copy", self.cmap )
        
        new_DS.set_max_S1S2distance( self.max_S1S2distance )
        new_DS.set_min_intensity( self.min_intensity )
        new_DS.set_min_amplitude( self.min_amplitude )
        return new_DS
    
    
    def text_output(self):
        with open("output.txt", 'w') as fout:
            for IPSE in self.IPSE_list:
                if self.loc_filter( np.append(IPSE.loc, [IPSE.T]) ) and IPSE.intensity>self.min_intensity and IPSE.S1_S2_distance<self.max_S1S2distance  \
                and IPSE.converged and IPSE.amplitude>self.min_amplitude and IPSE.RMS<self.max_RMS:
                    ID = IPSE.unique_index
                    X = IPSE.loc[0]
                    Y = IPSE.loc[1]
                    Z = IPSE.loc[2]
                    T = IPSE.T
                    I = IPSE.intensity
                    fout.write( str(ID)+' E '+str(X)+" "+str(Y)+" "+str(Z)+" "+str(T)+" " + str(I)+'\n' )
    
        
            
    
#class DataSet_IPSE_threeStage(DataSet_Type):
#    """This represents a set of simple dual-polarized point sources"""
#    
#    def __init__(self, IPSE_list, loc_stage, marker, marker_size, color_mode, name, cmap):
#        self.marker = marker
#        self.marker_size = marker_size
#        self.color_mode = color_mode
#        self.IPSE_list = IPSE_list
#        self.cmap = cmap
#        self.loc_stage = loc_stage ## says which stage (1,2,or 3) to plot locations from
#        
#        self.t_lims = [None, None]
#        self.x_lims = [None, None]
#        self.y_lims = [None, None]
#        self.z_lims = [None, None]
#        self.min_S1intensity = 0.0
#        self.max_S1S2distance = 100.0
#        self.min_S2intensity = 0.0
#        self.min_S3intensity = 0.0
#        self.max_S2S3distance = 100.0
#        self.require_S2convergence = 1
#        self.min_amplitude = 0.0
#        self.max_S2_RMS = 1.0
#        self.max_S3_RMS = 1.0
#        
#        ## probably should call previous constructor here
#        self.name = name
#        self.display = True
#        
#        
#        #### get the data ####
#        if self.loc_stage == 1:
#            self.locations = np.array([ IPSE.S1_XYZT for IPSE in IPSE_list])
#        elif self.loc_stage == 2:
#            self.locations = np.array([ IPSE.S2_XYZT for IPSE in IPSE_list])
#        elif self.loc_stage == 3:
#            self.locations = np.array([ IPSE.S3_XYZT for IPSE in IPSE_list])
#
#        self.S1_intensities = np.array([ IPSE.S1_intensity for IPSE in  IPSE_list])
#        self.S2_intensities = np.array([ IPSE.S2_intensity for IPSE in  IPSE_list])
#        self.S3_intensities = np.array([ IPSE.S3_intensity for IPSE in  IPSE_list])
#        self.amplitudes = np.array([ IPSE.amplitude for IPSE in  IPSE_list])
#        self.S1S2distances = np.array([ IPSE.S1_S2_distance for IPSE in  IPSE_list])
#        self.S2S3distances = np.array([ IPSE.S2_S3_distance for IPSE in  IPSE_list])
#        self.S2_converged = np.array([ IPSE.S2_converged for IPSE in  IPSE_list])
#        self.S2_RMS = np.array([ IPSE.S2_RMS for IPSE in  IPSE_list])
#        self.S3_RMS = np.array([ IPSE.S3_RMS for IPSE in  IPSE_list])
#        
#        
#        #### make maskes
#        bool_array = [True]*len(self.locations)
#        self.mask_on_X   = np.array(bool_array, dtype=bool)
#        self.mask_on_Y   = np.array(bool_array, dtype=bool)
#        self.mask_on_alt = np.array(bool_array, dtype=bool)
#        self.mask_on_T   = np.array(bool_array, dtype=bool)
#        
#        self.mask_on_S1_intensity   = np.array(bool_array, dtype=bool)
#        self.mask_on_S2_intensity   = np.array(bool_array, dtype=bool)
#        self.mask_on_S3_intensity   = np.array(bool_array, dtype=bool)
#        self.mask_on_S1S2distance = np.array(bool_array, dtype=bool)
#        self.mask_on_S2S3distance = np.array(bool_array, dtype=bool)
#        self.mask_on_amplitude = np.array(bool_array, dtype=bool)
#        self.mask_on_convergence = np.array(bool_array, dtype=bool)
#        self.mask_on_S2_RMS = np.array(bool_array, dtype=bool)
#        self.mask_on_S3_RMS = np.array(bool_array, dtype=bool)
#        
#        self.total_mask = np.array(bool_array, dtype=bool)
#        self.total_mask = np.stack([self.total_mask,self.total_mask,self.total_mask,self.total_mask], -1)
#        
#        
#        self.masked_loc_data   = np.ma.masked_array(self.locations, mask=self.total_mask, copy=False)
#        
#        self.set_show_all()
#    
#        #### some axis data ###
#        self.AltVsT_paths = None
#        self.AltVsEw_paths = None
#        self.NsVsEw_paths = None
#        self.NsVsAlt_paths = None
#        
#    def set_show_all(self):
#        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
#        
#        self.set_min_S1intensity( 0.0 )
#        self.set_min_S2intensity( 0.0 )
#        self.set_min_S3intensity( 0.0 )
#        self.set_max_S1S2distance( np.max(self.S1S2distances) )
#        self.set_max_S2S3distance( np.max(self.S2S3distances) )
#        self.set_min_amplitude( 0.0 )
#        self.set_max_S2_RMS( np.max(self.S2_RMS) )
#        self.set_max_S3_RMS( np.max(self.S3_RMS) )
#        
#        min_X = np.min(self.locations[:,0])
#        max_X = np.max(self.locations[:,0])
#        
#        min_Y = np.min(self.locations[:,1])
#        max_Y = np.max(self.locations[:,1])
#        
#        min_Z = np.min(self.locations[:,2])
#        max_Z = np.max(self.locations[:,2])
#        
#        min_T = np.min(self.locations[:,3])
#        max_T = np.max(self.locations[:,3])
#        
#        self.set_T_lims(min_T, max_T)
#        self.set_X_lims(min_X, max_X)
#        self.set_Y_lims(min_Y, max_Y)
#        self.set_alt_lims(min_Z, max_Z)
#        
#        return np.array([ [min_X, max_X], [min_Y, max_Y], [min_Z, max_Z], [min_T, max_T] ])
#    
#    
#    def bounding_box(self):
#        mask = np.logical_and(self.mask_on_S1_intensity, self.mask_on_S2_intensity)
#        mask = np.logical_and(mask, self.mask_on_S3_intensity, out=mask)
#        mask = np.logical_and(mask, self.mask_on_S1S2distance, out=mask)
#        mask = np.logical_and(mask, self.mask_on_S2S3distance, out=mask)
#        mask = np.logical_and(mask, self.mask_on_amplitude, out=mask)
#        mask = np.logical_and(mask, self.mask_on_convergence, out=mask)
#        mask = np.logical_and(mask, self.mask_on_S2_RMS, out=mask)
#        mask = np.logical_and(mask, self.mask_on_S3_RMS, out=mask)
#        
#        min_X = np.min(self.locations[mask,0])
#        max_X = np.max(self.locations[mask,0])
#        
#        min_Y = np.min(self.locations[mask,1])
#        max_Y = np.max(self.locations[mask,1])
#        
#        min_Z = np.min(self.locations[mask,2])
#        max_Z = np.max(self.locations[mask,2])
#        
#        min_T = np.min(self.locations[mask,3])
#        max_T = np.max(self.locations[mask,3])
#        
#        self.set_T_lims(min_T, max_T)
#        self.set_X_lims(min_X, max_X)
#        self.set_Y_lims(min_Y, max_Y)
#        self.set_alt_lims(min_Z, max_Z)
#        
#        return np.array([ [min_X, max_X], [min_Y, max_Y], [min_Z, max_Z], [min_T, max_T] ])
#    
#        
#    def set_T_lims(self, min, max):
#        self.t_lims = [min, max]
#        self.mask_on_T = np.logical_and( self.locations[:,3]>min, self.locations[:,3]<max )
#    
#    def set_X_lims(self, min, max):
#        self.x_lims = [min, max]
#        self.mask_on_X = np.logical_and( self.locations[:,0]>min, self.locations[:,0]<max )
#    
#    def set_Y_lims(self, min, max):
#        self.y_lims = [min, max]
#        self.mask_on_Y = np.logical_and( self.locations[:,1]>min, self.locations[:,1]<max )
#    
#    def set_alt_lims(self, min, max):
#        self.z_lims = [min, max]
#        self.mask_on_alt = np.logical_and( self.locations[:,2]>min, self.locations[:,2]<max )
#    
#    
#    def get_T_lims(self):
#        return list(self.t_lims)
#    
#    def get_X_lims(self):
#        return list(self.x_lims)
#    
#    def get_Y_lims(self):
#        return list(self.y_lims)
#    
#    def get_alt_lims(self):
#        return list(self.z_lims)
#    
#    
#    
#    def get_all_properties(self):
#        return {"marker size":str(self.marker_size),  "color mode":str(self.color_mode),  "min S1 intensity":str(self.min_S1intensity),
#                  "min S2 intensity":str(self.min_S2intensity),  "min S3 intensity":str(self.min_S3intensity),  
#                  "max S1S2 distance":str(self.max_S1S2distance), "max S2S3 distance":str(self.max_S2S3distance),
#                  "require S2 convergence":str(self.require_S2convergence),  "min amplitude":str(self.min_amplitude),
#                  "max S2 RMS":str(self.max_S2_RMS*(1e9)), "max S3 RMS":str(self.max_S3_RMS*1e9)}
#        ## need: marker type, color map
#    
#    def set_property(self, name, str_value):
#        
#        try:
#            if name == "marker size":
#                self.marker_size = int(str_value)
#            elif name == "color mode":
#                if str_value in ["time", "amplitude", "intensity"] or str_value[0] =="*":
#                    self.color_mode = str_value
#                    
#            elif name == "max S1S2 distance":
#                self.set_max_S1S2distance( float(str_value) )
#            elif name == "max S2S3 distance":
#                self.set_max_S2S3distance( float(str_value) )
#                
#            elif name == "min S1 intensity":
#                self.set_min_S1intensity( float(str_value) )
#            elif name == "min S2 intensity":
#                self.set_min_S2intensity( float(str_value) )
#            elif name == "min S3 intensity":
#                self.set_min_S3intensity( float(str_value) )
#                
#            elif name == "require S2 convergence":
#                self.set_req_convergence( int(str_value) )
#            elif name == "min amplitude":
#                self.set_min_amplitude( float(str_value) )
#            elif name == "max S2 RMS":
#                self.set_max_S2_RMS( float(str_value)*(1e-9) )
#            elif name == "max S3 RMS":
#                self.set_max_S3_RMS( float(str_value)*(1e-9) )
#            else:
#                print("do not have property:", name)
#        except:
#            print("error in setting property", name, str_value)
#            pass
#    
#    def set_max_S1S2distance(self, max_S1S2distance):
#        self.max_S1S2distance = max_S1S2distance
#        self.mask_on_S1S2distance = self.S1S2distances < max_S1S2distance
#    
#    def set_max_S2S3distance(self, max_S2S3distance):
#        self.max_S2S3distance = max_S2S3distance
#        self.mask_on_S2S3distance = self.S2S3distances < max_S2S3distance
#    
#    def set_min_S1intensity(self, min_S1intensity):
#        self.min_S1intensity = min_S1intensity
#        self.mask_on_S1_intensity = self.S1_intensities > min_S1intensity
#    
#    def set_min_S2intensity(self, min_S2intensity):
#        self.min_S2intensity = min_S2intensity
#        self.mask_on_S2_intensity = self.S2_intensities > min_S2intensity
#    
#    def set_min_S3intensity(self, min_S3intensity):
#        self.min_S3intensity = min_S3intensity
#        self.mask_on_S3_intensity = self.S3_intensities > min_S3intensity
#    
#    def set_max_S2_RMS(self, max_S2_RMS):
#        self.max_S2_RMS = max_S2_RMS
#        self.mask_on_S2_RMS = self.S2_RMS < max_S2_RMS
#    
#    def set_max_S3_RMS(self, max_S3_RMS):
#        self.max_S3_RMS = max_S3_RMS
#        self.mask_on_S3_RMS = self.S3_RMS < max_S3_RMS
#        
#    def set_req_convergence(self, req_convergence):
#        self.require_S2convergence = req_convergence
#        if req_convergence:
#            self.mask_on_convergence = np.ones(len(self.locations), dtype=bool )
#        else:
#            self.mask_on_convergence = self.S2_converged
#        
#    def set_min_amplitude(self, min_amplitude):
#        self.min_amplitude = min_amplitude
#        self.mask_on_amplitude = self.amplitudes > self.min_amplitude
#    
#    
#    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system):
#
#        ####  set total mask ####
#        self.total_mask[:,0] = self.mask_on_alt
#        np.logical_and(self.total_mask[:,0], self.mask_on_X, out=self.total_mask[:,0])
#        np.logical_and(self.total_mask[:,0], self.mask_on_Y, out=self.total_mask[:,0])
#        np.logical_and(self.total_mask[:,0], self.mask_on_T, out=self.total_mask[:,0])
#        np.logical_and(self.total_mask[:,0], self.mask_on_S1_intensity, out=self.total_mask[:,0])
#        np.logical_and(self.total_mask[:,0], self.mask_on_S2_intensity, out=self.total_mask[:,0])
#        np.logical_and(self.total_mask[:,0], self.mask_on_S3_intensity, out=self.total_mask[:,0])
#        np.logical_and(self.total_mask[:,0], self.mask_on_S1S2distance, out=self.total_mask[:,0])
#        np.logical_and(self.total_mask[:,0], self.mask_on_S2S3distance, out=self.total_mask[:,0])
#        np.logical_and(self.total_mask[:,0], self.mask_on_amplitude, out=self.total_mask[:,0])
#        np.logical_and(self.total_mask[:,0], self.mask_on_convergence, out=self.total_mask[:,0])
#        np.logical_and(self.total_mask[:,0], self.mask_on_S2_RMS, out=self.total_mask[:,0])
#        np.logical_and(self.total_mask[:,0], self.mask_on_S3_RMS, out=self.total_mask[:,0])
#            
#        np.logical_not(self.total_mask[:,0], out=self.total_mask[:,0]) ##becouse the meaning of the masks is flipped
#        self.total_mask[:,1] = self.total_mask[:,0]
#        self.total_mask[:,2] = self.total_mask[:,0]
#        self.total_mask[:,3] = self.total_mask[:,0]
#    
#        #### random book keeping ####
#        self.clear()
#        
#        if not self.display:
#            return
#        
#        if self.color_mode == "time":
#            color = self.locations[:,3]
#            
#        elif self.color_mode == "amplitude":
#            color = np.log( self.amplitudes )
#            
#        elif self.color_mode == "intensity":
#            color = self.intensities
#            
#        elif self.color_mode[0] == '*':
#            color = self.color_mode[1:]
#            
#            
#        #### plot ####
#        X_bar, Y_bar, Z_bar, T_bar = coordinate_system.transform( X=self.masked_loc_data[:,0], Y=self.masked_loc_data[:,1],
#                                                                 Z=self.masked_loc_data[:,2],  T=self.masked_loc_data[:,3])
#        
#        self.AltVsT_paths = AltVsT_axes.scatter(x=T_bar, y=Z_bar, c=color, marker=self.marker, s=self.marker_size, cmap=self.cmap)
#        self.AltVsEw_paths = AltVsEw_axes.scatter(x=X_bar, y=Z_bar, c=color, marker=self.marker, s=self.marker_size, cmap=self.cmap)
#        self.NsVsEw_paths = NsVsEw_axes.scatter(x=X_bar, y=Y_bar, c=color, marker=self.marker, s=self.marker_size, cmap=self.cmap)
#        self.NsVsAlt_paths = NsVsAlt_axes.scatter(x=Z_bar, y=Y_bar, c=color, marker=self.marker, s=self.marker_size, cmap=self.cmap)
#        
#
#            
#    def loc_filter(self, loc):
#        if self.x_lims[0]<loc[0]<self.x_lims[1] and self.y_lims[0]<loc[1]<self.y_lims[1] and self.z_lims[0]<loc[2]<self.z_lims[1] and self.t_lims[0]<loc[3]<self.t_lims[1]:
#            return True
#        else:
#            return False
#
#    def get_viewed_events(self):
#        print("get viewed events not implemented")
#        return []
##        return [PSE for PSE in self.PSE_list if 
##                (self.loc_filter(PSE.PolE_loc) and PSE.PolE_RMS<self.max_RMS and PSE.num_even_antennas>self.min_numAntennas) 
##                or (self.loc_filter(PSE.PolO_loc) and PSE.PolO_RMS<self.max_RMS and PSE.num_odd_antennas>self.min_numAntennas) ]
#
#    
#    def clear(self):
#        pass
##        if self.AltVsT_paths is not None:
##            self.AltVsT_paths.remove()
##            self.AltVsT_paths = None
##        if self.AltVsEw_paths is not None:
##            self.AltVsEw_paths.remove()
##            self.AltVsEw_paths = None
##        if self.NsVsEw_paths is not None:
##            self.NsVsEw_paths.remove()
##            self.NsVsEw_paths = None
##        if self.NsVsAlt_paths is not None:
##            self.NsVsAlt_paths.remove()
##            self.NsVsAlt_paths = None
#        
#    def use_ancillary_axes(self):
#        return False
#                
#
#    def toggle_on(self):
#        self.display = True
#    
#    def toggle_off(self):
#        self.display = False
#        self.clear()
#        
#    def search(self, ID):
#        try:
#            ID = int(ID)
#        except:
#            return None
#        
#        for IPSE in self.IPSE_list:
#            if IPSE.unique_index == ID:
#                break
#            
#        if IPSE.unique_index != ID:
#            print("cannot find:", ID)
#            return None
#        
#        new_DS = DataSet_IPSE_threeStage( [IPSE], self.loc_stage, self.marker, self.marker_size, "*k", self.name+"_S"+str(ID), self.cmap )
#        
#        new_DS.set_max_S1S2distance( 10000.0 )
#        new_DS.set_max_S2S3distance( 10000.0 )
#        
#        new_DS.set_min_S1intensity( 0.0 )
#        new_DS.set_min_S3intensity( 0.0 )
#        new_DS.set_min_S3intensity( 0.0 )
#        
#        new_DS.set_req_convergence( 0 )
#        new_DS.set_min_amplitude( 0.0 )
#        
#        new_DS.set_max_S2_RMS( 1.0 )
#        new_DS.set_max_S3_RMS( 1.0 )
#        return new_DS
#
#    def print_info(self):
#        print()
#        print()
#        i=0
#        for IPSE_i in range(len(self.locations)):
#            if self.loc_filter( self.locations[IPSE_i] ) and self.S1_intensities[IPSE_i]>self.min_S1intensity and self.S2_intensities[IPSE_i]>self.min_S2intensity \
#                 and self.S3_intensities[IPSE_i]>self.min_S3intensity and self.S1S2distances[IPSE_i]<self.max_S1S2distance \
#                 and self.S2S3distances[IPSE_i]<self.max_S2S3distance and (self.S2_converged[IPSE_i] or not self.require_S2convergence) \
#                 and self.S2_RMS[IPSE_i]<self.max_S2_RMS and self.S3_RMS[IPSE_i]<self.max_S3_RMS and self.amplitudes[IPSE_i]>self.min_amplitude:
#                
#                IPSE = self.IPSE_list[IPSE_i]
#                print("IPSE  block:", IPSE.block_index, "ID:", IPSE.IPSE_index, "unique index:", IPSE.unique_index )
#                print("    S1-S2 D:", IPSE.S1_S2_distance, "S2-S3 D:", IPSE.S2_S3_distance)
#                print("    S1 i:", IPSE.S1_intensity, "S2 i:",  IPSE.S2_intensity, "S3 i:", IPSE.S3_intensity)
#                print("    amplitude:", IPSE.amplitude)
#                print("    S2 RMS:", self.S2_RMS[IPSE_i], 'S3 RMS', self.S2_RMS[IPSE_i])
#                print("    XYZT:", self.locations[IPSE_i])
#                print()
#                i+=1
#        print(i, "sources")
#        
#    
#    def copy_view(self):
#        new_list = []
#        for IPSE_i in range(len(self.locations)):
#            if self.loc_filter( self.locations[IPSE_i] ) and self.S1_intensities[IPSE_i]>self.min_S1intensity and self.S2_intensities[IPSE_i]>self.min_S2intensity \
#                 and self.S3_intensities[IPSE_i]>self.min_S3intensity and self.S1S2distances[IPSE_i]>self.max_S1S2distance \
#                 and self.S2S3distances[IPSE_i]>self.max_S2S3distance and (self.S2_converged[IPSE_i] or not self.require_S2convergence)\
#                 and self.S2_RMS[IPSE_i]<self.max_S2_RMS and self.S3_RMS[IPSE_i]<self.max_S3_RMS and self.amplitudes[IPSE_i]>self.min_amplitude:
#                
#                new_list.append( self.IPSE_list[IPSE_i] )
#                
#        if len(new_list) ==0:
#            return None
#                
#        new_DS = DataSet_IPSE_threeStage( new_list, self.loc_stage, self.marker, self.marker_size, "*k", self.name+"_copy", self.cmap )
#        
#        new_DS.set_max_S1S2distance( self.max_S1S2distance )
#        new_DS.set_max_S2S3distance( self.max_S2S3distance )
#        
#        new_DS.set_min_S1intensity( self.min_S1intensity )
#        new_DS.set_min_S3intensity( self.min_S2intensity )
#        new_DS.set_min_S3intensity( self.min_S3intensity )
#        
#        new_DS.set_req_convergence( self.require_S2convergence )
#        new_DS.set_min_amplitude( self.min_amplitude )
#        return new_DS
        
class DataSet_arrow(DataSet_Type):
    """This represents a set of simple dual-polarized point sources"""
    
    def __init__(self, name, X, Y, Z, T, azimuth, zenith, length, color):
        self.XYZT = np.array([X,Y,Z,T])
        self.azimuth = azimuth
        self.zenith = zenith
        self.length = length
        self.color = color
        self.linewidth = 10
        
        self.set_arrow()
        
        ## probably should call previous constructor here
        self.name = name
        self.display = True
        
        self.in_bounds = True
        
        self.AltVsT_lines = None
        self.AltVsEw_lines = None
        self.NsVsEw_lines = None
        self.NsVsAlt_lines = None
        
        
    def set_show_all(self):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
        
        return np.array([ [self.X[0], self.X[1]], [self.Y[0], self.Y[1]], [self.Z[0], self.Z[1]], [self.T[0], self.T[1]] ])
    
    
    def bounding_box(self):
        return self.set_show_all()
    
        
    def set_T_lims(self, min, max):
        pass
#        self.in_bounds = self.in_bounds and min <= self.XYZT[3] <= max
    
    def set_X_lims(self, min, max):
        pass
#        self.in_bounds = self.in_bounds and min <= self.XYZT[0] <= max
    
    def set_Y_lims(self, min, max):
        pass
#        self.in_bounds = self.in_bounds and min <= self.XYZT[1] <= max
    
    def set_alt_lims(self, min, max):
        pass
#        self.in_bounds = self.in_bounds and min <= self.XYZT[2] <= max
    
    
    def get_T_lims(self):
        return list(self.T)
    
    def get_X_lims(self):
        return list(self.X)
    
    def get_Y_lims(self):
        return list(self.Y)
    
    def get_alt_lims(self):
        return list(self.Z)
    
    
    
    def get_all_properties(self):
        return {"X":str(self.XYZT[0]), "Y":str(self.XYZT[1]), "Z":str(self.XYZT[2]), "T":str(self.XYZT[3]), 'linewidth':float(self.linewidth),
                "length":str(self.length), "azimuth":str(self.azimuth), "zenith":str(self.zenith), "color":str(self.color)}
    
    def set_property(self, name, str_value):
        
        try:
                
            if name == "X":
                try:
                    self.XYZT[0] = float(str_value)
                except:
                    print("input error")
                    return
                self.set_arrow()
                
            elif name == "Y":
                try:
                    self.XYZT[1] = float(str_value)
                except:
                    print("input error")
                    return
                self.set_arrow()
                
            elif name == "Z":
                try:
                    self.XYZT[2] = float(str_value)
                except:
                    print("input error")
                    return
                self.set_arrow()
                
            elif name == "T":
                try:
                    self.XYZT[3] = float(str_value)
                except:
                    print("input error")
                    return
                self.set_arrow()
                
            elif name == "length":
                try:
                    self.length = float(str_value)
                except:
                    print("input error")
                    return
                self.set_arrow()
                
            elif name == "azimuth":
                try:
                    self.azimuth = float(str_value)
                except:
                    print("input error")
                    return
                self.set_arrow()
                
            elif name == "zenith":
                try:
                    self.zenith = float(str_value)
                except:
                    print("input error")
                    return
                self.set_arrow()
                
            elif name == "color":
                self.color = str_value
                
            elif name == 'linewidth':
                try:
                    self.linewidth = float(str_value)
                except:
                    print("input error")
                    return
                
            else:
                print("do not have property:", name)
        except:
            print("error in setting property", name, str_value)
            pass
        
    def set_arrow(self):
        dz = np.cos(self.zenith)*self.length*0.5
        dx = np.sin(self.zenith)*np.cos(self.azimuth)*self.length*0.5
        dy = np.sin(self.zenith)*np.sin(self.azimuth)*self.length*0.5

        X = self.XYZT[0]
        Y = self.XYZT[1]
        Z = self.XYZT[2]
        T = self.XYZT[3]

        self.X = np.array([X-dx,X+dx])
        self.Y = np.array([Y-dy,Y+dy])
        self.Z = np.array([Z-dz,Z+dz])
        self.T = np.array([T, T])
        
        print('[', self.XYZT[0], ',', self.XYZT[1], ',', self.XYZT[2], ',', self.XYZT[3], ']')

    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system):

    
        #### random book keeping ####
        self.clear()
        
        if not self.display or not self.in_bounds:
            return

            
            
        #### plot ####
        X_bar, Y_bar, Z_bar, T_bar = coordinate_system.transform( X=self.X, Y=self.Y, Z=self.Z, T=self.T)
        
        self.AltVsT_lines = AltVsT_axes.plot(T_bar, Z_bar, marker='o', ls='-', color=self.color, lw=self.linewidth)
        self.AltVsEw_lines = AltVsEw_axes.plot(X_bar, Z_bar, marker='o', ls='-', color=self.color, lw=self.linewidth)
        self.NsVsEw_lines = NsVsEw_axes.plot(X_bar, Y_bar, marker='o', ls='-', color=self.color, lw=self.linewidth)
        self.NsVsAlt_lines = NsVsAlt_axes.plot(Z_bar, Y_bar, marker='o', ls='-', color=self.color, lw=self.linewidth)

    def get_viewed_events(self):
        print("get viewed events not implemented")
        return []
    
    def clear(self):
        pass
#        if self.AltVsT_lines is not None:
#            self.AltVsT_lines = None
#            
#        if self.AltVsEw_lines is not None:
#            self.AltVsEw_lines = None
#            
#        if self.NsVsEw_lines is not None:
#            self.NsVsEw_lines = None
#            
#        if self.NsVsAlt_lines is not None:
#            self.NsVsAlt_lines = None
            
            
    def use_ancillary_axes(self):
        return False

    def print_info(self):
        print("not implented")
                
    def search(self, ID):
        print("not implmented")
        return None
         
    
    def toggle_on(self):
        self.display = True
    
    def toggle_off(self):
        self.display = False
        self.clear()
            
            
#class DataSet_temperature:
#    
#    def __init__(self, name, input_fname, lower_lim, upper_lim, color='b', width=10):
#        self.name = name
#        self.display = True
#        self.color = color
#        self.width = width
#        self.lower_lim = lower_lim
#        self.upper_lim = upper_lim
#        
#        self.input_fname = input_fname
#        alt = []
#        T = []
#        with open(input_fname, 'r') as fin:
#            for line in fin:
#                A,B = line.split()
#                alt.append( float(A) )
#                T.append( float(B) )
#                
#        self.altitude = np.array( alt )
#        self.temperature = np.array( T )
#        self.path = None
#        
#    def set_coordinate_system(self, coordinate_system):
#        self.coordinate_system = coordinate_system
#        self.transform_alt = np.array([ coordinate_system.transform(0.0,0.0,Z,0.0)[2] for Z in self.altitude ])
#    
#    def getSpaceBounds(self):
#        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
#        return [[np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan]]
#    
#    def getFilterBounds(self):
#        """return max_RMS, min_antennas"""
#        return [np.nan, np.nan]
#    
#    def set_T_lims(self, min, max):
#        pass
#    
#    def set_X_lims(self, min, max):
#        pass
#    
#    def set_Y_lims(self, min, max):
#        pass
#    
#    def set_alt_lims(self, min, max):
#        pass
#    
#    def set_max_RMS(self, max_RMS):
#        pass
#    
#    def set_min_numAntennas(self, min_numAntennas):
#        pass
#    
#    def plot(self, AltvsT_axes, AltvsEW_axes, NSvsEW_axes, NsvsAlt_axes, ancillary_axes):
#        self.path = ancillary_axes.plot(self.temperature, self.transform_alt, self.color+'o-', linewidth=self.width)
#        ancillary_axes.set_xlim( [self.lower_lim, self.upper_lim] )
#    
#    def get_viewed_events(self):
#        return []
#    
#    def clear(self):
#        if self.path is not None:
#            self.path.remove()
#    
#    def search(self, index):
#        return DataSet_Type(self.name+"_search", self.coordinate_system)
#    
#    def use_ancillary_axes(self):
#        return True
#    
#    def ancillary_label(self):
#        return "temperature (C)"

class FigureArea(FigureCanvas):
    """This is a widget that contains the central figure"""
    
    class previous_view_state(object):
        def __init__(self, axis_change, axis_data):
            self.axis_change = axis_change ## 0 is XY, 1 is altitude, and 2 is T
            self.axis_data = axis_data
    
    
    ## ALL spatial units are km, time is in seconds
    def __init__(self, coordinate_system, parent=None, width=5, height=4, dpi=100):
        
        #### some default settings, some can be changed
        
        self.axis_label_size = 15
        self.axis_tick_size = 12
        self.rebalance_XY = True ## keep X and Y have same ratio
        
#        self.default_marker_size = 5
        
#        cmap_name = 'plasma'
#        min_color = 0.0 ## start at zero means take cmap from beginning
#        max_color = 0.8 ## end at 1.0 means take cmap untill end
#        self.cmap = gen_cmap(cmap_name, min_color, max_color)
        
        
        self.coordinate_system = coordinate_system
        
        # initial state variables, in the above coordinate system
        self.alt_limits = np.array([0.0, 50.0])
        self.X_limits = np.array([0.0, 150.0])
        self.Y_limits = np.array([0.0, 150.0])
        self.T_limits = np.array([0.0, 20])
        
        self.previous_view_states = []
        
#        self.T_fraction = 1.0 ## if is 0, then do not plot any points. if 0.5, plot points halfway beteween tlims. if 1.0 plot all points
        
        
        self.data_sets = []
        
#        ## data sets
#        self.simple_pointSource_DataSets = [] ## this is a list of all Data Sets that are simple point sources
#        self.event_searches = [] ##data sets that represent events that were searched for
#        self.ancillary_data_sets = []
        
        #### setup figure and canvas
        self.fig = Figure(figsize=(width, height), dpi = dpi)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setFocusPolicy( QtCore.Qt.ClickFocus )
        self.setFocus()
        
        self.fig.subplots_adjust(top=0.97, bottom=0.07)
        
        #### setup axes. We need TWO grid specs to control heights properly
        self.top_gs = matplotlib.gridspec.GridSpec(4,1, hspace=0.3)
        self.middle_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec =self.top_gs[1], hspace=0.3, wspace=0.05)
        self.bottom_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec =self.top_gs[2:], wspace=0.05)
        
        self.AltVsT_axes = self.fig.add_subplot(self.top_gs[0])
        
        self.AltVsEw_axes= self.fig.add_subplot(self.middle_gs[0,:2])
        self.ancillary_axes = self.fig.add_subplot(self.middle_gs[0,2])
        
        self.NsVsEw_axes = self.fig.add_subplot(self.bottom_gs[0:,:2])
        self.NsVsAlt_axes = self.fig.add_subplot(self.bottom_gs[0:,2])
        
        self.AltVsT_axes.set_xlabel(self.coordinate_system.t_label, fontsize=self.axis_label_size)
        self.AltVsT_axes.set_ylabel(self.coordinate_system.z_label, fontsize=self.axis_label_size)
        self.AltVsT_axes.tick_params(labelsize = self.axis_tick_size)
        
        
        self.AltVsEw_axes.set_ylabel(self.coordinate_system.z_label, fontsize=self.axis_label_size)
        self.AltVsEw_axes.tick_params(labelsize = self.axis_tick_size)
        self.AltVsEw_axes.get_xaxis().set_visible(False)
        
        
        self.NsVsEw_axes.set_xlabel(self.coordinate_system.x_label, fontsize=self.axis_label_size)
        self.NsVsEw_axes.set_ylabel(self.coordinate_system.y_label, fontsize=self.axis_label_size)
        self.NsVsEw_axes.tick_params(labelsize = self.axis_tick_size)
        
        
        self.NsVsAlt_axes.set_xlabel(self.coordinate_system.z_label, fontsize=self.axis_label_size)
        self.NsVsAlt_axes.tick_params(labelsize = self.axis_tick_size)
        self.NsVsAlt_axes.get_yaxis().set_visible(False)
        
        
        self.ancillary_axes.get_yaxis().set_visible(False)
        self.ancillary_axes.tick_params(labelsize = self.axis_tick_size)
        self.ancillary_axes.set_axis_off()
            
        
        self.set_alt_lims(self.alt_limits[0], self.alt_limits[1])
        self.set_T_lims(self.T_limits[0], self.T_limits[1])
        self.set_X_lims(self.X_limits[0], self.X_limits[1])
        self.set_Y_lims(self.Y_limits[0], self.Y_limits[1])        
        
        #### create selectors on plots
        self.mouse_move = False
        
        self.TAlt_selector_rect = RectangleSelector(self.AltVsT_axes, self.TAlt_selector, useblit=False,
                    rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
        
        self.XAlt_selector_rect = RectangleSelector(self.AltVsEw_axes, self.XAlt_selector, useblit=False,
                    rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
        
        self.XY_selector_rect = RectangleSelector(self.NsVsEw_axes, self.XY_selector, useblit=False,
                      rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
        
        self.YAlt_selector_rect = RectangleSelector(self.NsVsAlt_axes, self.AltY_selector, useblit=False,
                    rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
        
        #### create button press/release events
        self.key_press = self.fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        self.key_release = self.fig.canvas.mpl_connect('key_release_event', self.key_release_event)
        self.button_press = self.fig.canvas.mpl_connect('button_press_event', self.button_press_event) ##mouse button
        self.button_release = self.fig.canvas.mpl_connect('button_release_event', self.button_release_event) ##mouse button
        
        ##state variables
        self.z_button_pressed = False
        self.right_mouse_button_location = None
        self.right_button_axis = None
        
        
    def add_dataset(self, new_dataset):
        self.data_sets.append( new_dataset )
        
        xlim, ylim, zlim, tlim = self.coordinate_system.invert( self.X_limits, self.Y_limits, self.alt_limits, self.T_limits )

        new_dataset.set_X_lims( *xlim )
        new_dataset.set_Y_lims( *ylim )
        new_dataset.set_alt_lims( *zlim)
        new_dataset.set_T_lims( *tlim )
        
        if new_dataset.use_ancillary_axes():
            self.ancillary_axes.set_xlabel(new_dataset.ancillary_label(), fontsize=self.axis_label_size)
            self.ancillary_axes.set_axis_on()
        
#    def add_simplePSE(self, PSE_list, name=None, markers=None, color_mode=None, size=None):
#        
#        if name is None:
#            name = "set "+str(len(self.simple_pointSource_DataSets))
#        if color_mode is None:
#            color_mode = 'time'
#        if size is None:
#            size = self.default_marker_size
#        if markers is None:
#            markers = ['s', 'D']
#        
#        new_simplePSE_dataset = DataSet_simplePointSources(PSE_list, markers, size, color_mode, name, self.coordinate_system, self.cmap)
#        self.simple_pointSource_DataSets.append(new_simplePSE_dataset)
#        
#        new_simplePSE_dataset.set_T_lims( *self.T_limits )
#        new_simplePSE_dataset.set_X_lims( *self.X_limits )
#        new_simplePSE_dataset.set_Y_lims( *self.Y_limits )
#        new_simplePSE_dataset.set_alt_lims( *self.alt_limits)
#        new_simplePSE_dataset.set_max_RMS( self.max_RMS )
#        new_simplePSE_dataset.set_min_numAntennas( self.min_numAntennas )
#        
#        self.replot_data()
#        
#        return name
#    
#    def add_ancillaryData(self, dataset):
#        self.ancillary_data_sets.append( dataset )
#        dataset.set_coordinate_system( self.coordinate_system )
#        
#        if dataset.use_ancillary_axes():
#            self.ancillary_axes.set_xlabel(dataset.ancillary_label(), fontsize=self.axis_label_size)
#            self.ancillary_axes.set_axis_on()
            
            
#    def show_all_PSE(self):
#        min_X = np.nan
#        max_X = np.nan
#        
#        min_Y = np.nan
#        max_Y = np.nan
#        
#        min_Z = np.nan
#        max_Z = np.nan
#        
#        min_T = np.nan
#        max_T = np.nan
#        
#        max_RMS = np.nan
#        
#        min_ant = np.nan
#        
#        for DS in self.simple_pointSource_DataSets:
#            [ds_xmin, ds_xmax], [ds_ymin, ds_ymax], [ds_zmin, ds_zmax], [ds_tmin, ds_tmax] = DS.getSpaceBounds()
#            ds_maxRMS, ds_minAnt = DS.getFilterBounds()
#
#            if np.isfinite( ds_xmin ):
#                if np.isfinite( min_X ):
#                    min_X = min(min_X, ds_xmin)
#                else:
#                    min_X = ds_xmin
#            if np.isfinite( ds_xmax ):
#                if np.isfinite( max_X ):
#                    max_X = max(max_X, ds_xmax)
#                else:
#                    max_X = ds_xmax
#            
#            if np.isfinite( ds_ymin ):
#                if np.isfinite( min_Y ):
#                    min_Y = min(min_Y, ds_ymin)
#                else:
#                    min_Y = ds_ymin
#            if np.isfinite( ds_ymax ):
#                if np.isfinite( max_Y ):
#                    max_Y = max(max_Y, ds_ymax)
#                else:
#                    max_Y = ds_ymax
#            
#            if np.isfinite( ds_zmin ):
#                if np.isfinite( min_Z ):
#                    min_Z = min(min_Z, ds_zmin)
#                else:
#                    min_Z = ds_zmin
#            if np.isfinite( ds_zmax ):
#                if np.isfinite( max_Z ):
#                    max_Z = max(max_Z, ds_zmax)
#                else:
#                    max_Z = ds_zmax
#            
#            if np.isfinite( ds_tmin ):
#                if np.isfinite( min_T ):
#                    min_T = min(min_T, ds_tmin)
#                else:
#                    min_T = ds_tmin
#            if np.isfinite( ds_tmax ):
#                if np.isfinite( max_T ):
#                    max_T = max(max_T, ds_tmax)
#                else:
#                    max_T = ds_tmax
#                    
#            if np.isfinite( ds_maxRMS ):
#                if np.isfinite( max_RMS ):
#                    max_RMS = max(max_RMS, ds_maxRMS)
#                else:
#                    max_RMS = ds_maxRMS
#                    
#            if np.isfinite( ds_minAnt ):
#                if np.isfinite( min_ant ):
#                    min_ant = min(min_ant, ds_minAnt)
#                else:
#                    min_ant = ds_minAnt
#        
#        self.set_T_lims(min_T, max_T, 0.1)
#        self.set_X_lims(min_X, max_X, 0.1)
#        self.set_Y_lims(min_Y, max_Y, 0.1)
#        self.set_alt_lims(min_Z, max_Z, 0.1)
#        self.set_max_RMS(max_RMS*1.1)
#        self.set_min_numAntennas( int(min_ant*0.9) )
#        
#        self.replot_data()

#    def get_viewed_events(self, data_set_name):
#        for DS in self.simple_pointSource_DataSets:
#            if DS.name == data_set_name:
#                return DS.get_viewed_events()
#            
#        return []

    #### set limits
    
    def set_all_dataset_coordinates(self):
        xlim, ylim, zlim, tlim = self.coordinate_system.invert( self.X_limits, self.Y_limits, self.alt_limits, self.T_limits )
        for DS in self.data_sets:    
            DS.set_X_lims( *xlim )
            DS.set_Y_lims( *ylim )
            DS.set_alt_lims( *zlim)
            DS.set_T_lims( *tlim )
        
    def set_T_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.T_limits[0]
        if upper is None:
            upper = self.T_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.T_limits = np.array([lower, upper])
        self.AltVsT_axes.set_xlim(self.T_limits)
        
        self.set_all_dataset_coordinates()
        
    def set_alt_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.alt_limits[0]
        if upper is None:
            upper = self.alt_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.alt_limits = np.array([lower, upper])
        self.AltVsT_axes.set_ylim(self.alt_limits)
        self.AltVsEw_axes.set_ylim(self.alt_limits)
        self.NsVsAlt_axes.set_xlim(self.alt_limits)
        self.ancillary_axes.set_ylim( self.alt_limits )
        
        self.set_all_dataset_coordinates()
        
    def set_X_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.X_limits[0]
        if upper is None:
            upper = self.X_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.X_limits = np.array([lower, upper])
        self.NsVsEw_axes.set_xlim(self.X_limits)
        self.AltVsEw_axes.set_xlim(self.X_limits)
        
        self.set_all_dataset_coordinates()
        
    def set_Y_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.Y_limits[0]
        if upper is None:
            upper = self.Y_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.Y_limits = np.array([lower, upper])
        self.NsVsAlt_axes.set_ylim(self.Y_limits)
        self.NsVsEw_axes.set_ylim(self.Y_limits)
        
        self.set_all_dataset_coordinates()
        
#    def set_max_RMS(self, max_RMS=None):
#        if max_RMS is not None:
#            self.max_RMS = max_RMS
#        
#        for DS in self.simple_pointSource_DataSets:
#            DS.set_max_RMS( self.max_RMS)
#        
#    def set_min_numAntennas(self, min_numAntennas=None):
#        if min_numAntennas is not None:
#            self.min_numAntennas = min_numAntennas
#        
#        for DS in self.simple_pointSource_DataSets:
#            DS.set_min_numAntennas( self.min_numAntennas )


        
    def replot_data(self):
        
        ### quick hacks ###
        
        self.AltVsT_axes.cla()
        self.AltVsEw_axes.cla()
        self.NsVsEw_axes.cla()
        self.NsVsAlt_axes.cla()
        self.ancillary_axes.cla()
        
        
        self.AltVsT_axes.set_xlabel(self.coordinate_system.t_label, fontsize=self.axis_label_size)
        self.AltVsT_axes.set_ylabel(self.coordinate_system.z_label, fontsize=self.axis_label_size)
        
        self.AltVsEw_axes.set_ylabel(self.coordinate_system.z_label, fontsize=self.axis_label_size)
        
        self.NsVsEw_axes.set_xlabel(self.coordinate_system.x_label, fontsize=self.axis_label_size)
        self.NsVsEw_axes.set_ylabel(self.coordinate_system.y_label, fontsize=self.axis_label_size)
        
        self.NsVsAlt_axes.set_xlabel(self.coordinate_system.z_label, fontsize=self.axis_label_size)
        
        
        #### create button press/release events
        self.key_press = self.fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        self.key_release = self.fig.canvas.mpl_connect('key_release_event', self.key_release_event)
        self.button_press = self.fig.canvas.mpl_connect('button_press_event', self.button_press_event) ##mouse button
        self.button_release = self.fig.canvas.mpl_connect('button_release_event', self.button_release_event) ##mouse button
        
        
        
        
        
        if self.rebalance_XY: ### if this is true, then we want to scale X and Y axis so that teh aspect ratio 1...I think
            bbox = self.NsVsEw_axes.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            ax_width, ax_height = bbox.width, bbox.height
            data_width = self.X_limits[1] - self.X_limits[0]
            data_height = self.Y_limits[1] - self.Y_limits[0]
            
            width_ratio = data_width/ax_width
            height_ratio = data_height/ax_height
            
            if width_ratio > height_ratio:
                new_height = ax_height*data_width/ax_width
                middle_y = (self.Y_limits[0] + self.Y_limits[1])/2.0
                self.set_Y_lims( middle_y-new_height/2.0, middle_y+new_height/2.0)
                
            else:
                new_width = ax_width*data_height/ax_height
                middle_X = (self.X_limits[0] + self.X_limits[1])/2.0
                self.set_X_lims( middle_X-new_width/2.0, middle_X+new_width/2.0 )
                
        for DS in self.data_sets:
            DS.plot( self.AltVsT_axes, self.AltVsEw_axes, self.NsVsEw_axes, self.NsVsAlt_axes, self.ancillary_axes, self.coordinate_system )
            
            
        ### anouther hadck ####
        self.NsVsEw_axes.set_xlim(self.X_limits)
        self.AltVsEw_axes.set_xlim(self.X_limits)
        
        self.NsVsAlt_axes.set_ylim(self.Y_limits)
        self.NsVsEw_axes.set_ylim(self.Y_limits)
        
        self.AltVsT_axes.set_ylim(self.alt_limits)
        self.AltVsEw_axes.set_ylim(self.alt_limits)
        self.NsVsAlt_axes.set_xlim(self.alt_limits)
        self.ancillary_axes.set_ylim( self.alt_limits )
        
        self.AltVsT_axes.set_xlim(self.T_limits)
        
        
        
        #### create selectors on plots
#        self.TAlt_selector_rect = None
        self.TAlt_selector_rect = RectangleSelector(self.AltVsT_axes, self.TAlt_selector, useblit=False,
                    rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
        
#        self.XAlt_selector_rect = None
        self.XAlt_selector_rect = RectangleSelector(self.AltVsEw_axes, self.XAlt_selector, useblit=False,
                    rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
        
#        self.XY_selector_rect = None
        self.XY_selector_rect = RectangleSelector(self.NsVsEw_axes, self.XY_selector, useblit=False,
                      rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
        
#        self.YAlt_selector_rect = None
        self.YAlt_selector_rect = RectangleSelector(self.NsVsAlt_axes, self.AltY_selector, useblit=False,
                    rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            
        
    #### calbacks for various events
    def TAlt_selector(self, eclick, erelease):
        minT = min(eclick.xdata, erelease.xdata)
        maxT = max(eclick.xdata, erelease.xdata)
        minAlt = min(eclick.ydata, erelease.ydata)
        maxAlt = max(eclick.ydata, erelease.ydata)
        
        if minT == maxT or minAlt==maxAlt: 
            self.mouse_move = False
            return
        self.mouse_move = True
        
        self.previous_view_states.append( self.previous_view_state(2, self.T_limits[:]) )
        self.previous_view_states.append( self.previous_view_state(1, self.alt_limits[:]) )
        
        if self.z_button_pressed: ## then we zoom out,
            
            minT = 2.0*self.T_limits[0] - minT
            maxT = 2.0*self.T_limits[1] - maxT
            
            minAlt = 2.0*self.alt_limits[0] - minAlt
            maxAlt = 2.0*self.alt_limits[1] - maxAlt
            
        self.set_T_lims(minT, maxT)
        self.set_alt_lims(minAlt, maxAlt)
        self.replot_data()
        self.draw()
        
    def XAlt_selector(self, eclick, erelease):
        
        minX = min(eclick.xdata, erelease.xdata)
        maxX = max(eclick.xdata, erelease.xdata)
        minA = min(eclick.ydata, erelease.ydata)
        maxA = max(eclick.ydata, erelease.ydata)
        
        if minA==maxA or minX==maxX: 
            self.mouse_move = False
            return
        self.mouse_move = True
        
        self.previous_view_states.append( self.previous_view_state(1, self.alt_limits[:]) )
        self.previous_view_states.append( self.previous_view_state(0, (self.X_limits[:],self.Y_limits[:]) ) ) 
            
        if self.z_button_pressed:
            minA = 2.0*self.alt_limits[0] - minA
            maxA = 2.0*self.alt_limits[1] - maxA
            
            minX = 2.0*self.X_limits[0] - minX
            maxX = 2.0*self.X_limits[1] - maxX
        
        self.set_alt_lims(minA, maxA)
        self.set_X_lims(minX, maxX)
        self.replot_data()
        self.draw()
        
    def AltY_selector(self, eclick, erelease):
        
        minA = min(eclick.xdata, erelease.xdata)
        maxA = max(eclick.xdata, erelease.xdata)
        minY = min(eclick.ydata, erelease.ydata)
        maxY = max(eclick.ydata, erelease.ydata)
        
        if minA==maxA or minY==maxY: 
            self.mouse_move = False
            return
        
        self.mouse_move = True
        
        self.previous_view_states.append( self.previous_view_state(1, self.alt_limits[:]) )
        self.previous_view_states.append( self.previous_view_state(0, (self.X_limits[:],self.Y_limits[:]) ) ) 
            
        if self.z_button_pressed:
            minA = 2.0*self.alt_limits[0] - minA
            maxA = 2.0*self.alt_limits[1] - maxA
            
            minY = 2.0*self.Y_limits[0] - minY
            maxY = 2.0*self.Y_limits[1] - maxY
        
        self.set_alt_lims(minA, maxA)
        self.set_Y_lims(minY, maxY)
        self.replot_data()
        self.draw()
    
    def XY_selector(self, eclick, erelease):
        
        minX = min(eclick.xdata, erelease.xdata)
        maxX = max(eclick.xdata, erelease.xdata)
        minY = min(eclick.ydata, erelease.ydata)
        maxY = max(eclick.ydata, erelease.ydata)
        
        if minX==maxX or minY==maxY:
            self.mouse_move = False
#            return
#            if len(self.data_sets)>0 and self.data_sets[0].name == "arrow":
#                print("BIGGLES", minX, minY)
#                C_X, C_Y, C_Z, C_T = self.coordinate_system.transform( [self.data_sets[0].XYZT[0]], [self.data_sets[0].XYZT[1]], [self.data_sets[0].XYZT[2]], [self.data_sets[0].XYZT[3]] )
#                print(C_X, C_Y, C_Z, C_T)
#                minX, minY, minZ, minT = self.coordinate_system.invert( [minX], [minY], C_Z, C_T )
#                print(minX, minY, minZ, minT)
#                self.data_sets[0].XYZT[0] = minX[0]
#                self.data_sets[0].XYZT[1] = minY[0]
#                self.data_sets[0].XYZT[2] = minZ[0]
#                self.data_sets[0].XYZT[3] = minT[0]
#                self.data_sets[0].set_arrow()
        else:
            self.mouse_move = True
            self.previous_view_states.append( self.previous_view_state(0, (self.X_limits[:],self.Y_limits[:]) ) ) 
            
            if self.z_button_pressed:
                minX = 2.0*self.X_limits[0] - minX
                maxX = 2.0*self.X_limits[1] - maxX
                
                minY = 2.0*self.Y_limits[0] - minY
                maxY = 2.0*self.Y_limits[1] - maxY
                
            self.set_X_lims(minX, maxX)
            self.set_Y_lims(minY, maxY)
            
            self.replot_data()
            self.draw()
        print("XY mouse moved:", self.mouse_move)
            
        
    def key_press_event(self, event):
#        print "key press:", event.key
        if event.key == 'z':
            self.z_button_pressed = True
        
    def key_release_event(self, event):
#        print "key release:", event.key
        if event.key == 'z':
            self.z_button_pressed = False
            
    def button_press_event(self, event):
#        print "mouse pressed:", event
        
        if event.button == 2 and len(self.previous_view_states)>0: ##middle mouse, back up
            previous_state = self.previous_view_states.pop(-1)
            if previous_state.axis_change == 0:
                xlims, ylims = previous_state.axis_data
                self.set_X_lims(xlims[0], xlims[1])
                self.set_Y_lims(ylims[0], ylims[1])
            elif  previous_state.axis_change == 1:
                self.set_alt_lims(previous_state.axis_data[0], previous_state.axis_data[1])
            elif  previous_state.axis_change == 2:
                
                self.set_T_lims(previous_state.axis_data[0], previous_state.axis_data[1])
                
            self.replot_data()
            self.draw()
            
        elif event.button == 3: ##right mouse, record location for drag
            self.right_mouse_button_location = [event.xdata, event.ydata]
            self.right_button_axis = event.inaxes
            
    def button_release_event(self, event):
        if event.button == 1:
            if len(self.data_sets)>0 and self.data_sets[0].name == "arrow" and not self.mouse_move:
                print("mouse moved:", self.mouse_move)
                C_X, C_Y, C_Z, C_T = self.coordinate_system.transform( [self.data_sets[0].XYZT[0]], [self.data_sets[0].XYZT[1]], [self.data_sets[0].XYZT[2]], [self.data_sets[0].XYZT[3]] )
                if event.inaxes is self.AltVsT_axes:
                    C_T[0] = event.xdata
                    C_Z[0] = event.ydata
                elif event.inaxes is self.NsVsEw_axes:
                    C_X[0] = event.xdata
                    C_Y[0] = event.ydata
                elif event.inaxes is self.AltVsEw_axes:
                    C_X[0] = event.xdata
                    C_Z[0] = event.ydata
                elif event.inaxes is self.NsVsAlt_axes:
                    C_Z[0] = event.xdata
                    C_Y[0] = event.ydata
                    
                minX, minY, minZ, minT = self.coordinate_system.invert( C_X, C_Y, C_Z, C_T )
                self.data_sets[0].XYZT[0] = minX[0]
                self.data_sets[0].XYZT[1] = minY[0]
                self.data_sets[0].XYZT[2] = minZ[0]
                self.data_sets[0].XYZT[3] = minT[0]
                self.data_sets[0].set_arrow()
                
#                self.replot_data()
#                self.draw()
                
        elif event.button == 3: ##drag
            if event.inaxes != self.right_button_axis: return
        
            deltaX = self.right_mouse_button_location[0] - event.xdata
            deltaY = self.right_mouse_button_location[1] - event.ydata
            
            if event.inaxes is self.AltVsT_axes:
                self.previous_view_states.append( self.previous_view_state(2, self.T_limits[:]) )
                self.previous_view_states.append( self.previous_view_state(1, self.alt_limits[:]) )
                self.set_T_lims( self.T_limits[0] + deltaX, self.T_limits[1] + deltaX)
                self.set_alt_lims( self.alt_limits[0] + deltaY, self.alt_limits[1] + deltaY)
                
                
            elif event.inaxes is self.AltVsEw_axes:
                self.previous_view_states.append( self.previous_view_state(1, self.alt_limits[:]) )
                ### TODO need to add y here
                self.set_X_lims( self.X_limits[0] + deltaX, self.X_limits[1] + deltaX)
                self.set_alt_lims( self.alt_limits[0] + deltaY, self.alt_limits[1] + deltaY)
                
            elif event.inaxes is self.NsVsEw_axes:
                self.previous_view_states.append( self.previous_view_state(0, (self.X_limits[:],self.Y_limits[:]) ) ) 
                self.set_X_lims( self.X_limits[0] + deltaX, self.X_limits[1] + deltaX)
                self.set_Y_lims( self.Y_limits[0] + deltaY, self.Y_limits[1] + deltaY)
                
            elif event.inaxes is self.NsVsAlt_axes:
                self.previous_view_states.append( self.previous_view_state(1, self.alt_limits[:]) )
                ##TODO need to add X here
                self.set_alt_lims( self.alt_limits[0] + deltaX, self.alt_limits[1] + deltaX)
                self.set_Y_lims( self.Y_limits[0] + deltaY, self.Y_limits[1] + deltaY)
                
            self.replot_data()
            self.draw()
            
        
        self.mouse_move = False
            
#    def event_search(self, event_index):
#        for DS in self.event_searches:
#            DS.clear()
        
#        self.event_searches = [DS.search(event_index, ['o','o'], self.default_marker_size, '*r') for DS in self.simple_pointSource_DataSets]
            
        
    #### external usefull functions
#    def set_T_fraction(self, t_fraction):
#        """plot points with all time less that the t_fraction. Where 0 is beginning of t-lim and 1 is end of t_lim"""
#        self.T_fraction = t_fraction
#        self.set_T_lims()
                
        

class Active3DPlotter(QtWidgets.QMainWindow):
    """This is the main window. Contains the figure window, and controlls all the menus and buttons"""
    
    def __init__(self, coordinate_system):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("LOFAR-LIM PSE plotter")
        self.setGeometry(300, 300, 1100, 1100)
        
#        self.statusBar().showMessage("All hail matplotlib!", 2000) ##this shows messages on bottom left of window



        #### menu bar ###
        ##file
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.file_menu.addAction('&Save Plot', self.savePlot)
        self.menuBar().addMenu(self.file_menu)
        
        ##plot settings
        self.plot_settings_menu = QtWidgets.QMenu('&Plot Settings', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.plot_settings_menu)
        
        ## analysis
        self.analysis_menu = QtWidgets.QMenu('&Analysis', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.analysis_menu)
        self.analysis_menu.addAction('&Print Info', self.printInfo)
        self.analysis_menu.addAction('&Copy View', self.copy_view)
        self.analysis_menu.addAction('&Output Text', self.textOutput)
#        self.analysis_menu.addAction('&Output SVG', self.svgOutput)
        ## new actions added through add_analysis

        ##help
        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)
        

        
        #### create layout ####
        self.main_widget = QtWidgets.QWidget(self)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        
        horizontal_divider = QtWidgets.QHBoxLayout(self.main_widget)
        
        ## space for buttons on left
        horizontal_divider.addSpacing(300)
        
        ##place figures on the right
        self.figure_space = FigureArea(coordinate_system, self.main_widget)
        horizontal_divider.addWidget( self.figure_space )
        
        
        
        
        #### data set controls ####
        self.current_variable_name = None
        
        # list of data sets
        self.current_data_set = -1
        self.DS_drop_list = QtWidgets.QComboBox(self)
        self.DS_drop_list.move(5, 50)
        self.DS_drop_list.resize(300, 25)
        self.refresh_analysis_list()
        self.DS_drop_list.activated[int].connect(self.DS_drop_list_choose)
        self.DS_drop_list_choose( self.DS_drop_list.currentIndex() )
        
#        increase/decrease priority
#        turn on/off

        ## show all PSE button
        self.showAll_button = QtWidgets.QPushButton(self)
        self.showAll_button.move(5, 80)
        self.showAll_button.setText("show all")
        self.showAll_button.resize(70,25)
        self.showAll_button.clicked.connect(self.showAllButtonPressed)
        
        ## toggel show data set
        self.toggleDS_button = QtWidgets.QPushButton(self)
        self.toggleDS_button.move(80, 80)
        self.toggleDS_button.setText("toggle")
        self.toggleDS_button.resize(50,25)
        self.toggleDS_button.clicked.connect(self.toggleButtonPressed)
        
        ## delete show data set
        self.deleteDS_button = QtWidgets.QPushButton(self)
        self.deleteDS_button.move(135, 80)
        self.deleteDS_button.setText("delete")
        self.deleteDS_button.resize(50,25)
        self.deleteDS_button.clicked.connect(self.deleteDSButtonPressed)
        
        ## list of variables in data set
        self.DS_variable_list = QtWidgets.QComboBox(self)
        self.DS_variable_list.move(5,120)
        self.DS_variable_list.resize(300, 25)
        self.DS_variable_list.activated[str].connect(self.DSvariable_get)
        self.DSvariable_get()
        
        ## text box to display variables
        self.variable_txtBox = QtWidgets.QLineEdit(self)
        self.variable_txtBox.move(5, 150)
        self.variable_txtBox.resize(300,25)
        
        ##set button
        self.set_button = QtWidgets.QPushButton(self)
        self.set_button.move(5, 180)
        self.set_button.setText("set")
        self.set_button.clicked.connect(self.setButtonPressed)
        
        ##set button
        self.get_button = QtWidgets.QPushButton(self)
        self.get_button.move(125, 180)
        self.get_button.setText("get")
        self.get_button.clicked.connect(self.getButtonPressed)


        #### set and get position ####
        
        ## X label
        self.XLabel = QtWidgets.QLabel(self)
        self.XLabel.move(5, 250)
        self.XLabel.resize(30,25)
        self.XLabel.setText("X:")
        ## X min box
        self.Xmin_txtBox = QtWidgets.QLineEdit(self)
        self.Xmin_txtBox.move(40, 250)
        self.Xmin_txtBox.resize(80,25)
        ## X max box
        self.Xmax_txtBox = QtWidgets.QLineEdit(self)
        self.Xmax_txtBox.move(140, 250)
        self.Xmax_txtBox.resize(80,25)
        
        ## Y label
        self.YLabel = QtWidgets.QLabel(self)
        self.YLabel.move(5, 280)
        self.YLabel.resize(30,25)
        self.YLabel.setText("Y:")
        ## Y min box
        self.Ymin_txtBox = QtWidgets.QLineEdit(self)
        self.Ymin_txtBox.move(40, 280)
        self.Ymin_txtBox.resize(80,25)
        ## Y max box
        self.Ymax_txtBox = QtWidgets.QLineEdit(self)
        self.Ymax_txtBox.move(140, 280)
        self.Ymax_txtBox.resize(80,25)
        
        ## Z label
        self.ZLabel = QtWidgets.QLabel(self)
        self.ZLabel.move(5, 310)
        self.ZLabel.resize(30,25)
        self.ZLabel.setText("Z:")
        ## Z min box
        self.Zmin_txtBox = QtWidgets.QLineEdit(self)
        self.Zmin_txtBox.move(40, 310)
        self.Zmin_txtBox.resize(80,25)
        ## Z max box
        self.Zmax_txtBox = QtWidgets.QLineEdit(self)
        self.Zmax_txtBox.move(140, 310)
        self.Zmax_txtBox.resize(80,25)
        
        ## T label
        self.TLabel = QtWidgets.QLabel(self)
        self.TLabel.move(5, 340)
        self.TLabel.resize(30,25)
        self.TLabel.setText("T:")
        ## T min box
        self.Tmin_txtBox = QtWidgets.QLineEdit(self)
        self.Tmin_txtBox.move(40, 340)
        self.Tmin_txtBox.resize(80,25)
        ## T max box
        self.Tmax_txtBox = QtWidgets.QLineEdit(self)
        self.Tmax_txtBox.move(140, 340)
        self.Tmax_txtBox.resize(80,25)
        
        ##set postion button
        self.setPos_button = QtWidgets.QPushButton(self)
        self.setPos_button.move(5, 370)
        self.setPos_button.setText("set pos.")
        self.setPos_button.clicked.connect(self.setPositionPressed)
        
        ##show all position button
        self.showAllPos_button = QtWidgets.QPushButton(self)
        self.showAllPos_button.move(140, 370)
        self.showAllPos_button.setText("show all position")
        self.showAllPos_button.resize(150,25)
        self.showAllPos_button.clicked.connect(self.showAllPositionsButtonPressed)
        
        ## add zoom-in or out controlls (pressing Z doesn't seem to work)
        
        
        #### SEARCH ####
        
        self.search_txtBox = QtWidgets.QLineEdit(self)
        self.search_txtBox.move(5, 420)
        self.search_txtBox.resize(80,25)
        
        self.search_button = QtWidgets.QPushButton(self)
        self.search_button.move(110, 420)
        self.search_button.setText("search")
        self.search_button.resize(150,25)
        self.search_button.clicked.connect(self.searchButtonPressed)
        
        
        
        ### zoom ###
        
        self.search_button = QtWidgets.QPushButton(self)
        self.search_button.move(5, 470)
        self.search_button.setText("zoom: in")
        self.search_button.resize(150,25)
        self.search_button.clicked.connect(self.zoomButtonPressed)
        
        ##TODO:
        #animation
        #custom analysis
        # rename datasets
        
        
        #### add an arrow data set for stuff
        source_arrow = DataSet_arrow('arrow', -15059, 10394, 5156, 1.25215163805, azimuth=-np.pi/4, zenith=np.pi/2, length=10, color='k')
        self.add_dataset( source_arrow )
        
        
        #### buttans and controlls on left
        ## max RMS box
#        self.max_RMS_input = QtWidgets.QLineEdit(self)
#        self.max_RMS_input.move(105, 120)
#        self.max_RMS_input.resize(50,25)
#        ## input txt box
#        self.max_RMS_label = QtWidgets.QLabel(self)
#        self.max_RMS_label.setText("max RMS (ns):")
#        self.max_RMS_label.move(5, 120)
#        
#        ## min numStations box
#        self.min_numAntennas_input = QtWidgets.QLineEdit(self)
#        self.min_numAntennas_input.move(125, 150)
#        self.min_numAntennas_input.resize(70,25)
#        ## input txt box
#        self.min_numAntennas_label = QtWidgets.QLabel(self)
#        self.min_numAntennas_label.setText("min. num antennas:")
#        self.min_numAntennas_label.move(5, 150)
#        
#        
#        ## show all PSE button
#        self.showAll_button = QtWidgets.QPushButton(self)
#        self.showAll_button.move(10, 30)
#        self.showAll_button.setText("show all")
#        self.showAll_button.clicked.connect(self.showAllButtonPressed)
#        
#        ## send the settings button
#        self.set_button = QtWidgets.QPushButton(self)
#        self.set_button.move(10, 70)
#        self.set_button.setText("set")
#        self.set_button.clicked.connect(self.setButtonPressed)
#        
#        ## refresh button
#        self.get_button = QtWidgets.QPushButton(self)
#        self.get_button.move(160, 70)
#        self.get_button.setText("get")
#        self.get_button.clicked.connect(self.getButtonPressed)
        
        
        
        ### animation controlls 
#        self.animation_label = QtWidgets.QLabel(self)
#        self.animation_label.setText("animation time (s):")
#        self.animation_label.resize(160,25)
#        self.animation_label.move(5, 500)
#        
#        self.animation_input = QtWidgets.QLineEdit(self)
#        self.animation_input.move(120, 500)
#        self.animation_input.resize(40,25)
#        
#        self.animation_FPS_label = QtWidgets.QLabel(self)
#        self.animation_FPS_label.setText("FPS:")
#        self.animation_FPS_label.resize(40,25)
#        self.animation_FPS_label.move(80, 530)
#        
#        self.animation_FPS_input = QtWidgets.QLineEdit(self)
#        self.animation_FPS_input.move(120, 530)
#        self.animation_FPS_input.resize(40,25)
#        
#        self.animate_button = QtWidgets.QPushButton(self)
#        self.animate_button.move(160, 500)
#        self.animate_button.setText("animate")
#        self.animate_button.clicked.connect(self.animateButtonPressed)
#        
        
#        ### search button
#        self.search_input = QtWidgets.QLineEdit(self)
#        self.search_input.move(120, 700)
#        self.search_input.resize(40,25)
#        
#        self.animate_button = QtWidgets.QPushButton(self)
#        self.animate_button.move(5, 700)
#        self.animate_button.setText("search")
#        self.animate_button.clicked.connect(self.searchButtonPressed)
        
        
        
        ### set the fields
#        self.getButtonPressed()
        
        
        #    #### menu bar call backs ####
    ## file
    def fileQuit(self):
        self.close()
        
    def savePlot(self):
        output_fname = "./plot_save.png"
        self.figure_space.fig.savefig(output_fname, format='png')
        
    ##help
    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """Bla Bla Bla""")

    def closeEvent(self, ce):
        self.fileQuit()
        
        
    #### add data sets ####
    
    def add_dataset(self, new_dataset):
        self.figure_space.add_dataset( new_dataset )
        self.refresh_analysis_list()
        self.DS_drop_list_choose( self.DS_drop_list.currentIndex() )
        
        
    #### control choosing data set and variables ####
    def refresh_analysis_list(self):
        self.DS_drop_list.clear()
        self.DS_drop_list.addItems( [DS.name for DS in self.figure_space.data_sets] )
        if len(self.figure_space.data_sets) > 0:
            self.DS_drop_list_choose( 0 )
        
    def DS_drop_list_choose(self, data_set_index):
        self.current_data_set = data_set_index
        self.refresh_DSvariable_list()
        
    
    def refresh_DSvariable_list(self):
        if self.current_data_set != -1 :
            variable_names = self.figure_space.data_sets[ self.current_data_set ].get_all_properties().keys()
            variable_names = list(variable_names)
            
            self.DS_variable_list.clear()
            self.DS_variable_list.addItems( variable_names )
            
            self.current_variable_name = variable_names[0]
            self.DSvariable_get()


    def DSvariable_get(self, variable_name=None):
        if self.current_data_set != -1:
            properties = self.figure_space.data_sets[ self.current_data_set ].get_all_properties()
            
            if variable_name is None:
                variable_name = self.current_variable_name
            else:
                self.current_variable_name = variable_name
            
            if variable_name is not None:
                value = str(properties[variable_name])
                self.variable_txtBox.setText( value )
                
    def DSposition_get(self):
        if self.current_data_set != -1:
            DS = self.figure_space.data_sets[ self.current_data_set ]
            tmin, tmax = DS.get_T_lims()
            xmin, xmax = DS.get_X_lims()
            ymin, ymax = DS.get_Y_lims()
            zmin, zmax = DS.get_alt_lims()
            
            self.Xmin_txtBox.setText( str(xmin) )
            self.Xmax_txtBox.setText( str(xmax) )
            
            self.Ymin_txtBox.setText( str(ymin) )
            self.Ymax_txtBox.setText( str(ymax) )
            
            self.Zmin_txtBox.setText( str(zmin) )
            self.Zmax_txtBox.setText( str(zmax) )
            
            self.Tmin_txtBox.setText( str(tmin) )
            self.Tmax_txtBox.setText( str(tmax) )
        
    
    
    #### buttons ####
    def showAllButtonPressed(self):
        DS = self.figure_space.data_sets[ self.current_data_set ]
        X_bounds, Y_bounds, Z_bounds, T_bounds = DS.set_show_all()
        X_bounds, Y_bounds, Z_bounds, T_bounds = self.figure_space.coordinate_system.transform( X_bounds, Y_bounds, Z_bounds, T_bounds )
        
        self.figure_space.set_X_lims( X_bounds[0], X_bounds[1], 0.05 )
        self.figure_space.set_Y_lims( Y_bounds[0], Y_bounds[1], 0.05 )
        self.figure_space.set_alt_lims( Z_bounds[0], Z_bounds[1], 0.05 )
        self.figure_space.set_T_lims( T_bounds[0], T_bounds[1], 0.05 )
        
        self.DSvariable_get()
        self.DSposition_get()
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def toggleButtonPressed(self):
        DS = self.figure_space.data_sets[ self.current_data_set ]
        if DS.display:
            DS.toggle_off()
        else:
            DS.toggle_on()
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def deleteDSButtonPressed(self):
        if self.current_data_set is not None and self.current_data_set != -1: ## is this needed?
            self.figure_space.data_sets.pop( self.current_data_set )
            self.refresh_analysis_list()
            self.figure_space.replot_data()
            self.figure_space.draw()
            
    def setButtonPressed(self):
        inputTXT = self.variable_txtBox.text()
        
        DS = self.figure_space.data_sets[ self.current_data_set ]
        DS.set_property(self.current_variable_name, inputTXT)
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def getButtonPressed(self):
        self.DSvariable_get()
        self.DSposition_get()
        
    def setPositionPressed(self):
        try:
            xmin = float( self.Xmin_txtBox.text() )
            xmax = float( self.Xmax_txtBox.text() )
             
            ymin = float( self.Ymin_txtBox.text() )
            ymax = float( self.Ymax_txtBox.text() )
            
            zmin = float( self.Zmin_txtBox.text() )
            zmax = float( self.Zmax_txtBox.text() )
            
            tmin = float( self.Tmin_txtBox.text() )
            tmax = float( self.Tmax_txtBox.text() )
        except:
            print("bad input")
            return
        
        
        X_bounds, Y_bounds, Z_bounds, T_bounds = self.figure_space.coordinate_system.transform( 
                np.array([xmin,xmax]), np.array([ymin,ymax]), np.array([zmin, zmax]), np.array([tmin,tmax]) )
        
        self.figure_space.set_X_lims( X_bounds[0], X_bounds[1], 0.00 )
        self.figure_space.set_Y_lims( Y_bounds[0], Y_bounds[1], 0.00 )
        self.figure_space.set_alt_lims( Z_bounds[0], Z_bounds[1], 0.00 )
        self.figure_space.set_T_lims( T_bounds[0], T_bounds[1], 0.00 )
        
        self.DSvariable_get()
        self.DSposition_get()
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def showAllPositionsButtonPressed(self):
        DS = self.figure_space.data_sets[ self.current_data_set ]
        X_bounds, Y_bounds, Z_bounds, T_bounds = DS.bounding_box()
        X_bounds, Y_bounds, Z_bounds, T_bounds = self.figure_space.coordinate_system.transform( X_bounds, Y_bounds, Z_bounds, T_bounds )
        
        self.figure_space.set_X_lims( X_bounds[0], X_bounds[1], 0.05 )
        self.figure_space.set_Y_lims( Y_bounds[0], Y_bounds[1], 0.05 )
        self.figure_space.set_alt_lims( Z_bounds[0], Z_bounds[1], 0.05 )
        self.figure_space.set_T_lims( T_bounds[0], T_bounds[1], 0.05 )
        
        self.DSvariable_get()
        self.DSposition_get()
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
        
    def searchButtonPressed(self):
        
        if self.current_data_set is not None and self.current_data_set != -1: ## is this needed?
            search_ID = self.search_txtBox.text()
            
            ret = self.figure_space.data_sets[ self.current_data_set ].search(search_ID)
            
            if ret is not None:
                self.add_dataset( ret )
                self.figure_space.replot_data()
                self.figure_space.draw()
        else:
            print("no dataset selected")
            
            
    def zoomButtonPressed(self):
        if self.figure_space.z_button_pressed:
            self.figure_space.z_button_pressed = False
            self.search_button.setText("zoom: in")
        else:
            self.figure_space.z_button_pressed = True
            self.search_button.setText("zoom: out")
            
        
        
        
    #### analysis
    def printInfo(self):
        DS = self.figure_space.data_sets[ self.current_data_set ]
        DS.print_info()
        
    def copy_view(self):
        
        if self.current_data_set is not None and self.current_data_set != -1: ## is this needed?
            ret = self.figure_space.data_sets[ self.current_data_set ].copy_view()
            
            if ret is not None:
                print("adding dataset")
                self.add_dataset( ret )
                self.figure_space.replot_data()
                self.figure_space.draw()
        else:
            print("no dataset selected")
            
    def textOutput(self):
        DS = self.figure_space.data_sets[ self.current_data_set ]
        DS.text_output()
            
#    def svgOutput(self):
#        DS = self.figure_space.data_sets[ self.current_data_set ]
#        DS.svgOutput(self.figure_space.AltVsT_axes, self.figure_space.AltVsEw_axes, self.figure_space.NsVsEw_axes, 
#                     self.figure_space.NsVsAlt_axes, self.figure_space.ancillary_axes, self.figure_space.coordinate_system, 
#                     self.figure_space.X_limits, self.figure_space.Y_limits, self.figure_space.alt_limits, self.figure_space.T_limits)
    
        
#    def print_number(self):
#        
#        if self.current_data_set is not None and self.current_data_set != -1: ## is this needed?
#            ret = self.figure_space.data_sets[ self.current_data_set ].copy_view()
#            
#            if ret is not None:
#                print("adding dataset")
#                self.add_dataset( ret )
#                self.figure_space.replot_data()
#                self.figure_space.draw()
#        else:
#            print("no dataset selected")
        
        
        
        
#    def add_analysis(self, analysis_name, function, data_set_names):
#        functor = lambda : self.run_analysis(function, data_set_names)
#        self.analysis_menu.addAction('&'+analysis_name, functor)
#        

#    def run_analysis(self, function, data_set_names):
#        viewed_events = []
#        for DS_name in data_set_names:
#            viewed_events += self.figure_space.get_viewed_events(DS_name)
#        function(viewed_events)
#        
#        

#        
#    
#    #### button callbacks ####
#    def setButtonPressed(self):
#        try:
#            maxRMS = float(self.max_RMS_input.text())*1.0E-9
#        except:
#            print( "invalid literal in max RMS field")
#            self.getButtonPressed()
#            return
#        
#        try:
#            minNumAntennas = int(self.min_numAntennas_input.text())
#        except:
#            print( "invalid literal in min num stations field")
#            self.getButtonPressed()
#            return
#   
#        self.figure_space.set_max_RMS( maxRMS )
#        self.figure_space.set_min_numAntennas( minNumAntennas )
#        self.figure_space.replot_data()
#        self.figure_space.draw()
#        
#    def showAllButtonPressed(self):
#        self.figure_space.show_all_PSE()
#        
#        self.getButtonPressed()
#        
#        self.figure_space.replot_data()
#        self.figure_space.draw()
#        
#    def getButtonPressed(self):
#        #### set text in max RMS field
#        self.max_RMS_input.setText( "{:.2f}".format(self.figure_space.max_RMS/(1.0E-9) ))
#        #### set text in min num stations
#        self.min_numAntennas_input.setText( str(self.figure_space.min_numAntennas ))
#        
#    def animateButtonPressed(self):
#        try:
#            animation_time = float(self.animation_input.text())
#        except:
#            print( "invalid literal in animation time field")
#            return
#        
#        try:
#            FPS = float(self.animation_FPS_input.text())
#        except:
#            print( "invalide interger in FPS field")
#        
#        n_frames = int(animation_time*FPS)
#        
#        frame_Tfractions, TfracStep = np.linspace(0.0, 1.0, n_frames, endpoint=True, retstep=True)
#        time_per_step = TfracStep*animation_time
#        FPS = 1.0/time_per_step
#        
#        print( "animate with FPS:", FPS)
#        
#        current_animation_index = 0
#        
##        print manimation.writers.list()
##        FFMpegWriter = manimation.writers['FFmpeg']
##        metadata = dict(title='LightningAnimation', artist='Matplotlib')
##        writer = FFMpegWriter(fps=FPS, metadata=metadata)
##        controller = writer.saving(self.figure_space.fig, "LightningAnimation.mp4", n_frames)
#        
#        self.figure_space.set_T_fraction( frame_Tfractions[ current_animation_index ] )
#        self.figure_space.replot_data()
#        self.figure_space.fig.canvas.draw()
#        self.figure_space.draw()
##        writer.grab_frame()
#
##        self.figure_space.fig.savefig("./animation_output/"+str(current_animation_index).zfill(4))
#        
#        timer = QtCore.QTimer(self)
#        timer.singleShot(time_per_step*1000, lambda: self.animateUpdate(current_animation_index, frame_Tfractions, time_per_step) )
#        
#    def animateUpdate(self, animation_index, Tfractions, time_per_step):
#        animation_index += 1
#        if animation_index == len(Tfractions):
#            print( "done animating")
#            return
#        
#        self.figure_space.set_T_fraction( Tfractions[ animation_index ] )
#        self.figure_space.replot_data()
#        self.figure_space.fig.canvas.draw()
#        self.figure_space.draw()
#        
##        self.figure_space.fig.set_size_inches(7.8, 10.6)
##        self.figure_space.fig.savefig("./animation_output/"+str(animation_index).zfill(4), dpi=100)
#        
#        timer = QtCore.QTimer(self)
#        timer.singleShot(time_per_step*1000, lambda: self.animateUpdate(animation_index, Tfractions, time_per_step) )
#        
#        
#    def searchButtonPressed(self):
#        try:
#            event_index = int(self.search_input.text())
#        except:
#            print( "invalid event index in search field")
#            return
#        
#        self.figure_space.event_search( event_index )

###### some default analysis ######
def print_details_analysis(PSE_list):
    for PSE in PSE_list:
        print( PSE.unique_index )
        print( " even fitval:", PSE.PolE_RMS )
        print( "   loc:", PSE.PolE_loc )
        print( " odd fitval:", PSE.PolO_RMS )
        print( "   loc:", PSE.PolO_loc )
        print()

class plot_data_analysis:
    def __init__(self, antenna_locations):
        self.ant_locs = antenna_locations
        
    def __call__(self, PSE_list):
        for PSE in PSE_list:
            print( "plotting:", PSE.unique_index )
            print( " even fitval:", PSE.PolE_RMS )
            print( "   loc:", PSE.PolE_loc )
            print( " odd fitval:", PSE.PolO_RMS )
            print( "   loc:", PSE.PolO_loc )
            print( " Even is Green, Odd is Magenta" )
            print() 
            PSE.plot_trace_data(self.ant_locs)

        
class print_ave_ant_delays:
    def __init__(self, ant_locs):
        self.ant_locs = ant_locs
        
    def __call__(self, PSE_list):
        max_RMS = 2.00e-9
        
        ant_delay_dict = {}
        
        for PSE in PSE_list:
            PSE.load_antenna_data(False)
            
            for ant_name, ant_info in PSE.antenna_data.items():
                loc = self.ant_locs[ant_name]
                
                if ant_name not in ant_delay_dict:
                    ant_delay_dict[ant_name] = [[],  []]
                
                if (ant_info.antenna_status==0 or ant_info.antenna_status==2) and PSE.PolE_RMS<max_RMS:
                    model = np.linalg.norm(PSE.PolE_loc[0:3] - loc)/v_air + PSE.PolE_loc[3]
                    data = ant_info.PolE_peak_time
                    ant_delay_dict[ant_name][0].append( data - model )
                
                if (ant_info.antenna_status==0 or ant_info.antenna_status==1) and PSE.PolO_RMS<max_RMS:
                    model = np.linalg.norm(PSE.PolO_loc[0:3] - loc)/v_air + PSE.PolO_loc[3]
                    data = ant_info.PolO_peak_time
                    ant_delay_dict[ant_name][1].append( data - model )
            
        for ant_name, (even_delays, odd_delays) in ant_delay_dict.items():
            PolE_ave = np.average(even_delays)
            PolE_std = np.std(even_delays)
            PolE_N = len(even_delays)
            PolO_ave = np.average(odd_delays)
            PolO_std = np.std(odd_delays)
            PolO_N = len(odd_delays)

            print(ant_name)
            print("   even:", PolE_ave, "+/-", PolE_std/np.sqrt(PolE_N), '(', PolE_std, PolE_N, ')')
            print("    odd:", PolO_ave, "+/-", PolO_std/np.sqrt(PolO_N), '(', PolO_std, PolO_N, ')')
        print("done")
        print()

class histogram_amplitudes:
    def __init__(self, station, pol=0):
        self.station=station
        self.polarization=pol #0 is even, 1 is odd
        
    def __call__(self, PSE_list):
        
        amplitudes = []
        for PSE in PSE_list:
            PSE.load_antenna_data(False)
            total = 0.0
            N = 0
            for antenna_name, data in PSE.antenna_data.items():
                if antenna_name[0:3] == self.station:
                    if self.polarization == 0 and (data.antenna_status==0 or data.antenna_status==2):
                        total += data.PolE_HE_peak
                        N += 1
                        
                    elif self.polarization == 1 and (data.antenna_status==0 or data.antenna_status==1):
                        total += data.PolO_HE_peak
                        N += 1
                        
            if N != 0:
                amplitudes.append( total/N )
            
        print(len(amplitudes), int(np.sqrt(len(amplitudes))) )
        print("average and std of distribution:", np.average(amplitudes), np.std(amplitudes))
        plt.hist(amplitudes, 2*int(np.sqrt(len(amplitudes))) )
        plt.xlim((0,1500))
        plt.tick_params(labelsize=40)
        plt.show()
        
#from scipy.stats import linregress
#from matplotlib.widgets import LassoSelector
#from matplotlib.path import Path
#class leader_speed_estimator:
#    def __init__(self, pol=0):
#        self.polarization = pol
#        self.PSE_list = None
#        
#    def __call__(self, PSE_list):
#        self.PSE_list = PSE_list
#        
#    def go(self):
#        
#        if self.PSE_list is None:
#            return
#        
#        segment_length = 4
#        
#        self.locs = np.empty( (len(self.PSE_list),5) )
#        for i,PSE in enumerate(self.PSE_list):
#            pos = PSE.PolE_loc
#            if self.polarization == 1:
#                pos = PSE.PolO_loc
#                
#            self.locs[i,:4] = pos
#            self.locs[i,4] = PSE.unique_index
#        
#        while True:
#            #### analyze
#            sort = np.argsort( self.locs[:,3])
#            self.locs = self.locs[sort]
#            
#            segments = int( len( self.locs )/segment_length )
#            
#            line_data = []
#            time_bounds = []
#            weighted_velocity_total = 0.0
#            time_sum = 0.0
#            for seg_i in range(segments):
#                segment_locs = self.locs[seg_i*segment_length:segment_length*(seg_i+1)]
#                X = segment_locs[:,0]
#                Y = segment_locs[:,1]
#                Z = segment_locs[:,2]
#                T = segment_locs[:,3]
#                
#                Xslope, Xintercept, R, P, stderr = linregress(T, X)
#                Yslope, Yintercept, R, P, stderr = linregress(T, Y)
#                Zslope, Zintercept, R, P, stderr = linregress(T, Z)
#                
#                line_data.append( [Xslope, Xintercept, Yslope, Yintercept, Zslope, Zintercept] )
#                time_bounds.append( [ T[0], T[-1] ] )
#                
#                V =  np.sqrt(Xintercept*Xintercept + Yintercept*Yintercept + Zintercept*Zintercept)
#                weight = T[-1]-T[0]
#                print(Xintercept, Yintercept, Zintercept, V, weight)
#                
#                weighted_velocity_total += V*weight
#                time_sum += weight
#                
#            print("av. 3D speed:", weighted_velocity_total/time_sum)
#            
#            #### make plots with lasso selector
#            
#            X_ax = plt.subplot(311)
#            Y_ax = plt.subplot(312, sharex=X_ax)
#            Z_ax = plt.subplot(313, sharex=X_ax)
#            
#            X_ax.scatter(self.locs[:,3], self.locs[:,0])
#            Y_ax.scatter(self.locs[:,3], self.locs[:,1])
#            Z_ax.scatter(self.locs[:,3], self.locs[:,2])
#            
#            for line,bounds in zip(line_data,time_bounds):
#                Xlow = line[0]*bounds[0] + line[1]
#                Xhigh= line[0]*bounds[1] + line[1]
#                X_ax.plot(bounds, [Xlow, Xhigh])
#                
#                Ylow = line[2]*bounds[0] + line[3]
#                Yhigh= line[2]*bounds[1] + line[3]
#                Y_ax.plot(bounds, [Ylow, Yhigh])
#                
#                Zlow = line[4]*bounds[0] + line[5]
#                Zhigh= line[4]*bounds[1] + line[5]
#                Z_ax.plot(bounds, [Zlow, Zhigh])
#                
#            
#            X_lasso = LassoSelector(X_ax, onselect=self.X_lasso_select)
#            Y_lasso = LassoSelector(Y_ax, onselect=self.Y_lasso_select)
#            Z_lasso = LassoSelector(Z_ax, onselect=self.Z_lasso_select)
#            
#            self.lasso_data = None
#            self.delete_mode = False
#            
#            plt.gcf().canvas.mpl_connect('key_press_event', self.press)
#            
#            plt.show()
#            print("done")
#            
#            X_lasso = None
#            Y_lasso = None
#            Z_lasso = None
#            
#            #### if data has been lasso-ed, then repeat
#            ## if not, quit
#            
#            if self.lasso_data is None:
#                print("quit filtering")
#                break
#            
#            if self.delete_mode:
#                self.lasso_data = np.logical_not(self.lasso_data)
#            
#            self.locs = self.locs[self.lasso_data]
#            
#            
#    def X_lasso_select(self, lasso_data):
#        path = Path(lasso_data)
#        locs_filter = np.array([path.contains_point([loc[3],loc[0]]) for loc in self.locs], dtype=bool)
#        self.lasso_data = locs_filter
#        print("X LASSO!")
#        
#    def Y_lasso_select(self, lasso_data):
#        path = Path(lasso_data)
#        locs_filter = np.nonzero([path.contains_point([loc[3],loc[1]]) for loc in self.locs])[0]
#        self.lasso_data = locs_filter
#        print("Y LASSO!")
#        
#    def Z_lasso_select(self, lasso_data):
#        path = Path(lasso_data)
#        locs_filter = np.array([path.contains_point([loc[3],loc[2]]) for loc in self.locs], dtype=bool)
#        self.lasso_data = locs_filter
#        print("Z LASSO!")
#        
#    def press(self, event):
#        print('press', event.key)
#        if event.key == 'd':
#            print("delete mode: ON!")
#            self.delete_mode = True
#        
#    def save(self, fname):
#        if self.locs is not None:
#            np.save(fname, self.locs, allow_pickle=False)


from scipy.stats import linregress
class leader_speed_estimator:
    def __init__(self, pol=0, RMS=2.0E-9):
        self.polarization = pol
        self.RMS_filter = RMS
        
    def __call__(self, PSE_list):
        X = []
        Y = []
        Z = []
        T = []
        unique_IDs = []
        
        for PSE in PSE_list:
            loc = PSE.PolE_loc
            RMS = PSE.PolE_RMS
            if self.polarization ==1:
                loc = PSE.PolO_loc
                RMS = PSE.PolO_RMS
                
            if RMS <= self.RMS_filter:
                X.append(loc[0])
                Y.append(loc[1])
                Z.append(loc[2])
                T.append(loc[3])
                unique_IDs.append( PSE.unique_index )
                
        T = np.array(T)   
        X = np.array(X)  
        Y = np.array(Y)  
        Z = np.array(Z)       
        
        sorter = np.argsort(T)
        T = T[sorter]
        X = X[sorter]
        Y = Y[sorter]
        Z = Z[sorter]
        
        Xslope, Xintercept, XR, P, stderr = linregress(T, X)
        Yslope, Yintercept, YR, P, stderr = linregress(T, Y)
        Zslope, Zintercept, ZR, P, stderr = linregress(T, Z)
        
        print("PSE IDs:", unique_IDs)
        print("X vel:", Xslope, XR)
        print("Y vel:", Yslope, YR)
        print("Z vel:", Zslope, ZR)
        print("3D speed:", np.sqrt(Xslope**2 + Yslope**2 + Zslope**2))
        print("time:", T[-1]-T[0])
        print("number sources", len(unique_IDs))
        
        X_ax = plt.subplot(311)
        plt.setp(X_ax.get_xticklabels(), visible=False)
        Y_ax = plt.subplot(312, sharex=X_ax)
        plt.setp(Y_ax.get_xticklabels(), visible=False)
        Z_ax = plt.subplot(313, sharex=X_ax)
        
        X_ax.scatter((T-3.819)*1000.0, X/1000.0)
        Y_ax.scatter((T-3.819)*1000.0, Y/1000.0)
        Z_ax.scatter((T-3.819)*1000.0, Z/1000.0)
        
        Xlow = Xintercept + Xslope*T[0]
        Xhigh = Xintercept + Xslope*T[-1]
        X_ax.plot( [ (T[0]-3.819)*1000.0, (T[-1]-3.819)*1000.0], [Xlow/1000.0,Xhigh/1000.0] )
        X_ax.tick_params('y', labelsize=25)
        
        Ylow = Yintercept + Yslope*T[0]
        Yhigh = Yintercept + Yslope*T[-1]
        Y_ax.plot( [(T[0]-3.819)*1000.0, (T[-1]-3.819)*1000.0], [Ylow/1000.0,Yhigh/1000.0] )
        Y_ax.tick_params('y', labelsize=25)
        
        Zlow = Zintercept + Zslope*T[0]
        Zhigh = Zintercept + Zslope*T[-1]
        Z_ax.plot( [(T[0]-3.819)*1000.0, (T[-1]-3.819)*1000.0], [Zlow/1000.0,Zhigh/1000.0])
        Z_ax.tick_params('both', labelsize=25)
        
        plt.show()
    
#### IDEA:
    # add setting to ignore time cut (to overlay things from different times)
    # Add way to save a current view (not the points, but the bounding box)
    # turn 1:1 aspect ratio on or off
    # show all position for all active datsets vs selected data set
    # increase number of available colors