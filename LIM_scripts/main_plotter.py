#!/usr/bin/env python

## on APP machine

from __future__ import unicode_literals
import sys
import os

from time import sleep

import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, RectangleSelector
import matplotlib.colors as colors

## mine
from utilities import v_air
from read_PSE import read_PSE_timeID

def gen_cmap(cmap_name, minval,maxval, num=100):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, num)))
    return new_cmap





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
        self.x_label = "distance East/West (km)"
        self.y_label = "distance North/South (km)"
        self.z_label = "altitude (km)"
        self.t_label = "time (ms)"
        
    def transform(self, X, Y, Z, T):
        return (X-self.space_center[0])/1000.0, (Y-self.space_center[1])/1000.0, (Z-self.space_center[2])/1000.0, (T-self.time_center)*1000.0
    
    def invert(self, X_bar, Y_bar, Z_bar, T_bar):
        return X_bar*1000+self.space_center[0], Y_bar*1000+self.space_center[1], Z_bar*1000+self.space_center[2], T_bar/1000.0+self.time_center
    
    
    


class DataSet_Type:
    """All data plotted will be part of a data set. There can be different types of data sets, this class is a wrapper over all data sets"""
    
    def __init__(self, name, coordinate_system):
        self.name = name
        self.display = True
        self.coordinate_system = coordinate_system
    
    def getSpaceBounds(self):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
        return [[np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan]]
    
    def getFilterBounds(self):
        """return max_RMS, min_antennas"""
        return [np.nan, np.nan]
    
    def set_T_lims(self, min, max):
        pass
    
    def set_X_lims(self, min, max):
        pass
    
    def set_Y_lims(self, min, max):
        pass
    
    def set_alt_lims(self, min, max):
        pass
    
    def set_max_RMS(self, max_RMS):
        pass
    
    def set_min_numAntennas(self, min_numAntennas):
        pass
    
    def plot(self, AltvsT_axes, AltvsEW_axes, NSvsEW_axes, NsvsAlt_axes):
        pass
    
    def get_viewed_events(self):
        return []
    
    def clear(self):
        pass
    
    def search(self, index):
        return DataSet_Type(self.name+"_search", self.coordinate_system)
    
class DataSet_simplePointSources(DataSet_Type):
    """This represents a set of simple dual-polarized point sources"""
    
    def __init__(self, PSE_list, markers, marker_size, color_mode, name, coordinate_system, cmap):
        self.markers = markers
        self.marker_size = marker_size
        self.color_mode = color_mode
        self.PSE_list = PSE_list
        self.cmap = cmap
        
        ## probably should call previous constructor here
        self.name = name
        self.display = True ##this is not implemented
        self.coordinate_system = coordinate_system
        
        
        #### get the data ####
        self.polE_loc_data = np.array([coordinate_system.transform(*PSE.PolE_loc) for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==2) ])
        self.polO_loc_data = np.array([coordinate_system.transform(*PSE.PolO_loc) for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==1) ])

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
    
    
    
        #### some axis data ###
        self.PolE_AltVsT_paths = None
        self.PolE_AltVsEw_paths = None
        self.PolE_NsVsEw_paths = None
        self.PolE_NsVsAlt_paths = None
        
        self.PolO_AltVsT_paths = None
        self.PolO_AltVsEw_paths = None
        self.PolO_NsVsEw_paths = None
        self.PolO_NsVsAlt_paths = None
        
        
    def getSpaceBounds(self):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
        min_X = min( np.min(self.polE_loc_data[:,0]), np.min(self.polO_loc_data[:,0]) )
        max_X = max( np.max(self.polE_loc_data[:,0]), np.max(self.polO_loc_data[:,0]) )
        
        min_Y = min( np.min(self.polE_loc_data[:,1]), np.min(self.polO_loc_data[:,1]) )
        max_Y = max( np.max(self.polE_loc_data[:,1]), np.max(self.polO_loc_data[:,1]) )
        
        min_Z = min( np.min(self.polE_loc_data[:,2]), np.min(self.polO_loc_data[:,2]) )
        max_Z = max( np.max(self.polE_loc_data[:,2]), np.max(self.polO_loc_data[:,2]) )
        
        min_T = min( np.min(self.polE_loc_data[:,3]), np.min(self.polO_loc_data[:,3]) )
        max_T = max( np.max(self.polE_loc_data[:,3]), np.max(self.polO_loc_data[:,3]) )
        
        return [[min_X, max_X], [min_Y, max_Y], [min_Z, max_Z], [min_T, max_T]]
    
    def getFilterBounds(self):
        """return max_RMS, min_antennas"""
        max_RMS = max( np.max(self.PolE_RMS_vals), np.max(self.PolO_RMS_vals) )
        min_antennas = min( np.min(self.PolE_numAntennas), np.min(self.PolO_numAntennas) )
        return [max_RMS, min_antennas]
        
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
    
    def set_max_RMS(self, max_RMS):
        self.max_RMS = max_RMS
        self.PolE_mask_on_RMS = self.PolE_RMS_vals<max_RMS
        self.PolO_mask_on_RMS = self.PolO_RMS_vals<max_RMS
    
    def set_min_numAntennas(self, min_numAntennas):
        self.min_numAntennas = min_numAntennas
        self.PolE_mask_on_min_numAntennas = self.PolE_masked_numAntennas>min_numAntennas
        self.PolO_mask_on_min_numAntennas = self.PolO_masked_numAntennas>min_numAntennas
    
    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes):
        ## set total mask ##
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
    
        self.clear()
            
        if self.color_mode == "time":
            color = self.PolE_masked_loc_data[:,3]
        elif self.color_mode[0] == '*':
            color = self.color_mode[1:]
            
        self.PolE_AltVsT_paths = AltVsT_axes.scatter(x=self.PolE_masked_loc_data[:,3], y=self.PolE_masked_loc_data[:,2], c=color, marker=self.markers[0], s=self.marker_size, cmap=self.cmap)
        self.PolE_AltVsEw_paths = AltVsEw_axes.scatter(x=self.PolE_masked_loc_data[:,0], y=self.PolE_masked_loc_data[:,2], c=color, marker=self.markers[0], s=self.marker_size, cmap=self.cmap)
        self.PolE_NsVsEw_paths = NsVsEw_axes.scatter(x=self.PolE_masked_loc_data[:,0], y=self.PolE_masked_loc_data[:,1], c=color, marker=self.markers[0], s=self.marker_size, cmap=self.cmap)
        self.PolE_NsVsAlt_paths = NsVsAlt_axes.scatter(x=self.PolE_masked_loc_data[:,2], y=self.PolE_masked_loc_data[:,1], c=color, marker=self.markers[0], s=self.marker_size, cmap=self.cmap)
        
        self.PolO_AltVsT_paths = AltVsT_axes.scatter(x=self.PolO_masked_loc_data[:,3], y=self.PolO_masked_loc_data[:,2], c=color, marker=self.markers[1], s=self.marker_size, cmap=self.cmap)
        self.PolO_AltVsEw_paths = AltVsEw_axes.scatter(x=self.PolO_masked_loc_data[:,0], y=self.PolO_masked_loc_data[:,2], c=color, marker=self.markers[1], s=self.marker_size, cmap=self.cmap)
        self.PolO_NsVsEw_paths = NsVsEw_axes.scatter(x=self.PolO_masked_loc_data[:,0], y=self.PolO_masked_loc_data[:,1], c=color, marker=self.markers[1], s=self.marker_size, cmap=self.cmap)
        self.PolO_NsVsAlt_paths = NsVsAlt_axes.scatter(x=self.PolO_masked_loc_data[:,2], y=self.PolO_masked_loc_data[:,1], c=color, marker=self.markers[1], s=self.marker_size, cmap=self.cmap)
        
        print("  data set:", self.name, "plotting", len(self.PolE_total_mask[:,0])-np.sum(self.PolE_total_mask[:,0]),  "even source and", len(self.PolO_total_mask[:,0])-np.sum(self.PolO_total_mask[:,0]), "odd sources")

    def loc_filter(self, loc):
        if self.x_lims[0]<loc[0]<self.x_lims[1] and self.y_lims[0]<loc[1]<self.y_lims[1] and self.z_lims[0]<loc[2]<self.z_lims[1] and self.t_lims[0]<loc[3]<self.t_lims[1]:
            return True
        else:
            return False

    def get_viewed_events(self):
        return [PSE for PSE in self.PSE_list if 
                (self.loc_filter(self.coordinate_system.transform(*PSE.PolE_loc)) and PSE.PolE_RMS<self.max_RMS and PSE.num_even_antennas>self.min_numAntennas) 
                or (self.loc_filter(self.coordinate_system.transform(*PSE.PolO_loc)) and PSE.PolO_RMS<self.max_RMS and PSE.num_odd_antennas>self.min_numAntennas) ]

    
    def clear(self):
        if self.PolE_AltVsT_paths is not None:
            self.PolE_AltVsT_paths.remove()
        if self.PolE_AltVsEw_paths is not None:
            self.PolE_AltVsEw_paths.remove()
        if self.PolE_NsVsEw_paths is not None:
            self.PolE_NsVsEw_paths.remove()
        if self.PolE_NsVsAlt_paths is not None:
            self.PolE_NsVsAlt_paths.remove()
            
        if self.PolO_AltVsT_paths is not None:
            self.PolO_AltVsT_paths.remove()
        if self.PolO_AltVsEw_paths is not None:
            self.PolO_AltVsEw_paths.remove()
        if self.PolO_NsVsEw_paths is not None:
            self.PolO_NsVsEw_paths.remove()
        if self.PolO_NsVsAlt_paths is not None:
            self.PolO_NsVsAlt_paths.remove()
    
    def search(self, index, marker, marker_size, color_mode, cmap=None):
        searched_PSE = [PSE for PSE in self.PSE_list if PSE.unique_index==index]
        return DataSet_simplePointSources( searched_PSE, [marker,marker], marker_size, color_mode, self.name+"_search", self.coordinate_system, cmap)

class FigureArea(FigureCanvas):
    """This is a widget that contains the central figure"""
    
    class previous_view_state(object):
        def __init__(self, axis_change, axis_data):
            self.axis_change = axis_change ## 0 is XY, 1 is altitude, and 2 is T
            self.axis_data = axis_data
    
    
    ## ALL spatial units are km, time is in seconds
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        
        #### some default settings, some can be changed
        
        self.axis_label_size = 15
        self.axis_tick_size = 12
        self.rebalance_XY = True ## keep X and Y have same ratio
        
        self.default_marker_size = 5
        
        cmap_name = 'plasma'
        min_color = 0.0 ## start at zero means take cmap from beginning
        max_color = 0.8 ## end at 1.0 means take cmap untill end
        self.cmap = gen_cmap(cmap_name, min_color, max_color)
        
        
        self.coordinate_system = typical_transform([0.0,0.0,0.0], 3.819)
        
        # initial state variables, in the above coordinate system
        self.alt_limits = [0.0, 50.0]
        self.X_limits = [0.0, 150.0]
        self.Y_limits = [0.0, 150.0]
        self.T_limits = [0.0, 20]
        self.max_RMS = 1.5E-9
        self.min_numAntennas = 50
        
        self.previous_view_states = []
        
        self.T_fraction = 1.0 ## if is 0, then do not plot any points. if 0.5, plot points halfway beteween tlims. if 1.0 plot all points
        
        
        ## data sets
        self.simple_pointSource_DataSets = [] ## this is a list of all Data Sets that are simple point sources
        self.event_searches = [] ##data sets that represent events that were searched for
        
        
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
        self.outer_gs = matplotlib.gridspec.GridSpec(4,1, hspace=0.3)
        self.lower_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec =self.outer_gs[1:])
        
        self.AltVsT_axes = self.fig.add_subplot(self.outer_gs[0])
        self.AltVsEw_axes= self.fig.add_subplot(self.lower_gs[0,:2])
        self.NsVsEw_axes = self.fig.add_subplot(self.lower_gs[1:,:2])
#        self.histogram_axes   = self.fig.add_subplot(self.lower_gs[0,2])
        self.NsVsAlt_axes = self.fig.add_subplot(self.lower_gs[1:,2])
        
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
        
        self.set_alt_lims(self.alt_limits[0], self.alt_limits[1])
        self.set_T_lims(self.T_limits[0], self.T_limits[1])
        self.set_X_lims(self.X_limits[0], self.X_limits[1])
        self.set_Y_lims(self.Y_limits[0], self.Y_limits[1])
        
        #### setup specific plots
#        self.PolE_AltVsT_paths = self.AltVsT_axes.scatter(x=[], y=[])
#        self.PolE_AltVsEw_paths = self.AltVsEw_axes.scatter(x=[], y=[])
#        self.PolE_NsVsEw_paths = self.NsVsEw_axes.scatter(x=[], y=[])
#        self.PolE_NsVsAlt_paths = self.NsVsAlt_axes.scatter(x=[], y=[])
#        
#        self.PolO_AltVsT_paths = self.AltVsT_axes.scatter(x=[], y=[])
#        self.PolO_AltVsEw_paths = self.AltVsEw_axes.scatter(x=[], y=[])
#        self.PolO_NsVsEw_paths = self.NsVsEw_axes.scatter(x=[], y=[])
#        self.PolO_NsVsAlt_paths = self.NsVsAlt_axes.scatter(x=[], y=[])
        
        
        #### create selectors on plots
        self.T_selector_span = SpanSelector(self.AltVsT_axes, self.T_selector, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'), button=1)
        
        self.alt_selector_span = SpanSelector(self.AltVsEw_axes, self.alt_selector, 'vertical', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'), button=1)
        
        self.XY_selector_rect = RectangleSelector(self.NsVsEw_axes, self.XY_selector, useblit=False,
                      rectprops=dict(alpha=0.5, facecolor='red'), button=1)
        
        #### create button press/release events
        self.key_press = self.fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        self.key_release = self.fig.canvas.mpl_connect('key_release_event', self.key_release_event)
        self.button_press = self.fig.canvas.mpl_connect('button_press_event', self.button_press_event) ##mouse button
        self.button_release = self.fig.canvas.mpl_connect('button_release_event', self.button_release_event) ##mouse button
        
        ##state variables
        self.z_button_pressed = False
        self.right_mouse_button_location = None
        self.right_button_axis = None
        
    def add_simplePSE(self, PSE_list, name=None, markers=None, color_mode=None, size=None):
        
        if name is None:
            name = "set "+str(len(self.simple_pointSource_DataSets))
        if color_mode is None:
            color_mode = 'time'
        if size is None:
            size = self.default_marker_size
        if markers is None:
            markers = ['s', 'D']
        
        new_simplePSE_dataset = DataSet_simplePointSources(PSE_list, markers, size, color_mode, name, self.coordinate_system, self.cmap)
        self.simple_pointSource_DataSets.append(new_simplePSE_dataset)
        
        new_simplePSE_dataset.set_T_lims( *self.T_limits )
        new_simplePSE_dataset.set_X_lims( *self.X_limits )
        new_simplePSE_dataset.set_Y_lims( *self.Y_limits )
        new_simplePSE_dataset.set_alt_lims( *self.alt_limits)
        new_simplePSE_dataset.set_max_RMS( self.max_RMS )
        new_simplePSE_dataset.set_min_numAntennas( self.min_numAntennas )
        
        self.replot_data()
        
        return name


    def show_all_PSE(self):
        min_X = np.nan
        max_X = np.nan
        
        min_Y = np.nan
        max_Y = np.nan
        
        min_Z = np.nan
        max_Z = np.nan
        
        min_T = np.nan
        max_T = np.nan
        
        max_RMS = np.nan
        
        min_ant = np.nan
        
        for DS in self.simple_pointSource_DataSets:
            [ds_xmin, ds_xmax], [ds_ymin, ds_ymax], [ds_zmin, ds_zmax], [ds_tmin, ds_tmax] = DS.getSpaceBounds()
            ds_maxRMS, ds_minAnt = DS.getFilterBounds()

            if np.isfinite( ds_xmin ):
                if np.isfinite( min_X ):
                    min_X = min(min_X, ds_xmin)
                else:
                    min_X = ds_xmin
            if np.isfinite( ds_xmax ):
                if np.isfinite( max_X ):
                    max_X = max(max_X, ds_xmax)
                else:
                    max_X = ds_xmax
            
            if np.isfinite( ds_ymin ):
                if np.isfinite( min_Y ):
                    min_Y = min(min_Y, ds_ymin)
                else:
                    min_Y = ds_ymin
            if np.isfinite( ds_ymax ):
                if np.isfinite( max_Y ):
                    max_Y = max(max_Y, ds_ymax)
                else:
                    max_Y = ds_ymax
            
            if np.isfinite( ds_zmin ):
                if np.isfinite( min_Z ):
                    min_Z = min(min_Z, ds_zmin)
                else:
                    min_Z = ds_zmin
            if np.isfinite( ds_zmax ):
                if np.isfinite( max_Z ):
                    max_Z = max(max_Z, ds_zmax)
                else:
                    max_Z = ds_zmax
            
            if np.isfinite( ds_tmin ):
                if np.isfinite( min_T ):
                    min_T = min(min_T, ds_tmin)
                else:
                    min_T = ds_tmin
            if np.isfinite( ds_tmax ):
                if np.isfinite( max_T ):
                    max_T = max(max_T, ds_tmax)
                else:
                    max_T = ds_tmax
                    
            if np.isfinite( ds_maxRMS ):
                if np.isfinite( max_RMS ):
                    max_RMS = max(max_RMS, ds_maxRMS)
                else:
                    max_RMS = ds_maxRMS
                    
            if np.isfinite( ds_minAnt ):
                if np.isfinite( min_ant ):
                    min_ant = min(min_ant, ds_minAnt)
                else:
                    min_ant = ds_minAnt
        
        self.set_T_lims(min_T, max_T, 0.1)
        self.set_X_lims(min_X, max_X, 0.1)
        self.set_Y_lims(min_Y, max_Y, 0.1)
        self.set_alt_lims(min_Z, max_Z, 0.1)
        self.set_max_RMS(max_RMS*1.1)
        self.set_min_numAntennas( int(min_ant*0.9) )
        
        self.replot_data()

    def get_viewed_events(self, data_set_name):
        for DS in self.simple_pointSource_DataSets:
            if DS.name == data_set_name:
                return DS.get_viewed_events()
            
        return []

    #### set limits
        
    def set_T_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.T_limits[0]
        if upper is None:
            upper = self.T_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.T_limits = [lower, upper]
        self.AltVsT_axes.set_xlim(self.T_limits)
        
        for DS in self.simple_pointSource_DataSets:
            DS.set_T_lims( self.T_limits[0], self.T_limits[0] + (self.T_limits[1]-self.T_limits[0])*self.T_fraction )
        
    def set_alt_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.alt_limits[0]
        if upper is None:
            upper = self.alt_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.alt_limits = [lower, upper]
        self.AltVsT_axes.set_ylim(self.alt_limits)
        self.AltVsEw_axes.set_ylim(self.alt_limits)
        self.NsVsAlt_axes.set_xlim(self.alt_limits)
        
        for DS in self.simple_pointSource_DataSets:
            DS.set_alt_lims( *self.alt_limits )
        
    def set_X_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.X_limits[0]
        if upper is None:
            upper = self.X_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.X_limits = [lower, upper]
        self.NsVsEw_axes.set_xlim(self.X_limits)
        self.AltVsEw_axes.set_xlim(self.X_limits)
        
        for DS in self.simple_pointSource_DataSets:
            DS.set_X_lims( *self.X_limits )
        
    def set_Y_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.Y_limits[0]
        if upper is None:
            upper = self.Y_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.Y_limits = [lower, upper]
        self.NsVsAlt_axes.set_ylim(self.Y_limits)
        self.NsVsEw_axes.set_ylim(self.Y_limits)
        
        for DS in self.simple_pointSource_DataSets:
            DS.set_Y_lims( *self.Y_limits )
        
    def set_max_RMS(self, max_RMS=None):
        if max_RMS is not None:
            self.max_RMS = max_RMS
        
        for DS in self.simple_pointSource_DataSets:
            DS.set_max_RMS( self.max_RMS)
        
    def set_min_numAntennas(self, min_numAntennas=None):
        if min_numAntennas is not None:
            self.min_numAntennas = min_numAntennas
        
        for DS in self.simple_pointSource_DataSets:
            DS.set_min_numAntennas( self.min_numAntennas )


        
    def replot_data(self):
        
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
                
        for DS in self.simple_pointSource_DataSets:
            DS.plot( self.AltVsT_axes, self.AltVsEw_axes, self.NsVsEw_axes, self.NsVsAlt_axes )
        
    #### calbacks for various events
    def T_selector(self, minT, maxT):
        if minT == maxT: return
        
        self.previous_view_states.append( self.previous_view_state(2, self.T_limits[:]) )
        
        if self.z_button_pressed: ## then we zoom out,
            minT = 2.0*self.T_limits[0] - minT
            maxT = 2.0*self.T_limits[1] - maxT
            
        self.set_T_lims(minT, maxT)
        self.replot_data()
        
    def alt_selector(self, minA, maxA):
        if minA==maxA: return
        self.previous_view_states.append( self.previous_view_state(1, self.alt_limits[:]) )
            
        if self.z_button_pressed:
            minA = 2.0*self.alt_limits[0] - minA
            maxA = 2.0*self.alt_limits[1] - maxA
        
        self.set_alt_lims(minA, maxA)
        self.replot_data()
    
    def XY_selector(self, eclick, erelease):
        minX = min(eclick.xdata, erelease.xdata)
        maxX = max(eclick.xdata, erelease.xdata)
        minY = min(eclick.ydata, erelease.ydata)
        maxY = max(eclick.ydata, erelease.ydata)
        
        if minX==maxX or minY==maxY: return
        
        self.previous_view_states.append( self.previous_view_state(0, (self.X_limits[:],self.Y_limits[:]) ) ) 
        
        if self.z_button_pressed:
            minX = 2.0*self.X_limits[0] - minX
            maxX = 2.0*self.X_limits[1] - maxX
            minY = 2.0*self.Y_limits[0] - minY
            maxY = 2.0*self.Y_limits[1] - maxY
            
        self.set_X_lims(minX, maxX)
        self.set_Y_lims(minY, maxY)
        self.replot_data()
        
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
        if event.button == 3: ##drag
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
            
#    def event_search(self, event_index):
#        for DS in self.event_searches:
#            DS.clear()
        
#        self.event_searches = [DS.search(event_index, ['o','o'], self.default_marker_size, '*r') for DS in self.simple_pointSource_DataSets]
            
        
    #### external usefull functions
    def set_T_fraction(self, t_fraction):
        """plot points with all time less that the t_fraction. Where 0 is beginning of t-lim and 1 is end of t_lim"""
        self.T_fraction = t_fraction
        self.set_T_lims()
                
        

class Active3DPlotter(QtWidgets.QMainWindow):
    """This is the main window. Contains the figure window, and controlls all the menus and buttons"""
    
    def __init__(self):
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
        self.figure_space = FigureArea(self.main_widget)
        horizontal_divider.addWidget( self.figure_space )
        
        
        
        
        #### buttans and controlls on left
        ## max RMS box
        self.max_RMS_input = QtWidgets.QLineEdit(self)
        self.max_RMS_input.move(105, 120)
        self.max_RMS_input.resize(50,25)
        ## input txt box
        self.max_RMS_label = QtWidgets.QLabel(self)
        self.max_RMS_label.setText("max RMS (ns):")
        self.max_RMS_label.move(5, 120)
        
        ## min numStations box
        self.min_numAntennas_input = QtWidgets.QLineEdit(self)
        self.min_numAntennas_input.move(125, 150)
        self.min_numAntennas_input.resize(70,25)
        ## input txt box
        self.min_numAntennas_label = QtWidgets.QLabel(self)
        self.min_numAntennas_label.setText("min. num antennas:")
        self.min_numAntennas_label.move(5, 150)
        
        
        ## show all PSE button
        self.showAll_button = QtWidgets.QPushButton(self)
        self.showAll_button.move(10, 30)
        self.showAll_button.setText("show all")
        self.showAll_button.clicked.connect(self.showAllButtonPressed)
        
        ## send the settings button
        self.set_button = QtWidgets.QPushButton(self)
        self.set_button.move(10, 70)
        self.set_button.setText("set")
        self.set_button.clicked.connect(self.setButtonPressed)
        
        ## refresh button
        self.get_button = QtWidgets.QPushButton(self)
        self.get_button.move(160, 70)
        self.get_button.setText("get")
        self.get_button.clicked.connect(self.getButtonPressed)
        
        
        
        ### animation controlls 
        self.animation_label = QtWidgets.QLabel(self)
        self.animation_label.setText("animation time (s):")
        self.animation_label.resize(160,25)
        self.animation_label.move(5, 500)
        
        self.animation_input = QtWidgets.QLineEdit(self)
        self.animation_input.move(120, 500)
        self.animation_input.resize(40,25)
        
        self.animation_FPS_label = QtWidgets.QLabel(self)
        self.animation_FPS_label.setText("FPS:")
        self.animation_FPS_label.resize(40,25)
        self.animation_FPS_label.move(80, 530)
        
        self.animation_FPS_input = QtWidgets.QLineEdit(self)
        self.animation_FPS_input.move(120, 530)
        self.animation_FPS_input.resize(40,25)
        
        self.animate_button = QtWidgets.QPushButton(self)
        self.animate_button.move(160, 500)
        self.animate_button.setText("animate")
        self.animate_button.clicked.connect(self.animateButtonPressed)
        
        
        ### search button
        self.search_input = QtWidgets.QLineEdit(self)
        self.search_input.move(120, 700)
        self.search_input.resize(40,25)
        
        self.animate_button = QtWidgets.QPushButton(self)
        self.animate_button.move(5, 700)
        self.animate_button.setText("search")
        self.animate_button.clicked.connect(self.searchButtonPressed)
        
        
        
        ### set the fields
        self.getButtonPressed()
        
        
        
    def add_analysis(self, analysis_name, function, data_set_names):
        functor = lambda : self.run_analysis(function, data_set_names)
        self.analysis_menu.addAction('&'+analysis_name, functor)
        
    def run_analysis(self, function, data_set_names):
        viewed_events = []
        for DS_name in data_set_names:
            viewed_events += self.figure_space.get_viewed_events(DS_name)
        function(viewed_events)
        
        
    #### menu bar call backs ####
    ## file
    def fileQuit(self):
        self.close()
        
    def savePlot(self):
        output_fname = "./plot_save.png"
        self.figure_space.fig.savefig(output_fname)
        
    ##help
    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """Bla Bla Bla""")

    def closeEvent(self, ce):
        self.fileQuit()
        
    def add_simplePSE(self, PSE_list, name=None):
        return self.figure_space.add_simplePSE(PSE_list, name=name)
        
    
    #### button callbacks ####
    def setButtonPressed(self):
        try:
            maxRMS = float(self.max_RMS_input.text())*1.0E-9
        except:
            print( "invalid literal in max RMS field")
            self.getButtonPressed()
            return
        
        try:
            minNumAntennas = int(self.min_numAntennas_input.text())
        except:
            print( "invalid literal in min num stations field")
            self.getButtonPressed()
            return
   
        self.figure_space.set_max_RMS( maxRMS )
        self.figure_space.set_min_numAntennas( minNumAntennas )
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def showAllButtonPressed(self):
        self.figure_space.show_all_PSE()
        
        self.getButtonPressed()
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def getButtonPressed(self):
        #### set text in max RMS field
        self.max_RMS_input.setText( "{:.2f}".format(self.figure_space.max_RMS/(1.0E-9) ))
        #### set text in min num stations
        self.min_numAntennas_input.setText( str(self.figure_space.min_numAntennas ))
        
    def animateButtonPressed(self):
        try:
            animation_time = float(self.animation_input.text())
        except:
            print( "invalid literal in animation time field")
            return
        
        try:
            FPS = float(self.animation_FPS_input.text())
        except:
            print( "invalide interger in FPS field")
        
        n_frames = int(animation_time*FPS)
        
        frame_Tfractions, TfracStep = np.linspace(0.0, 1.0, n_frames, endpoint=True, retstep=True)
        time_per_step = TfracStep*animation_time
        FPS = 1.0/time_per_step
        
        print( "animate with FPS:", FPS)
        
        current_animation_index = 0
        
#        print manimation.writers.list()
#        FFMpegWriter = manimation.writers['FFmpeg']
#        metadata = dict(title='LightningAnimation', artist='Matplotlib')
#        writer = FFMpegWriter(fps=FPS, metadata=metadata)
#        controller = writer.saving(self.figure_space.fig, "LightningAnimation.mp4", n_frames)
        
        self.figure_space.set_T_fraction( frame_Tfractions[ current_animation_index ] )
        self.figure_space.replot_data()
        self.figure_space.fig.canvas.draw()
        self.figure_space.draw()
#        writer.grab_frame()

#        self.figure_space.fig.savefig("./animation_output/"+str(current_animation_index).zfill(4))
        
        timer = QtCore.QTimer(self)
        timer.singleShot(time_per_step*1000, lambda: self.animateUpdate(current_animation_index, frame_Tfractions, time_per_step) )
        
    def animateUpdate(self, animation_index, Tfractions, time_per_step):
        animation_index += 1
        if animation_index == len(Tfractions):
            print( "done animating")
            return
        
        self.figure_space.set_T_fraction( Tfractions[ animation_index ] )
        self.figure_space.replot_data()
        self.figure_space.fig.canvas.draw()
        self.figure_space.draw()
        
#        self.figure_space.fig.set_size_inches(7.8, 10.6)
#        self.figure_space.fig.savefig("./animation_output/"+str(animation_index).zfill(4), dpi=100)
        
        timer = QtCore.QTimer(self)
        timer.singleShot(time_per_step*1000, lambda: self.animateUpdate(animation_index, Tfractions, time_per_step) )
        
        
    def searchButtonPressed(self):
        try:
            event_index = int(self.search_input.text())
        except:
            print( "invalid event index in search field")
            return
        
        self.figure_space.event_search( event_index )

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


if __name__=="__main__":
#    data = read_PSE_timeID("D20160712T173455.100Z", "subEvents", data_loc="/home/brian/processed_files")
#    PSE_list = data["sub_PSE_list"]
    
     
    data = read_PSE_timeID("D20160712T173455.100Z", "allPSE_new3") ## we read PSE from analysis "allPSE_new3" of flash that has timeID of "D20160712T173455.100Z"
    PSE_list = data["PSE_list"] ## this is a list of point source event (PSE) objects
    ant_locs = data["ant_locations"] ## this is a diction of XYZ antenna locations
    
    ## see the class plot_data_analysis, above, for an example of extracting source locations and plotting the pulse timeseries data.
    
    print( "opening data:", len(PSE_list))
    
    
    qApp = QtWidgets.QApplication(sys.argv)

    plotter = Active3DPlotter()
    plotter.setWindowTitle("LOFAR-LIM data viewer")
    
    plotter.add_analysis( "Print Details", print_details_analysis, ["PSE"]  )
    plotter.add_analysis( "Plot Trace Data", plot_data_analysis( ant_locs ), ["PSE"] )
    plotter.add_analysis( "Print Ave. Ant. Delays", print_ave_ant_delays(ant_locs), ["PSE"] )
    plotter.add_analysis( "plot amplitude histogram", histogram_amplitudes("188",1), ["PSE"] )
    plotter.add_analysis( "analyze speed", leader_speed_estimator(), ["PSE"] )
    
#    LSE = leader_speed_estimator()
#    plotter.add_analysis( "analyze leader speed", LSE, ["PSE"] )
    
    plotter.show()
    plotter.add_simplePSE( PSE_list, name="PSE" )
    
    qApp.exec_()
    
#    LSE.go()
#    LSE.save("../velocity_data/neg_seg_1")
    
    