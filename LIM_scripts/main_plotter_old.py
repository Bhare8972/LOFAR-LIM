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

class DataSet_Type:
    """All data plotted will be part of a data set. There can be different types of data sets, this class is a wrapper over all data sets"""
    
    def getlargestBounds(self):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
        return [[np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan]]
    
class DataSet_4DPoints(DataSet_Type):
    """This represents a set of 4D points"""
    
    def __init__():
        
class FigureArea(FigureCanvas):
    """This is a widget that contains the central figure"""
    
    class previous_view_state(object):
        def __init__(self, axis_change, axis_data):
            self.axis_change = axis_change ## 0 is XY, 1 is altitude, and 2 is T
            self.axis_data = axis_data
    
    
    ## ALL spatial units are km, time is in seconds
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        
        ##NO idea how to handle polarization. Right now, just pick the best fit
        
        #### some default settings, some can be changed
        self.T0 = 3819.0## in ms
        
        self.axis_label_size = 15
        self.axis_tick_size = 12
        self.rebalance_XY = True ## keep X and Y have same ratio
        
        self.marker_size = 5
        
        cmap_name = 'plasma'
        min_color = 0.0 ## start at zero means take cmap from beginning
        max_color = 0.8 ## end at 1.0 means take cmap untill end
        
        self.cmap = gen_cmap(cmap_name, min_color, max_color)
        
        # initial state variables
        self.alt_limits = [0.0, 50.0]
        self.X_limits = [0.0, 150.0]
        self.Y_limits = [0.0, 150.0]
        self.T_limits = [0.0, 20]
        self.max_RMS = 1.5E-9
        self.min_numAntennas = 50
        self.show_even_pol = True
        self.show_odd_pol = True
        
        self.previous_view_states = []
        
        #### all data ####
        self.PolE_alt_data = np.array( [], dtype=np.double)
        self.PolE_X_data = np.array( [], dtype=np.double )
        self.PolE_Y_data = np.array( [], dtype=np.double )
        self.PolE_T_data = np.array( [], dtype=np.double )
        self.PolE_RMS_vals = np.array( [], dtype=np.double )
        self.PolE_numAntennas = np.array([], dtype=np.long)
        self.PolE_color_data = np.array( [], dtype=np.double ) ## any set of doubles that can be matched to a variable
        
        self.PolO_alt_data = np.array( [], dtype=np.double)
        self.PolO_X_data = np.array( [], dtype=np.double )
        self.PolO_Y_data = np.array( [], dtype=np.double )
        self.PolO_T_data = np.array( [], dtype=np.double )
        self.PolO_RMS_vals = np.array( [], dtype=np.double )
        self.PolO_numAntennas = np.array([], dtype=np.long)
        self.PolO_color_data = np.array( [], dtype=np.double ) ## any set of doubles that can be matched to a variable
        
        #### individual masks ####
        self.PolE_mask_on_alt = np.array([], dtype=bool)
        self.PolE_mask_on_X   = np.array([], dtype=bool)
        self.PolE_mask_on_Y   = np.array([], dtype=bool)
        self.PolE_mask_on_T   = np.array([], dtype=bool)
        self.PolE_mask_on_RMS = np.array([], dtype=bool)
        self.PolE_mask_on_min_numAntennas = np.array([], dtype=bool)
        
        self.PolE_total_mask = np.array([], dtype=bool)
        
        
        self.PolO_mask_on_alt = np.array([], dtype=bool)
        self.PolO_mask_on_X   = np.array([], dtype=bool)
        self.PolO_mask_on_Y   = np.array([], dtype=bool)
        self.PolO_mask_on_T   = np.array([], dtype=bool)
        self.PolO_mask_on_RMS = np.array([], dtype=bool)
        self.PolO_mask_on_min_numAntennas = np.array([], dtype=bool)
        
        self.PolO_total_mask = np.array([], dtype=bool)
         
        
        
        self.PolE_masked_alt_data   = np.ma.masked_array(self.PolE_alt_data, mask=self.PolE_total_mask, copy=False)
        self.PolE_masked_X_data     = np.ma.masked_array(self.PolE_X_data, mask=self.PolE_total_mask, copy=False)
        self.PolE_masked_Y_data     = np.ma.masked_array(self.PolE_Y_data, mask=self.PolE_total_mask, copy=False)
        self.PolE_masked_T_data     = np.ma.masked_array(self.PolE_T_data, mask=self.PolE_total_mask, copy=False)
        self.PolE_masked_RMS_vals   = np.ma.masked_array(self.PolE_RMS_vals, mask=self.PolE_total_mask, copy=False)
        self.PolE_masked_color_data = np.ma.masked_array(self.PolE_color_data, mask=self.PolE_total_mask, copy=False)
        
        self.PolO_masked_alt_data   = np.ma.masked_array(self.PolO_alt_data, mask=self.PolO_total_mask, copy=False)
        self.PolO_masked_X_data     = np.ma.masked_array(self.PolO_X_data, mask=self.PolO_total_mask, copy=False)
        self.PolO_masked_Y_data     = np.ma.masked_array(self.PolO_Y_data, mask=self.PolO_total_mask, copy=False)
        self.PolO_masked_T_data     = np.ma.masked_array(self.PolO_T_data, mask=self.PolO_total_mask, copy=False)
        self.PolO_masked_RMS_vals   = np.ma.masked_array(self.PolO_RMS_vals, mask=self.PolO_total_mask, copy=False)
        self.PolO_masked_color_data = np.ma.masked_array(self.PolO_color_data, mask=self.PolO_total_mask, copy=False)
        
        
        self.PSE = []
        
        self.T_fraction = 1.0 ## if is 0, then do not plot any points. if 0.5, plot points halfway beteween tlims. if 1.0 plot all points
        
        
        
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
        
        self.AltVsT_axes.set_xlabel("time (ms)", fontsize=self.axis_label_size)
        self.AltVsT_axes.set_ylabel("altitude (km)", fontsize=self.axis_label_size)
        self.AltVsT_axes.tick_params(labelsize = self.axis_tick_size)
        
        
        self.AltVsEw_axes.set_ylabel("altitude (km)", fontsize=self.axis_label_size)
        self.AltVsEw_axes.tick_params(labelsize = self.axis_tick_size)
        self.AltVsEw_axes.get_xaxis().set_visible(False)
        
        
        self.NsVsEw_axes.set_xlabel("distance East/West (km)", fontsize=self.axis_label_size)
        self.NsVsEw_axes.set_ylabel("distance North/South (km)", fontsize=self.axis_label_size)
        self.NsVsEw_axes.tick_params(labelsize = self.axis_tick_size)
        
        
        self.NsVsAlt_axes.set_xlabel("altitude (km)", fontsize=self.axis_label_size)
        self.NsVsAlt_axes.tick_params(labelsize = self.axis_tick_size)
        self.NsVsAlt_axes.get_yaxis().set_visible(False)
        
        self.set_alt_lims(self.alt_limits[0], self.alt_limits[1])
        self.set_T_lims(self.T_limits[0], self.T_limits[1])
        self.set_X_lims(self.X_limits[0], self.X_limits[1])
        self.set_Y_lims(self.Y_limits[0], self.Y_limits[1])
        
        #### setup specific plots
        self.PolE_AltVsT_paths = self.AltVsT_axes.scatter(x=[], y=[])
        self.PolE_AltVsEw_paths = self.AltVsEw_axes.scatter(x=[], y=[])
        self.PolE_NsVsEw_paths = self.NsVsEw_axes.scatter(x=[], y=[])
        self.PolE_NsVsAlt_paths = self.NsVsAlt_axes.scatter(x=[], y=[])
        
        self.PolO_AltVsT_paths = self.AltVsT_axes.scatter(x=[], y=[])
        self.PolO_AltVsEw_paths = self.AltVsEw_axes.scatter(x=[], y=[])
        self.PolO_NsVsEw_paths = self.NsVsEw_axes.scatter(x=[], y=[])
        self.PolO_NsVsAlt_paths = self.NsVsAlt_axes.scatter(x=[], y=[])
        
        
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
        
        
    def add_PSE(self, PSE_list):
        
        #### this function should be fixed to respect limits that are in-place
        self.PSE += PSE_list
        
        self.PolE_X_data = np.append(self.PolE_X_data,     [PSE.PolE_loc[0]/1000.0 for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==2)] )
        self.PolE_Y_data = np.append(self.PolE_Y_data,     [PSE.PolE_loc[1]/1000.0 for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==2)] )
        self.PolE_alt_data = np.append(self.PolE_alt_data, [PSE.PolE_loc[2]/1000.0 for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==2)] )
        self.PolE_T_data = np.append(self.PolE_T_data,     [PSE.PolE_loc[3] for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==2)] )
        self.PolE_RMS_vals = np.append(self.PolE_RMS_vals ,    [PSE.PolE_RMS for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==2)] )
        self.PolE_numAntennas= np.append(self.PolE_numAntennas, [PSE.num_even_antennas for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==2)]) ##should be same between polarizations
        
        self.PolE_color_data = self.PolE_T_data ## color based on time for the moment
        
        
        
        self.PolO_X_data = np.append(self.PolO_X_data,     [PSE.PolO_loc[0]/1000.0 for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==1)] )
        self.PolO_Y_data = np.append(self.PolO_Y_data,     [PSE.PolO_loc[1]/1000.0 for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==1)] )
        self.PolO_alt_data = np.append(self.PolO_alt_data, [PSE.PolO_loc[2]/1000.0 for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==1)] )
        self.PolO_T_data = np.append(self.PolO_T_data,     [PSE.PolO_loc[3] for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==1)] )
        self.PolO_RMS_vals = np.append(self.PolO_RMS_vals ,    [PSE.PolO_RMS for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==1)] )
        self.PolO_numAntennas= np.append(self.PolO_numAntennas, [PSE.num_odd_antennas for PSE in PSE_list if (PSE.polarization_status==0 or PSE.polarization_status==1)]) ##should be same between polarizations
        
        self.PolO_color_data = self.PolO_T_data ## color based on time for the moment
        
        
        self.set_Y_lims()
        self.set_X_lims()
        self.set_alt_lims()
        self.set_T_lims()
        self.set_max_RMS()
        self.set_min_numAntennas()
        
        self.reset_total_mask()
        
        self.replot_data()
        
    def show_all_PSE(self):
        min_T = min( np.min(self.PolE_T_data), np.min(self.PolO_T_data) )
        max_T = max( np.max(self.PolE_T_data), np.max(self.PolO_T_data) ) 
        
        min_X = min( np.min(self.PolE_X_data), np.min(self.PolO_X_data) )
        max_X = max( np.max(self.PolE_X_data), np.max(self.PolO_X_data) )
        
        min_Y = min( np.min(self.PolE_Y_data), np.min(self.PolO_Y_data) )
        max_Y = max( np.max(self.PolE_Y_data), np.max(self.PolO_Y_data) )
        
        min_alt = min( np.min(self.PolE_alt_data), np.min(self.PolO_alt_data) )
        max_alt = max( np.max(self.PolE_alt_data), np.max(self.PolO_alt_data) )
        
        max_RMS = max( np.max( self.PolE_RMS_vals ), np.max( self.PolO_RMS_vals ))
        min_antennas = min( np.min( self.PolE_numAntennas ), np.min( self.PolO_numAntennas ) )
        
        self.set_T_lims(min_T, max_T, 0.1)
        self.set_X_lims(min_X, max_X, 0.1)
        self.set_Y_lims(min_Y, max_Y, 0.1)
        self.set_alt_lims(min_alt, max_alt, 0.1)
        self.set_max_RMS(max_RMS*1.1)
        self.set_min_numAntennas( int(min_antennas*0.9) )
        
        self.reset_total_mask()
        self.replot_data()
        
        
    def get_viewed_PSE(self):
        return [PSE for PSE,PolE_masked,PolO_masked in zip(self.PSE, self.PolE_total_mask, self.PolO_total_mask) if (not PolE_masked) or (not PolO_masked)]
        
    #### set limits
        
    def set_T_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.T_limits[0]
        if upper is None:
            upper = self.T_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.T_limits = [lower, upper]
        self.AltVsT_axes.set_xlim(np.array(self.T_limits)*1000.0-self.T0)
        
        self.PolE_mask_on_T = np.logical_and( self.PolE_T_data>self.T_limits[0], self.PolE_T_data<(self.T_limits[0]+(self.T_limits[1]-self.T_limits[0])*self.T_fraction))
        self.PolO_mask_on_T = np.logical_and( self.PolO_T_data>self.T_limits[0], self.PolO_T_data<(self.T_limits[0]+(self.T_limits[1]-self.T_limits[0])*self.T_fraction))
        
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
        
        self.PolE_mask_on_alt = np.logical_and( self.PolE_alt_data>self.alt_limits[0], self.PolE_alt_data<self.alt_limits[1] )
        self.PolO_mask_on_alt = np.logical_and( self.PolO_alt_data>self.alt_limits[0], self.PolO_alt_data<self.alt_limits[1] )
        
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
        
        self.PolE_mask_on_X = np.logical_and( self.PolE_X_data>self.X_limits[0], self.PolE_X_data<self.X_limits[1] )
        self.PolO_mask_on_X = np.logical_and( self.PolO_X_data>self.X_limits[0], self.PolO_X_data<self.X_limits[1] )
        
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
        
        self.PolE_mask_on_Y = np.logical_and( self.PolE_Y_data>self.Y_limits[0], self.PolE_Y_data<self.Y_limits[1] )
        self.PolO_mask_on_Y = np.logical_and( self.PolO_Y_data>self.Y_limits[0], self.PolO_Y_data<self.Y_limits[1] )
        
    def set_max_RMS(self, max_RMS=None):
        if max_RMS is not None:
            self.max_RMS = max_RMS
        self.PolE_mask_on_RMS = self.PolE_RMS_vals<self.max_RMS
        self.PolO_mask_on_RMS = self.PolO_RMS_vals<self.max_RMS
        
    def set_min_numAntennas(self, min_numAntennas=None):
        if min_numAntennas is not None:
            self.min_numAntennas = min_numAntennas
        self.PolE_mask_on_min_numAntennas = self.PolE_numAntennas>=self.min_numAntennas
        self.PolO_mask_on_min_numAntennas = self.PolO_numAntennas>=self.min_numAntennas
        

    #### plot data
    def reset_total_mask(self):
        
        if len(self.PolE_total_mask) != len(self.PolE_mask_on_alt): ### the amount of data has changed!
            self.PolE_total_mask = np.zeros(len(self.PolE_mask_on_alt), dtype=bool)
            
            self.PolE_masked_alt_data   = np.ma.masked_array(self.PolE_alt_data, mask=self.PolE_total_mask, copy=False)
            self.PolE_masked_X_data     = np.ma.masked_array(self.PolE_X_data, mask=self.PolE_total_mask, copy=False)
            self.PolE_masked_Y_data     = np.ma.masked_array(self.PolE_Y_data, mask=self.PolE_total_mask, copy=False)
            self.PolE_masked_T_data     = np.ma.masked_array(self.PolE_T_data, mask=self.PolE_total_mask, copy=False)
            
            self.PolE_masked_RMS_vals   = np.ma.masked_array(self.PolE_RMS_vals, mask=self.PolE_total_mask, copy=False)
            self.PolE_masked_color_data = np.ma.masked_array(self.PolE_color_data, mask=self.PolE_total_mask, copy=False)


        self.PolE_total_mask[:] = self.PolE_mask_on_alt
        np.logical_and(self.PolE_total_mask, self.PolE_mask_on_X, out=self.PolE_total_mask)
        np.logical_and(self.PolE_total_mask, self.PolE_mask_on_Y, out=self.PolE_total_mask)
        np.logical_and(self.PolE_total_mask, self.PolE_mask_on_T, out=self.PolE_total_mask)
        np.logical_and(self.PolE_total_mask, self.PolE_mask_on_RMS, out=self.PolE_total_mask)
        np.logical_and(self.PolE_total_mask, self.PolE_mask_on_min_numAntennas, out=self.PolE_total_mask)
        np.logical_not(self.PolE_total_mask, out=self.PolE_total_mask) ##becouse the meaning of the masks is flipped
        
        
        
        if len(self.PolO_total_mask) != len(self.PolO_mask_on_alt): ### the amount of data has changed!
            self.PolO_total_mask = np.zeros(len(self.PolO_mask_on_alt), dtype=bool)
            
            self.PolO_masked_alt_data   = np.ma.masked_array(self.PolO_alt_data, mask=self.PolO_total_mask, copy=False)
            self.PolO_masked_X_data     = np.ma.masked_array(self.PolO_X_data, mask=self.PolO_total_mask, copy=False)
            self.PolO_masked_Y_data     = np.ma.masked_array(self.PolO_Y_data, mask=self.PolO_total_mask, copy=False)
            self.PolO_masked_T_data     = np.ma.masked_array(self.PolO_T_data, mask=self.PolO_total_mask, copy=False)
            
            self.PolO_masked_RMS_vals   = np.ma.masked_array(self.PolO_RMS_vals, mask=self.PolO_total_mask, copy=False)
            self.PolO_masked_color_data = np.ma.masked_array(self.PolO_color_data, mask=self.PolO_total_mask, copy=False)


        self.PolO_total_mask[:] = self.PolO_mask_on_alt
        np.logical_and(self.PolO_total_mask, self.PolO_mask_on_X, out=self.PolO_total_mask)
        np.logical_and(self.PolO_total_mask, self.PolO_mask_on_Y, out=self.PolO_total_mask)
        np.logical_and(self.PolO_total_mask, self.PolO_mask_on_T, out=self.PolO_total_mask)
        np.logical_and(self.PolO_total_mask, self.PolO_mask_on_RMS, out=self.PolO_total_mask)
        np.logical_and(self.PolO_total_mask, self.PolO_mask_on_min_numAntennas, out=self.PolO_total_mask)
        np.logical_not(self.PolO_total_mask, out=self.PolO_total_mask) ##becouse the meaning of the masks is flipped
        
        
    def replot_data(self):
        
        print("showing", len(self.PolE_total_mask)-np.sum(self.PolE_total_mask), "PolE sources")
        print("showing", len(self.PolO_total_mask)-np.sum(self.PolO_total_mask), "PolO sources")
        
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
                self.Y_limits = [middle_y-new_height/2.0, middle_y+new_height/2.0]
                self.NsVsAlt_axes.set_ylim(self.Y_limits)
                self.NsVsEw_axes.set_ylim(self.Y_limits)
                
            else:
                new_width = ax_width*data_height/ax_height
                middle_X = (self.X_limits[0] + self.X_limits[1])/2.0
                self.X_limits = [middle_X-new_width/2.0, middle_X+new_width/2.0]
                self.NsVsEw_axes.set_xlim(self.X_limits)
                self.AltVsEw_axes.set_xlim(self.X_limits)

        self.PolE_AltVsT_paths.remove()
        self.PolO_AltVsT_paths.remove()
        self.PolE_AltVsT_paths = self.AltVsT_axes.scatter(x=self.PolE_masked_T_data*1000.0-self.T0, y=self.PolE_masked_alt_data, c=self.PolE_masked_color_data, marker='s', s=self.marker_size, cmap=self.cmap)
        self.PolO_AltVsT_paths = self.AltVsT_axes.scatter(x=self.PolO_masked_T_data*1000.0-self.T0, y=self.PolO_masked_alt_data, c=self.PolO_masked_color_data, marker='D', s=self.marker_size, cmap=self.cmap)
        
        self.PolE_AltVsEw_paths.remove()
        self.PolO_AltVsEw_paths.remove()
        self.PolE_AltVsEw_paths = self.AltVsEw_axes.scatter(x=self.PolE_masked_X_data, y=self.PolE_masked_alt_data, c=self.PolE_masked_color_data, marker='s', s=self.marker_size, cmap=self.cmap)
        self.PolO_AltVsEw_paths = self.AltVsEw_axes.scatter(x=self.PolO_masked_X_data, y=self.PolO_masked_alt_data, c=self.PolO_masked_color_data, marker='D', s=self.marker_size, cmap=self.cmap)
        
        self.PolE_NsVsEw_paths.remove()
        self.PolO_NsVsEw_paths.remove()
        self.PolE_NsVsEw_paths = self.NsVsEw_axes.scatter(x=self.PolE_masked_X_data, y=self.PolE_masked_Y_data, c=self.PolE_masked_color_data, marker='s', s=self.marker_size, cmap=self.cmap)
        self.PolO_NsVsEw_paths = self.NsVsEw_axes.scatter(x=self.PolO_masked_X_data, y=self.PolO_masked_Y_data, c=self.PolO_masked_color_data, marker='D', s=self.marker_size, cmap=self.cmap)
        
        self.PolE_NsVsAlt_paths.remove()
        self.PolO_NsVsAlt_paths.remove()
        self.PolE_NsVsAlt_paths = self.NsVsAlt_axes.scatter(x=self.PolE_masked_alt_data, y=self.PolE_masked_Y_data, c=self.PolE_masked_color_data, marker='s', s=self.marker_size, cmap=self.cmap)
        self.PolO_NsVsAlt_paths = self.NsVsAlt_axes.scatter(x=self.PolO_masked_alt_data, y=self.PolO_masked_Y_data, c=self.PolO_masked_color_data, marker='D', s=self.marker_size, cmap=self.cmap)
            
        
    #### calbacks for various events
    def T_selector(self, minT, maxT):
        if minT == maxT: return
        
        ##convert back to seconds
        minT += self.T0
        maxT += self.T0
        
        minT/=1000.0
        maxT/=1000.0
        
        self.previous_view_states.append( self.previous_view_state(2, self.T_limits[:]) )
        
        if self.z_button_pressed: ## then we zoom out,
            minT = 2.0*self.T_limits[0] - minT
            maxT = 2.0*self.T_limits[1] - maxT
            
        self.set_T_lims(minT, maxT)
        self.reset_total_mask()
        self.replot_data()
        
    def alt_selector(self, minA, maxA):
        if minA==maxA: return
        self.previous_view_states.append( self.previous_view_state(1, self.alt_limits[:]) )
            
        if self.z_button_pressed:
            minA = 2.0*self.alt_limits[0] - minA
            maxA = 2.0*self.alt_limits[1] - maxA
        
        self.set_alt_lims(minA, maxA)
        self.reset_total_mask()
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
        self.reset_total_mask()
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
                
            self.reset_total_mask()
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
                deltaX /= 1000.0
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
                
            self.reset_total_mask()
            self.replot_data()
            self.draw()
            
    #### external usefull functions
    def set_T_fraction(self, t_fraction):
        """plot points with all time less that the t_fraction. Where 0 is beginning of t-lim and 1 is end of t_lim"""
        self.T_fraction = t_fraction
        self.PolE_mask_on_T = np.logical_and( self.PolE_T_data>self.T_limits[0], self.PolE_T_data<(self.T_limits[0]+(self.T_limits[1]-self.T_limits[0])*self.T_fraction) )
        self.PolO_mask_on_T = np.logical_and( self.PolO_T_data>self.T_limits[0], self.PolO_T_data<(self.T_limits[0]+(self.T_limits[1]-self.T_limits[0])*self.T_fraction) )
            
#    def initialize_animation(self, animation_time):
#        
#        mask_invert = np.logical_not(self.total_mask)
#        
#        source_viewing_indeces = mask_invert.nonzero()[0]
#        source_viewing_times = self.T_data[ mask_invert ]
#        sorting_indeces = np.argsort( source_viewing_times )
#        
#        source_viewing_indeces = source_viewing_indeces[sorting_indeces]
#        source_viewing_times = source_viewing_times[sorting_indeces]
#        time_delays = source_viewing_times - self.T_limits[0]
#        time_delays[1:] = time_delays[1:] - time_delays[:-1]
#        time_delays *= animation_time / (self.T_limits[1] - self.T_limits[0])
#        
#        
#        self.total_mask.fill(True)
#        self.replot_data()
#        self.fig.canvas.draw()
#        self.draw()
#        
#        self.animate_indeces = source_viewing_indeces
#        self.animate_delays = time_delays
#        
#        self.animation_index = 0 
#        
#        if len(self.animate_delays) <= self.animation_index:
#            return None
#        else:
#            return self.animate_delays[self.animation_index]
#        
#    def continue_animation(self):
#        
#        self.total_mask[ self.animate_indeces[ self.animation_index ] ] = False
#        self.replot_data()
#        self.fig.canvas.draw()
#        self.draw()
#        
#        self.animation_index += 1
#        
#        if len(self.animate_delays) <= self.animation_index:
#            return None
#        else:
#            return self.animate_delays[self.animation_index]
        
        
#        for index, waittime in izip(source_viewing_indeces, time_delays):
#            time.sleep(waittime)
#            
#            
#            self.replot_data()
#            self.fig.canvas.draw()
#            self.draw()
#            
            
                
        
        
            
        

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
        
        ### set the fields
        self.getButtonPressed()
        
        
        
    def add_analysis(self, analysis_name, function):
        functor = lambda : self.run_analysis(function)
        self.analysis_menu.addAction('&'+analysis_name, functor)
        
    def run_analysis(self, function):
        PSE = self.figure_space.get_viewed_PSE()
        function(PSE)
        
        
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
        
    def add_PSE(self, PSE_list):
        self.figure_space.add_PSE(PSE_list)
        
    
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
        self.figure_space.reset_total_mask()
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
        self.figure_space.reset_total_mask()
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
        self.figure_space.reset_total_mask()
        self.figure_space.replot_data()
        self.figure_space.fig.canvas.draw()
        self.figure_space.draw()
        
#        self.figure_space.fig.set_size_inches(7.8, 10.6)
#        self.figure_space.fig.savefig("./animation_output/"+str(animation_index).zfill(4), dpi=100)
        
        timer = QtCore.QTimer(self)
        timer.singleShot(time_per_step*1000, lambda: self.animateUpdate(animation_index, Tfractions, time_per_step) )

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
    
    plotter.add_analysis( "Print Details", print_details_analysis )
    plotter.add_analysis( "Plot Trace Data", plot_data_analysis( ant_locs ) )
    plotter.add_analysis( "Print Ave. Ant. Delays", print_ave_ant_delays(ant_locs) )
    
    plotter.show()
    plotter.add_PSE( PSE_list )
    
    
    sys.exit(qApp.exec_())
    
    