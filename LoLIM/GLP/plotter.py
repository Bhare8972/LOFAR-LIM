#!/usr/bin/env python3

## on APP machine

### IDEA! running coordinate transformations all the time is expensive, and they rarely change.
### may be good idea to track previous coordinate system, and only update if it changes?

from __future__ import unicode_literals
import sys
import os

from time import sleep
from pickle import dumps, loads
import datetime

import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
#matplotlib.use('svg')
from PyQt5 import QtCore, QtWidgets
from PyQt5.Qt import QApplication, QClipboard

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, RectangleSelector
import matplotlib.colors as colors

## mine

def gen_cmap(cmap_name, minval,maxval, num=100):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, num)))
    return new_cmap

def gen_olaf_cmap(num=256):
    def RGB(V):
        cs = 0.9 + V*0.9
        R = 0.5 + 0.5*np.cos( 2*3.14*cs )
        G = 0.5 + 0.5*np.cos( 2*3.14*(cs+0.25)  )
        B = 0.5 + 0.5*np.cos( 2*3.14*(cs+0.5) )
        return R,G,B
    
    RGBlist = [RGB(x) for x in np.linspace(0.05,0.95,num)]
    return colors.LinearSegmentedColormap.from_list( 'OlafOfManyColors', RGBlist, N=num)

        

### TODO: make rebalancing x-y as part of coordinate system!
### make buttons to 'show all' for individual coordinates and settings
## toggle showing when a dataset is showing or not

### note that there are THREE coordinate systems. plot cordinates are the ones shown on the plotter.
### global coordinates are X, Y, Z, and T in m and s. 
### The coordinate system defines how to convert from global to plot coordinates
### local coordinats are those inside the dataset. The dataset must convert local to global.

## plot coordinates are described as: plotX, plotY, plotT, plotZ and plotZt (plotZ is for side-views and plotZt is for Z vs T. They may not be the same.)

class coordinate_transform:
    
    def __init__(self):
        self.x_label = ''
        self.y_label = ''
        self.z_label = ''
        self.zt_label = ''
        self.t_label = ''
        
        self.x_display_label = 'X'
        self.y_display_label = 'Y'
        self.z_display_label = 'Z'
        self.t_display_label = 'T'
        
        self.name = 'none'
        
    def make_workingMemory(self, N):
        return None
    
    def set_workingMemory(self, mem):
        pass
    
    def set_plotX(self, lower, upper):
        pass
    
    def set_plotY(self, lower, upper):
        pass
    
    def set_plotT(self, lower, upper):
        pass
    
    def set_plotZ(self, lower, upper):
        pass
    
    def set_plotZt(self, lower, upper):
        pass
    
    def get_plotX(self):
        return [0,0]
    
    def get_plotY(self):
        return [0,0]
    
    def get_plotT(self):
        return [0,0]
    
    def get_plotZ(self):
        return [0,0]
    
    def get_plotZt(self):
        return [0,0]
    
    def rebalance_X(self, window_width, window_height):
        pass
    
    def rebalance_Y(self, window_width, window_height):
        pass
    
    def rebalance_XY(self, window_width, window_height):
        pass
    
    def get_limits_plotCoords(self):
        ## return X, Y, Z, Zt, and T limits (in that order!)
        return [[0,0], [0,0], [0,0], [0,0], [0,0]]
    
    def get_displayLimits(self):
        return [[0,0], [0,0], [0,0], [0,0]]
    
    def set_displayLimits(self, Xminmax=None, Yminmax=None, Zminmax=None, Tminmax=None):
        pass
        
    def transform(self, Xglobal, Yglobal, Zglobal, Tglobal, make_copy=True):
        """transform global X, Y, Z, and T to plot coordinates. returns plotX, plotY, plotZ, plotZt, and plotT"""
        return [[0], [0], [0], [0], [0]]
    
    def transform_and_filter(self, Xglobal, Yglobal, Zglobal, Tglobal, 
            make_copy=True, ignore_T=False, bool_workspace=None):
        """like transform, but filters based on bounds"""
        return [[0], [0], [0], [0], [0]]
    
    def filter(self, Xplot, Yplot, Zplot, Ztplot, Tplot, 
                make_copy=True, ignore_T=False, bool_workspace=None):  
        """ only filters data in plot coordinates.. hopefully"""
        print("filter function in transform not implemented")
        quit()
    
class typical_transform( coordinate_transform ):
    """display in km and ms, with an arbitrary space zero and arbitrary time zero."""
    
    def __init__(self, space_center, time_center, t_unit='milli'):
        """center should be in LOFAR (m,s) coordinates"""
        
        self.space_center = space_center
        self.time_center = time_center
        
        # self.x_label = "distance east-west [km]"
        # self.y_label = "distance north-south [km]"
        # self.z_label = "altitude [km]"
        # self.zt_label = "altitude [km]"
        # self.t_label = "time [ms]"
        
        self.x_label = "Easting [km]"
        self.y_label = "Northing [km]"
        self.z_label = "Altitude [km]"
        self.zt_label = "Altitude [km]"

        if t_unit == 'milli':
            self.t_factor = 1000.0
            self.t_label = "Time [ms]"
        elif t_unit == 'micro':
            self.t_factor = 1.0e6
            self.t_label = "Time [$\mu$s]"
        
        ## these are in km and ms
        self.X_bounds = [0,1]
        self.Y_bounds = [0,1]
        self.Z_bounds = [0,1]
        self.T_bounds = [0,1]
        
#        self.rebalance_XY = True ## keep X and Y have same ratio
        
        self.x_display_label = 'X:'
        self.y_display_label = 'Y:'
        self.z_display_label = 'Z:'
        self.t_display_label = 'T:'
        
        self.name = 'XYZT'
        
    
    def set_plotX(self, lower, upper):
        self.X_bounds = [lower, upper]
        
    def get_plotX(self):
        return [self.X_bounds[0], self.X_bounds[1]]
    
    def set_plotY(self, lower, upper):
        self.Y_bounds = [lower, upper]
        
    def get_plotY(self):
        return [self.Y_bounds[0], self.Y_bounds[1]]
    
    def set_plotT(self, lower, upper):
        self.T_bounds = [lower, upper]
        
    def get_plotT(self):
        return [self.T_bounds[0], self.T_bounds[1]]
    
    def set_plotZ(self, lower, upper):
        self.Z_bounds = [lower, upper]
        
    def get_plotZ(self):
        return [self.Z_bounds[0], self.Z_bounds[1]]
    
    def set_plotZt(self, lower, upper):
        self.set_plotZ( lower, upper )
        
    def get_plotZt(self):
        return [self.Z_bounds[0], self.Z_bounds[1]]
    
    def rebalance_X(self, window_width, window_height):
        data_height = self.Y_bounds[1] - self.Y_bounds[0]
        
        new_width = window_width*data_height/window_height
        middle_X = (self.X_bounds[0] + self.X_bounds[1])/2.0
        self.set_plotX( middle_X-new_width/2.0, middle_X+new_width/2.0 )
    
    def rebalance_Y(self, window_width, window_height):
        data_width = self.X_bounds[1] - self.X_bounds[0]
        
        new_height = window_height*data_width/window_width
        middle_y = (self.Y_bounds[0] + self.Y_bounds[1])/2.0
        self.set_plotY( middle_y-new_height/2.0, middle_y+new_height/2.0 )
        
    def rebalance_XY(self, window_width, window_height):
        data_width = self.X_bounds[1] - self.X_bounds[0]
        data_height = self.Y_bounds[1] - self.Y_bounds[0]
        
        width_ratio = data_width/window_width
        height_ratio = data_height/window_height
        
        if width_ratio > height_ratio:
            new_height = window_height*data_width/window_width
            middle_y = (self.Y_bounds[0] + self.Y_bounds[1])/2.0
            self.set_plotY( middle_y-new_height/2.0, middle_y+new_height/2.0 )
            
        else:
            new_width = window_width*data_height/window_height
            middle_X = (self.X_bounds[0] + self.X_bounds[1])/2.0
            self.set_plotX( middle_X-new_width/2.0, middle_X+new_width/2.0 )
    
    def get_limits_plotCoords(self):
        """ return X, Y, Z, Zt, and T limits (in that order!) """
        return self.X_bounds, self.Y_bounds, self.Z_bounds, self.Z_bounds, self.T_bounds
    
    def get_displayLimits(self):
        #print('displayed coordinates are in m and s')
        Xout = [self.X_bounds[0]*1000.0+self.space_center[0], 
                self.X_bounds[1]*1000.0+self.space_center[0]]
        Yout = [self.Y_bounds[0]*1000.0+self.space_center[1], 
                self.Y_bounds[1]*1000.0+self.space_center[1]]
        Zout = [self.Z_bounds[0]*1000.0+self.space_center[2], 
                self.Z_bounds[1]*1000.0+self.space_center[2]]
        Tout = [self.T_bounds[0]/self.t_factor+self.time_center,
                self.T_bounds[1]/self.t_factor+self.time_center]
        return Xout, Yout, Zout, Tout
    
    def set_displayLimits(self, Xminmax=None, Yminmax=None, Zminmax=None, Tminmax=None):
        self.X_bounds = [(Xminmax[0]-self.space_center[0])/1000, 
                         (Xminmax[1]-self.space_center[0])/1000]
        self.Y_bounds = [(Yminmax[0]-self.space_center[1])/1000, 
                         (Yminmax[1]-self.space_center[1])/1000]
        self.Z_bounds = [(Zminmax[0]-self.space_center[2])/1000, 
                         (Zminmax[1]-self.space_center[2])/1000]
        self.T_bounds = [(Tminmax[0]-self.time_center)*self.t_factor,
                         (Tminmax[1]-self.time_center)*self.t_factor]
        
    def transform(self, Xglobal, Yglobal, Zglobal, Tglobal, make_copy=True):
        
        if make_copy:
            outX = np.array( Xglobal )
            outY = np.array( Yglobal )
            outZ = np.array( Zglobal )
            outT = np.array( Tglobal )
        else:
            outX = np.asanyarray( Xglobal )
            outY = np.asanyarray( Yglobal )
            outZ = np.asanyarray( Zglobal )
            outT = np.asanyarray( Tglobal )
            
        outX -= self.space_center[0]
        outX /= 1000.0
            
        outY -= self.space_center[1]
        outY /= 1000.0
            
        outZ -= self.space_center[2]
        outZ /= 1000.0
            
        outT -= self.time_center
        outT *= self.t_factor
        
        return outX, outY, outZ, outZ, outT
    
    def transform_and_filter(self, Xglobal, Yglobal, Zglobal, Tglobal, 
                make_copy=True, ignore_T=False, bool_workspace=None):
        """like transform, but filters based on bounds"""
        outX, outY, outZ, outZt, outT =  self.transform(Xglobal, Yglobal, Zglobal, Tglobal, make_copy)
        return self.filter(outX, outY, outZ, outZt, outT, make_copy=False, ignore_T=ignore_T, bool_workspace=bool_workspace)
        
#        if bool_workspace is None:
#            bool_workspace = np.ones( len(outX), dtype=bool )
#        else:
#            bool_workspace = bool_workspace[:len(outX)]
#            bool_workspace[:] = True
#            
#        np.greater_equal( outX, self.X_bounds[0], out=bool_workspace, where=bool_workspace )
#        np.greater_equal( self.X_bounds[1], outX, out=bool_workspace, where=bool_workspace )
#            
#        np.greater_equal( outY, self.Y_bounds[0], out=bool_workspace, where=bool_workspace )
#        np.greater_equal( self.Y_bounds[1], outY, out=bool_workspace, where=bool_workspace )
#            
#        np.greater_equal( outZ, self.Z_bounds[0], out=bool_workspace, where=bool_workspace )
#        np.greater_equal( self.Z_bounds[1], outZ, out=bool_workspace, where=bool_workspace )
#            
#        if not ignore_T:
#            np.greater_equal( outT, self.T_bounds[0], out=bool_workspace, where=bool_workspace )
#            np.greater_equal( self.T_bounds[1], outT, out=bool_workspace, where=bool_workspace )
#        
#        N = np.sum( bool_workspace )
#        
#        np.compress( bool_workspace, outX, out=outX[:N] )
#        np.compress( bool_workspace, outY, out=outY[:N] )
#        np.compress( bool_workspace, outZ, out=outZ[:N] )
#        np.compress( bool_workspace, outT, out=outT[:N] )
#        
#        return outX[:N], outY[:N], outZ[:N], outZ[:N], outT[:N]

    def get_filter(self, Xplot, Yplot, Zplot, Ztplot, Tplot, ignore_T=False, bool_workspace=None):

        if bool_workspace is None:
            bool_workspace = np.ones( len(Xplot), dtype=bool )
        else:
            bool_workspace = bool_workspace[:len(Xplot)]
            bool_workspace[:] = True
            
        np.greater_equal( Xplot, self.X_bounds[0], out=bool_workspace, where=bool_workspace )
        np.greater_equal( self.X_bounds[1], Xplot, out=bool_workspace, where=bool_workspace )
            
        np.greater_equal( Yplot, self.Y_bounds[0], out=bool_workspace, where=bool_workspace )
        np.greater_equal( self.Y_bounds[1], Yplot, out=bool_workspace, where=bool_workspace )
            
        np.greater_equal( Zplot, self.Z_bounds[0], out=bool_workspace, where=bool_workspace )
        np.greater_equal( self.Z_bounds[1], Zplot, out=bool_workspace, where=bool_workspace )
            
        if not ignore_T:
            np.greater_equal( Tplot, self.T_bounds[0], out=bool_workspace, where=bool_workspace )
            np.greater_equal( self.T_bounds[1], Tplot, out=bool_workspace, where=bool_workspace )

        return bool_workspace


    
    def filter(self, Xplot, Yplot, Zplot, Ztplot, Tplot, 
                make_copy=True, ignore_T=False, bool_workspace=None):  
        
        if make_copy:
            outX = np.array( Xplot )
            outY = np.array( Yplot )
            outZ = np.array( Zplot )
            outT = np.array( Tplot )
        else:
            outX = np.asanyarray( Xplot )
            outY = np.asanyarray( Yplot )
            outZ = np.asanyarray( Zplot )
            outT = np.asanyarray( Tplot )
        

        bool_workspace = self.get_filter(Xplot, Yplot, Zplot, Ztplot, Tplot, ignore_T, bool_workspace)
        
        N = np.sum( bool_workspace )
        
        np.compress( bool_workspace, outX, out=outX[:N] )
        np.compress( bool_workspace, outY, out=outY[:N] )
        np.compress( bool_workspace, outZ, out=outZ[:N] )
        np.compress( bool_workspace, outT, out=outT[:N] )
        
        return outX[:N], outY[:N], outZ[:N], outZ[:N], outT[:N]


class AzEl_transform( coordinate_transform ):
    
    def __init__(self, space_center, time_center):
        """center should be in LOFAR (m,s) coordinates"""
        
        self.space_center = space_center
        self.time_center = time_center
        
        self.x_label = "azimuth"
        self.y_label = "elivation"
        self.z_label = "elivation"
        self.zt_label = "elivation"
        self.t_label = "time [ms]"
        
        ## these are in km and ms
        self.az_bounds = [0,1]
        self.elivation_bounds = [0,1]
        self.T_bounds = [0,1]
        
#        self.rebalance_XY = True ## keep X and Y have same ratio
        
        self.x_display_label = 'Az:'
        self.y_display_label = '?:'
        self.z_display_label = 'El:'
        self.t_display_label = 'T:'
        
        self.working_memory = None
        
        
        self.name = 'AzElT'
    
    def set_plotX(self, lower, upper):
        self.az_bounds = [lower, upper]
        
    def get_plotX(self):
        return [self.az_bounds[0], self.az_bounds[1]]
    
    def set_plotY(self, lower, upper):
        self.elivation_bounds = [lower, upper]
        
    def get_plotY(self):
        return [self.elivation_bounds[0], self.elivation_bounds[1]]
    
    def set_plotT(self, lower, upper):
        self.T_bounds = [lower, upper]
        
    def get_plotT(self):
        return [self.T_bounds[0], self.T_bounds[1]]
    
    def set_plotZ(self, lower, upper):
        self.elivation_bounds = [lower, upper]
        
    def get_plotZ(self):
        return [self.elivation_bounds[0], self.elivation_bounds[1]]
    
    def set_plotZt(self, lower, upper):
        self.set_plotZ( lower, upper )
        
    def get_plotZt(self):
        return [self.elivation_bounds[0], self.elivation_bounds[1]]
    
    def rebalance_X(self, window_width, window_height):
        data_height = self.elivation_bounds[1] - self.elivation_bounds[0]
        
        new_width = window_width*data_height/window_height
        middle_X = (self.az_bounds[0] + self.az_bounds[1])/2.0
        self.set_plotX( middle_X-new_width/2.0, middle_X+new_width/2.0 )
    
    def rebalance_Y(self, window_width, window_height):
        data_width = self.az_bounds[1] - self.az_bounds[0]
        
        new_height = window_height*data_width/window_width
        middle_y = (self.elivation_bounds[0] + self.elivation_bounds[1])/2.0
        self.set_plotY( middle_y-new_height/2.0, middle_y+new_height/2.0 )
    
    def rebalance_XY(self, window_width, window_height):
        data_width = self.az_bounds[1] - self.az_bounds[0]
        data_height = self.elivation_bounds[1] - self.elivation_bounds[0]
        
        width_ratio = data_width/window_width
        height_ratio = data_height/window_height
        
        if width_ratio > height_ratio:
            new_height = window_height*data_width/window_width
            middle_y = (self.elivation_bounds[0] + self.elivation_bounds[1])/2.0
            self.set_plotY( middle_y-new_height/2.0, middle_y+new_height/2.0 )
            
        else:
            new_width = window_width*data_height/window_height
            middle_X = (self.az_bounds[0] + self.az_bounds[1])/2.0
            self.set_plotX( middle_X-new_width/2.0, middle_X+new_width/2.0 )
    
    def get_limits_plotCoords(self):
        """ return X, Y, Z, Zt, and T limits (in that order!) """
        return self.az_bounds, self.elivation_bounds, self.elivation_bounds, self.elivation_bounds, self.T_bounds
    
    def get_displayLimits(self):
        Tout = [self.T_bounds[0]/1000.0+self.time_center, 
            self.T_bounds[1]/1000.0+self.time_center]
        return self.az_bounds, self.elivation_bounds, self.elivation_bounds, Tout
    
    
    def set_displayLimits(self, Xminmax=None, Yminmax=None, Zminmax=None, Tminmax=None):
        self.set_plotX(Xminmax[0], Xminmax[1])
        self.set_plotY(Yminmax[0], Yminmax[1])
#        self.set_plotZ(Zminmax[0], Zminmax[1])
        tmin = (Tminmax[0]-self.time_center)*1000.0
        tmax = (Tminmax[1]-self.time_center)*1000.0
        self.set_plotT(tmin, tmax)
        
    def make_workingMemory(self, N):
        return np.empty( (4,N), dtype=np.double )
    
    def set_workingMemory(self, mem):
        self.working_memory = mem
        
    def transform(self, Xglobal, Yglobal, Zglobal, Tglobal, make_copy=True):
        
#        if make_copy:
#            outX = np.array( Xglobal )
#            outY = np.array( Yglobal )
#            outZ = np.array( Zglobal )
#            outT = np.array( Tglobal )
#        else:
#            outX = np.asanyarray( Xglobal )
#            outY = np.asanyarray( Yglobal )
#            outZ = np.asanyarray( Zglobal )
#            outT = np.asanyarray( Tglobal )
        
        tmp_memory = self.working_memory
        if tmp_memory is None or tmp_memory.shape[1] < len(Xglobal):
            tmp_memory = np.empty( (4,len(Xglobal)), dtype=np.double )
            
        if tmp_memory.shape[1] > len(Xglobal):
            tmp_memory = tmp_memory[:,:len(Xglobal)]
        
        outAlpha, outBeta, outEl, outT = tmp_memory
            
            ## first calculate rho
        outT[:] = Xglobal
        outT -= self.space_center[0]
        outBeta[:] = outT
        outT *= outT
        
        outEl[:] = Yglobal
        outEl -= self.space_center[1]
        outEl *= outEl
        
        outT += outEl # now rho squared
        
        outAlpha[:] = Zglobal
        outAlpha -= self.space_center[2]
        outAlpha *= outAlpha
        outAlpha += outT ## now R2
        
        np.sqrt( outT, out=outT ) # is now rho
        np.sqrt( outAlpha, out=outAlpha ) # is now R
        
        outEl[:] = outT
        outEl /= outAlpha ## is now cos El (I think)
        
        outBeta /= outT # is now cos Az
        np.arccos( outBeta, out=outAlpha )
#        np.sin(outAlpha, out=outAlpha) ## outAlpha is now sin Az
        
#        outBeta *= outEl
#        outAlpha *= outEl
        
        np.arccos(outEl, out=outEl)
        
        outT[:] = Tglobal
        outT -= self.time_center
        outT *= 1000.0
        
        return outAlpha, outEl, outEl, outEl, outT

    
    def transform_and_filter(self, Xglobal, Yglobal, Zglobal, Tglobal, 
                make_copy=True, ignore_T=False, bool_workspace=None):
        """like transform, but filters based on bounds"""
        outAz, outEl, throw, throw, outT =  self.transform(Xglobal, Yglobal, Zglobal, Tglobal, make_copy)
        
        if bool_workspace is None:
            bool_workspace = np.ones( len(outAz), dtype=bool )
        else:
            bool_workspace = bool_workspace[:len(outAz)]
            bool_workspace[:] = True

        
            
        np.greater_equal( outAz, self.az_bounds[0], out=bool_workspace, where=bool_workspace )
        np.greater_equal( self.az_bounds[1], outAz, out=bool_workspace, where=bool_workspace )
            
        np.greater_equal( outEl, self.elivation_bounds[0], out=bool_workspace, where=bool_workspace )
        np.greater_equal( self.elivation_bounds[1], outEl, out=bool_workspace, where=bool_workspace )
            
        if not ignore_T:
            np.greater_equal( outT, self.T_bounds[0], out=bool_workspace, where=bool_workspace )
            np.greater_equal( self.T_bounds[1], outT, out=bool_workspace, where=bool_workspace )
        
        N = np.sum( bool_workspace )
        
        np.compress( bool_workspace, outAz, out=outAz[:N] )
        np.compress( bool_workspace, outEl, out=outEl[:N] )
        np.compress( bool_workspace, outT, out=outT[:N] )
        
        return outAz[:N], outEl[:N], outEl[:N], outEl[:N], outT[:N]
    
## data sets work in lofar-centered mks. Except for plotting, where the coordinate system will be provided
class DataSet_Type:
    """All data plotted will be part of a data set. There can be different types of data sets, this class is a wrapper over all data sets"""
    
    def __init__(self, name):
        self.name = name
        self.display = True
        
    def set_show_all(self, coordinate_system, do_set_limits=True):
        """set bounds needed to show all data.
        Set all properties so that all points are shown"""
        pass
    
    def bounding_box(self, coordinate_system):
        """return bounding box in plot-coordinates"""
        return [[np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan], [np.nan,np.nan]]

    def T_bounds(self, coordinate_system):
        """return T-bounds given all other bounds, in plot-coordinats"""
        return [0.0, 0.0]
    
    def get_all_properties(self):
        return {}
    
    def set_property(self, name, str_value):
        pass
    
    def plot(self, AltvsT_axes, AltvsEW_axes, NSvsEW_axes, NsvsAlt_axes, ancillary_axes, coordinate_system, zorder):
        pass
    
    
#    def get_viewed_events(self):
#        return []
    
    def clear(self):
        pass
    
    def use_ancillary_axes(self):
        return False

    def print_info(self, coordinate_system):
        print( "not implemented" )
        
    def search(self, ID_list):
        print("not implemented")
        return None   
    
    def toggle_on(self):
        self.display = True
    
    def toggle_off(self):
        self.display = False
    
    def copy_view(self, coordinate_system):
        print("not implemented")
        return None
    
    def get_view_ID_list(self, coordinate_system):
        print("not implemented")
        return None
    
    def text_output(self):
        print("not implemented")
        
    def ignore_time(self, ignore=None):
        if ignore is not None:
            print("not implemented")
        return False
    
    def key_press_event(self, event):
        pass
    

def IPSE_to_DataSet(IPSE, name, cmap, marker='s', marker_size=5, color_mode='time'):
    X = np.empty(len(IPSE), dtype=np.double)
    Y = np.empty(len(IPSE), dtype=np.double)
    Z = np.empty(len(IPSE), dtype=np.double)
    T = np.empty(len(IPSE), dtype=np.double)
    intensity = np.empty(len(IPSE), dtype=np.double)
    S1S2_distance = np.empty(len(IPSE), dtype=np.double)
    amplitude = np.ones(len(IPSE), dtype=np.double)
    block = np.empty(len(IPSE), dtype=int)
    uniqueID = np.empty(len(IPSE), dtype=int)
    ID = np.empty(len(IPSE), dtype=int)
    
    for i,itPSE in enumerate(IPSE):
        X[i] = itPSE.XYZT[0]
        Y[i] = itPSE.XYZT[1]
        Z[i] = itPSE.XYZT[2]
        T[i] = itPSE.XYZT[3]
        
        intensity[i] = itPSE.intensity
        S1S2_distance[i] = itPSE.S1_S2_distance
        amplitude[i] = itPSE.amplitude
        block[i] = itPSE.block_index
        uniqueID[i] = itPSE.unique_index
        ID[i] = itPSE.IPSE_index
    
    new_dataset = DataSet_generic_PSE( X, Y, Z, T,
                                       marker=marker, marker_size=marker_size, color_mode=color_mode, 
                                       name=name, cmap=cmap,
                                       min_filters = {'amplitude':amplitude, 'intensity':intensity},
                                       max_filters = {'S1S3 distance':S1S2_distance},
                                       print_info = {'block':block, 'ID':ID},
                                       source_IDs = uniqueID
                                       )
    return new_dataset

def LMA_to_DataSet(LMA_data, name, cmap, zero_datetime=None, marker='s', marker_size=5, color_mode='time', center='LOFAR', header=None):
    """ note that zero_datetime should be a python datetime.datetime object, if used. center can be 'LOFAR' or 'LMA'"""
    X = np.empty(len(LMA_data), dtype=np.double)
    Y = np.empty(len(LMA_data), dtype=np.double)
    Z = np.empty(len(LMA_data), dtype=np.double)
    T = np.empty(len(LMA_data), dtype=np.double)
    power = np.empty(len(LMA_data), dtype=np.double)
    red_chi_squared = np.empty(len(LMA_data), dtype=np.double)
    RMS = np.empty(len(LMA_data), dtype=np.double)
    stations = np.empty(len(LMA_data), dtype=int)
    ID = np.empty(len(LMA_data), dtype=int)
    
    if header is not None:
        antenna_RMS = np.average( [ant.RMS_error for ant in header.antenna_info_list] )
        
    else:
        antenna_RMS = 70
    
    if zero_datetime is not None:
        midnight_datetime = datetime.datetime.combine(zero_datetime.date(),  datetime.time(0), tzinfo= datetime.timezone.utc )
    
    for i,source in enumerate(LMA_data):
        XYZ = source.get_XYZ(center=center)
        X[i] = XYZ[0]
        Y[i] = XYZ[1]
        Z[i] = XYZ[2]
        
        if zero_datetime is not None:
            TD = datetime.timedelta(seconds = source.time_of_day)
            excess = source.time_of_day - TD.total_seconds() 
            
            source_datetime = midnight_datetime + TD
            source_time = (source_datetime - zero_datetime).total_seconds() + excess
            T[i] = source_time
            
        else:
            T[i] = source.time_of_day
        
        power[i] = source.power
        red_chi_squared[i] = source.red_chi_squared
        ID[i] = i
        stations[i] = source.get_number_stations()
        RMS[i] = antenna_RMS*np.sqrt( source.red_chi_squared )
    
    new_dataset = DataSet_generic_PSE( X, Y, Z, T,
                                       marker=marker, marker_size=marker_size, color_mode=color_mode, 
                                       name=name, cmap=cmap,
                                       min_filters = {'power':power, 'stations':stations},
                                       max_filters = {'red. chi sq.':red_chi_squared, 'RMS':RMS},
                                       source_IDs = ID
                                       )
    return new_dataset
 
def iterPSE_to_DataSet(iterPSE, name, cmap, marker='s', marker_size=5, color_mode='time', eigMode='normal'):
    X = np.empty(len(iterPSE), dtype=np.double)
    Y = np.empty(len(iterPSE), dtype=np.double)
    Z = np.empty(len(iterPSE), dtype=np.double)
    T = np.empty(len(iterPSE), dtype=np.double)
    RMS = np.empty(len(iterPSE), dtype=np.double)
    RefAmp = np.empty(len(iterPSE), dtype=np.double)
    maxSqrtEig = np.ones(len(iterPSE), dtype=np.double)
    # FractionRadialError = np.zeros(len(iterPSE), dtype=np.double)
    # FractionZError = np.zeros(len(iterPSE), dtype=np.double)
    numRS = np.empty(len(iterPSE), dtype=int)
    block = np.empty(len(iterPSE), dtype=int)
    ID = np.empty(len(iterPSE), dtype=int)
    blockID = np.empty(len(iterPSE), dtype=int)
    numThrows = np.empty(len(iterPSE), dtype=int)
    
    eigMode = {'normal':1}[eigMode]
    for i,itPSE in enumerate(iterPSE):
        X[i] = itPSE.XYZT[0]
        Y[i] = itPSE.XYZT[1]
        Z[i] = itPSE.XYZT[2]
        T[i] = itPSE.XYZT[3]
        
        RMS[i] = itPSE.RMS
        RefAmp[i] = itPSE.refAmp
        numRS[i] = itPSE.numRS
        block[i] = itPSE.block
        ID[i] = itPSE.uniqueID
        blockID[i] = itPSE.ID
        numThrows[i] = itPSE.numThrows
        
        if eigMode == 1:
            A = itPSE.cov_eig()
            if A is None:
                maxSqrtEig[i] = np.inf
            else:
                eigVals, eigVecs = A
                
                Ai = np.argmax(eigVals)
                A = eigVals[Ai]
                # e = eigVecs[:,Ai]
                
                if A < 0 or np.iscomplex(A):
                    maxSqrtEig[i] = np.inf
                else:
                    maxSqrtEig[i] = np.sqrt( A )
                    # R_hat = itPSE.XYZT[:3]/np.linalg.norm( itPSE.XYZT[:3] )
                    # FractionRadialError[i] = np.dot( R_hat, e )
                    # FractionZError[ i ] = 
                    
                    
                if not np.isfinite(maxSqrtEig[i]):
                    maxSqrtEig[i] = np.inf ## make nans into infs
    
    new_dataset = DataSet_generic_PSE( X, Y, Z, T,
                                       marker=marker, marker_size=marker_size, color_mode=color_mode, 
                                       name=name, cmap=cmap,
                                       min_filters = {'amp':RefAmp, 'numRS':numRS},
                                       max_filters = {'RMS':RMS, 'sqrtEig':maxSqrtEig, 'numThrows':numThrows},
                                       print_info = {'block':block, 'blockID':blockID},
                                       source_IDs = ID,
                                        txtOut_info = {'RMS[ns]':RMS/1.0e-9, 'loc_error[m]':maxSqrtEig, 'numstations_excluded':numThrows, 'amplitude':RefAmp}
                                       )
    return new_dataset
    
def SPSF_to_DataSet(SPSF_file, name, cmap, marker='s', marker_size=5, color_mode='time'):

    SPSF_data = SPSF_file.get_data()

    num_points = len(SPSF_data)
    
    ID = np.empty(num_points, dtype=int)
    X = np.empty(num_points, dtype=np.double)
    Y = np.empty(num_points, dtype=np.double)
    Z = np.empty(num_points, dtype=np.double)
    T = np.empty(num_points, dtype=np.double)
    
    other_data = {cn:np.empty(num_points, dtype=np.double) for cn in SPSF_file.collums_headings[5:]}
    

    for i,dat in enumerate(SPSF_data):
        ID[i] = dat[0]
        X[i] = dat[1]
        Y[i] = dat[2]
        Z[i] = dat[3]
        T[i] = dat[4]
        
        for ci,cn in enumerate(SPSF_file.collums_headings[5:]):
            other_data[cn][i] = dat[5+ci]
        
    
    new_dataset = DataSet_generic_PSE( X, Y, Z, T,
                                       marker=marker, marker_size=marker_size, color_mode=color_mode, 
                                       name=name, cmap=cmap,
                                       min_filters = other_data,
                                       max_filters = other_data,
                                       print_info = other_data,
                                       source_IDs = ID,
                                        txtOut_info = other_data
                                       )
    return new_dataset

class DataSet_generic_PSE(DataSet_Type):
    
    def __init__(self, X_array, Y_array, Z_array, T_array, 
                 marker, marker_size, color_mode, name, cmap,
                 min_filters={}, max_filters={}, color_options={}, print_info={}, source_IDs = None, txtOut_info={}):
        
        self.marker = marker
        self.marker_size = marker_size
        self.color_mode = color_mode
        self.cmap = cmap
        self.name = name
        self._ignore_time = False
        self.display = True
        
        self.X_array = X_array
        self.Y_array = Y_array
        self.Z_array = Z_array
        self.T_array = T_array
        self.min_filters = min_filters
        self.max_filters = max_filters
        self.color_options = color_options
        self.print_data = print_info
        self.txtOut_info = txtOut_info
        
        self.max_num_points = 20000
        
        self.source_IDs = source_IDs
        if self.source_IDs is None:
            self.source_IDs = np.arange( len(X_array) )
        
        self.X_offset = 0.0
        self.Y_offset = 0.0
        self.Z_offset = 0.0
        self.T_offset = 0.0
        
        self.min_parameters = { name:np.min(values) for name,values in self.min_filters.items() }
        self.max_parameters = { name:np.max(values) for name,values in self.max_filters.items() }
        
        ## probably should call previous constructor here
        
        self.min_masks = { name:np.ones(len(self.X_array), dtype=bool) for name in self.min_filters.keys()}
        self.max_masks = { name:np.ones(len(self.X_array), dtype=bool) for name in self.max_filters.keys()}
        
        self.total_mask = np.ones(len(self.X_array), dtype=bool)

        self.X_TMP = np.empty(len(self.X_array), dtype=np.double)
        self.Y_TMP = np.empty(len(self.Y_array), dtype=np.double)
        self.Z_TMP = np.empty(len(self.Z_array), dtype=np.double)
        self.T_TMP = np.empty(len(self.T_array), dtype=np.double)
        
        
            
        self.decimation_TMP = np.empty( len(self.X_array), dtype=float )
        self.decimation_mask = np.zeros( len(self.X_array),dtype=bool )
        
#        self.masked_color_info = { name:np.ma.masked_array(data, mask=self.total_mask, copy=False)  for name,data in self.color_options.items() }
#        self.color_time_mask = np.ma.masked_array(self.T_array, mask=self.total_mask, copy=False)
        
#        self.set_show_all()
    
        #### some axis data ###
        self.AltVsT_paths = None
        self.AltVsEw_paths = None
        self.NsVsEw_paths = None
        self.NsVsAlt_paths = None
        
        self.transform_memory = None
        
    def set_show_all(self, coordinate_system, do_set_limits=True):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
        
        for name,data in self.min_filters.items():
            self.set_min_param( name, np.min(data) )
            
        for name,data in self.max_filters.items():
            self.set_max_param( name, np.max(data) )
            
        if do_set_limits:
            self.X_TMP[:] = self.X_array  
            self.Y_TMP[:] = self.Y_array  
            self.Z_TMP[:] = self.Z_array  
            self.T_TMP[:] = self.T_array  
            
            self.X_TMP += self.X_offset
            self.Y_TMP += self.Y_offset
            self.Z_TMP += self.Z_offset
            self.T_TMP += self.T_offset
            
            if self.transform_memory is None:
                self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
            coordinate_system.set_workingMemory( self.transform_memory )
            
            plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform( 
                self.X_TMP, self.Y_TMP, self.Z_TMP, self.T_TMP, 
                make_copy=False)
            
            if len(plotX) > 0:
                coordinate_system.set_plotX( np.min(plotX), np.max(plotX) )
                coordinate_system.set_plotY( np.min(plotY), np.max(plotY) )
                coordinate_system.set_plotZ( np.min(plotZ), np.max(plotZ) )
                coordinate_system.set_plotZt( np.min(plotZt), np.max(plotZt) )
                coordinate_system.set_plotT( np.min(plotT), np.max(plotT) )

    def bounding_box(self, coordinate_system):
        
        #### get cuts ###
        
        self.set_total_mask()
            
        ## filter and shift
        Ntmp = np.sum( self.total_mask )
            
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:Ntmp]
        Ytmp = self.Y_TMP[:Ntmp]
        Ztmp = self.Z_TMP[:Ntmp]
        Ttmp = self.T_TMP[:Ntmp]
        
        np.compress( self.total_mask, self.X_TMP, out=Xtmp )
        np.compress( self.total_mask, self.Y_TMP, out=Ytmp )
        np.compress( self.total_mask, self.Z_TMP, out=Ztmp )
        np.compress( self.total_mask, self.T_TMP, out=Ttmp )
        
        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        
        ## transform
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False)
        
        ## return actual bounds
        if len(plotX) > 0:
            Xbounds = [np.min(plotX), np.max(plotX)]
            Ybounds = [np.min(plotY), np.max(plotY)]
            Zbounds = [np.min(plotZ), np.max(plotZ)]
            Ztbounds = [np.min(plotZt), np.max(plotZt)]
            Tbounds = [np.min(plotT), np.max(plotT)]
        else:
            Xbounds = [0,1]
            Ybounds = [0,1]
            Zbounds = [0,1]
            Ztbounds = [0,1]
            Tbounds = [0,1]
        
        return Xbounds, Ybounds, Zbounds, Ztbounds, Tbounds
    
    def T_bounds(self, coordinate_system):
        #### get cuts ###
        self.set_total_mask()
            
        ## filter and shift
        Ntmp = np.sum( self.total_mask )
            
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:Ntmp]
        Ytmp = self.Y_TMP[:Ntmp]
        Ztmp = self.Z_TMP[:Ntmp]
        Ttmp = self.T_TMP[:Ntmp]
        
        np.compress( self.total_mask, self.X_TMP, out=Xtmp )
        np.compress( self.total_mask, self.Y_TMP, out=Ytmp )
        np.compress( self.total_mask, self.Z_TMP, out=Ztmp )
        np.compress( self.total_mask, self.T_TMP, out=Ttmp )
        
        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        
        ## transform and cut on bounds
        TMP = coordinate_system.get_plotT()
        A = TMP[0] ## need to copy
        B = TMP[1]
        coordinate_system.set_plotT(-np.inf, np.inf)
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False, ignore_T=self._ignore_time, bool_workspace=self.total_mask )
        coordinate_system.set_plotT(A, B)
        
        ## return actual bounds
        if len(plotT) > 0:
            return [np.min(plotT), np.max(plotT)]
        else:
            return [0, 1]
        
        
    def get_all_properties(self):
        ret =  {"marker size":str(self.marker_size),  "color mode":str(self.color_mode), 'name':self.name,
                "X offset":self.X_offset, "Y offset":self.Y_offset,"Z offset":self.Z_offset, 
                "T offset":self.T_offset, 'marker':self.marker, 'max points':self.max_num_points}
        
        for name, value in self.min_parameters.items():
            ret['min ' + name] = value
        for name, value in self.max_parameters.items():
            ret['max ' + name] = value
            
        return ret
        
        ## need: marker type, color map
    
    def set_property(self, name, str_value):
        
        try:
            if name == "marker size":
                self.marker_size = int(str_value)
                
            elif name == "color mode":
                self.color_mode = str_value
                    
            elif name == 'name':
                self.name = str_value
                
            elif name == "X offset":
                self.X_offset = float( str_value )
                
            elif name == "Y offset":
                self.Y_offset = float( str_value )
                
            elif name == "Z offset":
                self.Z_offset = float( str_value )
                
            elif name == "T offset":
                self.T_offset = float( str_value )
                
            elif name == "marker":
                self.marker = str_value
                
            elif name == "max points":
                self.max_num_points = int(str_value)
                
                
            elif name[:3]=='min' and (name[4:] in self.min_parameters):
                self.set_min_param( name[4:], float(str_value) )
                
            elif name[:3]=='max' and (name[4:] in self.max_parameters):
                self.set_max_param( name[4:], float(str_value) )
                
            else:
                print("do not have property:", name)
        except:
            print("error in setting property", name, str_value)
            pass
    
    def set_min_param(self, name, value):
        self.min_parameters[name] = value
#        self.min_masks[name][:] = self.min_filters[name] > value
        np.greater_equal( self.min_filters[name], value, out= self.min_masks[name])
    
    def set_max_param(self, name, value):
        self.max_parameters[name] = value
#        self.max_masks[name][:] = self.max_filters[name] < value 
        np.greater_equal( value, self.max_filters[name], out= self.max_masks[name])
        
    def set_total_mask(self):
        self.total_mask[:] = True
            
        for name,mask in self.min_masks.items():
            np.logical_and(self.total_mask, mask, out=self.total_mask)
            
        for name, mask in self.max_masks.items():
            np.logical_and(self.total_mask, mask, out=self.total_mask)
        
    
    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system, zorder):
        self.set_total_mask()
        N = np.sum( self.total_mask )    
        
        #### random book keeping ####
        self.clear()
        
    
        if self.color_mode in self.color_options:
            color = self.color_options[ self.color_mode ][self.total_mask] ## should fix this to not make memory
            
    
            
        #### set cuts and transforms
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:N]
        Ytmp = self.Y_TMP[:N]
        Ztmp = self.Z_TMP[:N]
        Ttmp = self.T_TMP[:N]
        
        np.compress( self.total_mask, self.X_TMP, out=Xtmp )
        np.compress( self.total_mask, self.Y_TMP, out=Ytmp )
        np.compress( self.total_mask, self.Z_TMP, out=Ztmp )
        np.compress( self.total_mask, self.T_TMP, out=Ttmp )
        
        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False, ignore_T=self._ignore_time, bool_workspace=self.total_mask )


        if (not self.display) or len(plotX)==0:
            
            print(self.name, "not display. have:", len(plotX))
            return
        
        ### get color!
        if self.color_mode == "time":
            color = plotT
            ARG = coordinate_system.get_plotT()
            color_min = ARG[0]
            color_max = ARG[1]
            
        elif self.color_mode[0] == '*':
            color = self.color_mode[1:]
            color_min = None
            color_max = None
            
        elif self.color_mode in self.color_options:
            color = color[self.total_mask[:N]]
            color_min = np.min( color )
            color_max = np.max( color )
            
        else:
            print("bad color mode. This should be interesting!")
            color = self.color_mode
            color_min = None
            color_max = None
            
        N_before = len(plotX)
        if self.max_num_points>0 and self.max_num_points<N_before:
            decimation_factor = self.max_num_points/float(N_before)
            
            data = self.decimation_TMP[:N_before]
            mask = self.decimation_mask[:N_before]
            
            data[:] = decimation_factor
            np.cumsum( data, out=data )
            np.floor(data, out=data)
            np.greater( data[1:] , data[:-1], out = mask[1:] )
            mask[0] = 0
            
            plotX = plotX[mask]
            plotY = plotY[mask]
            plotZ = plotZ[mask]
            plotZt = plotZt[mask]
            plotT = plotT[mask]
            
            
            try:
                B4 = color[mask]
                color = B4 ## hope this works?
            except:
                pass
            
        print(self.name, "plotting", len(plotX), "have:", N_before)
        
        try:
            if not self._ignore_time:
                self.AltVsT_paths = AltVsT_axes.scatter(x=plotT, y=plotZt, c=color, marker=self.marker, s=self.marker_size, 
                                                        cmap=self.cmap, vmin=color_min, vmax=color_max, zorder=zorder)
                
            self.AltVsEw_paths = AltVsEw_axes.scatter(x=plotX, y=plotZ, c=color, marker=self.marker, s=self.marker_size, 
                                                        cmap=self.cmap, vmin=color_min, vmax=color_max, zorder=zorder)
            
            self.NsVsEw_paths = NsVsEw_axes.scatter(x=plotX, y=plotY, c=color, marker=self.marker, s=self.marker_size, 
                                                        cmap=self.cmap, vmin=color_min, vmax=color_max, zorder=zorder)
            
            self.NsVsAlt_paths = NsVsAlt_axes.scatter(x=plotZ, y=plotY, c=color, marker=self.marker, s=self.marker_size, 
                                                        cmap=self.cmap, vmin=color_min, vmax=color_max, zorder=zorder)
        except Exception as e: print(e)
        

#    def get_viewed_events(self):
#        print("get viewed events not implemented")
#        return []
#        return [PSE for PSE in self.PSE_list if 
#                (self.loc_filter(PSE.PolE_loc) and PSE.PolE_RMS<self.max_RMS and PSE.num_even_antennas>self.min_numAntennas) 
#                or (self.loc_filter(PSE.PolO_loc) and PSE.PolO_RMS<self.max_RMS and PSE.num_odd_antennas>self.min_numAntennas) ]

    
    def clear(self):
        pass
        
    def use_ancillary_axes(self):
        return False
                

    def toggle_on(self):
        self.display = True
    
    def toggle_off(self):
        self.display = False
        self.clear()
        
    def search(self, ID_list):
        
#        outX = np.array([ self.X_array[index] ])
#        outY = np.array([ self.Y_array[index] ])
#        outZ = np.array([ self.Z_array[index] ])
#        outT = np.array([ self.T_array[index] ])
#        min_filter_datums = {name:np.array([ data[index] ]) for name,data in self.min_filters.items()}
#        max_filter_datums = {name:np.array([ data[index] ]) for name,data in self.max_filters.items()}
#        print_datums = {name:np.array([ data[index] ]) for name,data in self.print_data.items()}
#        color_datums = {name:np.array([ data[index] ]) for name,data in self.color_options.items()}
#        indeces = np.array([ID])
        
        N = len(ID_list)
        outX = np.empty(N, dtype=np.double)
        outY = np.empty(N, dtype=np.double)
        outZ = np.empty(N, dtype=np.double)
        outT = np.empty(N, dtype=np.double)
        min_filter_datums = {name:np.empty(N, data.dtype) for name,data in self.min_filters.items()}
        max_filter_datums = {name:np.empty(N, data.dtype) for name,data in self.max_filters.items()}
        print_datums = {name:np.empty(N, data.dtype) for name,data in self.print_data.items()}
        color_datums = {name:np.empty(N, data.dtype) for name,data in self.color_options.items()}
        indeces = np.empty(N, self.source_IDs.dtype)
        
#        print(self.source_IDs)
        
        i = 0
        for ID in ID_list:
            try:
                IDi = int(ID)
            except:
                print("ID", ID, 'is not an int')
                continue
            
            try:
                index = next((idx for idx, val in enumerate(self.source_IDs) if val==IDi))#[0]
            except Exception as e:
                print(e)
                print("ID", ID, 'cannot be found', IDi)
                continue
            
            outX[i] = self.X_array[index]
            outY[i] = self.Y_array[index]
            outZ[i] = self.Z_array[index]
            outT[i] = self.T_array[index]
            
            indeces[i] = ID
            
            for name,data in self.min_filters.items():
                min_filter_datums[name][i] = data[index]
                
            for name,data in self.print_data.items():
                print_datums[name][i] = data[index]
                
            for name,data in self.max_filters.items():
                max_filter_datums[name][i] = data[index]
                
            for name,data in self.color_options.items():
                color_datums[name][i] = data[index]
            
            i += 1
            
        if i==0:
            return None
            
        outX = outX[:i]
        outY = outY[:i]
        outZ = outZ[:i]
        outT = outT[:i]
        
        min_filter_datums = {name:data[:i] for name,data in min_filter_datums.items()}
        max_filter_datums = {name:data[:i] for name,data in max_filter_datums.items()}
        print_datums = {name:data[:i] for name,data in print_datums.items()}
        color_datums = {name:data[:i] for name,data in color_datums.items()}
        
        new_DS = DataSet_generic_PSE( outX, outY, outZ, outT, self.marker, self.marker_size, '*k', 
                 self.name+"_copy", self.cmap, 
                 min_filter_datums, max_filter_datums, print_datums, color_datums, source_IDs=indeces)
        
        new_DS.X_offset = self.X_offset
        new_DS.Y_offset = self.Y_offset
        new_DS.Z_offset = self.Z_offset
        new_DS.T_offset = self.T_offset
        
#        for name,param in self.min_parameters.items():
#            new_DS.set_min_param(name,param)
#        
#        for name,param in self.max_parameters.items():
#            new_DS.set_max_param(name,param)
            
        return new_DS
    
    def get_view_ID_list(self, coordinate_system):
        
        self.set_total_mask()
        
        bool_workspace = np.ones( len(self.total_mask), dtype=bool )
        
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:]
        Ytmp = self.Y_TMP[:]
        Ztmp = self.Z_TMP[:]
        Ttmp = self.T_TMP[:]
        
        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False, ignore_T=self._ignore_time, bool_workspace=bool_workspace )
            
        
        return [self.source_IDs[i] for i in range( len(self.X_array) ) if self.total_mask[i] and bool_workspace[i] ]
        
    
    def copy_view(self, coordinate_system):
        
        IDS = self.get_view_ID_list( coordinate_system )
        return self.search(IDS)
        
#        self.set_total_mask()
#        
#        bool_workspace = np.ones( len(self.total_mask), dtype=bool )
#        
#        self.X_TMP[:] = self.X_array  
#        self.Y_TMP[:] = self.Y_array  
#        self.Z_TMP[:] = self.Z_array  
#        self.T_TMP[:] = self.T_array 
#        
#        Xtmp = self.X_TMP[:]
#        Ytmp = self.Y_TMP[:]
#        Ztmp = self.Z_TMP[:]
#        Ttmp = self.T_TMP[:]
#        
##        np.compress( self.total_mask, self.X_TMP, out=Xtmp )
##        np.compress( self.total_mask, self.Y_TMP, out=Ytmp )
##        np.compress( self.total_mask, self.Z_TMP, out=Ztmp )
##        np.compress( self.total_mask, self.T_TMP, out=Ttmp )
#        
#        Xtmp += self.X_offset
#        Ytmp += self.Y_offset
#        Ztmp += self.Z_offset
#        Ttmp += self.T_offset
#        
#        if self.transform_memory is None:
#            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
#        coordinate_system.set_workingMemory( self.transform_memory )
#        
#        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter( 
#            Xtmp, Ytmp, Ztmp, Ttmp, 
#            make_copy=False, ignore_T=self._ignore_time, bool_workspace=bool_workspace )
#        
#        
#        outX = []
#        outY = []
#        outZ = []
#        outT = []
#        indeces = []
#        min_filter_datums = {name:[] for name in self.min_filters.keys()}
#        max_filter_datums = {name:[] for name in self.max_filters.keys()}
#        print_datums = {name:[] for name in self.print_data.keys()}
#        color_datums = {name:[] for name in self.color_options.keys()}
#
#        for i in range( len(self.X_array) ):
#            if not (self.total_mask[i] and bool_workspace[i]):
#                continue
#            
#            outX.append( self.X_array[i] )
#            outY.append( self.Y_array[i] )
#            outZ.append( self.Z_array[i] )
#            outT.append( self.T_array[i] )
#            indeces.append( self.source_IDs[i] )
#
#            for name, data in self.min_filters.items():
#                min_filter_datums[name].append( data[i] )
#            for name, data in self.max_filters.items():
#                max_filter_datums[name].append( data[i] )
#            for name, data in self.print_data.items():
#                print_datums[name].append( data[i] )
#            for name, data in self.color_options.items():
#                color_datums[name].append( data[i])
#                
#        if len(outX) == 0:
#            return None
#            
#        
#        outX = np.array( outX )
#        outY = np.array( outY )
#        outZ = np.array( outZ )
#        outT = np.array( outT )
#        indeces = np.array( indeces )
#        min_filter_datums = {name:np.array(data) for name,data in min_filter_datums.items()}
#        max_filter_datums = {name:np.array(data) for name,data in max_filter_datums.items()}
#        print_datums = {name:np.array(data) for name,data in print_datums.items()}
#        color_datums = {name:np.array(data) for name,data in color_datums.items()}
#        
#        new_DS = DataSet_generic_PSE( outX, outY, outZ, outT, self.marker, self.marker_size, '*k', 
#                 self.name+"_copy", self.cmap, 
#                 min_filter_datums, max_filter_datums, print_datums, color_datums, source_IDs=indeces)
#        
#        new_DS.X_offset = self.X_offset
#        new_DS.Y_offset = self.Y_offset
#        new_DS.Z_offset = self.Z_offset
#        new_DS.T_offset = self.T_offset
#        
#        for name,param in self.min_parameters.items():
#            new_DS.set_min_param(name,param)
#        
#        for name,param in self.max_parameters.items():
#            new_DS.set_max_param(name,param)
#            
#        return new_DS

    def print_info(self, coordinate_system):
        self.set_total_mask()
        
        bool_workspace = np.ones( len(self.total_mask), dtype=bool )
        
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:]
        Ytmp = self.Y_TMP[:]
        Ztmp = self.Z_TMP[:]
        Ttmp = self.T_TMP[:]
        
#        np.compress( self.total_mask, self.X_TMP, out=Xtmp )
#        np.compress( self.total_mask, self.Y_TMP, out=Ytmp )
#        np.compress( self.total_mask, self.Z_TMP, out=Ztmp )
#        np.compress( self.total_mask, self.T_TMP, out=Ttmp )
        
        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False, ignore_T=self._ignore_time, bool_workspace=bool_workspace )
        
        
        print()
        print()
        N = 0
        for i in range( len(self.X_array) ):
            if not (self.total_mask[i] and bool_workspace[i]):
                continue
            
            print('source:', self.source_IDs[i])
            print("  X:{:.2f} Y:{:.2f} Z:{:.2f} T:{:.10f}:".format( self.X_array[i], self.Y_array[i], self.Z_array[i], self.T_array[i]) )
            j = np.sum(bool_workspace[:i])
            print("  plot coords X, Y, Z, Zt, T", plotX[j], plotY[j], plotZ[j], plotZt[j], plotT[j])
            
            for name, data in self.min_filters.items():
                print("  ",name, data[i])
            for name, data in self.max_filters.items():
                print("  ",name, data[i])
            for name, data in self.print_data.items():
                print("  ",name, data[i])
            print()
            
            N +=1
        print(N, "source")
        
    
        
    
    def text_output(self):
        
        self.set_total_mask()
        
        # self.X_TMP[:] = self.X_array  
        # self.Y_TMP[:] = self.Y_array  
        # self.Z_TMP[:] = self.Z_array  
        # self.T_TMP[:] = self.T_array 
        
        # Xtmp = self.X_TMP[:]
        # Ytmp = self.Y_TMP[:]
        # Ztmp = self.Z_TMP[:]
        # Ttmp = self.T_TMP[:]
        
        # Xtmp += self.X_offset
        # Ytmp += self.Y_offset
        # Ztmp += self.Z_offset
        # Ttmp += self.T_offset

        
        with open("output.txt", 'w') as fout:
            fout.write("northing[m] easting[m] altitude[m] time[s]")
            
            for key in self.txtOut_info.keys():
                fout.write(" ")
                fout.write(key)
            fout.write('\n')
            
            for i in range( len(self.X_array) ):
                if not self.total_mask[i]:
                    continue

                fout.write( str(self.X_array[i]) )
                fout.write(', ')     

                fout.write(str(self.Y_array[i]) )
                fout.write(', ')     

                fout.write( str(self.Z_array[i]) )
                fout.write(', ')     

                fout.write( str(self.T_array[i]) )
                
                for value in self.txtOut_info.values():
                    fout.write(', ')     
                    fout.write( str(value[i]) )
                
                fout.write('\n')
                
        print('done writing')
                    
    def ignore_time(self, ignore=None):
        if ignore is not None:
            self._ignore_time = ignore
        return self._ignore_time


from .SPSF_readwrite import pointSource_data
def read_pol_data( fnames, name, cmap ):
    if isinstance(fnames, str) or not isinstance(fnames, list):
        fnames = [fnames] ## for backwards compatibility


    #
    all_Xlocs = None
    all_Ylocs = None
    all_Zlocs = None
    all_Tlocs = None
    all_intensity = None
    all_deg_pol = None
    all_lin_pol = None
    all_circ_pol = None
    all_dir_east = None
    all_dir_north = None
    all_dir_up = None
    allIDS = None

    for fname in fnames:

        if isinstance(fname, str):
            print('opening', fname, 'in', name )
            pol_data = pointSource_data( fname )
        else:
            pol_data = fname
            
        ID = pol_data.data['unique_id']
        X_locs = pol_data.data['distance_east']
        Y_locs = pol_data.data['distance_north']
        Z_locs = pol_data.data['distance_up']
        T = pol_data.data['time_from_second']
        intensity = pol_data.data['intensity']

        #CS002_amp = pol_data.data['CS002_amp']
        # para_fit = pol_data.data['para_fit']

        if 'deg_pol' in pol_data.collums_headings:
            deg_pol = pol_data.data['deg_pol']
        else:
            deg_pol = np.ones(len(ID))
        lin_pol = pol_data.data['lin_pol']
        circ_pol = pol_data.data['circ_pol']

        dir_east = pol_data.data['dir_east']
        dir_north = pol_data.data['dir_north']
        dir_up = pol_data.data['dir_up']

        for i in range(len(dir_east)):
            E = dir_east[i]
            N = dir_north[i]
            U = dir_up[i]

            norm = np.sqrt(E*E + N*N + U*U)

            dir_east[i] = E/norm
            dir_north[i] = N/norm
            dir_up[i] = U/norm

        if all_Xlocs is None:
            all_Xlocs = X_locs
            all_Ylocs = Y_locs
            all_Zlocs = Z_locs
            all_Tlocs = T
            all_intensity = intensity
            all_deg_pol = deg_pol
            all_lin_pol = lin_pol
            all_circ_pol = circ_pol
            all_dir_east = dir_east
            all_dir_north = dir_north
            all_dir_up = dir_up
            allIDS = ID
        else:
            all_Xlocs = np.append(all_Xlocs, X_locs )
            all_Ylocs = np.append(all_Ylocs, Y_locs )
            all_Zlocs = np.append(all_Zlocs, Z_locs )
            all_Tlocs = np.append(all_Tlocs, T )
            all_intensity = np.append(all_intensity, intensity )
            all_deg_pol = np.append(all_deg_pol, deg_pol )
            all_lin_pol = np.append(all_lin_pol, lin_pol )
            all_circ_pol = np.append(all_circ_pol, circ_pol )
            all_dir_east = np.append(all_dir_east, dir_east )
            all_dir_north = np.append(all_dir_north, dir_north )
            all_dir_up = np.append(all_dir_up, dir_up )
            allIDS = np.append(allIDS, ID )

    filters = { 'intensity':all_intensity,
                'lin_pol':all_lin_pol, 'circ_pol':all_circ_pol,
                'deg_pol':all_deg_pol}

    pol_dataset = DataSet_polarized_PSE(all_Xlocs,all_Ylocs,all_Zlocs,all_Tlocs,
        dirX_array=all_dir_east, dirY_array=all_dir_north, dirZ_array=all_dir_up,
        line_width=5, color_mode='time', name=name, cmap=cmap,
        extra_info=filters, source_IDs=allIDS)

    return pol_dataset

class DataSet_polarized_PSE(DataSet_Type):

    def __init__(self, X_array, Y_array, Z_array, T_array,
                 dirX_array, dirY_array, dirZ_array,
                line_width, color_mode, name, cmap,
                 extra_info={}, source_IDs=None, scale_key='intensity'):


        self.color_mode = color_mode
        self.cmap = cmap
        self.name = name
        self._ignore_time = False
        self.display = True

        self.X_array = X_array
        self.Y_array = Y_array
        self.Z_array = Z_array
        self.T_array = T_array

        self.dirX_array = dirX_array
        self.dirY_array = dirY_array
        self.dirZ_array = dirZ_array
        # self.intensity_array = intensity_array

        self.extra_info = extra_info
        # self.max_filters = max_filters
        # self.color_options = color_options
        # self.print_data = print_info
        # self.txtOut_info = txtOut_info

        # self.max_num_points = 10000

        self.source_IDs = source_IDs
        if self.source_IDs is None:
            self.source_IDs = np.arange(len(X_array))

        self.X_offset = 0.0
        self.Y_offset = 0.0
        self.Z_offset = 0.0
        self.T_offset = 0.0

        self.min_parameters = {name: np.min(values) for name, values in self.extra_info.items()}
        self.max_parameters = {name: np.max(values) for name, values in self.extra_info.items()}

        ## probably should call previous constructor here

        self.min_masks = {name: np.ones(len(self.X_array), dtype=bool) for name in self.extra_info.keys()}
        self.max_masks = {name: np.ones(len(self.X_array), dtype=bool) for name in self.extra_info.keys()}

        self.total_mask = np.ones(len(self.X_array), dtype=bool)

        self.X_TMP = np.empty(len(self.X_array), dtype=np.double)
        self.Y_TMP = np.empty(len(self.Y_array), dtype=np.double)
        self.Z_TMP = np.empty(len(self.Z_array), dtype=np.double)
        self.T_TMP = np.empty(len(self.T_array), dtype=np.double)

        self.scale_space = np.empty(len(self.X_array), dtype=np.double)
        self.time_scale_space = np.empty(len(self.X_array), dtype=np.double)
        self.dirX_memory = np.empty(len(self.X_array), dtype=np.double)
        self.dirY_memory = np.empty(len(self.X_array), dtype=np.double)
        self.dirZ_memory = np.empty(len(self.X_array), dtype=np.double)

        # self.decimation_TMP = np.empty(len(self.X_array), dtype=float)
        # self.decimation_mask = np.zeros(len(self.X_array), dtype=bool)

        #        self.masked_color_info = { name:np.ma.masked_array(data, mask=self.total_mask, copy=False)  for name,data in self.color_options.items() }
        #        self.color_time_mask = np.ma.masked_array(self.T_array, mask=self.total_mask, copy=False)

        #        self.set_show_all()

        #### some axis data ###
        self.AltVsT_paths = None
        self.AltVsEw_paths = None
        self.NsVsEw_paths = None
        self.NsVsAlt_paths = None

        self.transform_memory = None


        ### scale and shape info ###
        self.line_width = line_width # width of lines if plot polarozation

        self.time_dot_scale = line_width # scale of time dots
        self.space_line_scale = 0.1 # scale of the lines or dots in space
        self.space_dot_scale = line_width # scale of the lines or dots in space

        self.scale_mode = 'linear' # defines how the sizes of dots vary
        ## scale modes: linear, log, constant
        self.scale_key = scale_key ## defines which value to scale by
        if self.scale_key not in self.extra_info:
            self.scale_mode = 'constant'
        self.pol_mode = 'line' # line or none. Defines how to display polarization. none just plots dots


        self.help = 0 ## a trick to get this to print help docs

    def set_show_all(self, coordinate_system, do_set_limits=True):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""

        for name, data in self.extra_info.items():
            self.set_min_param(name, np.min(data))
            self.set_max_param(name, np.max(data))

        if do_set_limits:
            self.X_TMP[:] = self.X_array
            self.Y_TMP[:] = self.Y_array
            self.Z_TMP[:] = self.Z_array
            self.T_TMP[:] = self.T_array

            self.X_TMP += self.X_offset
            self.Y_TMP += self.Y_offset
            self.Z_TMP += self.Z_offset
            self.T_TMP += self.T_offset

            if self.transform_memory is None:
                self.transform_memory = coordinate_system.make_workingMemory(len(self.X_array))
            coordinate_system.set_workingMemory(self.transform_memory)

            plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform(
                self.X_TMP, self.Y_TMP, self.Z_TMP, self.T_TMP,
                make_copy=False)

            if len(plotX) > 0:
                coordinate_system.set_plotX(np.min(plotX), np.max(plotX))
                coordinate_system.set_plotY(np.min(plotY), np.max(plotY))
                coordinate_system.set_plotZ(np.min(plotZ), np.max(plotZ))
                coordinate_system.set_plotZt(np.min(plotZt), np.max(plotZt))
                coordinate_system.set_plotT(np.min(plotT), np.max(plotT))

    def bounding_box(self, coordinate_system):

        #### get cuts ###

        self.set_total_mask()

        ## filter and shift
        Ntmp = np.sum(self.total_mask)

        self.X_TMP[:] = self.X_array
        self.Y_TMP[:] = self.Y_array
        self.Z_TMP[:] = self.Z_array
        self.T_TMP[:] = self.T_array

        Xtmp = self.X_TMP[:Ntmp]
        Ytmp = self.Y_TMP[:Ntmp]
        Ztmp = self.Z_TMP[:Ntmp]
        Ttmp = self.T_TMP[:Ntmp]

        np.compress(self.total_mask, self.X_TMP, out=Xtmp)
        np.compress(self.total_mask, self.Y_TMP, out=Ytmp)
        np.compress(self.total_mask, self.Z_TMP, out=Ztmp)
        np.compress(self.total_mask, self.T_TMP, out=Ttmp)

        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset

        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory(len(self.X_array))
        coordinate_system.set_workingMemory(self.transform_memory)

        ## transform
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform(
            Xtmp, Ytmp, Ztmp, Ttmp,
            make_copy=False)

        ## return actual bounds
        if len(plotX) > 0:
            Xbounds = [np.min(plotX), np.max(plotX)]
            Ybounds = [np.min(plotY), np.max(plotY)]
            Zbounds = [np.min(plotZ), np.max(plotZ)]
            Ztbounds = [np.min(plotZt), np.max(plotZt)]
            Tbounds = [np.min(plotT), np.max(plotT)]
        else:
            Xbounds = [0, 1]
            Ybounds = [0, 1]
            Zbounds = [0, 1]
            Ztbounds = [0, 1]
            Tbounds = [0, 1]

        return Xbounds, Ybounds, Zbounds, Ztbounds, Tbounds

    def T_bounds(self, coordinate_system):
        #### get cuts ###
        self.set_total_mask()

        ## filter and shift
        Ntmp = np.sum(self.total_mask)

        self.X_TMP[:] = self.X_array
        self.Y_TMP[:] = self.Y_array
        self.Z_TMP[:] = self.Z_array
        self.T_TMP[:] = self.T_array

        Xtmp = self.X_TMP[:Ntmp]
        Ytmp = self.Y_TMP[:Ntmp]
        Ztmp = self.Z_TMP[:Ntmp]
        Ttmp = self.T_TMP[:Ntmp]

        np.compress(self.total_mask, self.X_TMP, out=Xtmp)
        np.compress(self.total_mask, self.Y_TMP, out=Ytmp)
        np.compress(self.total_mask, self.Z_TMP, out=Ztmp)
        np.compress(self.total_mask, self.T_TMP, out=Ttmp)

        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset

        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory(len(self.X_array))
        coordinate_system.set_workingMemory(self.transform_memory)

        ## transform and cut on bounds
        TMP = coordinate_system.get_plotT()
        A = TMP[0]  ## need to copy
        B = TMP[1]
        coordinate_system.set_plotT(-np.inf, np.inf)
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter(
            Xtmp, Ytmp, Ztmp, Ttmp,
            make_copy=False, ignore_T=self._ignore_time, bool_workspace=self.total_mask)
        coordinate_system.set_plotT(A, B)

        ## return actual bounds
        if len(plotT) > 0:
            return [np.min(plotT), np.max(plotT)]
        else:
            return [0, 1]

    def get_all_properties(self):
        ret = {"line width": str(self.line_width), "color mode": str(self.color_mode), 'name': self.name,
               "X offset": self.X_offset, "Y offset": self.Y_offset, "Z offset": self.Z_offset,
               "T offset": self.T_offset,  'time dot scale':self.time_dot_scale, 'space line scale':self.space_line_scale,
               'space dot scale':self.space_dot_scale, 'scale mode':self.scale_mode, 'scale key':self.scale_key,
               'pol_mode':self.pol_mode, "help":self.help}

        for name in self.extra_info.keys():
            ret['min ' + name] = self.min_parameters[name]
            ret['max ' + name] = self.max_parameters[name]

        return ret
        ## need: marker type, color map

    def set_property(self, name, str_value):

        try:
            if name == "line width":
                self.line_width = int(str_value)

            elif name == "color mode":
                self.color_mode = str_value

            elif name == 'name':
                self.name = str_value

            elif name == "X offset":
                self.X_offset = float(str_value)

            elif name == "Y offset":
                self.Y_offset = float(str_value)

            elif name == "Z offset":
                self.Z_offset = float(str_value)

            elif name == "T offset":
                self.T_offset = float(str_value)
            #
            # elif name == "max points":
            #     self.max_num_points = int(str_value)

            elif name == "time dot scale":
                self.time_dot_scale = float(str_value)

            elif name == "space line scale":
                self.space_line_scale = float(str_value)

            elif name == "space dot scale":
                self.space_dot_scale = float(str_value)

            elif name == "scale mode":
                if str_value in ['linear', 'log', 'constant']:
                    self.scale_mode = str_value
                else:
                    print('scale mode should be: linear, log, or constant')

            elif name == "scale key":
                if str_value in self.extra_info.keys():
                    self.scale_key = str_value
                else:
                    print('scale key should be a value in extra info')

            elif name == "pol_mode":
                if str_value in ['line', 'none']:
                    self.pol_mode = str_value
                elif str_value == "yo mama":
                    print( "Why did you belive this? What did you expect to happen???" )
                else:
                    print('pol_mode should be line, none, or "yo mama"')

            elif name == "help":
                if str_value == '1':
                    print('HELPING:')
                    print("  help not yet implemented")
                self.help = 0

            elif name[:3]=='min' and name[4:] in self.min_parameters:
                self.set_min_param(name[4:], float(str_value))

            elif name[:3]=='max' and name[4:] in self.max_parameters:
                self.set_max_param(name[4:], float(str_value))

            else:
                print("do not have property:", name)
        except:
            print("error in setting property", name, str_value)
            pass

    def set_min_param(self, name, value):
        self.min_parameters[name] = value
        #        self.min_masks[name][:] = self.min_filters[name] > value
        np.greater_equal(self.extra_info[name], value, out=self.min_masks[name])

    def set_max_param(self, name, value):
        self.max_parameters[name] = value
        #        self.max_masks[name][:] = self.max_filters[name] < value
        np.greater_equal(value, self.extra_info[name], out=self.max_masks[name])

    def set_total_mask(self):
        self.total_mask[:] = True

        for name, mask in self.min_masks.items():
            np.logical_and(self.total_mask, mask, out=self.total_mask)

        for name, mask in self.max_masks.items():
            np.logical_and(self.total_mask, mask, out=self.total_mask)

    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system, zorder):
        self.set_total_mask()
        N = np.sum(self.total_mask)

        #### random book keeping ####
        self.clear()

        # if self.color_mode in self.color_options:
        #     color = self.color_options[self.color_mode][self.total_mask]  ## should fix this to not make memory

        #### set cuts and transforms
        self.X_TMP[:] = self.X_array
        self.Y_TMP[:] = self.Y_array
        self.Z_TMP[:] = self.Z_array
        self.T_TMP[:] = self.T_array

        Xtmp = self.X_TMP[:N]
        Ytmp = self.Y_TMP[:N]
        Ztmp = self.Z_TMP[:N]
        Ttmp = self.T_TMP[:N]

        np.compress(self.total_mask, self.X_TMP, out=Xtmp)
        np.compress(self.total_mask, self.Y_TMP, out=Ytmp)
        np.compress(self.total_mask, self.Z_TMP, out=Ztmp)
        np.compress(self.total_mask, self.T_TMP, out=Ttmp)

        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset

        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory(len(self.X_array))
        coordinate_system.set_workingMemory(self.transform_memory)


        loc_mask = np.empty(N, dtype=bool)
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter(
            Xtmp, Ytmp, Ztmp, Ttmp,
            make_copy=False, ignore_T=self._ignore_time, bool_workspace=loc_mask)


        if (not self.display) or len(plotX) == 0:
            print(self.name, "not display. have:", len(plotX))
            return

        ### get color!
        if self.color_mode == "time":
            color = plotT
            ARG = coordinate_system.get_plotT()
            color_min = ARG[0]
            color_max = ARG[1]

        elif self.color_mode[0] == '*':
            color = self.color_mode[1:]
            color_min = None
            color_max = None

        elif self.color_mode in self.extra_info:
            color = self.extra_info[self.color_mode][self.total_mask][loc_mask]
            color_min = np.min(color)
            color_max = np.max(color)

        else:
            print("bad color mode. This should be interesting!")
            color = self.color_mode
            color_min = None
            color_max = None


        ## get scale!
        if self.scale_mode == 'constant':
            time_scale = 1
            space_scale = 1
        else:
            scale_data_1 = self.extra_info[self.scale_key][self.total_mask]
            max_scale = np.max( scale_data_1 )
            min_scale = np.min( scale_data_1 )

            scale_space = self.scale_space[:len(plotX)]
            scale_space[:] = scale_data_1[loc_mask]

            if self.scale_mode == 'linear':
                scale_space -= min_scale ## smallest value is 0
                scale_space *= 1.0/( max_scale - min_scale ) ## largest value is 1
            else: # log
                scale_space *= 1.0/min_scale ## smallist is 0
                np.log(scale_space, out=scale_space)
                scale_space *= 1.0/np.log( max_scale/min_scale ) ## largest is 1

            time_scale = self.time_scale_space[:len(plotX)]
            time_scale[:] = scale_space
            space_scale = scale_space

        time_scale *= self.time_dot_scale

        if self.pol_mode == 'line':
            space_scale *= self.space_line_scale
        else: ## none
            space_scale *= self.space_dot_scale





        N_before = len(plotX)
        # if self.max_num_points > 0 and self.max_num_points < N_before:
        #     decimation_factor = self.max_num_points / float(N_before)
        #
        #     data = self.decimation_TMP[:N_before]
        #     mask = self.decimation_mask[:N_before]
        #
        #     data[:] = decimation_factor
        #     np.cumsum(data, out=data)
        #     np.floor(data, out=data)
        #     np.greater(data[1:], data[:-1], out=mask[1:])
        #     mask[0] = 0
        #
        #     plotX = plotX[mask]
        #     plotY = plotY[mask]
        #     plotZ = plotZ[mask]
        #     plotZt = plotZt[mask]
        #     plotT = plotT[mask]
        #
        #     try:
        #         B4 = color[mask]
        #         color = B4  ## hope this works?
        #     except:
        #         pass

        print(self.name, "plotting", len(plotX), '/', len(self.X_array) )#, "have:", N_before)

        try:
            if not self._ignore_time:
                self.AltVsT_paths = AltVsT_axes.scatter(x=plotT, y=plotZt, c=color, marker='o',
                                                        s=time_scale, zorder=zorder,
                                                        cmap=self.cmap, vmin=color_min, vmax=color_max)

            if self.pol_mode == 'none':
                self.AltVsEw_paths = AltVsEw_axes.scatter(x=plotX, y=plotZ, c=color, marker='o', s=space_scale,
                                                          cmap=self.cmap, vmin=color_min, vmax=color_max, zorder=zorder)

                self.NsVsEw_paths = NsVsEw_axes.scatter(x=plotX, y=plotY, c=color, marker='o', s=space_scale,
                                                        cmap=self.cmap, vmin=color_min, vmax=color_max, zorder=zorder)

                self.NsVsAlt_paths = NsVsAlt_axes.scatter(x=plotZ, y=plotY, c=color, marker='o', s=space_scale,
                                                          cmap=self.cmap, vmin=color_min, vmax=color_max, zorder=zorder)

            else:
                print('a')
                dirX_tmp = self.dirX_memory[:len(plotX)]
                dirY_tmp = self.dirY_memory[:len(plotX)]
                dirZ_tmp = self.dirZ_memory[:len(plotX)]

                dirX_tmp[:] = self.dirX_array[self.total_mask][loc_mask]
                dirY_tmp[:] = self.dirY_array[self.total_mask][loc_mask]
                dirZ_tmp[:] = self.dirZ_array[self.total_mask][loc_mask]

                dirX_tmp *= 0.5
                dirY_tmp *= 0.5
                dirZ_tmp *= 0.5

                dirX_tmp *= space_scale
                dirY_tmp *= space_scale
                dirZ_tmp *= space_scale

                # dirX_tmp *= self.pol_scale*0.5
                # dirY_tmp *= self.pol_scale*0.5
                # dirZ_tmp *= self.pol_scale*0.5
                #
                # if self.pol_mode=='intensity' or self.pol_mode=='log_intensity':
                #     intensity_tmp = np.array( self.intensity_array[self.total_mask][loc_mask] )
                #
                #     if self.pol_mode=='log_intensity':
                #         np.log(intensity_tmp, out=intensity_tmp)
                #
                #     intensity_tmp /= np.max( intensity_tmp )
                #
                #     dirX_tmp *= intensity_tmp
                #     dirY_tmp *= intensity_tmp
                #     dirZ_tmp *= intensity_tmp

                X_low = np.array(plotX)
                X_high = np.array(plotX)
                X_low -= dirX_tmp
                X_high += dirX_tmp

                Y_low = np.array(plotY)
                Y_high = np.array(plotY)
                Y_low -= dirY_tmp
                Y_high += dirY_tmp

                Z_low = np.array(plotZ)
                Z_high = np.array(plotZ)
                Z_low -= dirZ_tmp
                Z_high += dirZ_tmp

                for i in range(len(X_low)):
                    X = [ X_low[i], X_high[i] ]
                    Y = [ Y_low[i], Y_high[i] ]
                    Z = [ Z_low[i], Z_high[i] ]

                    c = None
                    if (color_min is not None) and (color_max is not None):
                        c = color[i]
                        c -= color_min
                        c /= color_max-color_min

                        c = self.cmap( c )
                    else:
                        c = color

                    AltVsEw_axes.plot(X,Z,'-', lw=self.line_width, c=c, zorder=zorder)
                                 # cmap=self.cmap, vmin=color_min, vmax=color_max)

                    NsVsEw_axes.plot(X,Y,'-', lw=self.line_width, c=c, zorder=zorder)
                                       # cmap=self.cmap, vmin=color_min, vmax=color_max)

                    NsVsAlt_axes.plot(Z,Y,'-', lw=self.line_width, c=c, zorder=zorder)
                                         #   cmap=self.cmap, vmin=color_min, vmax=color_max)


        except Exception as e:
            print(e)

    #    def get_viewed_events(self):
    #        print("get viewed events not implemented")
    #        return []
    #        return [PSE for PSE in self.PSE_list if
    #                (self.loc_filter(PSE.PolE_loc) and PSE.PolE_RMS<self.max_RMS and PSE.num_even_antennas>self.min_numAntennas)
    #                or (self.loc_filter(PSE.PolO_loc) and PSE.PolO_RMS<self.max_RMS and PSE.num_odd_antennas>self.min_numAntennas) ]

    def clear(self):
        pass

    def use_ancillary_axes(self):
        return False

    def toggle_on(self):
        self.display = True

    def toggle_off(self):
        self.display = False
        self.clear()

    ## the res of this is wrong
    def ignore_time(self, ignore=None):
        if ignore is not None:
            self._ignore_time = ignore
        return self._ignore_time

    def print_info(self, coordinate_system):
        self.set_total_mask()
        N = np.sum(self.total_mask)

        #### random book keeping ####
        self.clear()

        # if self.color_mode in self.color_options:
        #     color = self.color_options[self.color_mode][self.total_mask]  ## should fix this to not make memory

        #### set cuts and transforms
        self.X_TMP[:] = self.X_array
        self.Y_TMP[:] = self.Y_array
        self.Z_TMP[:] = self.Z_array
        self.T_TMP[:] = self.T_array

        Xtmp = self.X_TMP[:N]
        Ytmp = self.Y_TMP[:N]
        Ztmp = self.Z_TMP[:N]
        Ttmp = self.T_TMP[:N]

        np.compress(self.total_mask, self.X_TMP, out=Xtmp)
        np.compress(self.total_mask, self.Y_TMP, out=Ytmp)
        np.compress(self.total_mask, self.Z_TMP, out=Ztmp)
        np.compress(self.total_mask, self.T_TMP, out=Ttmp)

        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset

        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory(len(self.X_array))
        coordinate_system.set_workingMemory(self.transform_memory)


        loc_mask = np.empty(N, dtype=bool)
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter(
            Xtmp, Ytmp, Ztmp, Ttmp,
            make_copy=False, ignore_T=self._ignore_time, bool_workspace=loc_mask)

        print( "dataset:", self.name )

        IDS = self.source_IDs[self.total_mask][loc_mask]

        inX = self.X_array[self.total_mask][loc_mask]
        inY = self.Y_array[self.total_mask][loc_mask]
        inZ = self.Z_array[self.total_mask][loc_mask]
        inT = self.T_array[self.total_mask][loc_mask]

        dirX = self.dirX_array[self.total_mask][loc_mask]
        dirY = self.dirY_array[self.total_mask][loc_mask]
        dirZ  = self.dirZ_array[self.total_mask][loc_mask]

        infoplot = { key:data[self.total_mask][loc_mask] for key,data in self.extra_info.items() }

        for i in range(len(IDS)):
            print('source', IDS[i])
            print(' XYZT:',  inX[i], ',', inY[i], ',', inZ[i], ',', inT[i])
            print(' dir:', dirX[i], ',', dirY[i], ',', dirZ[i])
            for key,d in infoplot.items():
                print(' ', key, d[i])
            print()

            # print('  plot at', plotX[i], plotY[i], plotZ[i], plotZt[i] )
            # print('    plot t:', plotT[i]  )
            # print('  at', inX[i], inY[i], inZ[i])
            # print('    T', inT[i])
            # for key, d in infoplot.items():
            #     print(' ', key, d[i])
            # print()


def LMAHeader_to_StationLocs(LMAheader, groupLOFARStations=False, center='LOFAR', name='stations', color=None, marker='s', textsize=None):
    # from LoLIM.utilities import SId_to_Sname

    station_names = []
    stationX = []
    stationY = []
    
    for station in LMAheader.antenna_info_list:
        name = station.name
        use = True
        
        if groupLOFARStations:
            # LOFAR_SID = int(name[:3])
            # name = SId_to_Sname[ LOFAR_SID ]
            name = name[:3]
            if name in station_names:
                use = False
                
        if use:
            station_names.append( name )
            stationXYZ = station.get_XYZ(center=center)
            
            stationX.append( stationXYZ[0] )
            stationY.append( stationXYZ[1] )
            
    return station_locations_DataSet( stationX, stationY, station_names, name='stations', color=color, marker=marker, textsize=textsize )

def IterMapper_StationLocs(header, name='stations', color=None, marker='s', textsize=None):
    station_names = []
    stationX = []
    stationY = []
    
    for name, antList in zip(header.station_names, header.antenna_info):
        station_names.append( name )
        
        ant = antList[0]
        stationX.append( ant.location[0] )
        stationY.append( ant.location[1] )
    
    return station_locations_DataSet( stationX, stationY, station_names, name='stations', color=color, marker=marker, textsize=textsize )
    
    

class station_locations_DataSet( DataSet_Type ):
    def __init__(self, station_X_array, station_Y_array, station_names, name, color, marker, textsize ):
         self.marker = marker
         self.marker_size = 10
         self.color = color
         self.name = name
         self.textsize = textsize
         self.display= True
         
         self.X_array = station_X_array
         self.Y_array = station_Y_array
         self.station_names = station_names
         
         self.X_TMP = np.empty(len(self.X_array), dtype=np.double)
         self.Y_TMP = np.empty(len(self.X_array), dtype=np.double)
         self.Z_TMP = np.empty(len(self.X_array), dtype=np.double)
         self.T_TMP = np.empty(len(self.X_array), dtype=np.double)
         
         self.transform_memory = None
         
         self.show_text = True
         
    def set_show_all(self, coordinate_system, do_set_limits=True):
        
        if do_set_limits:
            self.X_TMP[:] = self.X_array  
            self.Y_TMP[:] = self.Y_array  
            self.Z_TMP[:] = 0.0  
            self.T_TMP[:] = 0.0
            
            if self.transform_memory is None:
                self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
            coordinate_system.set_workingMemory( self.transform_memory )
            
            plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform( 
                self.X_TMP, self.Y_TMP, self.Z_TMP, self.T_TMP, 
                make_copy=False)
            
            if len(plotX) > 0:
                coordinate_system.set_plotX( np.min(plotX), np.max(plotX) )
                coordinate_system.set_plotY( np.min(plotY), np.max(plotY) )
                
    def bounding_box(self, coordinate_system):
        
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = 0.0  
        self.T_TMP[:] = 0.0
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        
        ## transform
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform( 
            self.X_TMP, self.Y_TMP, self.Z_TMP, self.T_TMP, 
            make_copy=False)
        
        ## return actual bounds
        if len(plotX) > 0:
            Xbounds = [np.min(plotX), np.max(plotX)]
            Ybounds = [np.min(plotY), np.max(plotY)]
        else:
            Xbounds = [0,1]
            Ybounds = [0,1]
            
        Zbounds = [np.nan,np.nan]
        Ztbounds = [np.nan,np.nan]
        Tbounds = [np.nan,np.nan]
        
        return Xbounds, Ybounds, Zbounds, Ztbounds, Tbounds
    
    def T_bounds(self, coordinate_system):
        return [np.nan, np.nan]
    
    def get_all_properties(self):
        ret =  {"marker size":str(self.marker_size),  'name':self.name, 'marker':self.marker,
                "color":self.color, "textsize":str(self.textsize), "show text": str(int(self.show_text))}
            
        return ret    
    
    def set_property(self, name, str_value):
        
        try:
            if name == "marker size":
                self.marker_size = int(str_value)
                
            elif name == "color":
                self.color = str_value
                    
            elif name == 'name':
                self.name = str_value
                
            elif name == 'marker':
                self.marker = str_value
                
            elif name == 'textsize':
                self.textsize = int(str_value)
                
            elif name == 'show text':
                self.show_text = bool(int(str_value))
                
            else:
                print("do not have property:", name)
        except:
            print("error in setting property", name, str_value)
            pass
                
    def clear(self):
        pass
        
    def use_ancillary_axes(self):
        return False

    def toggle_on(self):
        self.display = True
    
    def toggle_off(self):
        self.display = False
        self.clear()
        
    
    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system, zorder):
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = 0.0  
        self.T_TMP[:] = 0.0
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        
        ## transform
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform( 
            self.X_TMP, self.Y_TMP, self.Z_TMP, self.T_TMP, 
            make_copy=False)


        if (not self.display) or len(plotX)==0:
            
            print(self.name, "not display.")
            return
        
            
        print(self.name, "plotting")
        
        try:
            
            self.NsVsEw_paths = NsVsEw_axes.scatter(x=plotX, y=plotY, c=self.color, marker=self.marker, s=self.marker_size, zorder=zorder)
            
            if self.show_text:
                for sname, X, Y in zip(self.station_names, plotX, plotY):
                    NsVsEw_axes.annotate(sname, (X,Y), size=self.textsize)
            
        except Exception as e: print(e)
        
    
                
                
        

class linearSpline_DataSet(DataSet_Type):
    ### NOTE: this whole thing only works in plot-coordinates. Thus, only functional if consistantly used with cartesian system
    ## maybe replace this one with one in pol analysis. I think the code is better
    
    def __init__(self, name, color, linewidth=None, initial_points=None):
        self.name = name
        self.display = True
        self.color = color
        
        self.LW = linewidth
        if self.LW is not None:
            self.LW = float(self.LW)
        
        self.X_points = []
        self.Y_points = []
        self.Z_points = []
        if initial_points is not None:
            for p in initial_points:
                self.X_points.append( p[0] )
                self.Y_points.append( p[1] )
                self.Z_points.append( p[2] )
            
        self.mode = -1 ## -1 is off. -2 is edit new point. >0 means edit that point
        self.X_edit = 0
        self.Y_edit = 0
        self.Z_edit = 0
        self.transform_memory = None
        
    def set_show_all(self, coordinate_system, do_set_limits=True):
        """set bounds needed to show all data.
        Set all properties so that all points are shown"""
        pass
    
    def bounding_box(self, coordinate_system):
        """return bounding box in plot-coordinates"""
        X_bounds = [np.min(self.X_points), np.max(self.X_points)]
        Y_bounds = [np.min(self.Y_points), np.max(self.Y_points)]
        Z_bounds = [np.min(self.Z_points), np.max(self.Z_points)]
        return [X_bounds, Y_bounds, Z_bounds, [np.nan,np.nan], [np.nan,np.nan]]

    def T_bounds(self, coordinate_system):
        """return T-bounds given all other bounds, in plot-coordinats"""
        return [np.nan,np.nan]
    
    def get_all_properties(self):
        length = 0
        
        prevX = None
        prevY = None
        prevZ = None
        for X, Y, Z in zip(self.X_points, self.Y_points, self.Z_points):
            if prevX is not None:
                dx = X-prevX
                dy = Y-prevY
                dz = Z-prevZ
                
                length += np.sqrt( dx*dx + dy*dy + dz*dz )
                
            prevX = X
            prevY = Y
            prevZ = Z
        
        
        if self.mode == -1:
            mode = 'off'
        elif self.mode == -2:
            mode = 'edit'
        else:
            mode = str(self.mode)
        
        
        return {'color':self.color, 'name':self.name, 'lineWidth':self.LW, 'mode':mode, 'length':length}
    
    def set_property(self, name, str_value):
        
        try:
            if name == 'color':
                self.color = str_value
            elif name == 'name':
                self.name = str_value
            elif name == 'lineWidth':
                self.LW = float(str_value)
            elif name == 'mode':
                if str_value == 'off':
                    self.mode = -1
                elif str_value == 'edit':
                    self.mode = -2
                else: 
                    self.mode = int(str_value)
            elif name == 'length':
                print("The Erlemyer cannot be wonkydoodled. Please try again later.")
            else:
                print('no attribute:', name)
        except:
            print("error in setting property", name, str_value)
            pass
            
            
    
    def plot(self, AltvsT_axes, AltvsEW_axes, NSvsEW_axes, NsvsAlt_axes, ancillary_axes, coordinate_system, zorder):
        
        self.AltvsEW_axes = AltvsEW_axes
        self.NSvsEW_axes = NSvsEW_axes
        self.NsvsAlt_axes = NsvsAlt_axes
        
        if (not self.display):
            return
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_points) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        Xtmp = np.array( self.X_points )
        Ytmp = np.array( self.Y_points )
        Ztmp = np.array( self.Z_points )
        Ttmp = np.zeros( len(self.X_points) )
        
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.filter( 
            Xtmp, Ytmp, Ztmp, Ztmp, Ttmp, 
            make_copy=False, ignore_T=True )
        
        if len(plotX)==0:
            return

        try:
            self.AltVsEw_paths = AltvsEW_axes.plot(plotX, plotZ, c=self.color, linewidth = self.LW, zorder=zorder)
            
            self.NsVsEw_paths = NSvsEW_axes.plot(plotX, plotY, c=self.color, linewidth = self.LW, zorder=zorder)
            
            self.NsVsAlt_paths = NsvsAlt_axes.plot(plotZ, plotY, c=self.color, linewidth = self.LW, zorder=zorder)
        except Exception as e: print(e)
        
        
    
    
    def key_press_event(self, event):
        if event.key == 'a':
            print("append location to list: ")
            print("  ", self.X_edit, self.Y_edit, self.Z_edit)
            self.X_points.append( self.X_edit )
            self.Y_points.append( self.Y_edit )
            self.Z_points.append( self.Z_edit )
            
            self.X_edit=0
            self.Y_edit=0
            self.Z_edit=0
            
        elif event.key=='e' and self.mode!=-1:
            print('edit location')
            
            if event.inaxes == self.AltvsEW_axes:
                X = event.xdata
                Z = event.ydata
                print(' X:',X, 'Z:', Z)
                if self.mode == -2:
                    self.X_edit = X
                    self.Z_edit = Z
                else:
                    self.X_points[self.mode] = X
                    self.Z_points[self.mode] = Z
            
            elif event.inaxes == self.NSvsEW_axes:
                X = event.xdata
                Y = event.ydata
                print(' X:',X, 'Y:', Y)
                if self.mode == -2:
                    self.X_edit = X
                    self.Y_edit = Y
                else:
                    self.X_points[self.mode] = X
                    self.Y_points[self.mode] = Y
            
            elif event.inaxes == self.NsvsAlt_axes:
                Z = event.xdata
                Y = event.ydata
                print(' Y:',Y , 'Z:', Z)
                if self.mode == -2:
                    self.Y_edit = Y
                    self.Z_edit = Z
                else:
                    self.Y_points[self.mode] = Y
                    self.Z_points[self.mode] = Z
    
        
    def print_info(self, coordinate_system):
        for X,Y,Z in zip(self.X_points,self.Y_points,self.Z_points):
            print('[', X, ',', Y , ',', Z, '],')
            
        print()
        print('current test point:')
        print("  X:", self.X_edit, 'Y:', self.Y_edit, 'Z:', self.Z_edit)
        
        print()
        print(len(self.X_points), 'sources')
    
        length = 0
        
        prevX = None
        prevY = None
        prevZ = None
        for X, Y, Z in zip(self.X_points, self.Y_points, self.Z_points):
            if prevX is not None:
                dx = X-prevX
                dy = Y-prevY
                dz = Z-prevZ
                
                length += np.sqrt( dx*dx + dy*dy + dz*dz )
                
            prevX = X
            prevY = Y
            prevZ = Z
            
        print()
        print('length:', length)
        
    
    def clear(self):
        pass
    
    def use_ancillary_axes(self):
        return False
        
    
    def copy_view(self, coordinate_system):
        print( "not implemented" )
        return None
        
    def search(self, ID_list):
        print("not implemented")
        return None   
    
    def toggle_on(self):
        self.display = True
    
    def toggle_off(self):
        self.display = False
    
    def get_view_ID_list(self, coordinate_system):
        print("not implemented")
        return None
    
    def text_output(self):
        print("not implemented")
        
    def ignore_time(self, ignore=None):
        if ignore is not None:
            print("not implemented")
        return False
    



class DataSet_span(DataSet_Type):
    
    def __init__(self, name, T_starts, T_ends, color):
        
        self.name = name
        self.color = color
        self._ignore_time = False
        self.display = True
        
        self.T_starts = np.array(T_starts)
        self.T_ends = np.array(T_ends)
        
        self.T_offset = 0.0
        
        ## probably should call previous constructor here
        
        self.min_masks = { name:np.ones(len(self.X_array), dtype=bool) for name in self.min_filters.keys()}
        self.max_masks = { name:np.ones(len(self.X_array), dtype=bool) for name in self.max_filters.keys()}
        
        self.total_mask = np.ones(len(self.X_array), dtype=bool)

        self.X_TMP = np.empty(len(self.T_starts), dtype=np.double)
        self.Y_TMP = np.empty(len(self.T_starts), dtype=np.double)
        self.Z_TMP = np.empty(len(self.T_starts), dtype=np.double)
    
        #### some axis data ###
        self.AltVsT_paths = None
        self.AltVsEw_paths = None
        self.NsVsEw_paths = None
        self.NsVsAlt_paths = None
        
        self.transform_memory = None
        
    def set_show_all(self, coordinate_system, do_set_limits=True):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
             
        ## AM HERE
        
        if do_set_limits:
            self.X_TMP[:] = self.X_array  
            self.Y_TMP[:] = self.Y_array  
            self.Z_TMP[:] = self.Z_array  
            self.T_TMP[:] = self.T_array  
            
            self.X_TMP += self.X_offset
            self.Y_TMP += self.Y_offset
            self.Z_TMP += self.Z_offset
            self.T_TMP += self.T_offset
            
            if self.transform_memory is None:
                self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
            coordinate_system.set_workingMemory( self.transform_memory )
            
            plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform( 
                self.X_TMP, self.Y_TMP, self.Z_TMP, self.T_TMP, 
                make_copy=False)
            
            if len(plotX) > 0:
                coordinate_system.set_plotX( np.min(plotX), np.max(plotX) )
                coordinate_system.set_plotY( np.min(plotY), np.max(plotY) )
                coordinate_system.set_plotZ( np.min(plotZ), np.max(plotZ) )
                coordinate_system.set_plotZt( np.min(plotZt), np.max(plotZt) )
                coordinate_system.set_plotT( np.min(plotT), np.max(plotT) )

    def bounding_box(self, coordinate_system):
        
        #### get cuts ###
        
        self.set_total_mask()
            
        ## filter and shift
        Ntmp = np.sum( self.total_mask )
            
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:Ntmp]
        Ytmp = self.Y_TMP[:Ntmp]
        Ztmp = self.Z_TMP[:Ntmp]
        Ttmp = self.T_TMP[:Ntmp]
        
        np.compress( self.total_mask, self.X_TMP, out=Xtmp )
        np.compress( self.total_mask, self.Y_TMP, out=Ytmp )
        np.compress( self.total_mask, self.Z_TMP, out=Ztmp )
        np.compress( self.total_mask, self.T_TMP, out=Ttmp )
        
        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        
        ## transform
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False)
        
        ## return actual bounds
        if len(plotX) > 0:
            Xbounds = [np.min(plotX), np.max(plotX)]
            Ybounds = [np.min(plotY), np.max(plotY)]
            Zbounds = [np.min(plotZ), np.max(plotZ)]
            Ztbounds = [np.min(plotZt), np.max(plotZt)]
            Tbounds = [np.min(plotT), np.max(plotT)]
        else:
            Xbounds = [0,1]
            Ybounds = [0,1]
            Zbounds = [0,1]
            Ztbounds = [0,1]
            Tbounds = [0,1]
        
        return Xbounds, Ybounds, Zbounds, Ztbounds, Tbounds
    
    def T_bounds(self, coordinate_system):
        #### get cuts ###
        self.set_total_mask()
            
        ## filter and shift
        Ntmp = np.sum( self.total_mask )
            
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:Ntmp]
        Ytmp = self.Y_TMP[:Ntmp]
        Ztmp = self.Z_TMP[:Ntmp]
        Ttmp = self.T_TMP[:Ntmp]
        
        np.compress( self.total_mask, self.X_TMP, out=Xtmp )
        np.compress( self.total_mask, self.Y_TMP, out=Ytmp )
        np.compress( self.total_mask, self.Z_TMP, out=Ztmp )
        np.compress( self.total_mask, self.T_TMP, out=Ttmp )
        
        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        
        ## transform and cut on bounds
        TMP = coordinate_system.get_plotT()
        A = TMP[0] ## need to copy
        B = TMP[1]
        coordinate_system.set_plotT(-np.inf, np.inf)
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False, ignore_T=self._ignore_time, bool_workspace=self.total_mask )
        coordinate_system.set_plotT(A, B)
        
        ## return actual bounds
        if len(plotT) > 0:
            return [np.min(plotT), np.max(plotT)]
        else:
            return [0, 1]
        
        
    def get_all_properties(self):
        ret =  {"marker size":str(self.marker_size),  "color mode":str(self.color_mode), 'name':self.name,
                "X offset":self.X_offset, "Y offset":self.Y_offset,"Z offset":self.Z_offset, 
                "T offset":self.T_offset, 'marker':self.marker}
        
        for name, value in self.min_parameters.items():
            ret['min ' + name] = value
        for name, value in self.max_parameters.items():
            ret['max ' + name] = value
            
        return ret
        
        ## need: marker type, color map
    
    def set_property(self, name, str_value):
        
        try:
            if name == "marker size":
                self.marker_size = int(str_value)
                
            elif name == "color mode":
                self.color_mode = str_value
                    
            elif name == 'name':
                self.name = str_value
                
            elif name == "X offset":
                self.X_offset = float( str_value )
                
            elif name == "Y offset":
                self.Y_offset = float( str_value )
                
            elif name == "Z offset":
                self.Z_offset = float( str_value )
                
            elif name == "T offset":
                self.T_offset = float( str_value )
                
            elif name == "marker":
                self.marker = str_value
                
            elif name[4:] in self.min_parameters:
                self.set_min_param( name[4:], float(str_value) )
                
            elif name[4:] in self.max_parameters:
                self.set_max_param( name[4:], float(str_value) )
                
            else:
                print("do not have property:", name)
        except:
            print("error in setting property", name, str_value)
            pass
    
    def set_min_param(self, name, value):
        self.min_parameters[name] = value
#        self.min_masks[name][:] = self.min_filters[name] > value
        np.greater_equal( self.min_filters[name], value, out= self.min_masks[name])
    
    def set_max_param(self, name, value):
        self.max_parameters[name] = value
#        self.max_masks[name][:] = self.max_filters[name] < value 
        np.greater_equal( value, self.max_filters[name], out= self.max_masks[name])
        
    def set_total_mask(self):
        self.total_mask[:] = True
            
        for name,mask in self.min_masks.items():
            np.logical_and(self.total_mask, mask, out=self.total_mask)
            
        for name, mask in self.max_masks.items():
            np.logical_and(self.total_mask, mask, out=self.total_mask)
        
    
    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system, zorder):
        self.set_total_mask()
        N = np.sum( self.total_mask )    
        
        #### random book keeping ####
        self.clear()
        
    
        if self.color_mode in self.color_options:
            color = self.color_options[ self.color_mode ][self.total_mask] ## should fix this to not make memory
            
    
            
        #### set cuts and transforms
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:N]
        Ytmp = self.Y_TMP[:N]
        Ztmp = self.Z_TMP[:N]
        Ttmp = self.T_TMP[:N]
        
        np.compress( self.total_mask, self.X_TMP, out=Xtmp )
        np.compress( self.total_mask, self.Y_TMP, out=Ytmp )
        np.compress( self.total_mask, self.Z_TMP, out=Ztmp )
        np.compress( self.total_mask, self.T_TMP, out=Ttmp )
        
        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False, ignore_T=self._ignore_time, bool_workspace=self.total_mask )


        print(self.name, "plotting", len(plotX), 'sources. showing:', self.display)
        
        if (not self.display) or len(plotX)==0:
            return
        
        ### get color!
        if self.color_mode == "time":
            color = plotT
            ARG = coordinate_system.get_plotT()
            color_min = ARG[0]
            color_max = ARG[1]
            
        elif self.color_mode[0] == '*':
            color = self.color_mode[1:]
            color_min = None
            color_max = None
            
        elif self.color_mode in self.color_options:
            color = color[self.total_mask[:N]]
            color_min = np.min( color )
            color_max = np.max( color )
            
        else:
            print("bad color mode. This should be interesting!")
            color = self.color_mode
            color_min = None
            color_max = None
        
        
        try:
            if not self._ignore_time:
                self.AltVsT_paths = AltVsT_axes.scatter(x=plotT, y=plotZt, c=color, marker=self.marker, s=self.marker_size, 
                                                        cmap=self.cmap, vmin=color_min, vmax=color_max, zorder=zorder)
                
            self.AltVsEw_paths = AltVsEw_axes.scatter(x=plotX, y=plotZ, c=color, marker=self.marker, s=self.marker_size, 
                                                        cmap=self.cmap, vmin=color_min, vmax=color_max, zorder=zorder)
            
            self.NsVsEw_paths = NsVsEw_axes.scatter(x=plotX, y=plotY, c=color, marker=self.marker, s=self.marker_size, 
                                                        cmap=self.cmap, vmin=color_min, vmax=color_max, zorder=zorder)
            
            self.NsVsAlt_paths = NsVsAlt_axes.scatter(x=plotZ, y=plotY, c=color, marker=self.marker, s=self.marker_size, 
                                                        cmap=self.cmap, vmin=color_min, vmax=color_max, zorder=zorder)
        except Exception as e: print(e)
        

#    def get_viewed_events(self):
#        print("get viewed events not implemented")
#        return []
#        return [PSE for PSE in self.PSE_list if 
#                (self.loc_filter(PSE.PolE_loc) and PSE.PolE_RMS<self.max_RMS and PSE.num_even_antennas>self.min_numAntennas) 
#                or (self.loc_filter(PSE.PolO_loc) and PSE.PolO_RMS<self.max_RMS and PSE.num_odd_antennas>self.min_numAntennas) ]

    
    def clear(self):
        pass
        
    def use_ancillary_axes(self):
        return False
                

    def toggle_on(self):
        self.display = True
    
    def toggle_off(self):
        self.display = False
        self.clear()
        
    def search(self, ID_list):
        
#        outX = np.array([ self.X_array[index] ])
#        outY = np.array([ self.Y_array[index] ])
#        outZ = np.array([ self.Z_array[index] ])
#        outT = np.array([ self.T_array[index] ])
#        min_filter_datums = {name:np.array([ data[index] ]) for name,data in self.min_filters.items()}
#        max_filter_datums = {name:np.array([ data[index] ]) for name,data in self.max_filters.items()}
#        print_datums = {name:np.array([ data[index] ]) for name,data in self.print_data.items()}
#        color_datums = {name:np.array([ data[index] ]) for name,data in self.color_options.items()}
#        indeces = np.array([ID])
        
        N = len(ID_list)
        outX = np.empty(N, dtype=np.double)
        outY = np.empty(N, dtype=np.double)
        outZ = np.empty(N, dtype=np.double)
        outT = np.empty(N, dtype=np.double)
        min_filter_datums = {name:np.empty(N, data.dtype) for name,data in self.min_filters.items()}
        max_filter_datums = {name:np.empty(N, data.dtype) for name,data in self.max_filters.items()}
        print_datums = {name:np.empty(N, data.dtype) for name,data in self.print_data.items()}
        color_datums = {name:np.empty(N, data.dtype) for name,data in self.color_options.items()}
        indeces = np.empty(N, self.source_IDs.dtype)
        
#        print(self.source_IDs)
        
        i = 0
        for ID in ID_list:
            try:
                IDi = int(ID)
            except:
                print("ID", ID, 'is not an int')
                continue
            
            try:
                index = next((idx for idx, val in enumerate(self.source_IDs) if val==IDi))#[0]
            except Exception as e:
                print(e)
                print("ID", ID, 'cannot be found', IDi)
                continue
            
            outX[i] = self.X_array[index]
            outY[i] = self.Y_array[index]
            outZ[i] = self.Z_array[index]
            outT[i] = self.T_array[index]
            
            indeces[i] = ID
            
            for name,data in self.min_filters.items():
                min_filter_datums[name][i] = data[index]
                
            for name,data in self.print_data.items():
                print_datums[name][i] = data[index]
                
            for name,data in self.max_filters.items():
                max_filter_datums[name][i] = data[index]
                
            for name,data in self.color_options.items():
                color_datums[name][i] = data[index]
            
            i += 1
            
        if i==0:
            return None
            
        outX = outX[:i]
        outY = outY[:i]
        outZ = outZ[:i]
        outT = outT[:i]
        
        min_filter_datums = {name:data[:i] for name,data in min_filter_datums.items()}
        max_filter_datums = {name:data[:i] for name,data in max_filter_datums.items()}
        print_datums = {name:data[:i] for name,data in print_datums.items()}
        color_datums = {name:data[:i] for name,data in color_datums.items()}
        
        new_DS = DataSet_generic_PSE( outX, outY, outZ, outT, self.marker, self.marker_size, '*k', 
                 self.name+"_copy", self.cmap, 
                 min_filter_datums, max_filter_datums, print_datums, color_datums, source_IDs=indeces)
        
        new_DS.X_offset = self.X_offset
        new_DS.Y_offset = self.Y_offset
        new_DS.Z_offset = self.Z_offset
        new_DS.T_offset = self.T_offset
        
#        for name,param in self.min_parameters.items():
#            new_DS.set_min_param(name,param)
#        
#        for name,param in self.max_parameters.items():
#            new_DS.set_max_param(name,param)
            
        return new_DS
    
    def get_view_ID_list(self, coordinate_system):
        
        self.set_total_mask()
        
        bool_workspace = np.ones( len(self.total_mask), dtype=bool )
        
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:]
        Ytmp = self.Y_TMP[:]
        Ztmp = self.Z_TMP[:]
        Ttmp = self.T_TMP[:]
        
        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False, ignore_T=self._ignore_time, bool_workspace=bool_workspace )
        
        return [self.source_IDs[i] for i in range( len(self.X_array) ) if self.total_mask[i] and bool_workspace[i] ]
        
    
    def copy_view(self, coordinate_system):
        
        IDS = self.get_view_ID_list( coordinate_system )
        return self.search(IDS)
        
#        self.set_total_mask()
#        
#        bool_workspace = np.ones( len(self.total_mask), dtype=bool )
#        
#        self.X_TMP[:] = self.X_array  
#        self.Y_TMP[:] = self.Y_array  
#        self.Z_TMP[:] = self.Z_array  
#        self.T_TMP[:] = self.T_array 
#        
#        Xtmp = self.X_TMP[:]
#        Ytmp = self.Y_TMP[:]
#        Ztmp = self.Z_TMP[:]
#        Ttmp = self.T_TMP[:]
#        
##        np.compress( self.total_mask, self.X_TMP, out=Xtmp )
##        np.compress( self.total_mask, self.Y_TMP, out=Ytmp )
##        np.compress( self.total_mask, self.Z_TMP, out=Ztmp )
##        np.compress( self.total_mask, self.T_TMP, out=Ttmp )
#        
#        Xtmp += self.X_offset
#        Ytmp += self.Y_offset
#        Ztmp += self.Z_offset
#        Ttmp += self.T_offset
#        
#        if self.transform_memory is None:
#            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
#        coordinate_system.set_workingMemory( self.transform_memory )
#        
#        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter( 
#            Xtmp, Ytmp, Ztmp, Ttmp, 
#            make_copy=False, ignore_T=self._ignore_time, bool_workspace=bool_workspace )
#        
#        
#        outX = []
#        outY = []
#        outZ = []
#        outT = []
#        indeces = []
#        min_filter_datums = {name:[] for name in self.min_filters.keys()}
#        max_filter_datums = {name:[] for name in self.max_filters.keys()}
#        print_datums = {name:[] for name in self.print_data.keys()}
#        color_datums = {name:[] for name in self.color_options.keys()}
#
#        for i in range( len(self.X_array) ):
#            if not (self.total_mask[i] and bool_workspace[i]):
#                continue
#            
#            outX.append( self.X_array[i] )
#            outY.append( self.Y_array[i] )
#            outZ.append( self.Z_array[i] )
#            outT.append( self.T_array[i] )
#            indeces.append( self.source_IDs[i] )
#
#            for name, data in self.min_filters.items():
#                min_filter_datums[name].append( data[i] )
#            for name, data in self.max_filters.items():
#                max_filter_datums[name].append( data[i] )
#            for name, data in self.print_data.items():
#                print_datums[name].append( data[i] )
#            for name, data in self.color_options.items():
#                color_datums[name].append( data[i])
#                
#        if len(outX) == 0:
#            return None
#            
#        
#        outX = np.array( outX )
#        outY = np.array( outY )
#        outZ = np.array( outZ )
#        outT = np.array( outT )
#        indeces = np.array( indeces )
#        min_filter_datums = {name:np.array(data) for name,data in min_filter_datums.items()}
#        max_filter_datums = {name:np.array(data) for name,data in max_filter_datums.items()}
#        print_datums = {name:np.array(data) for name,data in print_datums.items()}
#        color_datums = {name:np.array(data) for name,data in color_datums.items()}
#        
#        new_DS = DataSet_generic_PSE( outX, outY, outZ, outT, self.marker, self.marker_size, '*k', 
#                 self.name+"_copy", self.cmap, 
#                 min_filter_datums, max_filter_datums, print_datums, color_datums, source_IDs=indeces)
#        
#        new_DS.X_offset = self.X_offset
#        new_DS.Y_offset = self.Y_offset
#        new_DS.Z_offset = self.Z_offset
#        new_DS.T_offset = self.T_offset
#        
#        for name,param in self.min_parameters.items():
#            new_DS.set_min_param(name,param)
#        
#        for name,param in self.max_parameters.items():
#            new_DS.set_max_param(name,param)
#            
#        return new_DS

    def print_info(self, coordinate_system):
        self.set_total_mask()
        
        bool_workspace = np.ones( len(self.total_mask), dtype=bool )
        
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:]
        Ytmp = self.Y_TMP[:]
        Ztmp = self.Z_TMP[:]
        Ttmp = self.T_TMP[:]
        
#        np.compress( self.total_mask, self.X_TMP, out=Xtmp )
#        np.compress( self.total_mask, self.Y_TMP, out=Ytmp )
#        np.compress( self.total_mask, self.Z_TMP, out=Ztmp )
#        np.compress( self.total_mask, self.T_TMP, out=Ttmp )
        
        Xtmp += self.X_offset
        Ytmp += self.Y_offset
        Ztmp += self.Z_offset
        Ttmp += self.T_offset
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform_and_filter( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False, ignore_T=self._ignore_time, bool_workspace=bool_workspace )
        
        
        print()
        print()
        N = 0
        for i in range( len(self.X_array) ):
            if not (self.total_mask[i] and bool_workspace[i]):
                continue
            
            print('source:', self.source_IDs[i])
            print("  X:{:.2f} Y:{:.2f} Z:{:.2f} T:{:.10f}:".format( self.X_array[i], self.Y_array[i], self.Z_array[i], self.T_array[i]) )
            j = np.sum(bool_workspace[:i])
            print("  plot coords X, Y, Z, Zt, T", plotX[j], plotY[j], plotZ[j], plotZt[j], plotT[j])
            
            for name, data in self.min_filters.items():
                print("  ",name, data[i])
            for name, data in self.max_filters.items():
                print("  ",name, data[i])
            for name, data in self.print_data.items():
                print("  ",name, data[i])
            print()
            
            N +=1
        print(N, "source")
        
    
        
    
#    def text_output(self):
#        
#        header = self.IPSE_list[0].header
#        for ant_data in header.antenna_data:
#            if ant_data.station == header.prefered_station_name: ## found the "prefered antenna"
#                pref_ant_loc = ant_data.location
#                break
#        
#        with open("output.txt", 'w') as fout:
#            for IPSE in self.IPSE_list:
#                if self.loc_filter( np.append(IPSE.loc, [IPSE.T]) ) and IPSE.intensity>self.min_intensity and IPSE.S1_S2_distance<self.max_S1S2distance  \
#                and IPSE.converged and IPSE.amplitude>self.min_amplitude and IPSE.RMS<self.max_RMS:
#                    ID = IPSE.unique_index
#                    X = IPSE.loc[0]
#                    Y = IPSE.loc[1]
#                    Z = IPSE.loc[2]
#                    T = IPSE.T
#                    I = IPSE.intensity
#                    
#                    R2 = (IPSE.loc[0]-pref_ant_loc[0])**2 + (IPSE.loc[1]-pref_ant_loc[1])**2 + (IPSE.loc[2]-pref_ant_loc[2])**2
#                    power = np.log10(4*np.pi*R2) + 2*np.log10(IPSE.amplitude)
#                    
#                    fout.write( str(ID)+' E '+str(X)+" "+str(Y)+" "+str(Z)+" "+str(T)+" " + str(I)+" "+str(power)+'\n' )
#        print("done writing")
                    
    def ignore_time(self, ignore=None):
        if ignore is not None:
            self._ignore_time = ignore
        return self._ignore_time
    
    
        
#class DataSet_arrow(DataSet_Type):
#    """This represents a set of simple dual-polarized point sources"""
#    
#    def __init__(self, name, X, Y, Z, T, azimuth, zenith, length, color):
#        self.XYZT = np.array([X,Y,Z,T])
#        self.azimuth = azimuth
#        self.zenith = zenith
#        self.length = length
#        self.color = color
#        self.linewidth = 10
#        
#        self.set_arrow()
#        
#        ## probably should call previous constructor here
#        self.name = name
#        self.display = True
#        
#        self.in_bounds = True
#        
#        self.AltVsT_lines = None
#        self.AltVsEw_lines = None
#        self.NsVsEw_lines = None
#        self.NsVsAlt_lines = None
#        
#        
#    def set_show_all(self):
#        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
#        
#        return np.array([ [self.X[0], self.X[1]], [self.Y[0], self.Y[1]], [self.Z[0], self.Z[1]], [self.T[0], self.T[1]] ])
#    
#    
#    def bounding_box(self):
#        return self.set_show_all()
#    
#        
#    def set_T_lims(self, min, max):
#        pass
##        self.in_bounds = self.in_bounds and min <= self.XYZT[3] <= max
#    
#    def set_X_lims(self, min, max):
#        pass
##        self.in_bounds = self.in_bounds and min <= self.XYZT[0] <= max
#    
#    def set_Y_lims(self, min, max):
#        pass
##        self.in_bounds = self.in_bounds and min <= self.XYZT[1] <= max
#    
#    def set_alt_lims(self, min, max):
#        pass
##        self.in_bounds = self.in_bounds and min <= self.XYZT[2] <= max
#    
#    
#    def get_T_lims(self):
#        return list(self.T)
#    
#    def get_X_lims(self):
#        return list(self.X)
#    
#    def get_Y_lims(self):
#        return list(self.Y)
#    
#    def get_alt_lims(self):
#        return list(self.Z)
#    
#    
#    
#    def get_all_properties(self):
#        return {"X":str(self.XYZT[0]), "Y":str(self.XYZT[1]), "Z":str(self.XYZT[2]), "T":str(self.XYZT[3]), 'linewidth':float(self.linewidth),
#                "length":str(self.length), "azimuth":str(self.azimuth), "zenith":str(self.zenith), "color":str(self.color)}
#    
#    def set_property(self, name, str_value):
#        
#        try:
#                
#            if name == "X":
#                try:
#                    self.XYZT[0] = float(str_value)
#                except:
#                    print("input error")
#                    return
#                self.set_arrow()
#                
#            elif name == "Y":
#                try:
#                    self.XYZT[1] = float(str_value)
#                except:
#                    print("input error")
#                    return
#                self.set_arrow()
#                
#            elif name == "Z":
#                try:
#                    self.XYZT[2] = float(str_value)
#                except:
#                    print("input error")
#                    return
#                self.set_arrow()
#                
#            elif name == "T":
#                try:
#                    self.XYZT[3] = float(str_value)
#                except:
#                    print("input error")
#                    return
#                self.set_arrow()
#                
#            elif name == "length":
#                try:
#                    self.length = float(str_value)
#                except:
#                    print("input error")
#                    return
#                self.set_arrow()
#                
#            elif name == "azimuth":
#                try:
#                    self.azimuth = float(str_value)
#                except:
#                    print("input error")
#                    return
#                self.set_arrow()
#                
#            elif name == "zenith":
#                try:
#                    self.zenith = float(str_value)
#                except:
#                    print("input error")
#                    return
#                self.set_arrow()
#                
#            elif name == "color":
#                self.color = str_value
#                
#            elif name == 'linewidth':
#                try:
#                    self.linewidth = float(str_value)
#                except:
#                    print("input error")
#                    return
#                
#            else:
#                print("do not have property:", name)
#        except:
#            print("error in setting property", name, str_value)
#            pass
#        
#    def set_arrow(self):
#        dz = np.cos(self.zenith)*self.length*0.5
#        dx = np.sin(self.zenith)*np.cos(self.azimuth)*self.length*0.5
#        dy = np.sin(self.zenith)*np.sin(self.azimuth)*self.length*0.5
#
#        X = self.XYZT[0]
#        Y = self.XYZT[1]
#        Z = self.XYZT[2]
#        T = self.XYZT[3]
#
#        self.X = np.array([X-dx,X+dx])
#        self.Y = np.array([Y-dy,Y+dy])
#        self.Z = np.array([Z-dz,Z+dz])
#        self.T = np.array([T, T])
#        
#        print('[', self.XYZT[0], ',', self.XYZT[1], ',', self.XYZT[2], ',', self.XYZT[3], ']')
#
#    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system):
#
#    
#        #### random book keeping ####
#        self.clear()
#        
#        if not self.display or not self.in_bounds:
#            return
#
#            
#            
#        #### plot ####
#        X_bar, Y_bar, Z_bar, T_bar = coordinate_system.transform( X=self.X, Y=self.Y, Z=self.Z, T=self.T)
#        
#        self.AltVsT_lines = AltVsT_axes.plot(T_bar, Z_bar, marker='o', ls='-', color=self.color, lw=self.linewidth)
#        self.AltVsEw_lines = AltVsEw_axes.plot(X_bar, Z_bar, marker='o', ls='-', color=self.color, lw=self.linewidth)
#        self.NsVsEw_lines = NsVsEw_axes.plot(X_bar, Y_bar, marker='o', ls='-', color=self.color, lw=self.linewidth)
#        self.NsVsAlt_lines = NsVsAlt_axes.plot(Z_bar, Y_bar, marker='o', ls='-', color=self.color, lw=self.linewidth)
#
#    def get_viewed_events(self):
#        print("get viewed events not implemented")
#        return []
#    
#    def clear(self):
#        pass
##        if self.AltVsT_lines is not None:
##            self.AltVsT_lines = None
##            
##        if self.AltVsEw_lines is not None:
##            self.AltVsEw_lines = None
##            
##        if self.NsVsEw_lines is not None:
##            self.NsVsEw_lines = None
##            
##        if self.NsVsAlt_lines is not None:
##            self.NsVsAlt_lines = None
#            
#            
#    def use_ancillary_axes(self):
#        return False
#
#    def print_info(self):
#        print("not implented")
#                
#    def search(self, ID):
#        print("not implmented")
#        return None
#         
#    
#    def toggle_on(self):
#        self.display = True
#    
#    def toggle_off(self):
#        self.display = False
#        self.clear()
#            
           

class FigureArea(FigureCanvas):
    """This is a widget that contains the central figure"""
    
    ## note that this works in plot coordinates, and that naming is not quite correct
    
#    class previous_view_state(object):
#        def __init__(self, axis_change, axis_data):
#            self.axis_change = axis_change ## 0 is XY, 1 is altitude, and 2 is T
#            self.axis_data = axis_data
    
    class previous_view_state:
        def __init__(self, lims=None):
            self.limits = lims
    
    
    ## ALL spatial units are km, time is in seconds
    def __init__(self, key_press_callback, parent=None, width=5, height=4, dpi=100, axis_label_size=12, axis_font_size=12):
        self.key_press_callback = key_press_callback

        #### some default settings, some can be changed
        
        self.axis_label_size = axis_label_size #15
        self.axis_tick_size = axis_font_size#12
        self.rebalance_XY = True
                
        
        
        self.previous_view_states = []
        self.previous_depth = 100
        
#        self.T_fraction = 1.0 ## if is 0, then do not plot any points. if 0.5, plot points halfway beteween tlims. if 1.0 plot all points
        
        self.data_sets = []
        
        #### setup figure and canvas
        self.fig = Figure(figsize=(width, height), dpi = dpi, constrained_layout=True)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setFocusPolicy( QtCore.Qt.ClickFocus )
        self.setFocus()
        
        #self.fig.subplots_adjust(top=0.97, bottom=0.07)
        
        #### setup axes. We need TWO grid specs to control heights properly

      #  self.top_gs = matplotlib.gridspec.GridSpec(4,1, hspace=0.3, figure=self.fig)
      #  self.middle_N_bottom_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec =self.top_gs[1:], hspace=0.0, wspace=0.05)
       # self.middle_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec =self.middle_N_bottom_gs[0], hspace=0.0, wspace=0.00)
       # self.bottom_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec =self.middle_N_bottom_gs[1:], wspace=0.00)
        
       # self.AltVsT_axes = self.fig.add_subplot(self.top_gs[0])
        
       # self.AltVsEw_axes= self.fig.add_subplot(self.middle_gs[0,:2])
       # self.ancillary_axes = self.fig.add_subplot(self.middle_gs[0,2])
        
       # self.NsVsEw_axes = self.fig.add_subplot(self.bottom_gs[0:,:2], sharex=self.AltVsEw_axes)
       # self.NsVsAlt_axes = self.fig.add_subplot(self.bottom_gs[0:,2], sharey=self.NsVsEw_axes)
        
       # self.AltVsT_axes.tick_params(labelsize = self.axis_tick_size)
        
        
        
        
        self.top_gs = matplotlib.gridspec.GridSpec(4,1, hspace=0.2, figure=self.fig)
        #self.middle_N_bottom_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec =self.top_gs[1:], hspace=0.0, wspace=0.05)
        self.under_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec =self.top_gs[1:], hspace=0.0, wspace=0.0)
        #self.middle_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec =self.middle_N_bottom_gs[0], hspace=0.0, wspace=0.00)
        #self.bottom_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec =self.middle_N_bottom_gs[1:], wspace=0.00)
        
        self.AltVsT_axes = self.fig.add_subplot(self.top_gs[0])
        
        self.AltVsEw_axes= self.fig.add_subplot(self.under_gs[0,:2])
        self.ancillary_axes = self.fig.add_subplot(self.under_gs[0,2])
        
        self.NsVsEw_axes = self.fig.add_subplot(self.under_gs[1:,:2], sharex=self.AltVsEw_axes)
        self.NsVsAlt_axes = self.fig.add_subplot(self.under_gs[1:,2], sharey=self.NsVsEw_axes)
        
        
        
        
        self.AltVsT_axes.tick_params(labelsize = self.axis_tick_size)
        
        self.AltVsEw_axes.tick_params(labelsize = self.axis_tick_size, top=True,right=True, labelbottom=False, direction='in')
        
        self.NsVsEw_axes.tick_params(labelsize = self.axis_tick_size, top=True,right=True, direction='in')
        
        self.NsVsAlt_axes.tick_params(labelsize = self.axis_tick_size, top=True,right=True, labelleft=False, direction='in')
        
        self.ancillary_axes.get_yaxis().set_visible(False)
        self.ancillary_axes.tick_params(labelsize = self.axis_tick_size)
        self.ancillary_axes.set_axis_off()
                   
        
        #### create selectors on plots
        self.mouse_move = False
        
        try:  ## different versions of PyQt have different API
            self.TAlt_selector_rect = RectangleSelector(self.AltVsT_axes, self.TAlt_selector, useblit=False,
                       rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            self.XAlt_selector_rect = RectangleSelector(self.AltVsEw_axes, self.XAlt_selector, useblit=False,
                       rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            self.XY_selector_rect = RectangleSelector(self.NsVsEw_axes, self.XY_selector, useblit=False,
                         rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            self.YAlt_selector_rect = RectangleSelector(self.NsVsAlt_axes, self.AltY_selector, useblit=False,
                       rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
                    
        except:
            self.TAlt_selector_rect = RectangleSelector(self.AltVsT_axes, self.TAlt_selector, useblit=False,
                props=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})

            self.XAlt_selector_rect = RectangleSelector(self.AltVsEw_axes, self.XAlt_selector, useblit=False,
                        props=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            self.XY_selector_rect = RectangleSelector(self.NsVsEw_axes, self.XY_selector, useblit=False,
                          props=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            self.YAlt_selector_rect = RectangleSelector(self.NsVsAlt_axes, self.AltY_selector, useblit=False,
                        props=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
        
        #### create button press/release events
        self.key_press = self.fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        self.key_release = self.fig.canvas.mpl_connect('key_release_event', self.key_release_event)
        self.button_press = self.fig.canvas.mpl_connect('button_press_event', self.button_press_event) ##mouse button
        self.button_release = self.fig.canvas.mpl_connect('button_release_event', self.button_release_event) ##mouse button
        
        ##state variables
        self.z_button_pressed = False
        self.right_mouse_button_location = None
        self.right_button_axis = None
        
        
        bbox = self.NsVsEw_axes.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.window_width = bbox.width
        self.window_height = bbox.width
            
    def set_coordinate_system(self, new_coordinate_system):
        
        self.coordinate_system = new_coordinate_system
        
        self.AltVsT_axes.set_xlabel(self.coordinate_system.t_label, fontsize=self.axis_label_size)
        self.AltVsT_axes.set_ylabel(self.coordinate_system.z_label, fontsize=self.axis_label_size)
        
        self.AltVsEw_axes.set_ylabel(self.coordinate_system.z_label, fontsize=self.axis_label_size)
        
        self.NsVsEw_axes.set_xlabel(self.coordinate_system.x_label, fontsize=self.axis_label_size)
        self.NsVsEw_axes.set_ylabel(self.coordinate_system.y_label, fontsize=self.axis_label_size)
        
        self.NsVsAlt_axes.set_xlabel(self.coordinate_system.z_label, fontsize=self.axis_label_size)
        
    def add_dataset(self, new_dataset):
        self.data_sets.append( new_dataset )
        
        if new_dataset.use_ancillary_axes():
            self.ancillary_axes.set_xlabel(new_dataset.ancillary_label(), fontsize=self.axis_label_size)
            self.ancillary_axes.set_axis_on()

    ## plot contollss
    def set_easting_numTicks(self, num_ticks):
        if num_ticks is None:
            self.NsVsEw_axes.locator_params(axis='x')
        else:
            self.NsVsEw_axes.locator_params(axis='x', nbins=num_ticks)
        self.draw()

    def set_northing_numTicks(self, num_ticks):
        if num_ticks is None:
            self.NsVsEw_axes.locator_params(axis='y')
        else:
            self.NsVsEw_axes.locator_params(axis='y', nbins=num_ticks)
        self.draw()

    def set_time_numTicks(self, num_ticks):
        if num_ticks is None:
            self.AltVsT_axes.locator_params(axis='x')
        else:
            self.AltVsT_axes.locator_params(axis='x', nbins=num_ticks)
        self.draw()

    def set_altVnorth_numTicks(self, num_ticks):
        if num_ticks is None:
            self.NsVsAlt_axes.locator_params(axis='x')
        else:
            self.NsVsAlt_axes.locator_params(axis='x', nbins=num_ticks)
        self.draw()

    def set_altVeast_numTicks(self, num_ticks):
        if num_ticks is None:
            self.AltVsEw_axes.locator_params(axis='y')
        else:
            self.AltVsEw_axes.locator_params(axis='y', nbins=num_ticks)
        self.draw()

    def set_altVtime_numTicks(self, num_ticks):
        if num_ticks is None:
            self.AltVsT_axes.locator_params(axis='y')
        else:
            self.AltVsT_axes.locator_params(axis='y', nbins=num_ticks)
        self.draw()


    def set_font_size(self, axis_label_size=None, axis_font_size=None):
        if axis_label_size is not None:
            self.axis_label_size = axis_label_size
        if axis_font_size is not None:
            self.axis_tick_size = axis_font_size

            self.AltVsT_axes.tick_params(labelsize=self.axis_tick_size)

            self.AltVsEw_axes.tick_params(labelsize=self.axis_tick_size, top=True, right=True, labelbottom=False,
                                          direction='in')
            self.NsVsEw_axes.tick_params(labelsize=self.axis_tick_size, top=True, right=True, direction='in')

            self.NsVsAlt_axes.tick_params(labelsize=self.axis_tick_size, top=True, right=True, labelleft=False,
                                          direction='in')
            self.ancillary_axes.tick_params(labelsize=self.axis_tick_size)

        self.replot_data()
        self.draw()

    #### set limits
        
    def set_T_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.T_limits[0]
        if upper is None:
            upper = self.T_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.coordinate_system.set_plotT(lower, upper)
        
        self.AltVsT_axes.set_xlim([lower, upper])
        
    def set_Z_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.alt_limits[0]
        if upper is None:
            upper = self.alt_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.coordinate_system.set_plotZ(lower, upper)
        
        self.AltVsEw_axes.set_ylim([lower, upper])
        self.NsVsAlt_axes.set_xlim([lower, upper])
#        self.ancillary_axes.set_ylim( self.alt_limits )
        
    def set_Zt_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.alt_limits[0]
        if upper is None:
            upper = self.alt_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.coordinate_system.set_plotZt(lower, upper)
        
        self.AltVsEw_axes.set_ylim([lower, upper])
        self.NsVsAlt_axes.set_xlim([lower, upper])  
        
#    def set_X_lims(self, lower=None, upper=None, extra_percent=0.0):
#        print('set lims depreciated')
#        if lower is None:
#            lower = self.X_limits[0]
#        if upper is None:
#            upper = self.X_limits[1]
#        
#        lower -= (upper-lower)*extra_percent
#        upper += (upper-lower)*extra_percent
#        
#        self.coordinate_system.set_plotX(lower, upper)
#        
#        self.NsVsEw_axes.set_xlim( [lower, upper] )
#        self.AltVsEw_axes.set_xlim( [lower, upper])
#        
#    def set_Y_lims(self, lower=None, upper=None, extra_percent=0.0):
#        print('set lims depreciated')
#        if lower is None:
#            lower = self.Y_limits[0]
#        if upper is None:
#            upper = self.Y_limits[1]
#        
#        lower -= (upper-lower)*extra_percent
#        upper += (upper-lower)*extra_percent
#        
#        self.coordinate_system.set_plotY(lower, upper)
#        
#        self.NsVsAlt_axes.set_ylim( [lower, upper] )
#        self.NsVsEw_axes.set_ylim( [lower, upper] )
        
    def set_just_X_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.X_limits[0]
        if upper is None:
            upper = self.X_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.coordinate_system.set_plotX(lower, upper)
        if self.rebalance_XY: ### if this is true, then we want to scale X and Y axis so that teh aspect ratio 1...I think
            bbox = self.NsVsEw_axes.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.window_width = bbox.width
            self.window_height = bbox.width
        
            self.coordinate_system.rebalance_Y(self.window_width, self.window_height)
            Ylims = self.coordinate_system.get_plotY()
            self.NsVsAlt_axes.set_ylim( Ylims )
            self.NsVsEw_axes.set_ylim( Ylims )
        
        self.NsVsEw_axes.set_xlim( [lower, upper] )
        self.AltVsEw_axes.set_xlim( [lower, upper])
        
    def set_just_Y_lims(self, lower=None, upper=None, extra_percent=0.0):
        if lower is None:
            lower = self.Y_limits[0]
        if upper is None:
            upper = self.Y_limits[1]
        
        lower -= (upper-lower)*extra_percent
        upper += (upper-lower)*extra_percent
        
        self.coordinate_system.set_plotY(lower, upper)
        if self.rebalance_XY: ### if this is true, then we want to scale X and Y axis so that teh aspect ratio 1...I think
            bbox = self.NsVsEw_axes.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.window_width = bbox.width
            self.window_height = bbox.width
            
            self.coordinate_system.rebalance_X(self.window_width, self.window_height)
            Xlims = self.coordinate_system.get_plotX()
            self.NsVsEw_axes.set_xlim( Xlims )
            self.AltVsEw_axes.set_xlim( Xlims)
        
        self.NsVsAlt_axes.set_ylim( [lower, upper] )
        self.NsVsEw_axes.set_ylim( [lower, upper] )
        
    def set_XY_lims(self, lowerX=None, upperX=None, lowerY=None, upperY=None,extra_percent=0.0):
        if lowerX is None:
            lowerX = self.X_limits[0]
        if upperX is None:
            upperX = self.X_limits[1]
            
        if lowerY is None:
            lowerY = self.Y_limits[0]
        if upperY is None:
            upperY = self.Y_limits[1]
        
        lowerX -= (upperX-lowerX)*extra_percent
        upperX += (upperX-lowerX)*extra_percent
        
        lowerY -= (upperY-lowerY)*extra_percent
        upperY += (upperY-lowerY)*extra_percent
        
        self.coordinate_system.set_plotX(lowerX, upperX)
        self.coordinate_system.set_plotY(lowerY, upperY)
        
        if self.rebalance_XY: ### if this is true, then we want to scale X and Y axis so that teh aspect ratio 1...I think
            
            bbox = self.NsVsEw_axes.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.window_width = bbox.width
            self.window_height = bbox.width
            
            self.coordinate_system.rebalance_XY(self.window_width, self.window_height)
            lowerX, upperX = self.coordinate_system.get_plotX()
            lowerY, upperY = self.coordinate_system.get_plotY()
        
        self.NsVsEw_axes.set_xlim( [lowerX, upperX] )
        self.AltVsEw_axes.set_xlim( [lowerX, upperX])
        
        self.NsVsAlt_axes.set_ylim( [lowerY, upperY] )
        self.NsVsEw_axes.set_ylim( [lowerY, upperY] )
        
    def replot_data(self):
        
        ### quick hacks ###
        
        self.AltVsT_axes.cla()
        self.AltVsEw_axes.cla()
        self.NsVsEw_axes.cla()
        self.NsVsAlt_axes.cla()
        self.ancillary_axes.cla()
        
        self.ancillary_axes.set_axis_off() ## probably should test if this needs to be on at some point
        
        
        self.AltVsT_axes.set_xlabel(self.coordinate_system.t_label, fontsize=self.axis_label_size)
        self.AltVsT_axes.set_ylabel(self.coordinate_system.z_label, fontsize=self.axis_label_size)
        
        self.AltVsEw_axes.set_ylabel(self.coordinate_system.zt_label, fontsize=self.axis_label_size)
        
        self.NsVsEw_axes.set_xlabel(self.coordinate_system.x_label, fontsize=self.axis_label_size)
        self.NsVsEw_axes.set_ylabel(self.coordinate_system.y_label, fontsize=self.axis_label_size)
        
        self.NsVsAlt_axes.set_xlabel(self.coordinate_system.zt_label, fontsize=self.axis_label_size)
        
        
        #### create button press/release events
        self.key_press = self.fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        self.key_release = self.fig.canvas.mpl_connect('key_release_event', self.key_release_event)
        self.button_press = self.fig.canvas.mpl_connect('button_press_event', self.button_press_event) ##mouse button
        self.button_release = self.fig.canvas.mpl_connect('button_release_event', self.button_release_event) ##mouse button
        
        print()
        current_z_order = 0
        for DS in self.data_sets:

            try:
                DS.plot( self.AltVsT_axes, self.AltVsEw_axes, self.NsVsEw_axes, self.NsVsAlt_axes, self.ancillary_axes, self.coordinate_system, zorder=current_z_order )
            except Exception as e:
                print('error for DS', DS.name, type(DS) )
                print(e)

            current_z_order += 10
                
            
        #### set limits ####
        X, Y, Z, Zt, T = self.coordinate_system.get_limits_plotCoords()
        self.NsVsEw_axes.set_xlim( X )
        self.AltVsEw_axes.set_xlim( X )
        
        self.NsVsAlt_axes.set_ylim( Y )
        self.NsVsEw_axes.set_ylim( Y )
        
        self.AltVsT_axes.set_ylim( Zt )
        
        self.AltVsEw_axes.set_ylim( Z )
        self.NsVsAlt_axes.set_xlim( Z )
#        self.ancillary_axes.set_ylim( self.alt_limits )
        
        self.AltVsT_axes.set_xlim( T )
        
        
        
        #### create selectors on plots
#        self.TAlt_selector_rect = None
        try:
            self.TAlt_selector_rect = RectangleSelector(self.AltVsT_axes, self.TAlt_selector, useblit=False,
                        rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            self.XAlt_selector_rect = RectangleSelector(self.AltVsEw_axes, self.XAlt_selector, useblit=False,
                        rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            self.XY_selector_rect = RectangleSelector(self.NsVsEw_axes, self.XY_selector, useblit=False,
                          rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            self.YAlt_selector_rect = RectangleSelector(self.NsVsAlt_axes, self.AltY_selector, useblit=False,
                        rectprops=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
        except:
            self.TAlt_selector_rect = RectangleSelector(self.AltVsT_axes, self.TAlt_selector, useblit=False,
                        props=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            self.XAlt_selector_rect = RectangleSelector(self.AltVsEw_axes, self.XAlt_selector, useblit=False,
                        props=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            self.XY_selector_rect = RectangleSelector(self.NsVsEw_axes, self.XY_selector, useblit=False,
                          props=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})
            
            self.YAlt_selector_rect = RectangleSelector(self.NsVsAlt_axes, self.AltY_selector, useblit=False,
                        props=dict(alpha=0.5, facecolor='red'), button=1, state_modifier_keys={'move':'', 'clear':'', 'square':'', 'center':''})

            
            
        
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
        
        self.previous_view_states.append( self.previous_view_state( self.coordinate_system.get_limits_plotCoords() ) )
        
        if len(self.previous_view_states) > self.previous_depth:
            N = len(self.previous_view_states) - self.previous_depth
            self.previous_view_states = self.previous_view_states[N:]
        
        if self.z_button_pressed: ## then we zoom out,
            Ztlims = self.coordinate_system.get_plotZt()
            Tlims = self.coordinate_system.get_plotT()
            
            minT = 2.0*Tlims[0] - minT
            maxT = 2.0*Tlims[1] - maxT
            
            minAlt = 2.0*Ztlims[0] - minAlt
            maxAlt = 2.0*Ztlims[1] - maxAlt
            
        self.set_T_lims(minT, maxT)
        self.set_Zt_lims(minAlt, maxAlt)
        
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
        
        self.previous_view_states.append( self.previous_view_state( self.coordinate_system.get_limits_plotCoords() ) )
        
        if len(self.previous_view_states) > self.previous_depth:
            N = len(self.previous_view_states) - self.previous_depth
            self.previous_view_states = self.previous_view_states[N:]
            
        if self.z_button_pressed:
            Xlims = self.coordinate_system.get_plotX()
            Zlims = self.coordinate_system.get_plotZ()
            
            minA = 2.0*Zlims[0] - minA
            maxA = 2.0*Zlims[1] - maxA
            
            minX = 2.0*Xlims[0] - minX
            maxX = 2.0*Xlims[1] - maxX
        
        self.set_Z_lims(minA, maxA)
        self.set_just_X_lims(minX, maxX)
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
    
        self.previous_view_states.append( self.previous_view_state( self.coordinate_system.get_limits_plotCoords() ) )
    
        if len(self.previous_view_states) > self.previous_depth:
            N = len(self.previous_view_states) - self.previous_depth
            self.previous_view_states = self.previous_view_states[N:]
            
        if self.z_button_pressed:
            Ylims = self.coordinate_system.get_plotY()
            Zlims = self.coordinate_system.get_plotZ()
            
            minA = 2.0*Zlims[0] - minA
            maxA = 2.0*Zlims[1] - maxA
            
            minY = 2.0*Ylims[0] - minY
            maxY = 2.0*Ylims[1] - maxY
        
        self.set_Z_lims(minA, maxA)
        self.set_just_Y_lims(minY, maxY)
        self.replot_data()
        self.draw()
    
    def XY_selector(self, eclick, erelease):
        
        minX = min(eclick.xdata, erelease.xdata)
        maxX = max(eclick.xdata, erelease.xdata)
        minY = min(eclick.ydata, erelease.ydata)
        maxY = max(eclick.ydata, erelease.ydata)
        
        if minX==maxX or minY==maxY:
            self.mouse_move = False

        else:
            self.mouse_move = True
            
            self.previous_view_states.append( self.previous_view_state( self.coordinate_system.get_limits_plotCoords() ) )
            
            if len(self.previous_view_states) > self.previous_depth:
                N = len(self.previous_view_states) - self.previous_depth
                self.previous_view_states = self.previous_view_states[N:]
            
            if self.z_button_pressed:
                Xlims = self.coordinate_system.get_plotX()
                Ylims = self.coordinate_system.get_plotY()
               
                minX = 2.0*Xlims[0] - minX
                maxX = 2.0*Xlims[1] - maxX
                
                minY = 2.0*Ylims[0] - minY
                maxY = 2.0*Ylims[1] - maxY
                
            self.set_XY_lims(minX, maxX, minY, maxY)
            
            self.replot_data()
            self.draw()
            
        
    def key_press_event(self, event):
        self.key_press_callback( event )
        
#        print "key press:", event.key
        if event.key == 'z':
            self.z_button_pressed = True
        elif event.key == 'c':
            print(event.inaxes)
            print(event.xdata)
            print(event.ydata)
        
    def key_release_event(self, event):
#        print "key release:", event.key
        if event.key == 'z':
            self.z_button_pressed = False
            
    def button_press_event(self, event):
#        print "mouse pressed:", event
        
        if event.button == 2 and len(self.previous_view_states)>0: ##middle mouse, back up
            previous_state = self.previous_view_states.pop(-1)
            X, Y, Z, Zt, T = previous_state.limits
                
            self.set_XY_lims(X[0], X[1], Y[0], Y[1])
            self.set_Z_lims(Z[0], Z[1])
            self.set_Zt_lims(Zt[0], Zt[1])
            self.set_T_lims(T[0], T[1])
                
            self.replot_data()
            self.draw()
            
        elif event.button == 3: ##right mouse, record location for drag
            self.right_mouse_button_location = [event.xdata, event.ydata]
            self.right_button_axis = event.inaxes
            
    def button_release_event(self, event):
        if event.button == 1:
            pass
#            if len(self.data_sets)>0 and self.data_sets[0].name == "arrow" and not self.mouse_move:
#                print("mouse moved:", self.mouse_move)
#                C_X, C_Y, C_Z, C_T = self.coordinate_system.transform( [self.data_sets[0].XYZT[0]], [self.data_sets[0].XYZT[1]], [self.data_sets[0].XYZT[2]], [self.data_sets[0].XYZT[3]] )
#                if event.inaxes is self.AltVsT_axes:
#                    C_T[0] = event.xdata
#                    C_Z[0] = event.ydata
#                elif event.inaxes is self.NsVsEw_axes:
#                    C_X[0] = event.xdata
#                    C_Y[0] = event.ydata
#                elif event.inaxes is self.AltVsEw_axes:
#                    C_X[0] = event.xdata
#                    C_Z[0] = event.ydata
#                elif event.inaxes is self.NsVsAlt_axes:
#                    C_Z[0] = event.xdata
#                    C_Y[0] = event.ydata
#                    
#                minX, minY, minZ, minT = self.coordinate_system.invert( C_X, C_Y, C_Z, C_T )
#                self.data_sets[0].XYZT[0] = minX[0]
#                self.data_sets[0].XYZT[1] = minY[0]
#                self.data_sets[0].XYZT[2] = minZ[0]
#                self.data_sets[0].XYZT[3] = minT[0]
#                self.data_sets[0].set_arrow()
                
#                self.replot_data()
#                self.draw()
                
        elif event.button == 3: ##drag
            if event.inaxes != self.right_button_axis: return
        
            deltaX = self.right_mouse_button_location[0] - event.xdata
            deltaY = self.right_mouse_button_location[1] - event.ydata
            
            lims = self.coordinate_system.get_limits_plotCoords()
            self.previous_view_states.append( self.previous_view_state( lims ) )
            Xlims, Ylims, Zlims, Ztlims, Tlims = lims
            
            if event.inaxes is self.AltVsT_axes:
                
                self.set_T_lims( Tlims[0] + deltaX, Tlims[1] + deltaX)
                self.set_Zt_lims( Ztlims[0] + deltaY, Ztlims[1] + deltaY)
                
            elif event.inaxes is self.AltVsEw_axes:
                self.set_just_X_lims( Xlims[0] + deltaX, Xlims[1] + deltaX)
                self.set_Z_lims( Zlims[0] + deltaY, Zlims[1] + deltaY)
                
            elif event.inaxes is self.NsVsEw_axes:
                self.set_XY_lims( Xlims[0] + deltaX, Xlims[1] + deltaX,  Ylims[0] + deltaY, Ylims[1] + deltaY)
                
            elif event.inaxes is self.NsVsAlt_axes:
                self.set_Z_lims( Zlims[0] + deltaX, Zlims[1] + deltaX)
                self.set_just_Y_lims( Ylims[0] + deltaY, Ylims[1] + deltaY)
                
            
            if len(self.previous_view_states) > self.previous_depth:
                N = len(self.previous_view_states) - self.previous_depth
                self.previous_view_states = self.previous_view_states[N:]
                
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
    
    def __init__(self, coordinate_system_s, width_height=None):
        
        self.qApp = QtWidgets.QApplication(sys.argv)
        
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Groninger Lightning Plotter")

        if width_height is None:
            H = QtWidgets.QDesktopWidget().screenGeometry(-1).height()
            width_height = [H*0.99, H*0.85]

        self.setGeometry(0, 0, int(width_height[0]), int(width_height[1]))
        
#        self.statusBar().showMessage("All hail matplotlib!", 2000) ##this shows messages on bottom left of window

        try:
            self.coordinate_systems = [cs for cs in coordinate_system_s]
        except:
            self.coordinate_systems = [ coordinate_system_s ]
#        self.current_coordinate_system = 0
            
#        self.coordinate_system = coordinate_system


        self.plot_save_location = "./plot_save"

        #### menu bar ###
     ##file
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.file_menu.addAction('&Save Plot PNG', self.savePlot)
        #self.menuBar().addMenu(self.file_menu)

        self.file_menu.addAction('&Save Plot SVG', self.savePlotSvg)
        #self.menuBar().addMenu(self.file_menu)

        self.file_menu.addAction('&Save Plot PDF', self.savePlotPdf)
        # self.menuBar().addMenu(self.file_menu)


        self.file_menu.addAction('&set plot save location', self.setPlotLocation_clicked)
        self.file_menu.addAction('&get plot save location', self.getPlotLocation_clicked)

        
        
   ##plot settings
        self.plot_settings_menu = QtWidgets.QMenu('&Plot Settings', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.plot_settings_menu)
     # ticks
        self.plot_settings_menu.addAction('&set easting num ticks', self.setEastingNumTicks)
        self.plot_settings_menu.addAction('&set northing num ticks', self.setNorthingNumTicks)
        self.plot_settings_menu.addAction('&set time num ticks', self.setTimeNumTicks)
        self.plot_settings_menu.addAction('&set alt(vsNorth) num ticks', self.setAltvsNorthNumTicks)
        self.plot_settings_menu.addAction('&set alt(vsEast) num ticks', self.setAltvsEastNumTicks)
        self.plot_settings_menu.addAction('&set alt(vsTime) num ticks', self.setAltvsTimeNumTicks)

        self.plot_settings_menu.addAction('&set label font size', self.set_labelSize_callback)
        self.plot_settings_menu.addAction('&set tick font size', self.set_tickFontSize_callback)

        
        ## analysis
        self.analysis_menu = QtWidgets.QMenu('&Analysis', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.analysis_menu)
        self.analysis_menu.addAction('&Print Info', self.printInfo)
        self.analysis_menu.addAction('&Copy View', self.copy_view)
        self.analysis_menu.addAction('&Output Text', self.textOutput)
        self.analysis_menu.addAction('&pickle IDs to clipboard', self.set_pickledIDs_to_clipboard)
        self.analysis_menu.addAction('&pickle IDs from clipboard', self.get_pickledIDs_from_clipboard)
        self.analysis_menu.addAction('&new linear spline', self.new_LinearSpline)
        self.next_linear_index = 0
        
        ## new actions added through add_analysis
        
        
        ## coordinate system
        self.coordinate_system_menu = QtWidgets.QMenu('&Coordinate Systems', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.coordinate_system_menu)
        self.coord_system_actions = []
        for i,CS in enumerate(self.coordinate_systems):
            ac = self.coordinate_system_menu.addAction(
                    '&'+CS.name, lambda j=i: self.set_coordinate_system(j) ) ## dfined strange so that i will be copied, not taken as referance
            
            self.coord_system_actions.append( ac )

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
        self.figure_space = FigureArea(self.key_press_event, self.main_widget)
#        self.figure_space.set_coordinate_system( coordinate_system )
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
        
        ## move data set up
        self.DSup_button = QtWidgets.QPushButton(self)
        self.DSup_button.move(250, 80)
        self.DSup_button.setText("U")
        self.DSup_button.resize(20,25)
        self.DSup_button.clicked.connect(self.DSupButtonPressed)
        
        ## move data set down
        self.DSdown_button = QtWidgets.QPushButton(self)
        self.DSdown_button.move(280, 80)
        self.DSdown_button.setText("D")
        self.DSdown_button.resize(20,25)
        self.DSdown_button.clicked.connect(self.DSdownButtonPressed)
        
        
        
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
        
        ##ignore time
        self.ignoreTimeCheckBox = QtWidgets.QCheckBox("ignore time", self)
        self.ignoreTimeCheckBox.stateChanged.connect( self.clickIgnoreTime )
        self.ignoreTimeCheckBox.move(5, 215)
        
        
        ##set button
        self.showallTime_Button = QtWidgets.QPushButton(self)
        self.showallTime_Button.move(125, 215)
        self.showallTime_Button.setText("show all time")
        self.showallTime_Button.resize(150,25)
        self.showallTime_Button.clicked.connect( self.showAllTimePressed )
        


        #### set and get position ####
        
        ## X label
        self.XLabel = QtWidgets.QLabel(self)
        self.XLabel.move(5, 250)
        self.XLabel.resize(30,25)
#        self.XLabel.setText( coordinate_system.x_display_label )
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
#        self.YLabel.setText( coordinate_system.y_display_label )
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
#        self.ZLabel.setText( coordinate_system.z_display_label )
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
#        self.TLabel.setText( coordinate_system.t_display_label )
        ## T min box
        self.Tmin_txtBox = QtWidgets.QLineEdit(self)
        self.Tmin_txtBox.move(40, 340)
        self.Tmin_txtBox.resize(80,25)
        ## T max box
        self.Tmax_txtBox = QtWidgets.QLineEdit(self)
        self.Tmax_txtBox.move(140, 340)
        self.Tmax_txtBox.resize(80,25)
        
        ##clip board
        self.toCB_button = QtWidgets.QPushButton(self)
        self.toCB_button.move(230, 265)
        self.toCB_button.setText("to CB")
        self.toCB_button.resize(75,25)
        self.toCB_button.clicked.connect(self.boundsToClipboard)
        
        self.fromCB_button = QtWidgets.QPushButton(self)
        self.fromCB_button.move(230, 325)
        self.fromCB_button.setText("from CB")
        self.fromCB_button.resize(75,25)
        self.fromCB_button.clicked.connect(self.boundsFromClipboard)
        
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
        
        
        ## zoom
        self.zoom_button = QtWidgets.QPushButton(self)
        self.zoom_button.move(5, 470)
        self.zoom_button.setText("zoom: in")
        self.zoom_button.resize(150,25)
        self.zoom_button.clicked.connect(self.zoomButtonPressed)
        
        
        ##aspect ratio
        self.aspectRatioCheckBox = QtWidgets.QCheckBox("1:1 aspect ratio", self)
        self.aspectRatioCheckBox.stateChanged.connect( self.clickAspectRatio )
        self.aspectRatioCheckBox.move(5, 520)
        self.aspectRatioCheckBox.setChecked( self.figure_space.rebalance_XY )
        self.aspectRatioCheckBox.resize(200,25)
        

        self.set_coordinate_system( 0 )
        
        ##TODO:
        #animation
        #custom analysis
	# save state in some way
        
        self.setWindowTitle("LOFAR-LIM data viewer")
        
        
        
    def add_analysis(self, name, func):
        """ call this, before finishing plotting calls, to add an analysis to tne analysis menu. 
        name must be a string.
        func must be a function that takes a coordinate system as first argument and dataset as second argument"""

        def noErrorDS():
            if self.current_data_set is None:
                return None
            else:
                return self.figure_space.data_sets[ self.current_data_set ]


        self.analysis_menu.addAction('&'+name, lambda: func( self.figure_space.coordinate_system,  noErrorDS()  ) )


        
        #### keyboard and mouse call backs ####
    def key_press_event(self, event):
        if self.current_data_set != -1:
            DS = self.figure_space.data_sets[ self.current_data_set ]
            DS.key_press_event(event)
        
        
        
            #### menu bar call backs ####

    def set_plotSaveLocation(self, location):
        """method can be called externally to set where plots are saved"""
        self.plot_save_location = location


    ## file
    def fileQuit(self):
        self.close()
        
    def savePlot(self):
        output_fname = self.plot_save_location+".png"
        self.figure_space.fig.savefig(output_fname, format='png')
        
    def savePlotSvg(self):
        output_fname = self.plot_save_location+".svg"
        self.figure_space.fig.savefig(output_fname, format='svg')
        
    def savePlotPdf(self):
        output_fname = self.plot_save_location+".pdf"
        self.figure_space.fig.savefig(output_fname, format='pdf')
        
    def set_coordinate_system(self, index):
        CS = self.coordinate_systems[index]
#        CS_ac = self.coord_system_actions[index]
        self.figure_space.set_coordinate_system( CS )
        
        self.XLabel.setText( CS.x_display_label )
        self.YLabel.setText( CS.y_display_label )
        self.ZLabel.setText( CS.z_display_label )
        self.TLabel.setText( CS.t_display_label )
        
#        self.coordinate_system_menu.setActiveAction( CS_ac )
        
        self.position_get()
        self.figure_space.replot_data()
        self.figure_space.draw()



    def setPlotLocation_clicked(self):
        inputTXT = self.variable_txtBox.text().strip()
        print('inputed', inputTXT)
        if len(inputTXT) == 0:
            print('NO LOCATION INPUTED')
        else:
            self.set_plotSaveLocation( inputTXT )



    def getPlotLocation_clicked(self):
        print('PSL:', self.plot_save_location)
        self.variable_txtBox.setText( self.plot_save_location )



    ### plot settings
    def __set_ticks_(self, func):
        inputTXT = self.variable_txtBox.text().strip()
        if inputTXT.strip() == "none":
            func( None )
        else:
            try:
                num = int(inputTXT)
            except:
                print('input not number!')
            else:
                func( num )

    def setEastingNumTicks(self):
        self.__set_ticks_( self.figure_space.set_easting_numTicks )

    def setNorthingNumTicks(self):
        self.__set_ticks_( self.figure_space.set_northing_numTicks )

    def setTimeNumTicks(self):
        self.__set_ticks_( self.figure_space.set_time_numTicks )

    def setAltvsNorthNumTicks(self):
        self.__set_ticks_( self.figure_space.set_altVnorth_numTicks )

    def setAltvsEastNumTicks(self):
        self.__set_ticks_( self.figure_space.set_altVeast_numTicks )

    def setAltvsTimeNumTicks(self):
        self.__set_ticks_( self.figure_space.set_altVtime_numTicks )



    def set_font_size(self, axis_label_size=None, axis_font_size=None):
        """set font size. Can be called externally"""
        self.figure_space.set_font_size( axis_label_size, axis_font_size)


    def set_labelSize_callback(self):
        inputTXT = self.variable_txtBox.text().strip()
        try:
            num = float(inputTXT)
        except:
            print('input not number!')
        else:
            self.figure_space.set_font_size( axis_label_size=num, axis_font_size=None)


    def set_tickFontSize_callback(self):
        inputTXT = self.variable_txtBox.text().strip()
        try:
            num = float(inputTXT)
        except:
            print('input not number!')
        else:
            self.figure_space.set_font_size( axis_label_size=None, axis_font_size=num)


    ##help
    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """Bla Bla Bla""")

    def closeEvent(self, ce):
        self.fileQuit()
        
        
    ## analysis
    
    def new_LinearSpline(self):
        
        name = "LineSpline_"+str(self.next_linear_index)
        color = 'C'+str(self.next_linear_index)
        
        self.add_dataset( linearSpline_DataSet(name, color) )
        self.next_linear_index += 1
        
    def printInfo(self):
        DS = self.figure_space.data_sets[ self.current_data_set ]
        DS.print_info( self.figure_space.coordinate_system )
        
    def copy_view(self):
        
        if self.current_data_set is not None and self.current_data_set != -1: ## is this needed?
            ret = self.figure_space.data_sets[ self.current_data_set ].copy_view( self.figure_space.coordinate_system )
            
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
        
    def set_pickledIDs_to_clipboard(self):
        IDs = self.figure_space.data_sets[ self.current_data_set ].get_view_ID_list( self.figure_space.coordinate_system )
        
        if IDs is not None:
            pickled_IDs = dumps(IDs)
            QApplication.clipboard().setText( pickled_IDs.hex() )
            
    
    def get_pickledIDs_from_clipboard(self):
        newDS = None
        try:
            IDs = loads( bytes.fromhex( QApplication.clipboard().text() ) )
            newDS = self.figure_space.data_sets[ self.current_data_set ].search( IDs )
        
        except:
            print('cannot load ids')
            return
        
        if newDS is not None:
            print("adding dataset")
            self.add_dataset( newDS )
            self.figure_space.replot_data()
            self.figure_space.draw()
        
        
        
        
    #### add data sets ####
    
    def add_dataset(self, new_dataset):
        self.figure_space.add_dataset( new_dataset )
        self.refresh_analysis_list( self.current_data_set )
        
        
    #### control choosing data set and variables ####
    def refresh_analysis_list(self, choose=None):
        self.DS_drop_list.clear()
        self.DS_drop_list.addItems( [DS.name for DS in self.figure_space.data_sets] )
        if len(self.figure_space.data_sets) > 0:
            if choose is None or choose<0:
                choose=0
            self.DS_drop_list_choose( choose )
        
    def DS_drop_list_choose(self, data_set_index):
        self.current_data_set = data_set_index
        self.refresh_DSvariable_list()
        
        if (self.current_data_set is not None) and (self.current_data_set!=-1) and (self.current_data_set<len(self.figure_space.data_sets)):
            self.DS_drop_list.setCurrentIndex( self.current_data_set )
            
            DS = self.figure_space.data_sets[ self.current_data_set ]
            self.ignoreTimeCheckBox.setChecked( DS.ignore_time() )
            
            self.DSvariable_get()
            self.toggleColorSet()
#            self.DSposition_get()
        
    
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
                
    def position_get(self):
        display_limits = self.figure_space.coordinate_system.get_displayLimits()
            
        self.Xmin_txtBox.setText( str(display_limits[0][0]) )
        self.Xmax_txtBox.setText( str(display_limits[0][1]) )
        
        self.Ymin_txtBox.setText( str(display_limits[1][0]) )
        self.Ymax_txtBox.setText( str(display_limits[1][1]) )
        
        self.Zmin_txtBox.setText( str(display_limits[2][0]) )
        self.Zmax_txtBox.setText( str(display_limits[2][1]) )
        
        self.Tmin_txtBox.setText( str(display_limits[3][0]) )
        self.Tmax_txtBox.setText( str(display_limits[3][1]) )
        
#        if self.current_data_set != -1:
#            DS = self.figure_space.data_sets[ self.current_data_set ]
#            tmin, tmax = DS.get_T_lims()
#            xmin, xmax = DS.get_X_lims()
#            ymin, ymax = DS.get_Y_lims()
#            zmin, zmax = DS.get_alt_lims()
#            
#            self.Xmin_txtBox.setText( str(xmin) )
#            self.Xmax_txtBox.setText( str(xmax) )
#            
#            self.Ymin_txtBox.setText( str(ymin) )
#            self.Ymax_txtBox.setText( str(ymax) )
#            
#            self.Zmin_txtBox.setText( str(zmin) )
#            self.Zmax_txtBox.setText( str(zmax) )
#            
#            self.Tmin_txtBox.setText( str(tmin) )
#            self.Tmax_txtBox.setText( str(tmax) )
        
    
    
    #### buttons ####
    def showAllButtonPressed(self):
        
#        X_bounds, Y_bounds, Z_bounds, T_bounds = DS.set_show_all()
#        X_bounds, Y_bounds, Z_bounds, T_bounds = self.figure_space.coordinate_system.transform( X_bounds, Y_bounds, Z_bounds, T_bounds )
#        
#        self.figure_space.set_X_lims( X_bounds[0], X_bounds[1], 0.05 )
#        self.figure_space.set_Y_lims( Y_bounds[0], Y_bounds[1], 0.05 )
#        self.figure_space.set_alt_lims( Z_bounds[0], Z_bounds[1], 0.05 )
#        self.figure_space.set_T_lims( T_bounds[0], T_bounds[1], 0.05 )
        
        
        DS = self.figure_space.data_sets[ self.current_data_set ]
        
        DS.set_show_all( self.figure_space.coordinate_system )
        
        
        X_bounds = self.figure_space.coordinate_system.get_plotX()
        Y_bounds = self.figure_space.coordinate_system.get_plotY()
        self.figure_space.set_XY_lims( X_bounds[0], X_bounds[1], Y_bounds[0], Y_bounds[1], 0.05 )
        Z_bounds = self.figure_space.coordinate_system.get_plotZ()
        self.figure_space.set_Z_lims( Z_bounds[0], Z_bounds[1], 0.05 )
        Zt_bounds = self.figure_space.coordinate_system.get_plotZt()
        self.figure_space.set_Zt_lims( Zt_bounds[0], Zt_bounds[1], 0.05 )
        T_bounds = self.figure_space.coordinate_system.get_plotT()
        self.figure_space.set_T_lims( T_bounds[0], T_bounds[1], 0.05 )
        
        self.DSvariable_get()
        self.position_get()
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def toggleButtonPressed(self):
        DS = self.figure_space.data_sets[ self.current_data_set ]
        if DS.display:
            DS.toggle_off()
        else:
            DS.toggle_on()
        
        self.toggleColorSet()
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def toggleColorSet(self):
        DS = self.figure_space.data_sets[ self.current_data_set ]
        if DS.display:
            self.toggleDS_button.setStyleSheet("background-color:green;");
        else:
            self.toggleDS_button.setStyleSheet("background-color:red;");
        
    def deleteDSButtonPressed(self):
        if self.current_data_set is not None and self.current_data_set != -1: ## is this needed?
            self.figure_space.data_sets.pop( self.current_data_set )
            self.refresh_analysis_list()
            self.figure_space.replot_data()
            self.figure_space.draw()
            
            
    def DSupButtonPressed(self):
        DSL = self.figure_space.data_sets
        
        if self.current_data_set > 0:
            DSL.insert(self.current_data_set-1, DSL.pop(self.current_data_set))
            
            self.refresh_analysis_list()
            self.DS_drop_list_choose(self.current_data_set-1)
            
            self.figure_space.replot_data()
            self.figure_space.draw()
            
            
    def DSdownButtonPressed(self):
        DSL = self.figure_space.data_sets
        
        if self.current_data_set != -1 and self.current_data_set< len(DSL)-1:
            DSL.insert(self.current_data_set+1, DSL.pop(self.current_data_set))
            
            self.refresh_analysis_list()
            self.DS_drop_list_choose(self.current_data_set+1)
            
            self.figure_space.replot_data()
            self.figure_space.draw()
            
            
    def setButtonPressed(self):
        inputTXT = self.variable_txtBox.text()
        
        DS = self.figure_space.data_sets[ self.current_data_set ]
        DS.set_property(self.current_variable_name, inputTXT)
        
        self.refresh_analysis_list( choose=self.current_data_set)
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def getButtonPressed(self):
        self.DSvariable_get()
        self.position_get()
        
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
        
        self.figure_space.coordinate_system.set_displayLimits( [xmin,xmax], [ymin,ymax], [zmin,zmax], [tmin,tmax] )
        
        X_bounds = self.figure_space.coordinate_system.get_plotX()
        Y_bounds = self.figure_space.coordinate_system.get_plotY()
        self.figure_space.set_XY_lims( X_bounds[0], X_bounds[1], Y_bounds[0], Y_bounds[1])
        Z_bounds = self.figure_space.coordinate_system.get_plotZ()
        self.figure_space.set_Z_lims( Z_bounds[0], Z_bounds[1])
        Zt_bounds = self.figure_space.coordinate_system.get_plotZt()
        self.figure_space.set_Zt_lims( Zt_bounds[0], Zt_bounds[1])
        T_bounds = self.figure_space.coordinate_system.get_plotT()
        self.figure_space.set_T_lims( T_bounds[0], T_bounds[1])
        
#        X_bounds, Y_bounds, Z_bounds, T_bounds = self.figure_space.coordinate_system.transform( 
#                np.array([xmin,xmax]), np.array([ymin,ymax]), np.array([zmin, zmax]), np.array([tmin,tmax]) )
        
#        self.figure_space.set_X_lims( X_bounds[0], X_bounds[1], 0.00 )
#        self.figure_space.set_Y_lims( Y_bounds[0], Y_bounds[1], 0.00 )
#        self.figure_space.set_alt_lims( Z_bounds[0], Z_bounds[1], 0.00 )
#        self.figure_space.set_T_lims( T_bounds[0], T_bounds[1], 0.00 )
        
        self.DSvariable_get()
        self.position_get()
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def clickIgnoreTime(self):
        DS = self.figure_space.data_sets[ self.current_data_set ]
        DS.ignore_time( self.ignoreTimeCheckBox.isChecked() )
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def showAllTimePressed(self):
        DS = self.figure_space.data_sets[ self.current_data_set ]
        
        Tbounds = DS.T_bounds( self.figure_space.coordinate_system )
        self.figure_space.coordinate_system.set_plotT( Tbounds[0], Tbounds[1] )
        
        self.figure_space.set_T_lims( Tbounds[0], Tbounds[1], 0.05 )
        
#        X_bounds = DS.get_X_lims()
#        Y_bounds = DS.get_Y_lims()
#        Z_bounds = DS.get_alt_lims()
#        trash, trash, trash, T_bounds = DS.bounding_box()
#        X_bounds, Y_bounds, Z_bounds, T_bounds = self.figure_space.coordinate_system.transform( X_bounds, Y_bounds, Z_bounds, T_bounds )
        
#        self.figure_space.set_X_lims( X_bounds[0], X_bounds[1], 0.05 )
#        self.figure_space.set_Y_lims( Y_bounds[0], Y_bounds[1], 0.05 )
#        self.figure_space.set_alt_lims( Z_bounds[0], Z_bounds[1], 0.05 )
#        self.figure_space.set_T_lims( T_bounds[0], T_bounds[1], 0.05 )
        
        self.DSvariable_get()
        self.position_get()
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def showAllPositionsButtonPressed(self):
        DS = self.figure_space.data_sets[ self.current_data_set ]
        X_bounds, Y_bounds, Z_bounds, Zt_bounds, T_bounds = DS.bounding_box( self.figure_space.coordinate_system )
#        X_bounds, Y_bounds, Z_bounds, T_bounds = self.figure_space.coordinate_system.transform( X_bounds, Y_bounds, Z_bounds, T_bounds )
#        

        if np.isfinite( X_bounds[0] ) and np.isfinite(X_bounds[1]):
            self.figure_space.coordinate_system.set_plotX( X_bounds[0], X_bounds[1] )
        if np.isfinite( Y_bounds[0] ) and np.isfinite(Y_bounds[1]):
            self.figure_space.coordinate_system.set_plotY( Y_bounds[0], Y_bounds[1] )
        if np.isfinite( T_bounds[0] ) and np.isfinite(T_bounds[1]):
            self.figure_space.coordinate_system.set_plotT( T_bounds[0], T_bounds[1] )
        if np.isfinite( Z_bounds[0] ) and np.isfinite(Z_bounds[1]):
            self.figure_space.coordinate_system.set_plotZ( Z_bounds[0], Z_bounds[1] )
        if np.isfinite( Zt_bounds[0] ) and np.isfinite(Zt_bounds[1]):
            self.figure_space.coordinate_system.set_plotZt( Zt_bounds[0], Zt_bounds[1] )
        
        if np.isfinite( X_bounds[0] ) and np.isfinite(X_bounds[1]) and np.isfinite( Y_bounds[0] ) and np.isfinite(Y_bounds[1]):
            self.figure_space.set_XY_lims( X_bounds[0], X_bounds[1], Y_bounds[0], Y_bounds[1],0.05 )
        if np.isfinite( Z_bounds[0] ) and np.isfinite(Z_bounds[1]):
            self.figure_space.set_Z_lims( Z_bounds[0], Z_bounds[1], 0.05 )
        if np.isfinite( Zt_bounds[0] ) and np.isfinite(Zt_bounds[1]):
            self.figure_space.set_Zt_lims( Zt_bounds[0], Zt_bounds[1], 0.05 )
        if np.isfinite( T_bounds[0] ) and np.isfinite(T_bounds[1]):
            self.figure_space.set_T_lims( T_bounds[0], T_bounds[1], 0.05 )
        
        self.DSvariable_get()
        self.position_get()
        
        self.figure_space.replot_data()
        self.figure_space.draw()
        
    def boundsToClipboard(self):
        Xmin = self.Xmin_txtBox.text( )
        Xmax = self.Xmax_txtBox.text( )
            
        Ymin = self.Ymin_txtBox.text( )
        Ymax = self.Ymax_txtBox.text( )
            
        Zmin = self.Zmin_txtBox.text( )
        Zmax = self.Zmax_txtBox.text( )
            
        Tmin = self.Tmin_txtBox.text( )
        Tmax = self.Tmax_txtBox.text( )
        
        result = ''.join(['[[',Xmin,',',Xmax,'],[',Ymin,',',Ymax,'],[',Zmin,',',Zmax,'],[',Tmin,',',Tmax,']]'])
        
        QApplication.clipboard().setText( result )
    
    def boundsFromClipboard(self):
        #### assume text is in same format as boundsToClipboard, but do not use python parsing (is dangerous)
        
        text = QApplication.clipboard().text()
        
        try:
            text = "".join(text.split()) ## remove all white space
            bits = text.split(',')
        except:
            print("clipboard text is not compatible")
            return
            
            
        if len(bits) != 8:
            print("clipboard text is not compatible")
            return
        
        try:
            Xmin = float( bits[0][2:]  )
            Xmax = float( bits[1][:-1] )
            
            Ymin = float( bits[2][1:]  )
            Ymax = float( bits[3][:-1] )
            
            Zmin = float( bits[4][1:]  )
            Zmax = float( bits[5][:-1] )
            
            Tmin = float( bits[6][1:]  )
            Tmax = float( bits[7][:-2] )
        except:
            print("clipboard text is not compatible")
            return
        
        self.Xmin_txtBox.setText( str(Xmin) )
        self.Xmax_txtBox.setText( str(Xmax) )
        
        self.Ymin_txtBox.setText( str(Ymin) )
        self.Ymax_txtBox.setText( str(Ymax) )
        
        self.Zmin_txtBox.setText( str(Zmin) )
        self.Zmax_txtBox.setText( str(Zmax) )
        
        self.Tmin_txtBox.setText( str(Tmin) )
        self.Tmax_txtBox.setText( str(Tmax) )
        
        
    def searchButtonPressed(self):
        
        if self.current_data_set is not None and self.current_data_set != -1: ## is this needed?
            search_ID = self.search_txtBox.text()
            
            ret = self.figure_space.data_sets[ self.current_data_set ].search( [search_ID] )
            
            if ret is not None:
                self.add_dataset( ret )
                self.figure_space.replot_data()
                self.figure_space.draw()
        else:
            print("no dataset selected")
            
    def zoomButtonPressed(self):
        if self.figure_space.z_button_pressed:
            self.figure_space.z_button_pressed = False
            self.zoom_button.setText("zoom: in")
        else:
            self.figure_space.z_button_pressed = True
            self.zoom_button.setText("zoom: out")
            
    def clickAspectRatio(self):
        self.figure_space.rebalance_XY = self.aspectRatioCheckBox.isChecked()
        
        
            
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
#def print_details_analysis(PSE_list):
#    for PSE in PSE_list:
#        print( PSE.unique_index )
#        print( " even fitval:", PSE.PolE_RMS )
#        print( "   loc:", PSE.PolE_loc )
#        print( " odd fitval:", PSE.PolO_RMS )
#        print( "   loc:", PSE.PolO_loc )
#        print()
#
#class plot_data_analysis:
#    def __init__(self, antenna_locations):
#        self.ant_locs = antenna_locations
#        
#    def __call__(self, PSE_list):
#        for PSE in PSE_list:
#            print( "plotting:", PSE.unique_index )
#            print( " even fitval:", PSE.PolE_RMS )
#            print( "   loc:", PSE.PolE_loc )
#            print( " odd fitval:", PSE.PolO_RMS )
#            print( "   loc:", PSE.PolO_loc )
#            print( " Even is Green, Odd is Magenta" )
#            print() 
#            PSE.plot_trace_data(self.ant_locs)

#        
#class print_ave_ant_delays:
#    def __init__(self, ant_locs):
#        self.ant_locs = ant_locs
#        
#    def __call__(self, PSE_list):
#        max_RMS = 2.00e-9
#        
#        ant_delay_dict = {}
#        
#        for PSE in PSE_list:
#            PSE.load_antenna_data(False)
#            
#            for ant_name, ant_info in PSE.antenna_data.items():
#                loc = self.ant_locs[ant_name]
#                
#                if ant_name not in ant_delay_dict:
#                    ant_delay_dict[ant_name] = [[],  []]
#                
#                if (ant_info.antenna_status==0 or ant_info.antenna_status==2) and PSE.PolE_RMS<max_RMS:
#                    model = np.linalg.norm(PSE.PolE_loc[0:3] - loc)/v_air + PSE.PolE_loc[3]
#                    data = ant_info.PolE_peak_time
#                    ant_delay_dict[ant_name][0].append( data - model )
#                
#                if (ant_info.antenna_status==0 or ant_info.antenna_status==1) and PSE.PolO_RMS<max_RMS:
#                    model = np.linalg.norm(PSE.PolO_loc[0:3] - loc)/v_air + PSE.PolO_loc[3]
#                    data = ant_info.PolO_peak_time
#                    ant_delay_dict[ant_name][1].append( data - model )
#            
#        for ant_name, (even_delays, odd_delays) in ant_delay_dict.items():
#            PolE_ave = np.average(even_delays)
#            PolE_std = np.std(even_delays)
#            PolE_N = len(even_delays)
#            PolO_ave = np.average(odd_delays)
#            PolO_std = np.std(odd_delays)
#            PolO_N = len(odd_delays)
#
#            print(ant_name)
#            print("   even:", PolE_ave, "+/-", PolE_std/np.sqrt(PolE_N), '(', PolE_std, PolE_N, ')')
#            print("    odd:", PolO_ave, "+/-", PolO_std/np.sqrt(PolO_N), '(', PolO_std, PolO_N, ')')
#        print("done")
#        print()
#
#class histogram_amplitudes:
#    def __init__(self, station, pol=0):
#        self.station=station
#        self.polarization=pol #0 is even, 1 is odd
#        
#    def __call__(self, PSE_list):
#        
#        amplitudes = []
#        for PSE in PSE_list:
#            PSE.load_antenna_data(False)
#            total = 0.0
#            N = 0
#            for antenna_name, data in PSE.antenna_data.items():
#                if antenna_name[0:3] == self.station:
#                    if self.polarization == 0 and (data.antenna_status==0 or data.antenna_status==2):
#                        total += data.PolE_HE_peak
#                        N += 1
#                        
#                    elif self.polarization == 1 and (data.antenna_status==0 or data.antenna_status==1):
#                        total += data.PolO_HE_peak
#                        N += 1
#                        
#            if N != 0:
#                amplitudes.append( total/N )
#            
#        print(len(amplitudes), int(np.sqrt(len(amplitudes))) )
#        print("average and std of distribution:", np.average(amplitudes), np.std(amplitudes))
#        plt.hist(amplitudes, 2*int(np.sqrt(len(amplitudes))) )
#        plt.xlim((0,1500))
#        plt.tick_params(labelsize=40)
#        plt.show()
        
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


#from scipy.stats import linregress
#class leader_speed_estimator:
#    def __init__(self, pol=0, RMS=2.0E-9):
#        self.polarization = pol
#        self.RMS_filter = RMS
#        
#    def __call__(self, PSE_list):
#        X = []
#        Y = []
#        Z = []
#        T = []
#        unique_IDs = []
#        
#        for PSE in PSE_list:
#            loc = PSE.PolE_loc
#            RMS = PSE.PolE_RMS
#            if self.polarization ==1:
#                loc = PSE.PolO_loc
#                RMS = PSE.PolO_RMS
#                
#            if RMS <= self.RMS_filter:
#                X.append(loc[0])
#                Y.append(loc[1])
#                Z.append(loc[2])
#                T.append(loc[3])
#                unique_IDs.append( PSE.unique_index )
#                
#        T = np.array(T)   
#        X = np.array(X)  
#        Y = np.array(Y)  
#        Z = np.array(Z)       
#        
#        sorter = np.argsort(T)
#        T = T[sorter]
#        X = X[sorter]
#        Y = Y[sorter]
#        Z = Z[sorter]
#        
#        Xslope, Xintercept, XR, P, stderr = linregress(T, X)
#        Yslope, Yintercept, YR, P, stderr = linregress(T, Y)
#        Zslope, Zintercept, ZR, P, stderr = linregress(T, Z)
#        
#        print("PSE IDs:", unique_IDs)
#        print("X vel:", Xslope, XR)
#        print("Y vel:", Yslope, YR)
#        print("Z vel:", Zslope, ZR)
#        print("3D speed:", np.sqrt(Xslope**2 + Yslope**2 + Zslope**2))
#        print("time:", T[-1]-T[0])
#        print("number sources", len(unique_IDs))
#        
#        X_ax = plt.subplot(311)
#        plt.setp(X_ax.get_xticklabels(), visible=False)
#        Y_ax = plt.subplot(312, sharex=X_ax)
#        plt.setp(Y_ax.get_xticklabels(), visible=False)
#        Z_ax = plt.subplot(313, sharex=X_ax)
#        
#        X_ax.scatter((T-3.819)*1000.0, X/1000.0)
#        Y_ax.scatter((T-3.819)*1000.0, Y/1000.0)
#        Z_ax.scatter((T-3.819)*1000.0, Z/1000.0)
#        
#        Xlow = Xintercept + Xslope*T[0]
#        Xhigh = Xintercept + Xslope*T[-1]
#        X_ax.plot( [ (T[0]-3.819)*1000.0, (T[-1]-3.819)*1000.0], [Xlow/1000.0,Xhigh/1000.0] )
#        X_ax.tick_params('y', labelsize=25)
#        
#        Ylow = Yintercept + Yslope*T[0]
#        Yhigh = Yintercept + Yslope*T[-1]
#        Y_ax.plot( [(T[0]-3.819)*1000.0, (T[-1]-3.819)*1000.0], [Ylow/1000.0,Yhigh/1000.0] )
#        Y_ax.tick_params('y', labelsize=25)
#        
#        Zlow = Zintercept + Zslope*T[0]
#        Zhigh = Zintercept + Zslope*T[-1]
#        Z_ax.plot( [(T[0]-3.819)*1000.0, (T[-1]-3.819)*1000.0], [Zlow/1000.0,Zhigh/1000.0])
#        Z_ax.tick_params('both', labelsize=25)
#        
#        plt.show()

