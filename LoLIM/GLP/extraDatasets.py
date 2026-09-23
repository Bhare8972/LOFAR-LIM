#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

from LoLIM.GLP.plotter import DataSet_Type

class DataSet_box(DataSet_Type):
    
    def __init__(self, x_bounds, y_bounds, z_bounds, t_bounds, 
                 lineStyle, lineColor, lineThickness, name,
                 do_ZvsT=True, do_ZvsEast=True, do_NorthvsEast=True, doNorthvsZ=True):

        self.x_bounds = np.array(x_bounds)
        self.y_bounds = np.array(y_bounds )
        self.z_bounds = np.array(z_bounds )
        self.t_bounds = np.array(t_bounds )

        X_array = []
        Y_array = []
        Z_array = []
        T_array = []

        patterns = [  [0,0,0,0], [0,1,0,0], [1,1,0,0], [1,0,0,0], [0,0,0,0],
                      [0,0,0,0], [0,0,1,0], [1,0,1,0], [1,0,0,0], [0,0,0,0],
                      [0,0,0,0], [0,1,0,0], [0,1,1,0], [0,0,1,0], [0,0,0,0],
                      [0,0,0,0], [0,0,0,1], [0,0,1,1], [0,0,1,0], [0,0,0,0],
                       ]

        #for x in self.x_bounds:
        #    for y in self.y_bounds:
        for p in patterns:
                x = self.x_bounds[p[0]]
                y = self.y_bounds[p[1]]
                z = self.z_bounds[p[2]]
                t = self.t_bounds[p[3]]
                #for z in self.z_bounds:
                #    for t in self.t_bounds:
                X_array.append(x)
                Y_array.append(y)
                Z_array.append(z)
                T_array.append(t)

        self.X_array = np.array( X_array)
        self.Y_array = np.array( Y_array)
        self.Z_array = np.array( Z_array)
        self.T_array = np.array( T_array)

        self.X_TMP = np.array(X_array)
        self.Y_TMP = np.array(Y_array )
        self.Z_TMP = np.array(Z_array )
        self.T_TMP = np.array(T_array )

        self.lineStyle = lineStyle
        self.lineColor = lineColor 
        self.lineThickness = lineThickness
        self.name = name

        self.do_ZvsT=do_ZvsT
        self.do_ZvsEast=do_ZvsEast
        self.do_NorthvsEast=do_NorthvsEast
        self.doNorthvsZ=doNorthvsZ

        self._ignore_time = False
        self.ignore_time_bounds = None
        self.display = True

        self.transform_memory = None
        
    def set_show_all(self, coordinate_system, do_set_limits=True):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
            
        if do_set_limits:
            self.X_TMP[:] = self.X_array  
            self.Y_TMP[:] = self.Y_array  
            self.Z_TMP[:] = self.Z_array  
            self.T_TMP[:] = self.T_array
            
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
        
            
        ## filter and shift]
            
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:]
        Ytmp = self.Y_TMP[:]
        Ztmp = self.Z_TMP[:]
        Ttmp = self.T_TMP[:]
        
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
        _, _ , _, Tbounds = self.bounding_box(coordinate_system)
        return Tbounds
        
        
    def get_all_properties(self):
        ret =  {"lineColor":str(self.lineColor),  "lineStyle":str(self.lineStyle), "lineThickness":str(self.lineThickness), "name":str(self.name),
        "do_ZvsT":str(self.do_ZvsT),"do_ZvsEast":str(self.do_ZvsEast),"do_NorthvsEast":str(self.do_NorthvsEast),"doNorthvsZ":str(self.doNorthvsZ),
                }

        return ret
        
        ## need: marker type, color map
    
    def set_property(self, name, str_value):
        
        try:
            if name == "lineColor":
                self.lineColor = str_value

            elif name == "lineStyle":
                self.lineStyle = str_value

            elif name == "lineThickness":
                self.lineStyle = float(str_value)

            elif name == "name":
                self.lineStyle = str_value
                

            elif name == "do_ZvsT":
                if str_value=='1' or str_value=='true' or str_value=='True':
                    self.do_ZvsT = True
                elif str_value=='0' or str_value=='false' or str_value=='False':
                    self.do_ZvsT = False
                else:
                    print('do_ZvsT must be one of: 0, 1, true, false, True, or False')
                    print()
                

            elif name == "do_NorthvsEast":
                if str_value=='1' or str_value=='true' or str_value=='True':
                    self.do_NorthvsEast = True
                elif str_value=='0' or str_value=='false' or str_value=='False':
                    self.do_NorthvsEast = False
                else:
                    print('do_NorthvsEast must be one of: 0, 1, true, false, True, or False')
                    print()
                

            elif name == "do_ZvsEast":
                if str_value=='1' or str_value=='true' or str_value=='True':
                    self.do_ZvsEast = True
                elif str_value=='0' or str_value=='false' or str_value=='False':
                    self.do_ZvsEast = False
                else:
                    print('do_ZvsEast must be one of: 0, 1, true, false, True, or False')
                    print()
                

            elif name == "doNorthvsZ":
                if str_value=='1' or str_value=='true' or str_value=='True':
                    self.doNorthvsZ = True
                elif str_value=='0' or str_value=='false' or str_value=='False':
                    self.doNorthvsZ = False
                else:
                    print('doNorthvsZ must be one of: 0, 1, true, false, True, or False')
                    print()

          
                
            else:
                print("do not have property:", name)
        except:
            print("error in setting property", name, str_value)
            pass

    
    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system, zorder):
        
        #### random book keeping ####
        self.clear()
    
        
    
            
    #### set cuts and transforms
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:]
        Ytmp = self.Y_TMP[:]
        Ztmp = self.Z_TMP[:]
        Ttmp = self.T_TMP[:]
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform( 
            Xtmp, Ytmp, Ztmp, Ttmp, make_copy=False)


        if (not self.display) or len(plotX)==0:
            
            print(self.name, "not display")
            return
        
    ## finally plot!
        
        try:
            if self.do_ZvsT and (not self._ignore_time):
                self.AltVsT_paths = AltVsT_axes.plot(plotT[15:20], plotZt[15:20], ls=self.lineStyle, lw=self.lineThickness, color=self.lineColor, zorder=zorder)
                
            if self.do_ZvsEast:
                self.AltVsEw_paths = AltVsEw_axes.plot(plotX[5:10], plotZ[5:10], ls=self.lineStyle, lw=self.lineThickness, color=self.lineColor, zorder=zorder)
            
            if self.do_NorthvsEast:
                self.NsVsEw_paths = NsVsEw_axes.plot(plotX[0:5], plotY[0:5], ls=self.lineStyle, lw=self.lineThickness, color=self.lineColor, zorder=zorder)
            
            if self.doNorthvsZ:
                self.NsVsAlt_paths = NsVsAlt_axes.plot(plotZ[10:15], plotY[10:15], ls=self.lineStyle, lw=self.lineThickness, color=self.lineColor, zorder=zorder)

        except Exception as e: 
            print(e)
            #print('len size:', len(size), 'len x', len(plotX))
            #print('size', size)
        

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
     
            
        return None
    
    def get_view_ID_list(self, coordinate_system):
        
       return None
        
    
    def copy_view(self, coordinate_system):
        
        return None

    def print_info(self, coordinate_system):
        print('no')
        
    
        
    
    def text_output(self):
        
        print('no')
                    
    def ignore_time(self, ignore=None, min=-np.inf, max=np.inf):
        if ignore is not None:
            self._ignore_time = ignore
            self.ignore_time_bounds = [min, max]
        return self._ignore_time


class DataSet_annotations(DataSet_Type):
    
    def __init__(self, annotation_XYZT_dictionary,
                       #annotation_argument_dictionary,
                       default_color, name, fontsize, otherArgs={},
                 do_ZvsT=True, do_ZvsEast=True, do_NorthvsEast=True, doNorthvsZ=True):

        self.annotation_XYZT_dictionary = annotation_XYZT_dictionary
        #self.annotation_argument_dictionary = annotation_argument_dictionary
        self.default_color = default_color
        self.name = name
        self.fontsize = fontsize
        self.otherArgs = otherArgs
        
        self._ignore_time = False
        self.ignore_time_bounds = None
        self.display = True

        self.do_ZvsT=do_ZvsT
        self.do_ZvsEast=do_ZvsEast
        self.do_NorthvsEast=do_NorthvsEast
        self.doNorthvsZ=doNorthvsZ

        self.annotation_labels = list( self.annotation_XYZT_dictionary.keys() )


    ### continue with some stuff
        self.X_array = np.array( [ self.annotation_XYZT_dictionary[L][0] for L in self.annotation_labels ], dtype=np.double )
        self.Y_array = np.array( [ self.annotation_XYZT_dictionary[L][1] for L in self.annotation_labels ], dtype=np.double)
        self.Z_array = np.array( [ self.annotation_XYZT_dictionary[L][2] for L in self.annotation_labels ], dtype=np.double)
        self.T_array = np.array( [ self.annotation_XYZT_dictionary[L][3] for L in self.annotation_labels ], dtype=np.double )

        self.X_TMP = np.empty(len(self.X_array), dtype=np.double)
        self.Y_TMP = np.empty(len(self.Y_array), dtype=np.double)
        self.Z_TMP = np.empty(len(self.Z_array), dtype=np.double)
        self.T_TMP = np.empty(len(self.T_array), dtype=np.double)
    
        #### some axis data ###
        self.AltVsT_paths = None
        self.AltVsEw_paths = None
        self.NsVsEw_paths = None
        self.NsVsAlt_paths = None
        
        self.transform_memory = None
        
    def set_show_all(self, coordinate_system, do_set_limits=True):
        """return bounds needed to show all data. Nan if not applicable returns: [[xmin, xmax], [ymin,ymax], [zmin,zmax],[tmin,tmax]]"""
            
        if do_set_limits:
            self.X_TMP[:] = self.X_array  
            self.Y_TMP[:] = self.Y_array  
            self.Z_TMP[:] = self.Z_array  
            self.T_TMP[:] = self.T_array  
            
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
        
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:]
        Ytmp = self.Y_TMP[:]
        Ztmp = self.Z_TMP[:]
        Ttmp = self.T_TMP[:]
        
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
            
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:]
        Ytmp = self.Y_TMP[:]
        Ztmp = self.Z_TMP[:]
        Ttmp = self.T_TMP[:]
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        
        ## transform and cut on bounds
        TMP = coordinate_system.get_plotT()
        A = TMP[0] ## need to copy
        B = TMP[1]
        coordinate_system.set_plotT(-np.inf, np.inf)
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False)
        coordinate_system.set_plotT(A, B)
        
        ## return actual bounds
        if len(plotT) > 0:
            return [np.min(plotT), np.max(plotT)]
        else:
            return [0, 1]
        
    def get_all_properties(self):
        ret =  { 'name':self.name,
                'default_color':self.default_color, 'fontsize':self.fontsize,
        "do_ZvsT":str(self.do_ZvsT),"do_ZvsEast":str(self.do_ZvsEast),"do_NorthvsEast":str(self.do_NorthvsEast),"doNorthvsZ":str(self.doNorthvsZ),}
            
        return ret
        
        ## need: marker type, color map
    
    def set_property(self, name, str_value):
        
        try:
            if name == "default_color size":
                self.default_color = str_value

            elif name == 'name':
                self.name = str_value

            elif name == 'fontsize':
                self.fontsize  = float(str_value )
                

            elif name == "do_ZvsT":
                if str_value=='1' or str_value=='true' or str_value=='True':
                    self.do_ZvsT = True
                elif str_value=='0' or str_value=='false' or str_value=='False':
                    self.do_ZvsT = False
                else:
                    print('do_ZvsT must be one of: 0, 1, true, false, True, or False')
                    print()
                

            elif name == "do_NorthvsEast":
                if str_value=='1' or str_value=='true' or str_value=='True':
                    self.do_NorthvsEast = True
                elif str_value=='0' or str_value=='false' or str_value=='False':
                    self.do_NorthvsEast = False
                else:
                    print('do_NorthvsEast must be one of: 0, 1, true, false, True, or False')
                    print()
                

            elif name == "do_ZvsEast":
                if str_value=='1' or str_value=='true' or str_value=='True':
                    self.do_ZvsEast = True
                elif str_value=='0' or str_value=='false' or str_value=='False':
                    self.do_ZvsEast = False
                else:
                    print('do_ZvsEast must be one of: 0, 1, true, false, True, or False')
                    print()
                

            elif name == "doNorthvsZ":
                if str_value=='1' or str_value=='true' or str_value=='True':
                    self.doNorthvsZ = True
                elif str_value=='0' or str_value=='false' or str_value=='False':
                    self.doNorthvsZ = False
                else:
                    print('doNorthvsZ must be one of: 0, 1, true, false, True, or False')
                    print()

          
                
            else:
                print("do not have property:", name)
        except:
            print("error in setting property", name, str_value)
            pass
    
        
    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system, zorder):
   
        
        #### random book keeping ####
        self.clear()
        
    
        
    #### set cuts and transforms
        self.X_TMP[:] = self.X_array  
        self.Y_TMP[:] = self.Y_array  
        self.Z_TMP[:] = self.Z_array  
        self.T_TMP[:] = self.T_array 
        
        Xtmp = self.X_TMP[:]
        Ytmp = self.Y_TMP[:]
        Ztmp = self.Z_TMP[:]
        Ttmp = self.T_TMP[:]
        
        if self.transform_memory is None:
            self.transform_memory = coordinate_system.make_workingMemory( len(self.X_array) )
        coordinate_system.set_workingMemory( self.transform_memory )
        
        plotX, plotY, plotZ, plotZt, plotT = coordinate_system.transform( 
            Xtmp, Ytmp, Ztmp, Ttmp, 
            make_copy=False)

        if (not self.display) or len(plotX)==0:
            
            print(self.name, "not display. have:", len(plotX))
            return
        
        print(self.name, "plotting")
        
        try:

            for label,x,y,z,zt,t in zip(self.annotation_labels,plotX,plotY,plotZ,plotZt,plotT):

                if not self._ignore_time and self.do_ZvsT:
                    self.AltVsT_paths = AltVsT_axes.annotate(label, xy=( t,zt ), size=self.fontsize, c=self.default_color, zorder=zorder, **self.otherArgs  )

                if self.do_ZvsEast:
                    self.AltVsEw_paths = AltVsEw_axes.annotate(label, xy=( x,z ), size=self.fontsize, c=self.default_color, zorder=zorder,**self.otherArgs  )

                if self.do_NorthvsEast:
                    self.NsVsEw_paths = NsVsEw_axes.annotate(label, xy=( x,y ), size=self.fontsize, c=self.default_color, zorder=zorder, **self.otherArgs  )

                if self.doNorthvsZ:
                    self.NsVsAlt_paths = NsVsAlt_axes.annotate(label, xy=( z,y ), size=self.fontsize , c=self.default_color, zorder=zorder,**self.otherArgs )

        except Exception as e: 
            print(e)
            #print('len size:', len(size), 'len x', len(plotX))
            #print('size', size)
        

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
        return None
    
    def get_view_ID_list(self, coordinate_system):
        
        return []
        
    
    def copy_view(self, coordinate_system):
        
        return None
    

    def print_info(self, coordinate_system):
        print('no')
        
    
    def text_output(self):
        
        print('no')
                    
    def ignore_time(self, ignore=None, min=-np.inf, max=np.inf):
        if ignore is not None:
            self._ignore_time = ignore
            self.ignore_time_bounds = [min, max]
        return self._ignore_time

class DataSet_extraTextBox(DataSet_Type):
    
    def __init__(self, text, name, axis="ancillary_axes", txtX=0, txtY=1, txtArgs={'fontsize':12}):

        self.text = text 
        self.axis = axis 
        allowed_axes = "AltVsT_axes", "AltVsEw_axes", "NsVsEw_axes", "NsVsAlt_axes", "ancillary_axes"

        if self.axis not in allowed_axes:
            print('axis:', axis, 'not in allowed axes:', allowed_axes)
        
        self.name = name
        self.display = True

        self.txtX = txtX
        self.txtY = txtY
        self.txtArgs = txtArgs

        self.returnedTxtObj = None

    def ancillary_label(self):
        return ""

    def get_all_properties(self):
        ret =  {"txtX":str(txtX),  "txtY":str(txtY), 'name':self.name}
        
        for k,v in self.txtArgs:
            ret[k]=str(v)
            
        return ret
        
    
    def set_property(self, name, str_value):
        
        try:
            if name == "txtX":
                self.txtX = float(str_value)
                
            elif name == "txty":
                self.txtY = float(str_value)
                    
            elif name == 'name':
                self.name = str_value
               
            else:
                print("property:", name, 'sent to text arguments with value', str_value)
                self.txtArgs[name]=str_value

        except:
            print("error in setting property", name, str_value)
            pass
    
 
        
    
    def plot(self, AltVsT_axes, AltVsEw_axes, NsVsEw_axes, NsVsAlt_axes, ancillary_axes, coordinate_system, zorder):

        if not self.display:
            self.clear()
            return 

        if self.axis == 'AltVsT_axes':
            axisObj = AltVsT_axes
        elif self.axis == 'AltVsEw_axes':
            axisObj = AltVsEw_axes
        elif self.axis == 'NsVsEw_axes':
            axisObj = NsVsEw_axes
        elif self.axis == 'NsVsAlt_axes':
            axisObj = NsVsAlt_axes
        elif self.axis == 'ancillary_axes':
            axisObj = ancillary_axes
        
        try:
            self.returnedTxtObj = axisObj.text(x=self.txtX, y=self.txtY, s=self.text, transform=axisObj.transAxes, **self.txtArgs)

        except Exception as e: 
            print(e)

    
    def clear(self):
        self.returnedTxtObj = None
        
    def use_ancillary_axes(self):
        return self.axis == 'ancillary_axes'
                

    def toggle_on(self):
        self.display = True
    
    def toggle_off(self):
        self.display = False
        self.clear()
   