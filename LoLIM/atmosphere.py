#!/usr/bin/env python3
""" This module is to account for different atmospheres. It exists now becouse different codes have different values for speed of light in air. In future this should be expanded for more complex atmospheres."""

import numpy as np
from LoLIM.utilities import C

class base_atmosphere:
    """base class. Inherit from this, do not use."""
    def __init__(self):
        pass

    def get_effective_lightSpeed(self, emission_XYZ, antenna_XYZs):
        """given an emission location, and antenna location, give effective speed of light. If antenna_XYZs is 2D[i,j], than i should be an antenna index, and j should be 0,1,2 (x,y,z)"""
        return C

class simple_atmosphere( base_atmosphere ):
    """ a constant speed of light."""
    def __init__(self, v_air):
        self.v_air = v_air

    def get_effective_lightSpeed(self, emission_XYZ, antenna_XYZs):
        """given an emission location, and antenna location, give effective speed of light."""
        return self.v_air

default_atmosphere = simple_atmosphere( C/1.000293 )
olaf_constant_atmosphere = simple_atmosphere( 299792458.0/1.0003 )




class corsika_atmosphere( base_atmosphere ):
    def __init__(self):
        self.height_step = 10 # m
        self.num_heights = 1000

        self.C = 299792458.0

        self.RefIndex_by_height = np.zeros(  self.num_heights )
        self.RefIndex_by_height[0] = 1.0003 # index of refraction at sea level

        mass = 0

        RefOrho = (self.RefIndex_by_height[0] -1.0)*9941.8638/1222.6562
        for i in range(1,self.num_heights):
            height = i*self.height_step

            if (height > 10000.0):
                b = 1305.5948
                c = 6361.4304
            elif height > 4000.0 :
                b = 1144.9069
                c = 8781.5355
            else :
                b = 1222.6562
                c = 9941.8638


            mass = mass + (b/c) * np.exp(-height/c) * self.height_step # assume density is about constant

            self.RefIndex_by_height[i] = 1.0+ RefOrho* mass/height   # mean index of refraction for rays from height to 0


   # xi(0)=Refrac
   # mass=0.
   # RefOrho=(xi(0)-1.d0)*9941.8638d0/1222.6562d0
   # do i = 1, AtmHei_dim
   #    height=i*AtmHei_step  ! distance to ground along shower axis
   #     if (height.ge.10d3) then
   #          b = 1305.5948; c = 6361.4304
   #     elseif (height.ge.4d3) then
   #          b = 1144.9069d0; c = 8781.5355d0
   #     else
   #          b = 1222.6562d0; c = 9941.8638d0
   #     endif
   #    mass = mass + (b/c) * exp(-height/c) * AtmHei_step ! assume density is about constant

   #    xi(i) = 1.d0+ RefOrho* mass/height   ! mean index of refraction for rays from height to 0
   #  end do



    def get_effective_lightSpeed(self, emission_XYZ, antenna_XYZs):

        hi = emission_XYZ[2] / self.height_step
        i = int(hi)

        if i >= self.num_heights-1:
            RefracIndex = self.RefIndex_by_height[ -1 ]
        elif i>=0:
            a = self.RefIndex_by_height[ i ]
            b = self.RefIndex_by_height[ i+1 ]
            RefracIndex = a*(i+1.-hi) + b*(hi-i)
            
        else: 
            RefracIndex = self.RefIndex_by_height[ 0 ]
        return self.C/RefracIndex


       # i=INT(hi)
       # If(i.ge.AtmHei_Dim) then
       #   RefracIndex=xi(AtmHei_Dim)
       # ElseIf(i.gt.0) then
       #   RefracIndex=xi(i)*(i+1.-hi) + xi(i+1)*(hi-i)
       # Else
       #   RefracIndex=Xi(0)



olaf_varying_atmosphere = corsika_atmosphere()
