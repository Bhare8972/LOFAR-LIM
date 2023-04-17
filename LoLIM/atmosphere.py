#!/usr/bin/env python3
""" This module is to account for different atmospheres. It exists now becouse different codes have different values for speed of light in air. In future this should be expanded for more complex atmospheres."""

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
olaf_atmosphere = simple_atmosphere( 299792458.0/1.0003 )




