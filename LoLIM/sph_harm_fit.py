#!/usr/bin/env python3
""" this contains code for fitting functions to spherical harmonics. Note this uses numeriallu solves a matrix instead of solving analytically"""

from scipy.special import sph_harm
import numpy as np


class spherical_harmonics_fitter:
    def __init__(self, degree, zeniths, azimuths, angle_units=1):
        """degree should be >=0. is max degree of spherical harms. zeniths, azimuths are numpy arrays of angles where the function is sampled.
        angle_units should be a factor to convert the angles to radians"""
        self.degree = degree
        self.num_params = np.sum([2*i+1 for i in range(degree+1) ])
        self.num_measurments = len( zeniths )*len( azimuths )

        ## sort angles
        all_azimuths = np.empty(self.num_measurments , dtype=np.double)
        all_zeniths = np.empty(self.num_measurments , dtype=np.double)
        total_i = 0
        for ze in zeniths:
            for az in azimuths:
                all_azimuths[total_i] = az*angle_units
                all_zeniths[total_i] =  ze*angle_units
                total_i += 1

        ## make matrix
        self.matrix = np.empty( (self.num_measurments, self.num_params), dtype=complex )
        par_i = 0
        for d in range(degree+1):
            ## order zero
            sph_harm(0, d, all_azimuths, all_zeniths, out=self.matrix[:,par_i] )
            par_i += 1

            ## all others
            for order in range(1, d+1):
                ## positive
                sph_harm( order, d, all_azimuths, all_zeniths, out=self.matrix[:,par_i] )
                par_i += 1
                ## negative
                sph_harm(-order, d, all_azimuths, all_zeniths, out=self.matrix[:,par_i] )
                par_i += 1

        self.all_zeniths = all_zeniths
        self.all_azimuths = all_azimuths


        self.SA_TMP = np.empty(self.num_measurments , dtype=complex)

        d_az = np.average( azimuths[1:] - azimuths[:-1] )
        d_ze = np.average( zeniths[1:] -  zeniths[:-1] )

        self.dSolidAngle = np.sin(self.all_zeniths)
        self.dSolidAngle *= d_az*d_ze

    def get_num_params(self):
        """Return number of spherical harmonic parameters used"""
        return self.num_params

    def solve_numeric(self, F):
        """Returns spherical harmonic coefients. F should be a complex-valued 2D numpy array. First index should be zenith angle, second should be azimuth.
        returned spherical harmonic coefients are 1D numpy array starting with degree 0 and increassing. Each value is the next order. e.g: [(d=0,o=0),(d=1,o=1),(d=1,o=1),(d=1,o=-1), etc...]
        This functin shoudl probably be prefered over semianalytic."""
        F_view = F.flatten()
        weights, residuals, rank, singularValues = np.linalg.lstsq(self.matrix, F_view, rcond=None)
        return weights

    def solve_semianalytic(self, F, out=None):
        """ Uses properties of spherical harmonics to find best fit. WARNING: currently assumes zenith and azimuth have equal (but not same) spacing.
        May not be as stable as solve_numeric"""
        F_view = F.flatten()

        ## assume uniform grid


        out = np.empty( self.num_params, dtype=complex )

        def GO(param_i):
            nonlocal self
            np.conj(self.matrix[ :, param_i ], out=self.SA_TMP)
            self.SA_TMP *= self.dSolidAngle
            self.SA_TMP *= F_view
            out[param_i] = np.sum( self.SA_TMP )

        par_i = 0
        for d in range(self.degree+1):
            ## order zero
            GO(par_i)
            par_i += 1

            ## all others
            for order in range(1, d+1):
                ## positive
                GO(par_i)
                par_i += 1
                ## negative
                GO(par_i)
                par_i += 1

        return out


def evaluate_SphHarms_manyAngles(weights, zeniths, azimuths, out=None, memory=None):
    """ Weights should be first output of  sph erical_harmonics_fitter.solve.  zeniths and azimuths should be radians. Can be single values or numpy arrays of same length.
    Out should be complex-valued of same lenght as zeniths and azimuths if used. memory will speed up calculation, should be same shape out. No arrays can alias."""

    try:
        L = len(zeniths)
    except:
        L = None

    if L is not None:
        tmp = memory
        if tmp is None:
            tmp = np.empty(L, dtype=complex)

        if out is None:
            out = np.zeros(L, dtype=complex)
        else:
            out[:] = 0.0
    else:
        tmp = None
        out = 0


    par_i = 0
    d = 0
    while True:
        ## order zero
        r = sph_harm(0, d, azimuths, zeniths, out=tmp )
        r *= weights[ par_i ]
        out += r
        par_i += 1

        ## all others
        for order in range(1, d + 1):
            ## positive
            r = sph_harm( order, d, azimuths, zeniths, out=tmp )
            r *= weights[ par_i ]
            out += r
            par_i += 1

            ## negative
            r = sph_harm(-order, d, azimuths, zeniths, out=tmp )
            r *= weights[par_i]
            out += r
            par_i += 1

        d += 1
        if par_i == len(weights):
            break

    return out


def evaluate_SphHarms_oneAngle(weights, zenith, azimuth, out=None, memory=None):
    """
     Evaluate multiple spherical harmonic functions at one zenith/azimuth.
     weights should be a (N, M) complex-valued array, where N is the number of spherical harmonic coeficents and M is number of functions.
            (i.e. spherical_harmonics_fitter.solve ran M times)
    zenith and azimuth should be single numbers in degrees.
    out, if given should be complex valued M long
    memory is same. Note that no inputs should alias
        """


    num_sph, num_funcs = weights.shape
    if out is None:
        out = np.zeros(num_funcs, dtype=np.complex)
    else:
        out[:] = 0.0

    if memory is None:
        tmp = np.empty(num_funcs, dtype=np.complex)
    else:
        tmp = memory

    par_i = 0
    d = 0
    while True:
        ## order zero
        SphH = sph_harm(0, d, azimuth, zenith)
        tmp[:] = weights[par_i, :]
        tmp *= SphH
        out += tmp
        par_i += 1

        ## all others
        for order in range(1, d + 1):
            ## positive
            SphH = sph_harm(-order, d, azimuth, zenith)
            tmp[:] = weights[par_i, :]
            tmp *= SphH
            out += tmp
            par_i += 1

            ## negative
            SphH = sph_harm(-order, d, azimuth, zenith)
            tmp[:] = weights[par_i, :]
            tmp *= SphH
            out += tmp
            par_i += 1

        d += 1
        if par_i == num_sph:
            break

    return out

