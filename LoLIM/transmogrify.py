#!/usr/bin/env python3

""" a series of useful coordinate transform functions. Based on Transmogrify package by Tim Schellart. Originally in C"""


import numpy as np

from LoLIM.utilities import RTD

def gregorian2jd(year, month, fractional_day):
  if month==1 or month==2:
    Y = year - 1
    M = month + 12
  else:
    Y = year
    M = month

  a = (int)(Y / 100);
  b = 2-a+(int)(a/4);

  return int(365.25 * (Y + 4716)) + int(30.6001 * (M + 1)) + fractional_day + b - 1524.5

def delta_tai_utc(utc):
    
  ## Leap second table from the IERS Earth Orientation Centre bulletin C
  ls = [
      2441499.5,  # 1972  Jul.   1              - 1s
      2441683.5,  # 1973  Jan.   1              - 1s
      2442048.5,  # 1974  Jan.   1              - 1s
      2442413.5,  # 1975  Jan.   1              - 1s
      2442778.5,  # 1976  Jan.   1              - 1s
      2443144.5,  # 1977  Jan.   1              - 1s
      2443509.5,  # 1978  Jan.   1              - 1s
      2443874.5,  # 1979  Jan.   1              - 1s
      2444239.5,  # 1980  Jan.   1              - 1s
      2444786.5,  # 1981  Jul.   1              - 1s
      2445151.5,  # 1982  Jul.   1              - 1s
      2445516.5,  # 1983  Jul.   1              - 1s
      2446247.5,  # 1985  Jul.   1              - 1s
      2447161.5,  # 1988  Jan.   1              - 1s
      2447892.5,  # 1990  Jan.   1              - 1s
      2448257.5,  # 1991  Jan.   1              - 1s
      2448804.5,  # 1992  Jul.   1              - 1s
      2449169.5,  # 1993  Jul.   1              - 1s
      2449534.5,  # 1994  Jul.   1              - 1s
      2450083.5,  # 1996  Jan.   1              - 1s
   	  2450630.5,  # 1997  Jul.   1              - 1s
      2451179.5,  # 1999  Jan.   1              - 1s
      2453736.5,  # 2006  Jan.   1              - 1s
      2454832.5,  # 2009  Jan.   1              - 1s
   ]

  # Loop through the leap seconds table
  i = 0
  while (i<24 and utc>ls[i]):
    i += 1

  # before 2441499.5 TAI - UTC = 10s
  return 10 + i


def delta_tt_utc(utc):
  return 32.184 + float(delta_tai_utc(utc))

def julian2jd(year, month, fractional_day):

  if (month==1 or month==2):
    Y = year - 1
    M = month + 12
  else:
    Y = year
    M = month

  return int(365.25 * (Y + 4716)) + int(30.6001 * (M + 1)) + fractional_day - 1524.5



def date2jd(year, month, fractional_day):
  jd = 0.0

  # Gregorian calendar
  if (year > 1582):
    jd = gregorian2jd(year, month, fractional_day)
    
  # Julian calendar
  elif (year < 1582):
    jd = julian2jd(year, month, fractional_day)
    
  # Gregorian calendar
  elif month > 10:
    jd = gregorian2jd(year, month, fractional_day)
    
  ## Julian calendar
  elif month < 10:
    jd = julian2jd(year, month, fractional_day)

  # Gregorian calendar
  elif fractional_day >= 15:
    jd = gregorian2jd(year, month, fractional_day)
 
  # Julian calendar
  elif fractional_day <= 4:
    jd = julian2jd(year, month, fractional_day)

  return jd

def tmf_nutation(jde):
  i = 0
  Dphi = 0.0
  arg = 0.0
  T = (jde - 2451545.) / 36525.
  T2 = T * T
  T3 = T * T2

  # Mean elongation of the Moon from the Sun
  D = (297.85036 + 445267.111480 * T - 0.0019142 * T2 + T3 / 189474.) * np.pi / 180.

  # Mean anomaly of the Sun (Earth)
  M = (357.52772 + 35999.050340 * T - 0.0001603 * T2 + T3 / 300000.) * np.pi / 180.

  # Mean anomaly of the Moon
  Mm = (134.96298 + 477198.867398 * T - 0.0086972 * T2 + T3 / 56250.) * np.pi / 180.

  # Moon's argument of latitude
  F = (93.27191 + 483202.017538 * T - 0.0036825 * T2 + T3 / 327270.) * np.pi / 180.

  ## Longitude of the ascending node of the Moon's mean orbit on the
  ##   ecliptic, measured from the mean equinox of the date 
  Omega = (125.04452 - 1934.136261 * T + 0.0020708 * T2 + T3 / 450000.) * np.pi / 180.

  ## Argument for trigonometric functions in terms of 
  ##   multiples of D, M, Md, F, Omega 
  mul = [
    [ 0, 0, 0, 0, 1],
    [-2, 0, 0, 2, 2],
    [ 0, 0, 0, 2, 2],
    [ 0, 0, 0, 0, 2],
    [ 0, 1, 0, 0, 0],
    [ 0, 0, 1, 0, 0],
    [-2, 1, 0, 2, 2],
    [ 0, 0, 0, 2, 1],
    [ 0, 0, 1, 2, 2],
    [-2,-1, 0, 2, 2],
    [-2, 0, 1, 0, 0],
    [-2, 0, 0, 2, 1],
    [ 0, 0,-1, 2, 2],
    [ 2, 0, 0, 0, 0],
    [ 0, 0, 1, 0, 1],
    [ 2, 0,-1, 2, 2],
    [ 0, 0,-1, 0, 1],
    [ 0, 0, 1, 2, 1],
    [-2, 0, 2, 0, 0],
    [ 0, 0,-2, 2, 1],
    [ 2, 0, 0, 2, 2],
    [ 0, 0, 2, 2, 2],
    [ 0, 0, 2, 0, 0],
    [-2, 0, 1, 2, 2],
    [ 0, 0, 0, 2, 0],
    [-2, 0, 0, 2, 0],
    [ 0, 0,-1, 2, 1],
    [ 0, 2, 0, 0, 0],
    [ 2, 0,-1, 0, 1],
    [-2, 2, 0, 2, 2],
    [ 0, 1, 0, 0, 1],
    [-2, 0, 1, 0, 1],
    [ 0,-1, 0, 0, 1],
    [ 0, 0, 2,-2, 0],
    [ 2, 0,-1, 2, 1],
    [ 2, 0, 1, 2, 2],
    [ 0, 1, 0, 2, 2],
    [-2, 1, 1, 0, 0],
    [ 0,-1, 0, 2, 2],
    [ 2, 0, 0, 2, 1],
    [ 2, 0, 1, 0, 0],
    [-2, 0, 2, 2, 2],
    [-2, 0, 1, 2, 1],
    [ 2, 0,-2, 0, 1],
    [ 2, 0, 0, 0, 1],
    [ 0,-1, 1, 0, 0],
    [-2,-1, 0, 2, 1],
    [-2, 0, 0, 0, 1],
    [ 0, 0, 2, 2, 1],
    [-2, 0, 2, 0, 1],
    [-2, 1, 0, 2, 1],
    [ 0, 0, 1,-2, 0],
    [-1, 0, 1, 0, 0],
    [-2, 1, 0, 0, 0],
    [ 1, 0, 0, 0, 0],
    [ 0, 0, 1, 2, 0],
    [ 0, 0,-2, 2, 2],
    [-1,-1, 1, 0, 0],
    [ 0, 1, 1, 0, 0],
    [ 0,-1, 1, 2, 2],
    [ 2,-1,-1, 2, 2],
    [ 0, 0, 3, 2, 2],
    [ 2,-1, 0, 2, 2]
  ]

  # Coefficients of sine and cosine for Dphi in units of 0''.00001 */
  c = [
    -171996 - 174.2 * T,
       -13187 - 1.6 * T,
        -2274 - 0.2 * T,
         2062 + 0.2 * T,
         1426 - 3.4 * T,
          712 + 0.1 * T,
         -517 + 1.2 * T,
         -386 - 0.4 * T,
                   -301,
          217 - 0.5 * T,
                   -158,
          129 + 0.1 * T,
                    123,
                     63,
           63 + 0.1 * T,
                    -59,
          -58 - 0.1 * T,
                    -51,
                     48,
                     46,
                    -38,
                    -31,
                     29,
                     29,
                     26,
                    -22,
                     21,
           17 - 0.1 * T,
                     16,
          -16 + 0.1 * T,
                    -15,
                    -13,
                    -12,
                     11,
                    -10,
                     -8,
                      7,
                     -7,
                     -7,
                     -7,
                      6,
                      6,
                      6,
                     -6,
                     -6,
                      5,
                     -5,
                     -5,
                     -5,
                      4,
                      4,
                      4,
                     -4,
                     -4,
                     -4,
                      3,
                     -3,
                     -3,
                     -3,
                     -3,
                     -3,
                     -3,
                     -3
  ]

  # Calculate sum over sine and cosine for Dph and Depsilon respectively
  for i in range(63):
    arg = mul[i][0] * D  + \
          mul[i][1] * M  + \
          mul[i][2] * Mm + \
          mul[i][3] * F  + \
          mul[i][4] * Omega

    Dphi += c[i] * np.sin(arg)

  return (Dphi / 3.6e7)/RTD

def tmf_obliquity(jde):
  Depsilon = 0.0
  arg = 0.0
  T = (jde - 2451545.) / 36525.
  T2 = T * T
  T3 = T * T2

  # Mean elongation of the Moon from the Sun
  D = (297.85036 + 445267.111480 * T - 0.0019142 * T2 + T3 / 189474.) * np.pi / 180.

  # Mean anomaly of the Sun (Earth)
  M = (357.52772 + 35999.050340 * T - 0.0001603 * T2 + T3 / 300000.) * np.pi / 180.

  # Mean anomaly of the Moon
  Mm = (134.96298 + 477198.867398 * T - 0.0086972 * T2 + T3 / 56250.) * np.pi / 180.

  # Moon's argument of latitude
  F = (93.27191 + 483202.017538 * T - 0.0036825 * T2 + T3 / 327270.) * np.pi / 180.

  ## Longitude of the ascending node of the Moon's mean orbit on the
  ##   ecliptic, measured from the mean equinox of the date 
  Omega = (125.04452 - 1934.136261 * T + 0.0020708 * T2 + T3 / 450000.) * np.pi / 180.

  ## Argument for trigonometric functions in terms of 
  ##   multiples of D, M, Md, F, Omega */
  mul = [
    [ 0, 0, 0, 0, 1],
    [-2, 0, 0, 2, 2],
    [ 0, 0, 0, 2, 2],
    [ 0, 0, 0, 0, 2],
    [ 0, 1, 0, 0, 0],
    [ 0, 0, 1, 0, 0],
    [-2, 1, 0, 2, 2],
    [ 0, 0, 0, 2, 1],
    [ 0, 0, 1, 2, 2],
    [-2,-1, 0, 2, 2],
    [-2, 0, 0, 2, 1],
    [ 0, 0,-1, 2, 2],
    [ 0, 0, 1, 0, 1],
    [ 2, 0,-1, 2, 2],
    [ 0, 0,-1, 0, 1],
    [ 0, 0, 1, 2, 1],
    [ 0, 0,-2, 2, 1],
    [ 2, 0, 0, 2, 2],
    [ 0, 0, 2, 2, 2],
    [-2, 0, 1, 2, 2],
    [ 0, 0,-1, 2, 1],
    [ 2, 0,-1, 0, 1],
    [-2, 2, 0, 2, 2],
    [ 0, 1, 0, 0, 1],
    [-2, 0, 1, 0, 1],
    [ 0,-1, 0, 0, 1],
    [ 2, 0,-1, 2, 1],
    [ 2, 0, 1, 2, 2],
    [ 0, 1, 0, 2, 2],
    [ 0,-1, 0, 2, 2],
    [ 2, 0, 0, 2, 1],
    [-2, 0, 2, 2, 2],
    [-2, 0, 1, 2, 1],
    [ 2, 0,-2, 0, 1],
    [ 2, 0, 0, 0, 1],
    [-2,-1, 0, 2, 1],
    [-2, 0, 0, 0, 1],
    [ 0, 0, 2, 2, 1] 
  ]

  ## Coefficients of sine and cosine for Depsilon in units of 0''.00001 ##
  c = [
     92025 + 8.9 * T,
      5736 - 3.1 * T,
       977 - 0.5 * T,
      -895 + 0.5 * T,
        54 - 0.1 * T,
                  -7,
       224 - 0.6 * T,
                 200,
       129 - 0.1 * T,
       -95 + 0.3 * T,
                 -70,
                 -53,
                 -33,
                  26,
                  32,
                  27,
                 -24,
                  16,
                  13,
                 -12,
                 -10,
                  -8,
                   7,
                   9,
                   7,
                   6,
                   5,
                   3,
                  -3,
                   3,
                   3,
                  -3,
                  -3,
                   3,
                   3,
                   3,
                   3,
                   3
  ]

  # Calculate sum over sine and cosine for Dph and Depsilon respectively
  for i in range(38):
    arg = mul[i][0] * D  + \
          mul[i][1] * M  + \
          mul[i][2] * Mm + \
          mul[i][3] * F  + \
          mul[i][4] * Omega

    Depsilon += c[i] * np.cos(arg)

  return (Depsilon / 3.6e7)/RTD

def tmf_mean_obliquity(jde):
  epsilon_0 = (23. * 3600.) + (26. * 60.) + 21.448

  T = (jde - 2451545.) / 36525.;
  U = T / 100;
  a = [
    -4680.93,
    -1.55,
     1999.25,
    -51.38,
    -249.67,
    -39.05,
     7.12,
     27.87,
     5.79,
     2.45
  ]

  # Ufac = { U, U^2, U^3, ... , U^10 }
  Ufac = 1.0;

  for i in range(10):
    Ufac *= U
    epsilon_0 += a[i]*Ufac

  return (epsilon_0 / 3600)/RTD

def tmf_true_obliquity(jde):
  epsilon_0 = tmf_mean_obliquity(jde)
  Depsilon = tmf_obliquity(jde)
  return epsilon_0 + Depsilon





def gmst(jd):
  T = (jd - 2451545.) / 36525.
  return 4.8949612127 + 6.3003880989850 * (jd - 2451545.0) + 0.00000677071 * T * T - (T * T * T / 675616.95)

def gast(jd, jde):
  # Get true obliquity of the ecliptic
  epsilon = tmf_true_obliquity(jde)

  # Get nutation of the ecliptic
  Dphi = tmf_nutation(jde);

  # Get Greenwhich Mean Siderial Time and add equation of the equinoxes
  return gmst(jd) + Dphi * np.cos(epsilon)



def last(jd, jde, longitude):
  # Get Greenwich Apparent Siderial Time  and correct for observer's longitude */
  return gast(jd, jde) + longitude

def rad2circle(phi):
  p = np.fmod(phi, 2*np.pi)
  return 2*np.pi + p if p<0 else p


