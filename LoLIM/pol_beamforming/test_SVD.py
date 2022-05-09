#!/usr/bin/env python3

import numpy as np

import LoLIM.pol_beamforming.cython_beamforming_tools as cyt


def tst_SVD():
     #  input matrix:
    # 0.4032 + 0.0876i   0.1678 + 0.0390i   0.5425 + 0.5118i
    # 0.3174 + 0.3352i   0.9784 + 0.4514i  -0.4416 - 1.3188i
    # 0.4008 - 0.0504i   0.0979 - 0.2558i   0.2983 + 0.7800i
         
     # correct output
    # 0.4318 - 0.1398i   0.2869 - 0.1360i   0.3897 - 0.0370i
    # 0.3507 - 0.0725i   0.4354 - 0.1329i   0.2877 + 0.0565i
    # 0.3116 - 0.2626i   0.0042 + 0.2584i   0.2388 - 0.2806i
     
    A = np.array( [[ 0.4032 + 0.0876j, 0.1678 + 0.0390j, 0.5425 + 0.5118j],
                   [ 0.3174 + 0.3352j, 0.9784 + 0.4514j,-0.4416 - 1.3188j],
                   [ 0.4008 - 0.0504j, 0.0979 - 0.2558j, 0.2983 + 0.7800j]], dtype=np.cdouble )
     
     # cdef double complex[:,:] A = np.array( [[ 1],
     #                                         [ 1],
     #                                         [ 2] ,
     #                                         [ 3] ,
     #                                         [ np.exp(1j*np.pi/4)  ] ], dtype=np.cdouble )
     
     # cdef double complex[:,:] A = np.array( [[ 0.5, 0, 0],
     #                                         [ 0, 1+0.4j, 0],
     #                                         [ 0, 0, 6j]], dtype=np.cdouble )
     
     # cdef double complex[:,:] A = np.array( [[-5.40+7.40j, 1.09+1.55j, 9.88+1.91j],
     #                                         [ 6.00+6.38j, 2.60+0.07j, 4.92+6.31j],
     #                                         [ 9.91+0.61j, 3.98-5.26j,-2.11+7.39j],
     #                                         [-5.28-4.16j, 2.03+1.11j,-9.81-8.98j]],  dtype=np.cdouble)
     
     # cdef double complex[:,:] A = np.array( [[-5.40+7.40j, 6.00+6.38j, 9.91+0.61j, -5.28-4.16j],
     #                                         [ 1.09+1.55j, 2.60+0.07j, 3.98-5.26j, 2.03+1.11j],
     #                                         [ 9.88+1.91j, 4.92+6.31j,-2.11+7.39j, -9.81-8.98j]],  dtype=np.cdouble)
     
    B = np.array([0,1,0.5], dtype=np.cdouble)
    
    conditioning = -1 ## =1: use rcond, 0, full inversion, 1: throw one SV out
    
     
    M = A.shape[0]
    N = A.shape[1]
     
     
    inversinator = cyt.SVD_psuedoinversion( M, N, rcond = 1.0/100 )
    
    info = inversinator.set_matrix( A )
    print('SVD info', info)
    
    
     
    INVERSE = np.empty( (N,M), dtype=np.cdouble )
    rank = inversinator.get_psuedoinverse( INVERSE, conditioning)
    
    print('inverse. rank:', rank)
    print(INVERSE)
    print('CN', inversinator.rank_to_CN(rank) )
    print()
    print('dot with B')
    print( np.dot(INVERSE, B) )
    print()
    print()
   
    
    SOLUTION = np.empty(M, dtype=np.cdouble)
    inversinator.solve(B, SOLUTION, conditioning)
    
    print('solved')
    print(SOLUTION)
    print()
    
     
if __name__ == "__main__":
    tst_SVD()
    