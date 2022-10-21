#!/usr/bin/env python3


import numpy as np
from LoLIM.FFT import complex_fft_obj

def check_within(DATA, value, error):
    if np.any( DATA < (value-error) ):
        return False
    elif np.any( DATA > (value+error) ):
        return False
    else:
        return True

if __name__ == "__main__":
    N = 100

    FFT_OBJ = complex_fft_obj( N )
    print('USING GSL:', not FFT_OBJ.using_numpy)

    A = np.full(N, 1, dtype=complex)
    FFT_OBJ.fft(A)
    print('TEST1 (fft)') ## real 100 followed by all zero
    print("  ", check_within( np.real(A[0]), 100, 0.1) and check_within(np.real(A[1:]), 0, 0.1) and check_within(np.imag(A), 0, 0.1))
    print()

    FFT_OBJ.ifft(A) ## all 1
    print('TEST2 (ifft)')
    print("  ", check_within( np.real(A[0]), 1, 0.1) and check_within(np.imag(A), 0, 0.1))
    print()


    B = np.full(2*N, 1, dtype=complex)
    B[1::2] = 2.0
    S = B[::2]
    FFT_OBJ.fft(S)
    print('TEST3 (slicing)')
    print("  ", check_within( np.real(S[0]), 100, 0.1) and check_within(np.real(S[1:]), 0, 0.1) and check_within(np.imag(S), 0, 0.1))
    print("  ", check_within( np.real(B[1::2]), 2, 0.1) and check_within(np.imag(B[1::2]), 0, 0.1) )
