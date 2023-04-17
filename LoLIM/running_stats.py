#!/usr/bin/env python3
""" This is a set of tools to calculate mean and standard devaitoin in a running fashion, i.e. not saving every variable"""

import numpy as np

class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())

class ManyRunningStats:
    """calculate runing stats for many variables"""

    def __init__(self, n_stats, dtype=float):
        self.n_stats = n_stats
        self.dtype = dtype
        
        self.n = 0
        self.M = np.zeros(self.n_stats, dtype=self.dtype)
        self.S = np.zeros(self.n_stats, dtype=self.dtype)

        self.TMP1 = np.empty(self.n_stats, dtype=self.dtype)
        self.TMP2 = np.empty(self.n_stats, dtype=self.dtype)

    def clear(self):
        self.n = 0
        self.M[:] = 0
        self.S[:] = 0

    def push(self, x):
        """x should be a numpy array of lenght n_stats"""
        self.n += 1

        if self.n == 1:
            self.M[:] = x
            self.S[:] = 0.0
        else:
            self.TMP1[:] = x
            self.TMP1 -= self.M

            self.TMP2[:] = self.TMP1
            self.TMP2 *= 1.0/self.n
            self.M += self.TMP2

            self.TMP2[:] = self.M
            self.TMP2 *= -1
            self.TMP2 += x
            self.TMP2 *= self.TMP1

            self.S += self.TMP2

    def mean(self):
        return self.M

    def variance(self):
        return self.S / (self.n - 1) if self.n > 1 else np.zeros(self.n_stats, dtype=self.dtype)

    def standard_deviation(self):
        V = self.variance()
        np.sqrt(V, out=V)
        return V


if __name__ == "__main__":
    ## TEST!

    A = np.random.normal(loc=5, scale=10, size=(3,1000))

    T1 = RunningStats()
    for x in A.flatten():
        T1.push(x)
    print('t1 ave:', T1.mean() )
    print('  should be 5')
    print('t1 std:', T1.standard_deviation() )
    print('  should be 10')

    T2 = ManyRunningStats(3, float)
    for i in range( A.shape[1] ):
        T2.push( A[:,i] )
    print('t2 ave:', T2.mean() )
    print('  should be 5, 5, 5')
    print('t2 std:', T2.standard_deviation() )
    print('  should be 10, 10, 10')






