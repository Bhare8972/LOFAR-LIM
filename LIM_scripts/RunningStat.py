#!/usr/bin/env python3

import numpy as np

class RunningStat(object):
        def __init__(self):
            self.m_n = 0
            self.m_oldM = 0.0 
            self.m_newM = 0.0  
            self.m_oldS = 0.0  
            self.m_newS = 0.0

        def Push(self, x):
            self.m_n += 1

            ## See Knuth TAOCP vol 2, 3rd edition, page 232
            if (self.m_n == 1):
                self.m_oldM = x
                self.m_newM = x
                self.m_oldS = 0.0
            else:
                self.m_newM = self.m_oldM + (x - self.m_oldM)/self.m_n
                self.m_newS = self.m_oldS + (x - self.m_oldM)*(x - self.m_newM)
    
                ## set up for next iteration
                self.m_oldM = self.m_newM
                self.m_oldS = self.m_newS
                
                
        def NumDataValues(self):
            return self.m_n

        def Mean(self):
            if self.m_n > 0:
                return self.m_newM
            else:
                return 0.0

        def Variance(self):
            if self.m_n > 1:
                return self.m_newS/(self.m_n - 1)
            else:
                return 0.0

        def StandardDeviation(self):
            return np.sqrt( self.Variance() )
        
        
        
if __name__ == "__main__":
    N = 100000
    ave = 0.5
    std = 0.1
    
    nums = np.random.normal(ave, std, N)
    
    running_stats = RunningStat()
    for x in nums:
        running_stats.Push(x)
        
    print( running_stats.Mean(), running_stats.StandardDeviation() )
    print( np.average(nums), np.std(nums) )
