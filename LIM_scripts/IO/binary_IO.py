#!/usr/bin/env python3

##on APP machine

from struct import Struct, unpack, pack
import numpy as np

class BinaryIOError(Exception):
    def __init__(self, err):
        self.err=err
        
    def __str__(self):
        return str(self.err)
    
    def __repr(self):
        return str(self)

##codes:
## 1: long
## 2: double
## 3: long array
## 4: double array
## 5: string

long_struct = Struct('<l')
double_struct = Struct('<d')
NPdouble_type = np.dtype( '<f8' )

def at_eof(fin):
    A = fin.read(1)
    if len(A)!=1:
        return True
    else:
        fin.seek(-1,1)
        return False

#### reading single numbers
def next_is_long(fin):
    cde = long_struct.unpack(fin.read(4))[0]
    fin.seek(-4, 1)
    return cde == 1

def read_long(fin):
    cde = long_struct.unpack(fin.read(4))[0]
    if cde!=1:
        raise BinaryIOError( "long binary IO error")
        
    return long_struct.unpack( fin.read(4))[0]
    
def read_double(fin):
    cde = long_struct.unpack(fin.read(4))[0]
    if cde!=2:
        raise BinaryIOError( "double binary IO error")
        
    return double_struct.unpack( fin.read(8))[0]
    
#### read array of numbers
def read_long_array(fin):
    cde = long_struct.unpack(fin.read(4))[0]
    if cde!=3:
        raise BinaryIOError( "long array binary IO error")
        
    N = long_struct.unpack( fin.read(4) )[0]
    out = np.empty(N, dtype=np.int32)
    for i in range(N):
        out[i] = long_struct.unpack( fin.read(4) )[0]
        
    return out
    
def read_double_array(fin):
    cde = long_struct.unpack(fin.read(4))[0]
    if cde!=4:
        raise BinaryIOError( "double array binary IO error")
        
    N = long_struct.unpack( fin.read(4))[0]
    
#    out=np.array([ double_struct.unpack(fin.read(4+8)[-8:])[0] for i in xrange(N) ] ,dtype=np.double)

    data = np.fromfile(fin, dtype=NPdouble_type, count=N)

    return data

def skip_double_array(fin):
    
    cde = long_struct.unpack(fin.read(4))[0]
    if cde!=4:
        raise BinaryIOError( "double array binary IO error")
        
    N = long_struct.unpack( fin.read(4))[0]
    
    fin.seek(N*8  ,1)
    
####read a string
def read_string(fin):
	
    cde=long_struct.unpack(fin.read(4))[0]
    if cde!=5:
        raise BinaryIOError( "string binary IO error")
        
    N = long_struct.unpack( fin.read(4))[0]
    return fin.read(N).decode()
    
##codes:
## 1: long
## 2: double
## 3: long array
## 4: double array
## 5: string
    
####write single numbers
def write_long(fout, num):
    fout.write( long_struct.pack( 1 ))
    fout.write( long_struct.pack( num ))
    
def write_double(fout, num):
    fout.write( long_struct.pack( 2 ))
    fout.write( double_struct.pack( num ))
    
####write array of numbers
def write_long_array(fout, long_array):
    N = len(long_array)
    ##should probably do some error checking here....
    fout.write( long_struct.pack( 3 ))
    fout.write( long_struct.pack( N ))
    for i in range(N):
        fout.write( long_struct.pack( long_array[i] ) )
    
def write_double_array(fout, double_array):
    N= len(double_array)
    ##should probably do some error checking here....
    fout.write( long_struct.pack( 4 ))
    fout.write( long_struct.pack( N ))
    for i in range(N):
        fout.write( double_struct.pack( double_array[i] ))
        
####write a string
def write_string(fout, string):
    data = bytes(string, 'utf-8')
    N = len(data)
    fout.write(long_struct.pack( 5 ))
    fout.write( long_struct.pack( N ))
    fout.write(data)
        
        
