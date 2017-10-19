#!/usr/bin/env python3

##ON APP MACHINE

#This is a module for generating python code and saving data to be transfered between computers.
#Primary purpose is to be able to make plots on a server, and be able to transfer them to personal computer and have them still be interactive

#### WARNING ####
## this code is very unsafe. It literaly just runs arbtrary python commands. So do NOT run any porta_code files that arn't from sources that you would trust with complete controll over your computer.
## this module needs to be replaced with something that has more.....limited.....functionality

import pickle

class code_logger(object):
    def __init__(self, fname):
        self.fname=fname
        self.lines=[] ##is a list of lines. Each line is a list. First item of the list is an integer, that says the kind of line and info to follow
        ##types:
        ## 0: statement, list also contains a string to be run
        ## 1: function, list contains function name, option argument list, and optional argument dictionary
        ## 2: save data as variable. List contains name of variable and data to save to it
        
#    def __del__(self):
#        self.save()

    def add_statement(self, STR):
        """ add a statement to be run. STR should be a string that can contain any valid python code"""
        self.lines.append( [0, STR] )
        
    def add_function(self, func, *largs, **dargs):
        """run a function on external data. first variable should be name of function, without parenthesis or arguments. 
        Following arguments should be just function arguments. Do not call print from here, use "output" instead"""
        self.lines.append( [1, func, largs, dargs] )
        
    def add_variable(self, var, data):
        """save external data as a variable. var should be variable name, and data should be the data to save to var"""
        self.lines.append( [2, var, data] )
        
    def save(self):
        with open(self.fname, "wb") as fout:
            pickle.dump(self.lines, fout, 2)
        
def read_file(fname):
    with open(fname, "rb") as fin:
        data=pickle.load(fin)
        
    for line in data:
        if line[0]==0:
            exec(line[1])
        elif line[0]==1:
            global __TMP1__, __TMP2__
            __TMP1__=line[2]
            __TMP2__=line[3]
            func=line[1]+"(*__TMP1__, **__TMP2__)"
            exec(func)
        elif line[0]==2:
            global __TMP__
            __TMP__=line[2]
            exec(line[1]+"=__TMP__")
            
            
class pyplot_emulator:
    """contains a series of functions that look like matplotlib, so that code that uses matplotlib functional interface can be easly replaced with code_logger"""
    
    def __init__(self, CL):
        self.CL = CL
        self.CL.add_statement("from matplotlib import pyplot as plt")
        
    def plot(self, *largs, **dargs):
        self.CL.add_function("plt.plot", *largs, **dargs)
        
    def scatter(self, *largs, **dargs):
        self.CL.add_function("plt.scatter", *largs, **dargs)
        
    def hist(self, *largs, **dargs):
        self.CL.add_function("plt.hist", *largs, **dargs)
        
    def show(self, *largs, **dargs):
        self.CL.add_function("plt.show", *largs, **dargs)
        
    def annotate(self, *largs, **dargs):
        self.CL.add_function("plt.annotate", *largs, **dargs)
    

#### todo: add other functionality that allows one to program in python, but code is written to file
#### E.G. add a class that emulates classes. make is so that these are returned from pyplot_emulator so that pyplots full interface can be emulated!!!
## with some clever coding one could emulate many behaviours of arbitrary objects. Including getters and setters. 


if __name__=="__main__":
    
#    data=[1,2,3]
#    
#    CL=code_logger("out.pybin")
#    CL.add_statement("print( 3 )") ### print '3' to screen
#    CL.add_function("print", data) ##runs function with data as argument.
#    CL.add_variable("A", data) ## save data to a variable name
#    CL.add_statement("print( A )") ##print the data again
#    CL.save()
#    
#    read_file("out.pybin")

#    read_file("/vol/astro3/lofar/lightning/noise_distributions/D20130619T100456.232Z/plots/TSTL148008_D20130619T100456.232Z_CS004_R000_tbb.h5")

    read_file(argv[1])