#!/usr/bin/env python3

##ON APP MACHINE

#This is a module for generating python code and saving data to be transfered between computers.
#Primary purpose is to be able to make plots on a server, and be able to transfer them to personal computer and have them still be interactive

#### WARNING ####
## this code is very unsafe. It literaly just runs arbtrary python commands. So do NOT run any porta_code files that arn't from sources that you would trust with complete controll over your computer.
## this module needs to be replaced with something that has more.....limited.....functionality

from sys import argv
import pickle

class unique_var_name:
    """return a string that could represent a variable name. Each call gives a unique name. each name starts with '__var__'"""
    def __init__(self):
        self.num = 0
        
    def __call__(self):
        name = "__var__" + hex(self.num)
        self.num += 1
        return name

class code_logger(object):
    
    class var_handle:
        def __init__(self, name):
            self.name = name
    
    def __init__(self, fname):
        self.fname=fname
        self.lines=[] ##is a list of lines. Each line is a list. First item of the list is an integer, that says the kind of line and info to follow
        self.var_generator = unique_var_name()
        
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
        Returns a handle representing the return of the function. This handle can be used in future add_function calls"""
        var_name = self.var_generator()
        self.lines.append( [1, var_name, func, largs, dargs] )
        return self.var_handle( var_name )
        
    def add_variable(self, var_name, data):
        """save external data as a variable. var should be variable name, and data should be the data to save to var. return a handle that can be used in 'add_function'"""
        self.lines.append( [2, var_name, data] )
        return self.var_handle( var_name ) 
        
    def save(self):
        with open(self.fname, "wb") as fout:
            pickle.dump(self.lines, fout, 2)
        
def read_file(fname):
    with open(fname, "rb") as fin:
        data=pickle.load(fin)
        
    for line in data:
        if line[0]==0: ## general statment
            exec(line[1])
            
        elif line[0]==1: ## function
            global __TMP1__, __TMP2__
            __TMP1__ = line[3]
            for i,v in enumerate(__TMP1__):
                if isinstance(v, code_logger.var_handle):
                    __TMP1__[i] = eval( v.name )
                
            __TMP2__=line[4]
            for i,v in __TMP2__.items():
                if isinstance(v, code_logger.var_handle):
                    __TMP1__[i] = eval( v.name )
                
            
            func = line[1] + '=' + line[2]+"(*__TMP1__, **__TMP2__)"
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
        return self.CL.add_function("plt.plot", *largs, **dargs)
        
    def scatter(self, *largs, **dargs):
        return self.CL.add_function("plt.scatter", *largs, **dargs)
        
    def hist(self, *largs, **dargs):
        return self.CL.add_function("plt.hist", *largs, **dargs)
        
    def show(self, *largs, **dargs):
        return self.CL.add_function("plt.show", *largs, **dargs)
        
    def annotate(self, *largs, **dargs):
        return self.CL.add_function("plt.annotate", *largs, **dargs)
    
    def gca(self):
        return self.CL.add_function("plt.gca")
    

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