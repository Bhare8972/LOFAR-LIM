#!/usr/bin/env python3
"""this is code to read the standard point-source format."""

## needs to be moved to main folder

import numpy as np


## a few helper functions
def format_to_dtypes( format_list ):

    def HELPER(f):
        if f == 'i':
            return int 
        elif f == 'd':
            return np.double
        elif f[0] == 's':
            return np.dtype('U'+f[1:])
        else:
            print('No format with name', f)
            quit()

    return [ HELPER(f) for f in format_list ]

def format_to_functions( format_list ):

    def HELPER(f):
        if f == 'i':
            return int 
        elif f == 'd':
            return float
        elif f[0] == 's':
            return str
        else:
            print('No format with name', f)
            quit()

    return [ HELPER(f) for f in format_list ]


def format_to_dtypes( format_list ):

    def HELPER(f):
        if f == 'i':
            return int 
        elif f == 'd':
            return np.double
        elif f[0] == 's':
            return np.dtype('U'+f[1:])
        else:
            print('No format with name', f)
            quit()

    return [ HELPER(f) for f in format_list ]


## Use this class to read the data
class pointSource_data:
    def __init__(self, input_fname=None, read_all_pointsources=True):
        

        ## make an empty object. Fill it later if possible
        self.version = None
        self.notes = {}
        self.comments = []

        self.max_num_data = None
        self.timeID = None

        self.collums_headings = []
        self.collumn_dataFormats = None
        self.is_default_collumnFormat = None
        self.dtype = None
        self.data = None

        ### in case we want an empty file. Usefull for data-injection reasons
        if input_fname is None:
            self.version = 1
            return


        # if isinstance(input_fname, str):
        datafile = open(input_fname, 'r')

        version_line = datafile.readline().split()
        self.version = int( version_line[-1] )

        for line in datafile:
            line_data = line.split()

            if line_data[0][0] == '#':
                self.comments.append( line )
            elif line_data[0][0] == '%':
                self.notes[line_data[1]] = line_data[2:]
            elif line_data[0][0] == '!':
                if line_data[1] == 'timeID':
                    self.timeID = line_data[2]
                elif line_data[1] == 'max_num_data':
                    self.max_num_data = int( line_data[2] )
                elif line_data[1] == 'data_format':
                    self.collumn_dataFormats = line_data[2:]  
                    self.is_default_collumnFormat = False

            else:
                ## should be descrptive row
                self.collums_headings = line_data
                if self.collumn_dataFormats is None:
                    self.is_default_collumnFormat = True
                    self.collumn_dataFormats = ['i']+['d']*(len(self.collums_headings)-1)
                self.dtype = np.dtype({ 'names':self.collums_headings,  'formats':format_to_dtypes(self.collumn_dataFormats) })

                break



        self.datafile = datafile
        self.dataReadFuncs = format_to_functions(self.collumn_dataFormats)
        if read_all_pointsources:
            self.get_data()


    def get_data(self):
        if self.data is None:
            if self.max_num_data is not None:
                next_data_I = 0
                self.data = np.empty( self.max_num_data, dtype=self.dtype )
            else:
                data_tmp = []


            for line in self.datafile:
                line_data = line.split()

                D = [ self.dataReadFuncs[0](line_data[0]) ]
                D += [ self.dataReadFuncs[i](line_data[i]) for i in range(1,len(self.collums_headings)) ]
                D = tuple(D)

                if self.max_num_data is not None:
                    self.data[ next_data_I ] = D
                    next_data_I += 1
                else:
                    data_tmp.append( D )

            if self.max_num_data is None:
                self.data = np.array( data_tmp, dtype= self.dtype )
            else:
                self.data = self.data[:next_data_I]

        return self.data

    def iterdata(self):
        if self.data is None:

            for line in self.datafile:
                line_data = line.split()

                D = [ self.dataReadFuncs[0](line_data[0]) ]
                D += [ self.dataReadFuncs[i](line_data[i]) for i in range(1,len(self.collums_headings)) ]

                yield D

        else:
            for d in self.data:
                yield  d


    def write_to_file(self, out_fname):
        fout = open(out_fname, 'w')

        if self.version is None:
            print('cannot write SPSF, version not defined')
            return
        if self.timeID is None:
            print('cannot write SPSF, timeID not defined')
            return


        ## version
        fout.write('v ')
        fout.write(str( self.version ))
        fout.write('\n')

        ## comments
        for c in self.comments:
            if c[0] != '#':
                fout.write('# ')
            fout.write(c)
            if c[-1] != '\n':
                fout.write('\n')


        ## notes
        for label, data in self.notes.items():
            fout.write('% ')
            fout.write( label )
            for d in data:
                fout.write(' ')
                fout.write(d)
            fout.write('\n')

        ## commands
        if self.timeID is not None:
            fout.write('! timeID ')
            fout.write( self.timeID )
            fout.write('\n')

        if self.max_num_data is not None:
            if self.data is not None:
                towrite = max(self.max_num_data, len(self.data) )

            fout.write('! max_num_data ')
            fout.write( str(towrite) )
            fout.write( '\n' )
        elif self.data is not None:

            fout.write('! max_num_data ')
            fout.write( str( len(self.data) ) )
            fout.write( '\n' )

        if not self.is_default_collumnFormat:
            fout.write('! data_format')
            for sym in self.collumn_dataFormats:
                fout.write(' ')
                fout.write(sym)
            fout.write( '\n' )


        ## write collumn heading names
        for n in self.collums_headings:
            fout.write(n)
            fout.write(' ')
        fout.write('\n')

        ## WRITE DATA!!
        if self.data is not None:
            for point in self.data:
                for d in point:
                    fout.write( str(d) )
                    fout.write(' ')
                fout.write('\n')

  


## this function is to make a new SPSF dataset       
def make_SPSF_from_data(timeID, collumn_names, data_iterator, data_format=None, extra_notes=None, comments=None):
    """give a timeID (as string), collumn names (list of strings), and a data iterator.
    each item of teh data iterator should be a 1D iterator that can be cast to tuple. Each item should correspond, in order, to the collums.
    Note the first six collums shoudl be defined according to teh SPSF format, which is not checked! (first is cast to int, following ones are cast to double)
    This function then returns a SPSF object
    data_format should be None, or a list same length as collumn_names. Each item describes type of collumn. Options are 'i', 'd', 's' for integer, double, and string
    extra_notes should be a dictionary of lists of stings.
    comments should be list of strings"""

    new_SPSF = pointSource_data()
    new_SPSF.timeID = timeID

    new_SPSF.collums_headings = collumn_names


    if data_format is None:
        new_SPSF.collumn_dataFormats = ['i']+['d']*(len(self.collums_headings)-1)
        new_SPSF.is_default_collumnFormat = True
    else:
        new_SPSF.is_default_collumnFormat = False
        new_SPSF.collumn_dataFormats = data_format


    new_SPSF.dtype = np.dtype({'names': new_SPSF.collums_headings, 'formats': format_to_dtypes( new_SPSF.collumn_dataFormats)  })
    new_SPSF.dataReadFuncs = format_to_functions(new_SPSF.collumn_dataFormats)

    data_tmp = []
    for item in data_iterator:
        pointData = list( item )

        D = [  new_SPSF.dataReadFuncs[0](pointData[0])]
        D += [ new_SPSF.dataReadFuncs[i](pointData[i]) for i in range(1, len(new_SPSF.collums_headings))]
        D = tuple(D)

        data_tmp.append(D)

    new_SPSF.data = np.array(data_tmp, dtype=new_SPSF.dtype)
    new_SPSF.comments = comments
    new_SPSF.notes = extra_notes

    return new_SPSF