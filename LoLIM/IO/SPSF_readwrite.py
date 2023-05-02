#!/usr/bin/env python3
"""this is code to read the standard point-source format."""

## needs to be moved to main folder

import numpy as np

class pointSource_data:
    def __init__(self, input_fname=None, read_all_pointsources=True):

        ## TODO: make a reader object that can be sub-classed. an instance of that reader object is sent to this constructor. If use base-class, then this is empty. There is SPSF sub-class for reading files.
        ## then, to make a "spoofed" object (convert to this data-format). you just need to write a new sub-class

        ## also need a method to write this data to a file
        

        ## make an empty object. Fill it later if possible
        self.version = None
        self.notes = {}
        self.comments = []

        self.max_num_data = None
        self.timeID = None

        self.collums_headings = []
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

            else:
                ## should be descrptive row
                self.collums_headings = line_data
                self.dtype = np.dtype({ 'names':self.collums_headings,  'formats':[np.int]+[np.double]*(len(self.collums_headings)-1) })

                break



        self.datafile = datafile
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

                D = [ int(line_data[0]) ]
                D += [ float(line_data[i]) for i in range(1,len(self.collums_headings)) ]
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

                D = [ int(line_data[0]) ]
                D += [ float(line_data[i]) for i in range(1,len(self.collums_headings)) ]

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
            fout.write(c) ## this should include the newline

        ## notes
        for label, data in self.notes.items():
            fout.write('! ')
            fout.write( label )
            for d in date:
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

        
def make_SPSF_from_data(timeID, collumn_names, data_iterator):
    """give a timeID (as string), collumn names (list of strings), and a data iterator.
    each item of teh data iterator should be a 1D iterator that can be cast to tuple. Each item should correspond, in order, to the collums.
    Note the first six collums shoudl be defined according to teh SPSF format, which is not checked! (first is cast to int, following ones are cast to double)
    This function then returns a SPSF object"""

    new_SPSF = pointSource_data()
    new_SPSF.timeID = timeID

    new_SPSF.collums_headings = collumn_names
    new_SPSF.dtype = np.dtype({'names': new_SPSF.collums_headings, 'formats': [np.int] + [np.double] * (len(new_SPSF.collums_headings) - 1)})

    data_tmp = []
    for item in data_iterator:
        pointData = list( item )

        D = [int(pointData[0])]
        D += [float(pointData[i]) for i in range(1, len(new_SPSF.collums_headings))]
        D = tuple(D)

        data_tmp.append(D)

    new_SPSF.data = np.array(data_tmp, dtype=new_SPSF.dtype)

    return new_SPSF