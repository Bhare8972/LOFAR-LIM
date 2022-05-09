#!/usr/bin/env python3
"""this is code to read the standard point-source format."""

import numpy as np

class pointSource_data:
    def __init__(self, file):
        if isinstance(file, str):
            data = open(file, 'r')

        version_line = data.readline().split()
        self.version = int( version_line[-1] )

        mode = 0 ## 0: reading header, 1: reading data

        self.notes = {}
        self.max_num_data = None

        for line in data:
            line_data = line.split()
            if mode == 0:
                if line_data[0] == '#':
                    continue
                elif line_data[0] == '%':
                    self.notes[line_data[1]] = line_data[1:]
                elif line_data[0] == '!':
                    if line_data[1] == 'timeID':
                        self.timeID = line_data[2]
                    elif line_data[1] == 'max_num_data':
                        self.max_num_data = int( line_data[2] )

                else:
                    ## should be descrptive row
                    self.collums_headings = line_data
                    mode = 1
                    self.dtype = np.dtype({ 'names':self.collums_headings,  'formats':[np.int]+[np.double]*(len(self.collums_headings)-1) })

                    if self.max_num_data is not None:
                        next_data_I = 0
                        self.data = np.empty( self.max_num_data, dtype=self.dtype )
                    else:
                        data_tmp = []

            elif mode == 1:
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

