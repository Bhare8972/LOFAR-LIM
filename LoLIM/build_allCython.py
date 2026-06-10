#!/usr/bin/env python3

import pathlib
import subprocess
from os.path import join, isfile
from os import rename, chdir
import traceback



### this thing is a horrific hack. But only way I know to make this work ####

def doInstall():
    print('trying to install cython things!')

    # cython_files = [ 'NumLib/FFT/GSL_FFT.pyx' ]
    # setup_files =  ['NumLib/FFT/setup.py']
    # locs        =  ['NumLib/FFT/']

    # current_location = pathlib.Path(__file__).parent.resolve()
    # print('  i am at:', current_location)


    # for c,f,l in zip(cython_files, setup_files,locs):

    #     has_init = False
    #     total_loc = join(join(current_location,l)
    #     if isfile(join(total_loc,'__init__.py')):
    #         ## rename init, bcz messes up this hack
    #         rename(join(total_loc,'__init__.py'), join(total_loc,'__tmp__'))
    #         print('doing __init__ rename')
    #         has_init = True


    #     R1 = ['cython', '-a', join(current_location,c)]
    #     R2 = ['python',  join(current_location,f), 'build_ext', '--inplace'] ## this one is extra stupid
      
    #     subprocess.run(R1, capture_output=False, check=True)
    #     subprocess.run(R2, capture_output=False, check=True)


    #     if has_init:
    #         ## undo rename
    #         rename( join(total_loc,'__tmp__'), join(total_loc,'__init__.py'))


    cython_files = ['GSL_FFT.pyx' , 'GSL_LeastSquares.pyx',   'cython_utils.pyx',  'autoCorrelator_tools.pyx',      'cythonPipelineTools.pyx']
    setup_files =  ['setup.py',    'setup.py',                 'setup_utils.py',   'setup_autoCorrelator_tools.py', 'setup_utils.py'  ]
    locs        =  ['NumLib/FFT/', 'NumLib/LeastSquares/', 'iterativeMapper/',      'stationTimings/',               'pipeline/cythonTools/'] 

    current_location = pathlib.Path(__file__).parent.resolve()
    print('  i am at:', current_location)


    for c,f,l in zip(cython_files, setup_files,locs):

        print('building', c, 'in', l)

        total_loc = join(current_location,l)
        chdir(total_loc)


        has_init = False
        if isfile('__init__.py'):
            ## rename init, bcz messes up this hack
            #print('doing __init__ rename')
            rename('__init__.py', '__tmp__')
            has_init = True


        R1 = ['cython', '-a', c]
        R2 = ['python',  f, 'build_ext', '--inplace'] ## this one is extra stupid
      
        try:
            subprocess.run(R1, capture_output=False, check=True)
            subprocess.run(R2, capture_output=False, check=True)
        except Exception:
            print('cannot compile! : ', c)
            print('  exception:')
            print(traceback.format_exc())
            print()
            print('file exists:', isfile(c) )
            print('ls:')
            subprocess.run(['ls'], capture_output=False, check=True)



        if has_init:
            ## undo rename
            rename( '__tmp__', '__init__.py')

if __name__ == "__main__":
    doInstall()