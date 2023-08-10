#!/usr/bin/env python3

from distutils.sysconfig import get_python_lib 
from os import symlink
from os.path import dirname, abspath

if __name__ == "__main__":

    print("This install file just places a sym-link named GLP in the python directory.")

    i=''
    while not (i=='y' or i=='n'):
        i = input('type y to continue. n to quit')

    if i=='n':
        quit()

    location = get_python_lib()
    print("placing symbolic link in path:", location)
    location += '/GLP'
    current_path = dirname(abspath(__file__))
    symlink(current_path, location,  target_is_directory=True)