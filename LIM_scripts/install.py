#!/usr/bin/env python3

from distutils.sysconfig import get_python_lib 
from os import symlink
from os.path import dirname, abspath

if __name__ == "__main__":
    location = get_python_lib()
    print("placing symbolic link in path:", location)
    location += '/LoLIM'
    current_path = dirname(abspath(__file__))
    symlink(current_path, location,  target_is_directory=True)