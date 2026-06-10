#!/usr/bin/env python3

from . import mapper_header
try:
	from . import iterative_mapper
except Exception as e: 
	print('WARNING: cannot import LoLIM.iterative_mapper.iterative_mapper')
	print(e)
