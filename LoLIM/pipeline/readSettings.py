#!/usr/bin/env python3

import json

def readSettings(fname):
	return json.load( open(fname, 'r') )


