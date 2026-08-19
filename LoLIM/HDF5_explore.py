#!/usr/bin/env python3
""" this is a command-line tool to explore HDF5 files."""
import sys
import os


os.environ["HDF5_EXTFILE_PREFIX"]='${ORIGIN}'  ## need this for raw files to be relative location to header files
import h5py

try:
	import matplotlib.pyplot as plt
	has_plotting = True
except:
	print('plotting not available')
	has_plotting = False

def continuation(key, list):
	ret = []
	for l in list:
		if l.startswith(key):
			ret.append( l )

	return ret


if __name__=="__main__":
	if len(sys.argv) != 2:
		print('wrong number arguments given. Expected 2, got:', len(sys.argv), 'Second object should be file name of the HDF5 file.' )
		quit()

	print('opening file:', sys.argv[1])
	file = h5py.File(sys.argv[1], "r")
	directory = os.path.dirname(sys.argv[1])
	print('   at directory', directory)
	current_obj = file

	while True:
		print()
		command = input("cmd:")
		words = command.split()

		try:

			if words[0] == 'attr':
				if len(words)==1:
					print('attributes:')
					print( current_obj.attrs.keys() )
				elif len(words)==2:
					if words[1] in current_obj.attrs:
						print( current_obj.attrs[words[1]] )
					else:
						print('attribute does not exist:', words[1])

				elif len(words)==3 and words[2]=='c':
					c_list = continuation(words[1], current_obj.attrs.keys())
					if len(c_list) == 0:
						print('No attribute starts with:', words[1])
					elif len(c_list) > 1:
						print('possible attributes:')
						print( c_list )
					elif len(c_list) == 1:
						print('attribute', c_list[0] )
						print( current_obj.attrs[ c_list[0] ] )


			elif words[0] == 'all_attrs':
				for k, v in current_obj.attrs.items():
					print('attr key:', k)
					print('   value:', v)

				print()


			elif words[0] == 'name':
				print('current name:', current_obj.name)

			elif words[0] == 'parent':
				print('parent:', current_obj.parent)

			elif words[0] == 'child':
				if len(words)==1:
					print("child names:")
					print(current_obj.keys())
				elif len(words)==2:
					if not words[1] in current_obj:
						print('child not in object:', words[1])
					else:
						child = current_obj[ words[1] ]
						if child is None:
							print('child is none!')
						else:
							current_obj = child

				elif len(words)==3 and words[2]=='c':
					c_list = continuation(words[1], current_obj.keys())
					if len(c_list) == 0:
						print('No child starts with:', words[1])
					elif len(c_list) > 1:
						print('possible children:')
						print( c_list )
					elif len(c_list) == 1:
						print('child', c_list[0] )
						child = current_obj[ c_list[0] ]
						if child is None:
							print('child is none!')
						else:
							current_obj = child

			elif words[0] == 'goto_parent':
				print('Go to parent')
				current_obj = current_obj.parent

			elif words[0] == 'goto_file':
				current_obj = file
				print('am now at file level')

			elif words[0] == 'type':
				try:
					S = current_obj.shape
					dt = current_obj.dtype
					print('is dataset of size:', S, 'and type', dt)
				except:
					try:
						t = current_obj.keys()
						print('is a group')
					except:
						print('I don"t know, got an error')

			elif words[0] == 'plot1D':

				CWD = os.getcwd()

				#os.chdir(directory)

				S = current_obj.shape
				if len(S)!=1:
					print('Can only plot 1D dataset')
					continue

				if len(words)==2:
					start = int(words[1])
					L = 1000

					if start+L > S[0]:
						L = S[0]-start

				elif len(words)==3:
					start = int(words[1])
					L = int(words[2])


				print('data from', start, 'to', start+L)
				DATA = current_obj[start:start+L]

				#os.chdir(CWD)
				print('  len data read:', len(DATA))

				plt.plot( DATA )
				plt.savefig('./out.pdf')
				plt.show()
				print('plotting complete')



				

			elif words[0] == 'help':
				print('This is a simple command-line script to explore the structure of HDF5 files')
				print('  attr [name] [c]')
				print('    Print names of all attributes of current object. If followed by name of an attribute, then that attribute will be printed.')
				print('      If there is a third argument that \'c\', then attribute will be guessed based on partial naming.')
				print('  all_attrs')
				print('     Prints names and values of all attributes.')
				print('  name')
				print('    Print name of current object')
				print('  parent')
				print('    Print name of parent')
				print('  child [name] [c]')
				print('    Print name of all children of the current object. If followed by name of a child, then the child will be made the current object.')
				print('      If there is a third argument that \'c\', then child will be guessed based on partial naming.')
				print('  goto_parent')
				print('    Make the parent the current object')
				print('  goto_file')
				print('    Undo all progression. Make the original file object the current object.')
				print('  type')
				print('      prints "dataset" or "group", depending on what current object is. If dataset, also give shape and dtype')
				print('  plot1D (stat_index) [number]')
				print('      if this is a datset, and that dataset is 1D, plot some data. first argument is an integer, and is the starting index. Second argument is optional, and is an integer. It is the number of datapoints. It defaults to 1000 (or max of datset)')
				print('  exit')
				print('    It is very difficult to describe what this command does.')

			elif words[0] == 'exit':
				exit()
			else:
				print('command not recgnized!')

		except Exception as e:
			print('ERROR try again!')
			print(e)

