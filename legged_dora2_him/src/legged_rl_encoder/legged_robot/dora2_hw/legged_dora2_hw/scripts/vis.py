import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
import numpy as np
import pandas as pd
import sys

def read_data(file_path, skip_row_val):
	column_names = ['time',

					'lhipz_pos', 'lhipx_pos', 'lhipy_pos', 'lknee_pos', 'lankley_pos',
					'rhipz_pos', 'rhipx_pos', 'rhipy_pos', 'rknee_pos', 'rankley_pos',
					'lhipz_vel', 'lhipx_vel', 'lhipy_vel', 'lknee_vel', 'lankley_vel',
					'rhipz_vel', 'rhipx_vel', 'rhipy_vel', 'rknee_vel', 'rankley_vel',
					'lhipz_tor', 'lhipx_tor', 'lhipy_tor', 'lknee_tor', 'lankley_tor',
					'rhipz_tor', 'rhipx_tor', 'rhipy_tor', 'rknee_tor', 'rankley_tor',

					'test_phase',

					'lhipz_poscmd', 'lhipx_poscmd', 'lhipy_poscmd', 'lknee_poscmd', 'lankley_poscmd',
					'rhipz_poscmd', 'rhipx_poscmd', 'rhipy_poscmd', 'rknee_poscmd', 'rankley_poscmd',
					'lhipz_velcmd', 'lhipx_velcmd', 'lhipy_velcmd', 'lknee_velcmd', 'lankley_velcmd',
					'rhipz_velcmd', 'rhipx_velcmd', 'rhipy_velcmd', 'rknee_velcmd', 'rankley_velcmd',
					'lhipz_torcmd', 'lhipx_torcmd', 'lhipy_torcmd', 'lknee_torcmd', 'lankley_torcmd',
					'rhipz_torcmd', 'rhipx_torcmd', 'rhipy_torcmd', 'rknee_torcmd', 'rankley_torcmd',

					'larm1_poscmd', 'larm2_poscmd', 'larm3_poscmd', 'larm4_poscmd',
					'rarm1_poscmd', 'rarm2_poscmd', 'rarm3_poscmd', 'rarm4_poscmd',
					'larm1_velcmd', 'larm2_velcmd', 'larm3_velcmd', 'larm4_velcmd',
					'rarm1_velcmd', 'rarm2_velcmd', 'rarm3_velcmd', 'rarm4_velcmd',
					'larm1_torcmd', 'larm2_torcmd', 'larm3_torcmd', 'larm4_torcmd',
					'rarm1_torcmd', 'rarm2_torcmd', 'rarm3_torcmd', 'rarm4_torcmd',

					'larm1_pos', 'larm2_pos', 'larm3_pos', 'larm4_pos',
					'rarm1_pos', 'rarm2_pos', 'rarm3_pos', 'rarm4_pos',
					'larm1_vel', 'larm2_vel', 'larm3_vel', 'larm4_vel',
					'rarm1_vel', 'rarm2_vel', 'rarm3_vel', 'rarm4_vel',
					'larm1_tor', 'larm2_tor', 'larm3_tor', 'larm4_tor',
					'rarm1_tor', 'rarm2_tor', 'rarm3_tor', 'rarm4_tor',
					
					'not_useful'
                    ]
	data = pd.read_csv(file_path, header = None, names = column_names, skiprows=skip_row_val)

	return data 

	

#‘b’ blue 
#‘g’ green 
#‘r’ red 
#‘c’ cyan 
#‘m’ magenta 
#‘y’ yellow 
#‘k’ black 
#‘w’ white  
def visSingle(name):
	#@ para name: specify which joint, such as rlegz/rlegx/rlegy...
	#@ para kind: specify which item viewed, such as tor/pos/vel..
		plt.figure(name)
		plt.title(name)
		plt.plot(dataset['time'].values, dataset[name].values, 'r', label=name, linewidth = 1.2)
		plt.legend()
def visCmdAndActual(name):
		plt.figure(name)
		plt.title(name)
		plt.subplot(3, 1, 1)
		plt.plot(dataset['time'].values, dataset[name+'_poscmd'].values, 'r', label=name+'_poscmd', linewidth = 1.2)
		plt.plot(dataset['time'].values, dataset[name+'_pos'].values, 'g', label=name+'_pos', linewidth = 1.2)
		plt.legend()
		plt.subplot(3, 1, 2)
		plt.plot(dataset['time'].values, dataset[name+'_velcmd'].values, 'r', label=name+'_velcmd', linewidth = 1.2)
		plt.plot(dataset['time'].values, dataset[name+'_vel'].values, 'g', label=name+'_vel', linewidth = 1.2)
		plt.legend()
		plt.subplot(3, 1, 3)
		plt.plot(dataset['time'].values, dataset[name+'_torcmd'].values, 'r', label=name+'_torcmd', linewidth = 1.2)
		plt.plot(dataset['time'].values, dataset[name+'_tor'].values, 'g', label=name+'_tor', linewidth = 1.2)
		plt.legend()

def func():
	visSingle('test_phase')

	visCmdAndActual('lhipz')
	visCmdAndActual('lhipx')
	visCmdAndActual('lhipy')
	visCmdAndActual('lknee')
	visCmdAndActual('lankley')

	visCmdAndActual('rhipz')
	visCmdAndActual('rhipx')
	visCmdAndActual('rhipy')
	visCmdAndActual('rknee')
	visCmdAndActual('rankley')

	visCmdAndActual('larm1')
	visCmdAndActual('larm2')
	visCmdAndActual('larm3')
	visCmdAndActual('larm4')

	visCmdAndActual('rarm1')
	visCmdAndActual('rarm2')
	visCmdAndActual('rarm3')
	visCmdAndActual('rarm4')


	plt.show(block=False)
	plt.pause(1)
	input("press ctrl+c to quit...")
	plt.close(all)

dataset = read_data('/home/noetix/noetix-core/build/user/dora2-core/log.csv', 0)


while True:
	func()