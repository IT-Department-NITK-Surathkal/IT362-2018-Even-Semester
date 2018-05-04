import sys
from os import listdir
from matplotlib import pyplot as plt
from matplotlib import style
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import pandas as pd
style.use('ggplot')


#For Plotting Graphs
for file in listdir('[ABSOLUTE-PATH-FOR-DATASET]'):
	file_path='[ABSOLUTE-PATH-FOR-DATASET]'+"/"+file

	df = pd.read_csv(file_path)
	x = df.Year
	y = df.Value
	x=x.tolist()
	y=y.tolist()
	fig = plt.figure()
	plt.plot(x,y)
	plt.subplots_adjust(bottom=.25, left=.25)
	file=file.split('.')
	plt.title(file[0]+' Population')
	plt.ylabel('Population')
	plt.xlabel('Year')
	fig.savefig('[ABSOLUTE-PATH-FOR-STORING-THE-GRAPHS'+file[0])


