import sys
from os import listdir
from matplotlib import pyplot as plt
from matplotlib import style
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import pandas as pd
style.use('ggplot')

#Fetching data from Wikipedia using wikiapi
from wikiapi import WikiApi
wiki = WikiApi()
wiki = WikiApi({ 'locale' : 'en'})


#For Plotting Graphs
for file in listdir('/Users/aimanabdullahanees/Desktop/IR-Project/Dataset/Population'):
	file_path='/Users/aimanabdullahanees/Desktop/IR-Project/Dataset/Population'+"/"+file
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
	plt.grid(True)
	from matplotlib2tikz import save as tikz_save
	tikz_save('/Users/aimanabdullahanees/Desktop/IR-Project/Latex/Population/'+file[0]+'.tex')



