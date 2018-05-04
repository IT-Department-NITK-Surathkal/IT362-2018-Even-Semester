import sys
from os import listdir

#For plotting Graphs using Matplotlib
from matplotlib import pyplot as plt
from matplotlib import style
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import pandas as pd
style.use('ggplot')

#Fetching data from Wikipedia using wikiapi
"""
from wikiapi import WikiApi
wiki = WikiApi()
wiki = WikiApi({ 'locale' : 'en'})
"""
import wikipedia

#Support for creation of PDFs
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
Story=[]

incomplete=[]
for file in listdir('Graphs/Population'):
	file_path='Graphs/Population'+"/"+file
	
	#Paragraph Generator
	copy_file=file
	file=file.split('.')
	if wikipedia.page(file[0]):
		result = wikipedia.page(file[0])
		Content=result.content
	
		#PDF Generator
		doc = SimpleDocTemplate("/Users/aimanabdullahanees/Desktop/IR-Project/PDFs/Population_PDFs/"+file[0]+".pdf",pagesize=letter,
								rightMargin=72,leftMargin=72,
								topMargin=72,bottomMargin=18)
			
		title="<center><strong>"+file[0]+"</strong></center>"
		Story.append(Paragraph(title, styles["Title"]))

		logo = file_path
		im = Image(logo, 5*inch, 5*inch)
		Story.append(im)
		Story.append(Paragraph(Content, styles["Normal"]))
		doc.build(Story)
		print("Done:--->",file[0])
	else:
		incomplete.append(file[0])
print(incomplete)
	
