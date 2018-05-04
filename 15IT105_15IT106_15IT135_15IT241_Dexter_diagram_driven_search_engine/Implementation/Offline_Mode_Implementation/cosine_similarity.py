import csv
import numpy as np
import math
from operator import itemgetter
import ast

documents=[]
file_path='tfidf.csv'
file=open(file_path, "r")
reader = csv.reader(file)
for line in reader:
	documents.append(line)

###Accessing the TFIDF of Query###
file_path='tfidf.txt'
f1 = open(file_path, 'r')
query=f1.read()
query = ast.literal_eval(query)
query=list(map(float, query))
f1.close()

del documents[0]
flag=1
COS_SIM=[]
for i in range(len(documents)):
	document=list(map(float, documents[i]))
	numerator=np.multiply(document,query)
	numerator=numerator.tolist()
	numerator=sum(numerator)
	a=np.square(query)
	a=a.tolist()
	a=sum(a)
	a=math.sqrt(a)
	b=np.square(document)
	b=b.tolist()
	b=sum(b)
	b=math.sqrt(b)
	if (a*b) == 0:
		flag=0
		break
	cos_sim=numerator/(a*b)
	COS_SIM.append([i+1,cos_sim])

import json
data = []
# data['query'] = []
	

if flag == 0:
	print("Your search did not match any documents.")
else:
	# print("top-10")
	COS_SIM.sort(key=itemgetter(1), reverse=True)
	print(COS_SIM[0:11])
	for i in range(10):
		new_data = json.load(open('key_fields.json'))
		
		data.append({
			'id': COS_SIM[i][0],
			'score': COS_SIM[i][1],
			'title': new_data[COS_SIM[i][0]]["Title"],
			'description': new_data[COS_SIM[i][0]]["Description"],
			'x_label': new_data[COS_SIM[i][0]]["X-Label"],
			'y_label': new_data[COS_SIM[i][0]]["Y-Label"],
			'x_min': new_data[COS_SIM[i][0]]["X-min"],
			'x_max': new_data[COS_SIM[i][0]]["X-max"],
			'y_min': new_data[COS_SIM[i][0]]["Y-min"],
			'y_max': new_data[COS_SIM[i][0]]["Y-max"],
			'file-location': new_data[COS_SIM[i][0]]["File_location"]
		})
	with open('static/js/data.json', 'w') as outfile:
		outfile.write("var results = ")
		json.dump(data, outfile)
