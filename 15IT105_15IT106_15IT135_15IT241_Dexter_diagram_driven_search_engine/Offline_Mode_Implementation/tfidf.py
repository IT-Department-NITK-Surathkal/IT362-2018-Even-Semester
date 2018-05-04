import sys
import ast
from collections import Counter
from os import listdir
import simplejson
import math
import numpy as np
import csv
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


###Term Frequency###
def term_frequency(documents,vocabulary_list):
	TF=[]
	for document in documents:
		tf_per_document=[]
		for word in vocabulary_list:
			fij=document.count(word)
			if fij>0:
				tf=1+math.log(fij,2)
			else:
				tf=0
			tf_per_document.append(tf)
		TF.append(tf_per_document)

	return TF


###Inverse Document Frequency###
def inverse_document_frequency(documents,vocabulary_list):
	IDF=[]
	N=len(documents)
	for word in vocabulary_list:
		count=0
		for document in documents:
			if word in document:
				count+=1
		if count == 0:
			idf=0
		else:
			idf=math.log(N/count,2)
		IDF.append(idf)
	return IDF


###Term Frequency Inverse Document Frequency###
def term_frequency_inverse_document_frequency(TF,IDF):
	IDF=np.array(IDF)
	TFIDF=[]
	for tf in TF:
		tf=np.array(tf)
		tfidf=tf*IDF
		TFIDF.append(tfidf.tolist())
	return TFIDF


###Accessing the Vocabulary###
file_path='vocabulary.txt'
f1 = open(file_path, 'r')
vocabulary_list=f1.read()
vocabulary_list = ast.literal_eval(vocabulary_list)
f1.close()

###Accessing the Documents###
file_path='documents.txt'
f1 = open(file_path, 'r')
documents=f1.read()
documents = ast.literal_eval(documents)
f1.close()



###Performing TFIDF###
TF=term_frequency(documents,vocabulary_list)
IDF=inverse_document_frequency(documents,vocabulary_list)
TFIDF=term_frequency_inverse_document_frequency(TF,IDF)


file_path='idf.txt'
f = open(file_path, 'w')
simplejson.dump(IDF, f)
f.close()


###Generating CSV###
file_path='tfidf.csv'
with open(file_path, 'w') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(vocabulary_list)
	
	for tfidf in TFIDF:
		wr.writerow(tfidf)
	#wr.writerow(TFIDF_query)

