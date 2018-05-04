from inverted_index import build_dataset,build_inverted_index,get_query_rel_docs,trim_query
import re
import os
import string
import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import math
import numpy as np
import pickle
def build():
	documents=build_dataset()
	print("Total number of documents : ",len(documents))
	inverted_index,avg_len=build_inverted_index(documents)
	file = open('documents.pkl', 'wb')
	pickle.dump(documents, file)
	file.close()

	file = open('index.pkl','wb')
	pickle.dump(inverted_index,file)
	file.close()
	file = open('avg_len.pkl', 'wb')
	pickle.dump(avg_len, file)
	file.close()
def weight(term,document):
	tf=0
	if document.docID not in inverted_index[term]:
		return 0
	title=[t.lower() for t in nltk.word_tokenize(document.title) if t==term]
	abstract=[t.lower() for t in nltk.word_tokenize(document.abstract) if t==term]
	tf=len(title)+len(abstract)	
	if tf==0:
		return 0
	k=2
	b=0.5
	doc_length=document.length
	avg_len_docs=avg_len
	total_docs=len(documents)
	term_docs=len(inverted_index[term])
	score=(tf/(k*((1-b)+b*(doc_length/avg_len_docs))+tf))*(math.log((total_docs-term_docs+0.5)/term_docs+0.5)/math.log(2))
	return score

def weight_score():
	file = open('index.pkl','rb')
	f = file.read()
	if f=='':
		print("no index created yet")
		return
	file.close()
	global inverted_index	
	inverted_index = pickle.loads(f)
	file = open('documents.pkl','rb')
	f = file.read()
	if f=='':
		print("no documents created yet")
		return
	file.close()	
	global documents
	documents = pickle.loads(f)
	file = open('avg_len.pkl','rb')
	f = file.read()
	if f=='':
		print("no avg_len calculated yet")
		return
	file.close()	
	global avg_len
	avg_len = pickle.loads(f)	
	score = {}
	count = 0
	for t in inverted_index:
		score[t] = {}
		for d in documents:
			score[t][d.docID] = weight(t,d)
		print(str(count)+"\t"+t+"term......\n")
		count= count + 1	
	file = open('score1.pkl','wb')
	pickle.dump(score,file)
	file.close()		

def query_score():
	file = open('index.pkl','rb')
	f = file.read()
	if f=='':
		print("no index created yet")
		return
	file.close()
	global inverted_index	
	inverted_index = pickle.loads(f)
	file = open('documents.pkl','rb')
	f = file.read()
	if f=='':
		print("no documents created yet")
		return
	file.close()	
	global documents
	documents = pickle.loads(f)
	file = open('avg_len.pkl','rb')
	f = file.read()
	if f=='':
		print("no avg_len calculated yet")
		return
	file.close()	
	global avg_len
	avg_len = pickle.loads(f)	
	score = {}
	count = 0
	countq = 0
	query_rel_dictionary=get_query_rel_docs()
	queries=list(query_rel_dictionary.keys())
	print(len(queries),'length')
	for query in queries:
		query=trim_query(query)
		for t in query:
			if t not in score:
				score[t] = {}
				for d in documents:
					score[t][d.docID] = weight(t,d)
				print(str(count)+"\t"+t+"term......\n")
				count= count + 1
		print(str(countq)+'\t'+"query")	
	file = open('scoreq.pkl','wb')
	pickle.dump(score,file)
	file.close()	

#build()
query_score()




