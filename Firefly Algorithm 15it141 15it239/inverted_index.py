import re
import os
import string
import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from collections import defaultdict

#Document class which stores each document object
class Document:
	def __init__(self,docID,title,abstract,mesh):
		self.table=str.maketrans(" "," ",string.punctuation)
		self.docID=docID
		self.title=title.translate(self.table)
		self.abstract=abstract.translate(self.table)
		self.mesh=mesh.translate(self.table)
		self.length=len(abstract.split())

	def __str__(self):
		return self.title

def build_dataset():
	file=open('ohsumed.87','r').read()
	f=file.split('.I')
	documents=[]
	for i in f:
		docID=re.search(r'(?<=\.U\n)[0-9]+',i)
		title=re.search(r'(?<=\.T\n).*',i)
		abstract=re.search(r'(?<=\.W\n).*',i)
		mesh=re.search(r'(?<=\.M\n).*',i)
		if docID and title and abstract and mesh:
			documents.append(Document(docID.group(),title.group(),abstract.group(),mesh.group()))
	return documents

def trim_query(query):
	query=[t.lower() for t in nltk.word_tokenize(query)]
	trimed_query=[]
	for token in query:
		if token in stopwords.words('english'):
			continue
		trimed_query.append(EnglishStemmer().stem(token))
	return trimed_query

def build_inverted_index(documents):
	inverted_index=defaultdict(list)
	len_docs=0
	for document in documents:
		print(document)
		docID=document.docID
		title=[t.lower() for t in nltk.word_tokenize(document.title)]
		abstract=[t.lower() for t in nltk.word_tokenize(document.abstract)]
		mesh=[t.lower() for t in nltk.word_tokenize(document.mesh)]
		for token in title:
			if token in stopwords.words('english'):
				continue
			token=EnglishStemmer().stem(token)
			if docID not in inverted_index[token]:
				inverted_index[token].append(docID)
		for token in abstract:
			if token in stopwords.words('english'):
				continue
			token=EnglishStemmer().stem(token)
			if docID not in inverted_index[token]:
				inverted_index[token].append(docID)
		for token in mesh:
			if token in stopwords.words('english'):
				continue
			token=EnglishStemmer().stem(token)
			if docID not in inverted_index[token]:
				inverted_index[token].append(docID)
		len_docs=len_docs+document.length
	avg_len_docs=len_docs/len(documents)
	return inverted_index,avg_len_docs

def generate_vocabulary(R):
	V=[]
	for document in R:
		title=[t.lower() for t in nltk.word_tokenize(document.title)]
		abstract=[t.lower() for t in nltk.word_tokenize(document.abstract)]
		mesh=[t.lower() for t in nltk.word_tokenize(document.mesh)]
		for token in title:
			if token in stopwords.words('english') or token in V:
				continue
			V.append(EnglishStemmer().stem(token))
		for token in abstract:
			if token in stopwords.words('english') or token in V:
				continue
			V.append(EnglishStemmer().stem(token))
		for token in mesh:
			if token in stopwords.words('english') or token in V:
				continue
			V.append(EnglishStemmer().stem(token))
	return V

#Not required as of now
def get_mesh():
	file=open('qrels.mesh.adapt.87','r').read()
	f=file.split('\n')
	dictionary=defaultdict(list)
	for i in f[:]:
		line=i.split('\t')
		dictionary[line[0]].append(line[1])
	return dictionary

def get_query_rel_docs():
	dictionary={}
	file=open('qrels.ohsu.batch.87','r').read()
	f=file.split('\n')
	for i in f[:]:
		line=i.split('\t')
		if line[0] not in dictionary:
			if(line == ['']):
				break
			dictionary[line[0]]=list()
			#print(line)
		dictionary[line[0]].append(line[1])
	
	# Key : QueryID 		Value : Relevant documents
	file=open('query.ohsu.1-63','r').read()
	f=file.split('<top>')
	for i in f[1:]:
		number=re.search(r'(?<=Number: ).*',i)
		title=re.search(r'(?<=Description:\n).*',i)
		dictionary[title.group()]=dictionary.pop(number.group())
	# print(dictionary)
	return dictionary	

# print(get_query_rel_docs())
# print(get_mesh())
# documents=build_dataset()
# inverted_index,avg_len=build_inverted_index(documents)
# print("Constructed inverted index : ")
# print(inverted_index)
