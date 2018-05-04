import ast
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import simplejson
import csv


###Accessing the Corpus###
file_path='key_fields.json'
f1 = open(file_path, 'r')
corpus=f1.read()
corpus = ast.literal_eval(corpus)
f1.close()


###Normalization###
for i in range(len(corpus)):
	corpus[i]['Title']=corpus[i]['Title'].lower()
	corpus[i]['Description']=corpus[i]['Description'].lower()



###Tokenization###
for i in range(len(corpus)):
	corpus[i]['Title']=word_tokenize(corpus[i]['Title'])
	corpus[i]['Description']=word_tokenize(corpus[i]['Description'])



###Lemmatization###
lemmatizer=WordNetLemmatizer()
for i in range(len(corpus)):
	for j in range(len(corpus[i]['Title'])):
		corpus[i]['Title'][j]=lemmatizer.lemmatize(corpus[i]['Title'][j])
	for k in range(len(corpus[i]['Description'])):
		corpus[i]['Description'][k]=lemmatizer.lemmatize(corpus[i]['Description'][k])



"""
Cleaning should have the following steps:
(i) Removing tokens that contains punctuations
(ii) Removing tokens that are just punctuations
(iii) Removing tokens that have one character 
"""

###(i) & (ii)###
translation = str.maketrans("","", string.punctuation)
for i in range(len(corpus)):
	for j in range(len(corpus[i]['Title'])):
		corpus[i]['Title'][j]=corpus[i]['Title'][j].translate(translation)
	for k in range(len(corpus[i]['Description'])):
		corpus[i]['Description'][k]=corpus[i]['Description'][k].translate(translation)


###(iii)###
for i in range(len(corpus)):
	corpus[i]['Title']=[token for token in corpus[i]['Title'] if len(token)>2]
	corpus[i]['Description']=[token for token in corpus[i]['Description'] if len(token)>2]


###Stopword Removal###
stop_words = set(stopwords.words('english'))
for i in range(len(corpus)):
	corpus[i]['Title']=[token for token in corpus[i]['Title'] if not token in stop_words]
	corpus[i]['Description']=[token for token in corpus[i]['Description'] if not token in stop_words]


###Creating Vocabulary###
vocabulary = Counter()
for i in range(len(corpus)):
	vocabulary.update(corpus[i]['Title'])
	vocabulary.update(corpus[i]['Description'])
	X_Label=[corpus[i]['X-Label']]
	vocabulary.update(X_Label)
	Y_Label=[corpus[i]['Y-Label']]
	vocabulary.update(Y_Label)
	X_min=[corpus[i]['X-min']]
	vocabulary.update(X_min)
	Y_min=[corpus[i]['Y-min']]
	vocabulary.update(Y_min)
	X_max=[corpus[i]['X-max']]
	vocabulary.update(X_max)
	Y_max=[corpus[i]['Y-max']]
	vocabulary.update(Y_max)

vocabulary_list = [word for word,frequency in vocabulary.items() if frequency >=1 and len(word)>1]
file_path='vocabulary.txt'
f = open(file_path, 'w')
simplejson.dump(vocabulary_list, f)
f.close()

documents=[]
for i in range(len(corpus)):
	corpus[i]['Description'].append(corpus[i]['X-Label'])
	corpus[i]['Description'].append(corpus[i]['Y-Label'])
	corpus[i]['Description'].append(corpus[i]['X-min'])
	corpus[i]['Description'].append(corpus[i]['Y-min'])
	corpus[i]['Description'].append(corpus[i]['X-max'])
	corpus[i]['Description'].append(corpus[i]['Y-max'])
	documents.append(corpus[i]['Description'])

file_path='documents.txt'
f = open(file_path, 'w')
simplejson.dump(documents, f)
f.close()

print("Preprocessing Successful!")






