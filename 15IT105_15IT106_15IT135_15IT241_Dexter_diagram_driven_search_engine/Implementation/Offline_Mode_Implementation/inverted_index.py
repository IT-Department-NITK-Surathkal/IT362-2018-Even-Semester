import ast
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

###Accessing the Documents###
file_path='key_fields.json'
f1 = open(file_path, 'r')
documents=f1.read()
documents = ast.literal_eval(documents)
f1.close()

tag='Description'
###Normalization###
for i in range(len(documents)):
	documents[i][tag]=documents[i][tag].lower()


###Tokenization###
for i in range(len(documents)):
	documents[i][tag]=word_tokenize(documents[i][tag])


###Stemming###
ps=PorterStemmer()
for i in range(len(documents)):
	for j in range(len(documents[i][tag])):
		documents[i][tag][j]=ps.stem(documents[i][tag][j])


###Lemmatization###
lemmatizer=WordNetLemmatizer()
for i in range(len(documents)):
	for j in range(len(documents[i][tag])):
		documents[i][tag][j]=lemmatizer.lemmatize(documents[i][tag][j])


"""
Cleaning should have the following steps:
(i) Removing tokens that contains punctuations
(ii) Removing tokens that are just punctuations
(iii) Removing tokens that have one character 
"""

###(i) & (ii)###
translation = str.maketrans("","", string.punctuation)
for i in range(len(documents)):
	for j in range(len(documents[i][tag])):
		documents[i][tag][j]=documents[i][tag][j].translate(translation)


###(iii)###
for i in range(len(documents)):
	documents[i][tag]=[token for token in documents[i][tag] if len(token)>2]



###Stopword Removal###
stop_words = set(stopwords.words('english'))
for i in range(len(documents)):
	documents[i][tag]=[token for token in documents[i][tag] if not token in stop_words]


Inverted_Index=dict()
###Inverted Index###
###Format:- token : [Doc_id]###
for i in range(len(documents)):
	for j in range(len(documents[i][tag])):
		if documents[i][tag][j] not in Inverted_Index:
			Inverted_Index[documents[i][tag][j]] = []
			Inverted_Index[documents[i][tag][j]].append(documents[i]['doc_id'])
		else:
			Inverted_Index[documents[i][tag][j]].append(documents[i]['doc_id'])




print("Inverted Index")
print(Inverted_Index)



