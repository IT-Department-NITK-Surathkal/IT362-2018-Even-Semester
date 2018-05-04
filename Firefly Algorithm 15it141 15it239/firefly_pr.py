import numpy as np
import math
import nltk
from inverted_index import build_dataset,build_inverted_index,trim_query,generate_vocabulary,get_query_rel_docs
from nltk.corpus import stopwords
import pickle
import csv
#Fixed constants
alpha0=1
gamma=1
theta=0.95
N = 5 #For p@5 and map for varying Q' length
T = 10#For p@5 and map for varying Q' length
#Okapi BM25 Model for Original Query
def bm(doc,orig_query):
	score=0
	for t in orig_query:
		score=score+weight(t,doc)
	return score

def bm_fast(doc,query,score_all_term):
	score = 0
	for t in query:
		#print(t)
		if t in score_all_term:
			if doc.docID in score_all_term[t]:
				score = score + score_all_term[t][doc.docID]
				#print("fast",t,doc.docID)
			else:
				#print("second fast",t,doc.docID)				
				score_all_term[t][doc.docID] = weight(t,doc)
				score = score + score_all_term[t][doc.docID]	
				
		else:
			#print(score_all_term)
			print("slow",t,doc.docID)
			score_all_term[t] = {}			
			score_all_term[t][doc.docID] = weight(t,doc)
			score = score + score_all_term[t][doc.docID]
						
	#print(score)
	return score			

def rocchio_fast(term,R,score_all_term):
	score = 0
	for doc in R:
		if term in score_all_term:
			if doc.docID in score_all_term[term]:
				score = score + score_all_term[term][doc.docID]
			else:
				print("seconf fast",term,doc.docID)				
				score_all_term[term][doc.docID] = weight(term,doc)
				score = score + score_all_term[term][doc.docID]
		else:
			print("slow",term,doc.docID)
			score_all_term[term] = {}			
			score_all_term[term][doc.docID] = weight(term,doc)
			score = score + score_all_term[term][doc.docID]
	#print(score)			
	return score			

#Rochhio weight for expansion terms
def rocchio(term,R):
	score=0
	for doc in R:
		score=score+weight(term,doc)
	return score

def rsj(term,R):
	N=len(documents)
	term_docs=len(inverted_index[term])
	R_=len(R)
	R_title=[doc.docID for doc in R]
	term_rel_docs=len([doc for doc in inverted_index[term] if doc in R_title])
	score=math.log(((term_rel_docs+0.5)*(N-R_-term_docs+term_rel_docs+0.5))/((term_docs-term_rel_docs+0.5)*(R_-term_rel_docs+0.5)))/math.log(2)
	return score
def weight(term,document):
	tf=0
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

def search_index(doc,term):
	doc_list=inverted_index.get(term)
	if doc_list == None:
		return False
	for j in doc_list:
		if j==doc:
			return True
	return False

#Fitness function
def fitness_function(orig_query,query,R):
	S={}
	for j in query:
		for k in R:
			if k.docID not in S.keys() :
				if search_index(k.docID,j) == True:
					S[k.docID]=bm(k,orig_query)+bm(k,query)
	L=S.values()
	return max(L)

def move(eq1,eq2,alpha,V):
	beta=1/(1+gamma*hamming_distance(eq1,eq2))
	for i,t in enumerate(eq1):
		if t not in eq2:
			random_value=np.random.random_sample()
			if random_value<beta:
				# count = np.random.random_integers(len(eq2)-1)
				# time=0
				# while(eq2[count] in eq1 and time<5):
				# 	count = np.random.random_integers(len(eq2)-1)
				# 	time=time+1
				options=[term for term in eq2 if term not in eq1]
				if len(options)>1:
					eq1[i]=options[np.random.random_integers(len(options)-1)]
				elif len(options)==1:
					eq1[i]=options[0]
			if random_value<alpha:
				rand=np.random.random_integers(len(V)-1)
				time = 0
				while V[rand] in eq1:
					rand=np.random.random_integers(len(V)-1)
					time = time + 1
					if(time==5):
						break
				# time=0
				# while (V[rand] in eq1 and time<5):
				# 	rand=np.random.random_integers(len(V)-1)
				# 	time=time+1
				eq1[i]=V[rand]
	return eq1	

def hamming_distance(eq1,eq2):
	union=[]
	for j in eq1:
		if j not in union:
			union.append(j)
	for j in eq2:
		if j not in union:
			union.append(j)
	distance=len(eq1)+len(eq2)-len(union)
	return distance

def initialize_population(V,size=30,length=3):
	fireflies=[[] for i in range(size)]
	for i in range(size):
		firefly=np.random.permutation(V)[:length].tolist()
		time = 0
		while True:
			if firefly not in fireflies:
				fireflies[i]=firefly
				break
			time = time + 1			
			firefly=np.random.permutation(V)[:length].tolist()
	return fireflies

def firefly_algorithm(fireflies,V,R,query,max_iter=10):
	population_size=len(fireflies)
	#Calculate initial fitness of each firefly
	fitness=np.zeros(population_size)
	for i in range(population_size):
		fitness[i]=fitness_function(query,fireflies[i],R)
	best_firefly=[]
	best_fitness=0
	for k in range(max_iter):
		alpha=alpha0*gamma**k
		for i in range(population_size):
			for j in range(population_size):
				if i!=j and fitness[i]<fitness[j]:
					fireflies[i]=move(fireflies[i],fireflies[j],alpha,V)
					fitness[i]=fitness_function(query,fireflies[i],R)
		best_position=np.argmax(np.array(fitness))
		if fitness[best_position]>best_fitness:
			best_fitness=fitness[best_position]
			best_firefly=fireflies[best_position]
		print("Current best firefly : ",best_firefly)
	return best_firefly

def common(T):
	print("Generated fireflies : ",fireflies)
	additional_query=firefly_algorithm(fireflies,V,R,query,T)
	expanded_query=query+additional_query
	print("*******************************************************************")
	print("Best expanded query : ",expanded_query)
	print("*******************************************************************")
	#Get the final relevant documents
	R_=[]
	score2 = {}
	#print(score_all_term)
	for i,document in enumerate(documents):
		score2[i]=score[i]+bm_fast(document,additional_query,score_all_term)
		#print(score2[i],'score\n\n'+str(i)+'\n')
	for key,value in sorted(score2.items(),key=lambda x:x[1],reverse=True):
		R_.append(documents[key])

	actual_relevant=0
	relevant_docs=quer[1]
	print("relevant documents",relevant_docs)
	print("Relevant documents : ")
	ap1 = 0
	for i in range(10):
		print(R_[i].docID)
		if R_[i].docID in relevant_docs:
			actual_relevant=actual_relevant+1
			ap1 = ap1 + actual_relevant*1.0/(i+1)
	if actual_relevant > 0:	
		ap1 = ap1/actual_relevant	
	print("Actually relevant documents = ",actual_relevant,ap1)	
	#return actual_relevant	
	return ap1

R_len = [10,30,50]
#Get the corpus and build inverted index
documents=pickle.loads(open('documents.pkl','rb').read())
print("Total number of documents : ",len(documents))
inverted_index = pickle.loads(open('index.pkl','rb').read())
avg_len = pickle.loads(open('avg_len.pkl','rb').read())
#inverted_index,avg_len=build_inverted_index(documents)
print("*******************************************************************")
print("Constructed inverted index :")
#print(inverted_index)
print("*******************************************************************")
score_all_term = pickle.loads(open('scoreq.pkl','rb').read())
query_rel_dictionary=get_query_rel_docs()
#print(query_rel_dictionary,'\n\n\n')
query_rel_dictionary = sorted(query_rel_dictionary.items(),key=lambda x:len(x[1]),reverse=True) # it is a list with tuples query and docs.
qc = 8
p_5_all_queries = [[] for i in range(qc)]
p_5_all_queries2 = [[] for i in range(qc)]
p_5_all_queries3 = [[] for i in range(qc)]
print("Beginning analysis")
for idx,quer in enumerate(query_rel_dictionary[qc:qc+2]):
	print("Opened csv file")	
	query_b4 = quer[0]
	query = trim_query(query_b4)
	print("Preprocesses query : ",query)
	#To retrieve the pseudo relevant documents : Okapi BM25 Model
	score={}
	#R : List of relevant documents in sorted order
	R=[]
	for i,document in enumerate(documents):
		score[i]=bm_fast(document,query,score_all_term)
		#print(score[i],'score\n\n'+str(i)+'\n')
	num_top_results=5

	for num_pseudo_doc in R_len:
		print("Starting for ",num_pseudo_doc)	
		count_pseudo = 0
		for key,value in sorted(score.items(),key=lambda x:x[1],reverse=True):
			R.append(documents[key])
			count_pseudo = count_pseudo + 1
			if(count_pseudo == num_pseudo_doc):
				break
		print("Analysis for q(length of expanded query) and R = [10,30,50]")		
		
		print("*******************************************************************")
		print("Pseudo relevant documents : ")
		for i in range(num_pseudo_doc):
			print(R[i])
		print("*******************************************************************")
		print("Firefly algorithm to find expanded query : ")
		V=generate_vocabulary(R)			
		for i in range(2,7,2):			
		#Step 1 : Generate the vocabulary of Pseudo relevant documents

			print("varying query lengths")
			fireflies=initialize_population(V,N,i)
			actual_relevant = common(T)
			print("generate ",actual_relevant," query length")
			p_5_all_queries[idx].append(actual_relevant)


		T = 10
		q = 4
		#population size
		for i in range(5,26,5):
			print("varying population")
			fireflies=initialize_population(V,i,q)
			actual_relevant = common(T)
			print("generate ",actual_relevant," population")
			p_5_all_queries2[idx].append(actual_relevant)

		N = 10
		q = 4
		#number of iterations
		for i in range(10,51,10):
			print("varying iterations")
			fireflies = initialize_population(V,N,q)
			actual_relevant = common(i)
			print("generate ",actual_relevant," iterations")
			p_5_all_queries3[idx].append(actual_relevant)	
		
	with open('Analysis.csv','a') as f:
		writer = csv.writer(f,quoting=csv.QUOTE_ALL)
		writer.writerow(p_5_all_queries[idx]+p_5_all_queries2[idx]+p_5_all_queries3[idx])
			



print(p_5_all_queries)
print(p_5_all_queries2)
print(p_5_all_queries3)
'''
Now plot for q = 4 t = 10 and n=5-35 Find the best value for n
'''			
file = open('scoreq.pkl','wb')
pickle.dump(score_all_term,file)
file.close()
