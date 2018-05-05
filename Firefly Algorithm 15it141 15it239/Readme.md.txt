Source folders:
Firefly Algorithm 15it141 15it239 : Files used for experimental evaluation of results
IR Mini Project PPT : Contains the end semester presentation
IR Mini Project Report : Latex final report
Dataset.zip : Medline Dataset
Web Model.zip : Information retrieval web interface for seach queries
inverted_index_stored.zip,documents_score.zip,scorequeries.rar and avg_len.pkl are precalculated parameters of firefly algorithm used for evaluation

Source files:
	in Web Model folder: run python manage.py runserver for starting interface. Click on the link in terminal to open the browser interface
		Important Files : 
		inverted_index.py : Python file to generate inverted index of the vocabulary of the Medline Dataset
		views.py : Contains the required functions to enable web search interface for Medline Dataset
	
	in Firefly A
		firefly_pr.py : Python file used to calculate the precision@5 and MAP@10 measure for varying firefly parameters
		firefly_anal.py : Python file used to compare precision@5 measure of firefly algorithm vs rocchio and rsj
		inverted_index.py : Python file to generate inverted index of the vocabulary of the Medline Dataset
		store_index.py : Python file to store inverted index of medline dataset and relevance scores of anticipated query terms
	in Dataset : 
		ohsumed.87 : Medline dataset
		qrels.ohsu.* : Queries for ir system and corresponding relevant documents
		query.ohsu : Queries for ir system
