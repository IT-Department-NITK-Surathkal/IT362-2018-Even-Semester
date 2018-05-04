Source Files and Folders:
Python files
firefly_pr.py : Python file used to calculate the precision@5 and MAP@10 measure for varying firefly parameters
firefly_anal.py : Python file used to compare precision@5 measure of firefly algorithm vs rocchio and rsj
inverted_index.py : Python file to generate inverted index of the vocabulary of the Medline Dataset
store_index.py : Python file to store inverted index of medline dataset and relevance scores of anticipated query terms
In IR folder:
	IR/manage.py : command line utility to start server and manage migrations.
		Command 'python manage.py runserver' sets up the localhost server
	Important Files in IR/project : 
		inverted_index.py : Python file to generate inverted index of the vocabulary of the Medline Dataset
		views.py : Contains the required functions to enable web search interface for Medline Dataset

graphs folder: Contains graphical representation of all results
Mini Project PPT : Presentation for end semester evaluation 
*.pkl : Pickle files containing results from store_index
ohsumed.87 : Medline dataset
qrels.ohsu.* : Queries for ir system and corresponding relevant documents
query.ohsu : Queries for ir system
Mini Project Report : Folder containing all latex files along with final report
Mini Project PPT.ppt : End semester powerpoint presentation

