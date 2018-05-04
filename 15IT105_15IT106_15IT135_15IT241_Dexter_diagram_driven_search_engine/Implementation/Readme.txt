IR Mini Project
---------------

Team Members:
-------------
Aiman Abdullah Anees [15IT106]
Salman Shah [15IT241]
Rashika Chowlek [15IT135]
Adwaith CD [15IT105]


Project Topic: Dexter: Data-driven Search Engine
--------------

Report_PPT_Reference_Paper:
---------------------------
Folder Structure:
-----------------
(1) Reference Paper
(2) Mini_Project_Report
(3) IR_final_presentation


Dataset: 
--------
Folder Structure:
-----------------
(1) Population_CSV_Files: A folder containing population statisitcs of all the countries in the form of csv.
(2) PDFs: A folder containing all the pdfs.
(3) key_fields.json: A JSON file containing 8-key fields for all the PDFs present.


Dataset_Generation_Files:
-------------------------
Folder Structure:
-----------------
(1) graph_generator.py: for generating graphs.
(2) key_fields_generator.py: for generating key-fields.
(3) latex_generator.py: for generating latex files.
(4) pdf_generator.py: for generating pdf documents.

Offline_Mode_Implementation:
----------------------------
Folder Structure:
-----------------
(1) inverted_index.py: implementation of Boolean Model
(2) VSM.py: implementation of VSM Model
(3) tfidf.py: implementation of TFIDF term weighting scheme and applied on all the documents.
(4) tfidf_query.py: implementation of TFIDF term weighting scheme applied on the input query.
(5) cosine_similarity.py: implementation of Cosine Similarity
(6) tfidf.csv: result produced by tfidf.py stored in the form of csv file.
(7) tfidf.txt: storing the TFIDF of input query.
(8) idf.txt: storing the IDF of all the documents.
(9) documents.txt: storing all the documents in the form of a multidimensional array.
(10) vocabulary.txt: vocabulary produced after preprocessing.

Online_Mode_Implementation:
---------------------------
Folder Structure:
-----------------
(1) server.py: backend of the website.
(2) static: Folder containing css, images, and javascript-related files.
(3) templates: Folder containing HTML files. 








