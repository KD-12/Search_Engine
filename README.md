# Search_Engine
Dataset is extracted using scrapy.
The entire dataset is represented using Vector Space Model, 
where each document is a vector in the vector space. Further, 
for computational purposes we represent our dataset in the form of a Term Frequency- 
Inverse Document Frequency(tf-idf) matrix. 

The first step involves  obtaining the 
most coherent sequence of words of the  search query entered. 
The entered query is processed using Front End algorithms this includes-Spell Checker,
Text Segmentation and Language Modeling. Back End processing includes Similarity Modeling,
Clustering, Indexing and Retrieval. 

The relationship between documents and words is established 
using cosine similarity measured between the documents and words in Vector Space. 

Clustering performed is used to suggest books that are similar to the search query entered by the user. 

Lastly, the Lucene Based Elasticsearch engine is used for indexing on the documents. 
This allows faster retrieval of data. Elasticsearch returns a dictionary and creates a TF-IDF matrix.
The processed query is compared with the dictionary obtained and tf idf matrix is used to calculate
the score for each match to give most relevant result.

Performing Data Visualisation using d3.js
