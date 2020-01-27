from nltk import word_tokenize
from nltk.corpus import reuters 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re
import csv
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
new_words = open("stopwords_en.txt", "r").read().split()
new_stopwords = stop.union(new_words)

def tokenize(text):
	min_length = 3
	words = map(lambda word: word.lower(), word_tokenize(text))
	words = [word for word in words if word not in new_stopwords]
	tokens =(list(map(lambda token: PorterStemmer().stem(token), words)))
	p = re.compile('[a-zA-Z]+')
	filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens))
	return filtered_tokens

def tf_idf(docs):
	tfidf = TfidfVectorizer(tokenizer=tokenize, use_idf=True, sublinear_tf=True);
	tfidf.fit(docs);
	return tfidf;

def feature_values(doc, representer):
	doc_representation = representer.transform([doc])
	features = representer.get_feature_names()
	return [(features[index], doc_representation[0, index]) for index in doc_representation.nonzero()[1]]

def collection_stats():
	documents = reuters.fileids()
	print(str(len(documents)) + " documents");
	
	train_docs = list(filter(lambda doc: doc.startswith("train"), documents));
	print(str(len(train_docs)) + " total train documents");
	
	test_docs = list(filter(lambda doc: doc.startswith("test"), documents));	
	print(str(len(test_docs)) + " total test documents");

	categories = reuters.categories();
	print(str(len(categories)) + " categories");

	category_docs = reuters.fileids("acq");

	document_id = category_docs[0]
	document_words = reuters.words(category_docs[0]);
	print(document_words);	

	print(reuters.raw(document_id));

def preProcesamiento():
	l_docs = []
	lista_docs = []
	train_docs = []
	test_docs = []
	file = open('pre_eric_ds.csv','w')

	with open('eric_ds.csv', 'r', encoding = "ISO-8859-1") as csvfile:
		l_docs = list(csv.reader(csvfile,delimiter = ","))
		
	flat_list = []
	for sublist in l_docs:
		lista_docs.append(sublist[0])
		flat_list.append(sublist[1])

	representer = tf_idf(flat_list)
	print(representer)

	i = 0 

	for doc in flat_list:
		file.write(lista_docs[i] + ", " + str(feature_values(doc, representer))+ "\n")
		i = i + 1

if __name__ == '__main__':
	preProcesamiento()