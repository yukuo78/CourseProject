import sys, string
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import re
import os
stemmer = PorterStemmer()

#load all review texts
def load_file(file):
	reviews = []
	ratings = []
	f = open(file,'r')
	for line in f:
		l = line.strip().split('>')
		if l[0] == '<Content':
			s = str(l[1])
			reviews.append(s)
		elif l[0] == '<Rating':
			r = l[1].split('\t')
			ratings.append(int(r[1]))
	f.close()
	return reviews , ratings
	#print(len(reviews), reviews[1])

def parse_to_sentence(reviews):
	review_processed = []
	actual = []
	only_sent = []
	for r in reviews:
		sentences = nltk.sent_tokenize(r)
		actual.append(sentences)
		sent = []
		for s in sentences:
			#words to lower case
			s = s.lower()
			#remove punctuations and stopwords
			replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
			s = s.translate(replace_punctuation)
			stop_words	 = list(stopwords.words('english'))
			additional_stopwords = ["'s","...","'ve","``","''","'m",'--',"'ll","'d"]
			# additional_stopwords = []
			stop_words = set(stop_words + additional_stopwords)
			# print stop_words
			# sys.exit()
			word_tokens = word_tokenize(s)
			s = [w for w in word_tokens if not w in stop_words]
			#Porter Stemmer
			stemmed = [stemmer.stem(w) for w in s]
			if len(stemmed)>0:
				sent.append(stemmed)
			#print(len(only_sent), stemmed)
		review_processed.append(sent)
		only_sent.extend(sent)
	return review_processed, actual, only_sent

# sent = parse_to_sentence(reviews)
# print len(sent), sent[2]

def create_vocab(sent):
	words = []
	for s in sent:
		words += s
	freq = FreqDist(words)
	vocab = []
	for k,v in freq.items():
		if v > 5:
			vocab.append(k)
	#Assign a number corresponding to each word. Makes counting easier.
	vocab_dict = dict(zip(vocab, range(len(vocab))))
	return vocab, vocab_dict

def load_path(path, max_review=5000):
	total_reviews = []
	path = "TripAdvisor/Texts"
	for f in os.listdir(path):
		reviews, ratings = load_file(os.path.join(path, f))
		total_reviews += reviews
		if (len(total_reviews) >= max_review):
			break
	return total_reviews

if __name__ == '__main__':
	import sys
	import os
	total_reviews = load_path(path)
	print(len(total_reviews), total_reviews[2])
	#sys.exit(1)
	processed_reviews, actual, only_sent = parse_to_sentence(total_reviews)
	print("total reviews:", len(processed_reviews))
	print("total scentences:", len(only_sent))
	vocab, vocab_dict = create_vocab(only_sent)
	print("total terms:", len(vocab))
