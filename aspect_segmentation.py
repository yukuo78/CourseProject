from preprocess import *
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

#goal: map sentences to corresponding aspect.

def get_aspect_terms(file, vocab_dict):
	aspect_terms = []
	w_notfound = []
	f = open(file, "r")
	for line in f:
		s = line.strip().split(",")
		stem = [stemmer.stem(w.strip().lower()) for w in s]
		#we store words by their corresponding number.
		# aspect = [vocab_dict[w] for w in stem]
		aspect = []
		for w in stem:
			if w in vocab_dict:
				aspect.append(w)
			else:
				w_notfound.append(w)
		aspect_terms.append(aspect)
	#We are only using one hotel review file, as we keep inceasing the number of files words not found will decrease.
	# print "Words not found in vocab:", ' '.join(w_notfound)
	f.close()
	return aspect_terms

def chi_sq(a,b,c,d):
	"""
	a: count of word[j] in aspect[i]
	b: count of word[j]
	c: count of scentences of aspect[i]
	d: count of scentences
	"""
	# C1: count of w in Ai
	c1 = a

	# C2: count of w in ~Ai
	c2 = b - a

	# C3: count of sentences in Ai without w.
	c3 = c - a

	# C4: count of sentences in ~Ai without w.
	c4 = d - b - c + a

	# C: totoal number of word
	nc =  b

	return nc * (c1*c4 - c2*c3) * (c1*c4 - c2*c3)/((c1+c3) * (c2+c4) * (c1+c2) * (c3+c4))

## returns a K*V matrix
def chi_sq_mat(K, aspect_words, aspect_sent, num_words, vocab, sent):
	asp_rank = np.zeros(aspect_words.shape)
	for i in range(K):
		for j in range(len(vocab)):
			asp_rank[i][j] = chi_sq(aspect_words[i][j], num_words[j], aspect_sent[i], len(sent))
	return asp_rank

def aspect_segmentaion():

	#INPUT
	#review, this algo needs all the review. Please process dataset.
	file="TripAdvisor/Texts/hotel_72572_parsed.txt"
	#file="Review_Texts/hotel_218524.dat"
	#reviews, all_ratings = load_file(file)
	reviews = load_path("TripAdvisor/Texts")

	#selection threshold
	p = 5
	#p = 2
	#Iterations 
	I = 10
	I = 1

	#Create Vocabulary
	review_sent, review_actual, only_sent = parse_to_sentence(reviews)
	vocab, vocab_dict = create_vocab(only_sent)

	#Aspect Keywords
	aspect_file = "aspect_keywords.csv"
	aspect_file = "init_aspect_keywords.csv"
	aspect_terms = get_aspect_terms(aspect_file, vocab_dict)

	label_text = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Check in/Front Desk', 'Service', 'Business Service']
	print("initial aspect_terms:", aspect_terms)

	#ALGORITHM
	review_labels = []
	K = len(aspect_terms)
	V = len(vocab)
	print("vocab size:", V)
	aspect_words = np.zeros((K,V))
	aspect_sent = np.zeros(K)
	num_words = np.zeros(V)

	for iter in range(I):
		for review in review_sent:
			# each scentence belongs to which label
			labels = []
			for sent in review:
				count = np.zeros(K)
				i = 0
				for a in aspect_terms:
					for w in sent:
						if w in vocab_dict:
							num_words[vocab_dict[w]] += 1
							if w in a:
								count[i] += 1
					i = i + 1
				if max(count) > 0:
					la = np.where(np.max(count) == count)[0].tolist()
					labels.append(la)
					print("assign sent to label", la, sent)
					for i in la:
						aspect_sent[i] += 1
						for w in sent:
							if w in vocab_dict:
								aspect_words[i][vocab_dict[w]] += 1
				else:
					labels.append([])
			review_labels.append(labels)

		## all sentences assigned to labels/aspects
		## now we should update chi square, add terms to aspects
		aspect_w_rank = chi_sq_mat(K, aspect_words, aspect_sent, num_words, vocab, only_sent)
		print("aspect with rank:", aspect_w_rank)
		new_labels = []
		new_terms = []
		for i, na in enumerate(aspect_w_rank):
			x = np.argsort(na)[::-1][:p]
			terms = []
			new_labels.append(x)
			new_terms.append(terms)
			print("add new terms to label %d:" % i , x, extract_list(x, na))
			for k,v in vocab_dict.items():
				if vocab_dict[k] in x:
					print("add %s %d to %s" % (k,v,label_text[i]), aspect_words[i][v])
					if (not k in aspect_terms[i]):
						aspect_terms[i].append(k)
		#print("new_labels", new_labels)
		print("aspect_terms (iter %d):" % iter, aspect_terms)
		# sys.exit()


def estimate_reviews():
	#Sentiment analysis
	sid = SIA()
	#print("review_labels", review_labels)
	ratings_sentiment = []
	for r in review_actual:
		sentiment = []
		#aspect ratings based on sentiment
		for s in r:
			ss = sid.polarity_scores(s)
			sentiment.append(ss['compound'])
		ratings_sentiment.append(sentiment)

	#Aspect Ratings Per Review
	aspect_ratings = []
	for i,r in enumerate(review_labels):
		rating = np.zeros(7)
		count = np.zeros(7)
		rs = ratings_sentiment[i] 
		for j,l in enumerate(r):
			for k in range(7):
				if k in l:
					rating[k] += rs[j]
			for k in range(7):
				if count[k] != 0:
					rating[k] /= count[k]
		#Map from -[-1,1] to [1,5]
		for k in range(7):
			if rating[k] != 0:
				rating[k] = int(round((rating[k]+1)*5/2))
		aspect_ratings.append(rating)
	#return aspect_ratings, all_ratings

	#n = 0
	#print(review_actual[n], '\n', review_labels[n])
	#print(ratings_sentiment[n], '\n', aspect_ratings[n])
	#print(len(all_ratings), len(reviews), all_ratings[0])
	sys.exit()
	return aspect_ratings

	# print sent[5:9], labels[5:9]
	# print zip(actual_sent, labels)[:10]
	# print zip(actual_sent, sentiment)[:10]

def extract_list(subs, list):
	result = []
	for i in subs:
		result.append(list[i])
	return result
	
aspect_segmentaion()
