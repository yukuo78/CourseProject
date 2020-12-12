import sys
from preprocess import *
import numpy as np

#goal: map sentences to corresponding aspect.
label_text = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Check in/Front Desk', 'Service', 'Business Service']
REVIEW_SIZE=5000

## load aspect definition from a file
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

def chi_sq(aspect, term, a,b,c,d):
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

	chi = nc * (c1*c4 - c2*c3) * (c1*c4 - c2*c3)/((c1+c3) * (c2+c4) * (c1+c2) * (c3+c4))
	sumC = c1 + c2 + c3 + c4
	#print("chi for %s %s" % (aspect, term), c1, c2, c3, c4, nc, sumC, chi)
	return chi

## returns a K*V matrix
def chi_sq_mat(K, V, aspect_words, aspect_sent, num_words, only_sent):
	asp_rank = np.zeros(aspect_words.shape)
	for i in range(K):
		for j in range(V):
			asp_rank[i][j] = chi_sq(label_text[i], vocab[j], aspect_words[i][j], num_words[j], aspect_sent[i], len(only_sent))
	return asp_rank

## only review_lables is important for LRR
def assign_sent_to_aspects(review_sent, aspect_terms, vocab_dict, K, V):
	review_labels = []
	num_words = np.zeros(V)
	aspect_words = np.zeros((K,V))
	aspect_sent = np.zeros(K)

	for review in review_sent:
		# each scentence belongs to which label
		labels = []
		for sent in review:
			for w in set(sent):
				if w in vocab_dict:
					num_words[vocab_dict[w]] += 1

			count = np.zeros(K)
			i = 0
			for a in aspect_terms:
				for w in set(sent):
					if w in vocab_dict:
						if w in a:
							count[i] += 1
				i = i + 1
			if max(count) > 0:
				la = np.where(np.max(count) == count)[0].tolist()
				labels.append(la)
				#print("assigned sent to label", la, sent, extract_list(la, label_text))
				for i in la:
					aspect_sent[i] += 1
					for w in set(sent):
						if w in vocab_dict:
							aspect_words[i][vocab_dict[w]] += 1
			else:
				labels.append([])
		review_labels.append(labels)
	return num_words, aspect_words, aspect_sent, review_labels


vocab=[]
vocab_dict={}

def aspect_segmentaion():

	#INPUT
	#review, this algo needs all the review. Please process dataset.
	file="TripAdvisor/Texts/hotel_72572_parsed.txt"
	#file="Review_Texts/hotel_218524.dat"
	#reviews, all_ratings = load_file(file)
	reviews, labels = load_path("TripAdvisor/Texts", REVIEW_SIZE)

	#selection threshold
	p = 5
	#p = 2
	#Iterations 
	I = 10
	#I = 1

	#Create Vocabulary
	review_sent, review_actual, only_sent = parse_to_sentence(reviews)
	global vocab, vocab_dict
	vocab, vocab_dict = create_vocab(only_sent)

	#Aspect Keywords
	#aspect_file = "aspect_keywords.csv"
	aspect_file = "init_aspect_keywords.csv"
	aspect_terms = get_aspect_terms(aspect_file, vocab_dict)

	print("initial aspect_terms:", aspect_terms)

	#ALGORITHM
	K = len(aspect_terms)
	V = len(vocab)
	print("vocab size:", V)
	print("review scent:", len(review_sent))

	for iter in range(I):
		num_words, aspect_words, aspect_sent, review_labels = assign_sent_to_aspects(review_sent, aspect_terms, vocab_dict, K, V)
		changed = reassign_aspect_terms(K, V, p, aspect_words, aspect_sent, num_words, only_sent, vocab_dict, aspect_terms, iter)
		if not changed:
			break

	return Data(num_words, aspect_words, aspect_sent, aspect_terms, review_sent, vocab, vocab_dict, labels)

## all sentences assigned to labels/aspects
## now we should update chi square, add terms to aspects
def reassign_aspect_terms(K, V, p, aspect_words, aspect_sent, num_words, only_sent, vocab_dict, aspect_terms, iter):
	aspect_w_rank = chi_sq_mat(K, V, aspect_words, aspect_sent, num_words, only_sent)
	print("aspect with rank:", aspect_w_rank)

	new_labels = []
	new_terms = []
	changed = False
	for i, na in enumerate(aspect_w_rank):
		x = np.argsort(na)[::-1][:p]
		terms = []
		new_labels.append(x)
		new_terms.append(terms)
		print("add new terms to label %s:" % label_text[i] , x, extract_list(x, na))
		for k,v in vocab_dict.items():
			if vocab_dict[k] in x:
				if (not k in aspect_terms[i]):
					print("add %s %d to %s" % (k,v,label_text[i]), aspect_words[i][v])
					aspect_terms[i].append(k)
					changed = True
	#print("new_labels", new_labels)
	print("total_sent(iter %d):" % iter, len(only_sent))
	print("aspect_sent(iter %d):" % iter, aspect_sent)
	print("aspect_terms (iter %d):" % iter)
	for i, asp_term in enumerate(aspect_terms):
		print(label_text[i], asp_term)
	# sys.exit()
	return changed

## use indexs to retrive values in a list
def extract_list(subs, list):
	result = []
	for i in subs:
		result.append(list[i])
	return result
	
## Data for LRR model regression
class Data:
	def __init__(this, num_words, aspect_words, aspect_sent, aspect_terms, review_sent, vocab, vocab_dict, labels):
		## shape (V), global term count
		this.num_words = num_words

		## shape (K, V), per aspect term count
		this.aspect_words = aspect_words

		## shape (K, V), per aspect term count
		this.aspect_sent = aspect_sent

		## shape (K, []), per aspect critical terms
		this.aspect_terms = aspect_terms

		## shape(D, [[]])
		this.review_sent = review_sent

		## Vocabulary
		this.vocab = vocab
		this.vocab_dict = vocab_dict

		this.labels = labels

if __name__ == "__main__":
	data = aspect_segmentaion()
	if (len(sys.argv) > 1):
		REVIEW_SIZE = sys.argv[1]
	import pickle
	pickle.dump(data, open("data_dump.pickle", "wb"))
