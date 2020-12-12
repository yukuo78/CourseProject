import numpy as np
from aspect_segmentation import Data

label_text = ['Value', 'Rooms', 'Location', 'Cleanliness', 'Check in/Front Desk', 'Service', 'Business Service']

## alpha is variable, of shape (K)
## other are considered constant
## output is shape (K)
def f_x_derivative(lrr, alpha, rd, d):
	deltaR = np.dot(alpha , lrr.S[d]) - rd
	a1 = - deltaR * lrr.S[d]/lrr.Sigma
	a2 = - np.matmul(np.linalg.inv(lrr.Omega) , (alpha - lrr.Mu))
	return a1 + a2
	 
def f_x(lrr, alpha, rd, d):
	deltaR = np.dot(alpha , lrr.S[d]) - rd
	a1 = - deltaR * deltaR / (2 * lrr.Sigma)

	b2 = np.matmul(np.linalg.inv(lrr.Omega) , (alpha - lrr.Mu))
	a2 = - np.matmul( (alpha - lrr.Mu).T, b2)
	return a1 + a2

def gradient_precision(x_start, max_iter, precision, learning_rate, f_x, dfunc):

	# These x and y value lists will be used later for visualisation.
	x_grad = [x_start]
	y_grad = [f_x(x_start)]

	iter = 0
	while iter < max_iter:

		# Get the Slope value from the derivative function for x_start
		# Since we need negative descent (towards minimum), we use '-' of derivative
		x_start_derivative = - dfunc(x_start)

		print("x_start:", x_start)
		print("y_grad: ", f_x(x_start))
		print("x_start_derivative: ", x_start_derivative)

		# calculate x_start by adding the previous value to 
		# the product of the derivative and the learning rate calculated above.
		x_start = x_start + (learning_rate * x_start_derivative)
		print("next x_start:", x_start)

		x_grad.append(x_start)
		y_grad.append(f_x(x_start))
		# Break out of the loop as soon as we meet precision.
		#if abs(x_grad[len(x_grad)-1] - x_grad[len(x_grad)-2]) <= precision:
		delta = np.sum(x_grad[len(x_grad)-1] - x_grad[len(x_grad)-2])
		print(delta, x_grad[len(x_grad)-1] , x_grad[len(x_grad)-2])
		print("total x_grad", len(x_grad), x_grad)
		if delta <= precision:
			break
		iter += 1

	print ("Local minimum occurs at: ", (x_start))
	print ("Number of steps taken: ",len(x_grad)-1)
	#plot_gradient(x, f_x(x) ,x_grad, y_grad)

class LRR:
	def __init__(this, K, V, data):
		this.Mu = np.random.rand(K)
		this.Omega = np.random.rand(K,K)
		this.Sigma = np.random.rand()
		this.Beta = np.random.rand(K,V)	
		this.data = data
		this.K = K
		this.V = V
		this.D = len(data.review_sent)

	## W is ground truth for LRR
	def calculateW(this):
		vocab_dict = this.data.vocab_dict
		review_sent = this.data.review_sent
		aspect_terms = this.data.aspect_terms
		K, V = this.K, this.V
		W = []

		for review in review_sent:
			# each scentence belongs to which label
			aspect_words = np.zeros((K,V))
			for sent in review:
				# print("assigned sent:", sent)
				count = np.zeros(K)
				for i, asp in enumerate(aspect_terms):
					for w in set(sent):
						if w in vocab_dict:
							if w in asp:
								count[i] += 1
							else:
								pass
								#print("non word:", w, asp)
				
				if max(count) > 0:
					la = np.where(np.max(count) == count)[0].tolist()
					#print("assigned sent to", sent, extract_list(la, label_text))
					for i in la:
						for w in set(sent):
							if w in vocab_dict:
								aspect_words[i][vocab_dict[w]] += 1
				else:
					pass
					#print("no counts > 0")
		
			W.append(aspect_words)
		return W

	def Expectation(this):
		# for every review calculate Si and Alpha[d]
		# S[d][i] = W[d][i] dot Beta[i]
		# Alpha[d] is infered using MAP estimation

		S = np.zeros((this.D, this.K));
		for d in range(this.D):
			for i in range(7):
				S[d][i] = np.dot(lrr.Beta[i], W[d][i])
			#print("S[%d]" % d, S[d])
		this.S = S

		d = 0
		rd = this.data.labels[d]
		alpha0 = np.zeros(this.K) + 1/7
		func = lambda x: f_x(this, x, rd, d)
		gradient_precision(alpha0, 100, 0.001, 0.1, func, lambda x: f_x_derivative(this, x, rd, d))

	def Maximization(this):
		# Calculate Mu, Omega
		# Mu = 1/len(D) * sum_over_all(Alpha[d])
		# Omega = 1/len(D) * sum_over((Alpha[d]-Mu) * (Alpha[d] - Mu).transpose)

		# Calulate Sigma	
		
		# Estimate Beta
		pass

def extract_list(subs, list):
	result = []
	for i in subs:
		result.append(list[i])
	return result
	
if __name__ == "__main__":
	import pickle
	#print(lrr.Omega)
	#print(lrr.Beta)
	data = pickle.load(open("data_dump.pickle", "rb"))
	lrr = LRR(7, 3656, data)
	W = lrr.calculateW()	
	lrr.Expectation()
	
