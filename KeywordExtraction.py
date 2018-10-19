# Keyword Extraction

# Author: Boyang Lu, Zhangyang Wei, Jie Zhou

import string
import nltk
from nltk.collocations import *
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk.wsd import lesk


############
## Part 0 ##
############
# Task: Tokenize the data


# dictionary of (business id, list of review ids)
with open('data/business_review.data', 'rb') as input:
	business_review = pickle.load(input)

# dictionary of (review id, review contents)
with open('data/id_content.data', 'rb') as input:
	id_content = pickle.load(input)


all_tokens = []
all_reviews = []

iterator = iter(business_review)
for i in range(len(business_review)):
	review = ""
	tokens = []
	for review_id in business_review[next(iterator)]:
		# Remove punctuations and uppercase all sentences
		text = id_content[review_id]["text"].translate(str.maketrans("", "", string.punctuation)).upper()
		review = review + text
		tokens = tokens + nltk.word_tokenize(text)
	all_reviews.append(review)
	all_tokens.append(tokens)





############
## Part 1 ##
############
# Task: For each business get common ngrams (up to 3)
# Method: nltk collocations (pmi)

'''

# A list of strings, each corresponding to all reviews of a business
with open('data/all_reviews.data', 'rb') as input:
	all_reviews = pickle.load(input)

# A list of list of strings, each corresponding to the tokenized reviews of a business
with open('data/all_tokens.data', 'rb') as input:
	all_tokens = pickle.load(input)

'''

stopword_upper = list(map(lambda x : x.upper(), stopwords.words('english')))

# We get the 500 most common n-grams for each business
unigrams = []
bigrams = []
trigrams = []

for i in range(len(all_tokens)):
	
	# Get the top 500 unigrams
	unigramFreq = nltk.FreqDist(all_tokens[i])
	# Remove words in stoplist
	for word in unigramFreq:
		if word in stopword_upper:
			unigramFreq[word] = 0
	unigrams.append(list(map(lambda x : x[0], unigramFreq.most_common(500))))

	# Get the top 500 bigram collocations
	bigram_finder = BigramCollocationFinder.from_words(all_tokens[i])
	bigram_finder.apply_freq_filter(5)
	bigrams.append(bigram_finder.nbest(nltk.collocations.BigramAssocMeasures().pmi, 500))

	# Get the top 500 trigram collocations 
	trigram_finder = TrigramCollocationFinder.from_words(all_tokens[i])
	trigram_finder.apply_freq_filter(5)
	trigrams.append(trigram_finder.nbest(nltk.collocations.TrigramAssocMeasures().pmi, 500))

	# Note: We do not simply remove bigrams and trigrams with stopwords because dish names may include stopwords, e.g. fish and chips.





############
## Part 2 ##
############
# Task: Find words that are common for one particular restaurant
# Method: TF-IDF


'''

# A list of strings, each corresponding to all reviews of a business
with open('data/all_reviews.data', 'rb') as input:
	all_reviews = pickle.load(input)

# A list of list of strings, each corresponding to the tokenized reviews of a business
with open('data/all_tokens.data', 'rb') as input:
	all_tokens = pickle.load(input)

# list of 500 most common unigrams
with open('data/unigrams.data', 'rb') as input:
	unigrams = pickle.load(input)

# list of 500 most common bigrams
with open('data/bigrams.data', 'rb') as input:
	bigrams = pickle.load(input)

# list of 500 most common trigrams
with open('data/trigrams.data', 'rb') as input:
	trigrams = pickle.load(input)

print("loading file done")

'''

# Find the terms with a non-zero entry in the term-document matrix
# These are terms that bear important document-specific information


tfidf_unigram = TfidfVectorizer(ngram_range = (1,1))
doc_term_matrix = tfidf_unigram.fit_transform(all_reviews)
idf_unigram = tfidf_unigram.inverse_transform(doc_term_matrix)
for i in range(len(idf_unigram)):
	idf_unigram[i] = list(map(lambda x: x.upper(), idf_unigram[i]))


tfidf_bigram = TfidfVectorizer(ngram_range = (2,2))
doc_term_matrix = tfidf_bigram.fit_transform(all_reviews)
idf_bigram = tfidf_bigram.inverse_transform(doc_term_matrix)
for i in range(len(idf_bigram)):
	idf_bigram[i] = list(map(lambda x: x.upper(), idf_bigram[i]))


tfidf_trigram = TfidfVectorizer(ngram_range = (3,3))
doc_term_matrix = tfidf_trigram.fit_transform(all_reviews)
idf_trigram = tfidf_trigram.inverse_transform(doc_term_matrix)
for i in range(len(idf_trigram)):
	idf_trigram[i] = list(map(lambda x: x.upper(), idf_trigram[i]))


for i in range(10):
	unigrams[i] = list(filter(lambda x : x in idf_unigram[i], unigrams[i]))
	bigrams[i] = list(filter(lambda x : (x[0] + ' ' + x[1]) in idf_bigram[i], bigrams[i]))
	trigrams[i] = list(filter(lambda x : (x[0] + ' ' + x[1] + ' ' + x[2]) in idf_trigram[0], trigrams[i]))




############
## Part 3 ##
############
# Task 3: Determine if ngrams are actually dishes

'''
# Read in data
# list of filtered unigrams
with open('data/unigrams_filtered.data', 'rb') as input:
	unigrams = pickle.load(input)

# list of filtered bigrams
with open('data/bigrams_filtered.data', 'rb') as input:
	bigrams = pickle.load(input)

# list of filtered trigrams
with open('data/trigrams_filtered.data', 'rb') as input:
	trigrams = pickle.load(input)

print("loading file done")

'''

# Method 1: WordNet
# We find the dish terms based on its semantic similarity with another dish name
# For each type of n-grams, we manually select one term that is a dish as our example, then we use wordNet to find all words with high similarity with our example


def unigram_similarity(word1, word2):
	word1_synset = lesk(word1, word1)
	word2_synset = lesk(word2, word2)
	try: 
		sim = wordnet.wup_similarity(word1_synset, word2_synset)
	except TypeError:
		return 0
	except AttributeError:
		return 0
	if (sim is None):
		return 0
	return sim

unigram_dishes = []
example_dish_name = 'steak'
for unigram_list in unigrams:
	unigram_dish = []
	for i in range(len(unigram_list)):
		if (unigram_similarity(unigram_list[i], example_dish_name) > 0.5):
			unigram_dish.append(unigram_list[i])
	unigram_dishes.append(unigram_dish)
print("unigram dishes finished")

# We define the similarity between bigrams the max similarity bewteen each individual words
def bigram_similarity(bigram1, bigram2):
	return max(unigram_similarity(bigram1[0], bigram2[0]), unigram_similarity(bigram1[0], bigram2[1]), unigram_similarity(bigram1[1], bigram2[0]), unigram_similarity(bigram1[1], bigram2[1]))

bigram_dishes = []
example_dish_name = ('pulled', 'pork')
for bigram_list in bigrams:
	bigram_dish = []
	for i in range(len(bigram_list)):
		if (bigram_similarity(bigram_list[i], example_dish_name) > 0.5):
			bigram_dish.append(bigram_list[i][0] + ' ' + bigram_list[i][1])
	bigram_dishes.append(bigram_dish)
print("bigram dishes finished")

# We give a similar definition for trigram similarities
def trigram_similarity(trigram1, trigram2):
	return max(unigram_similarity(trigram1[0], trigram2[0]), unigram_similarity(trigram1[0], trigram2[1]), unigram_similarity(trigram1[0], trigram2[2]), unigram_similarity(trigram1[1], trigram2[0]), unigram_similarity(trigram1[1], trigram2[1]), unigram_similarity(trigram1[1], trigram2[2]), unigram_similarity(trigram1[2], trigram2[0]), unigram_similarity(trigram1[2], trigram2[1]), unigram_similarity(trigram1[2], trigram2[2]))

trigram_dishes = []
example_dish_name = ('caramel', 'creme', 'brulee')
for trigram_list in trigrams:
	trigram_dish = []
	for i in range(len(trigram_list)):
		if (trigram_similarity(trigram_list[i], example_dish_name) > 0.5):
			trigram_dish.append(trigram_list[i][0] + ' ' + trigram_list[i][1] + ' ' + trigram_list[i][2])
	trigram_dishes.append(trigram_dish)
print("trigram dishes finished")

# Generate a single list for each restaurant
wordnet_dishes = []
for i in range(10):
	wordnet_dishes.append(unigram_dishes[i] + bigram_dishes[i] + trigram_dishes[i])


# Method 2: Gensim.word2vec


