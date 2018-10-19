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
from textblob import TextBlob
from random import *



############
## Part 0 ##
############
# Task: Extract the data


# dictionary of (business id, list of review ids)
with open('business_review.data', 'rb') as input:
	business_review = pickle.load(input)

# dictionary of (review id, review contents)
with open('id_content.data', 'rb') as input:
	id_content = pickle.load(input)





############
## Part 1 ##
############

# Task: Using textblob to calculate the sentimental score of each review of a given business
# And then get the average score of that restaurant review.

bussiness_score_LS=[]

for bussid in business_review:
    print("business id", bussid)
    bussiness_score=0
    
    #count how many predicted score is withhin 0.5 from the true review star.
    true_count=0
    rand_true_count=0
    for rv_id in business_review[bussid]:
        text=id_content[rv_id]['text']
        stars=id_content[rv_id]['stars']
        
        testimonial=TextBlob(text)
        polar=testimonial.sentiment.polarity
        subject=testimonial.sentiment.subjectivity
        
        
        
        # model we choose to generate rv_score
        
        #rv_score=polar*2.5*(3*subject)+2.5
        #rv_score=polar*2*(1+3*subject)+3
        #rv_score=polar*2.5*(1+3*subject)+2.5
        #rv_score=polar*1.5*(1+subject)+3.5
        rv_score=polar*2*(1+subject)+3
        
        

        
        bussiness_score+=rv_score

        if stars-0.5<rv_score<stars+0.5:
            true_count+=1
        
        #Random Baseline as comparision
        randx = randint(0, 10)
        randscore=randx*0.5
        
        if stars-0.5<randscore<stars+0.5:
            rand_true_count+=1
        


        
    # Get the average result of the bussiness score
    bussiness_score=bussiness_score/len(business_review[bussid])
    bussiness_score_LS.append(bussiness_score)
    true_count_rate=true_count/len(business_review[bussid])
    rand_true_count_rate=rand_true_count/len(business_review[bussid])





    print("Average Processed Review Score:", str(bussiness_score))
    print("Sentimental Analyzed Review Accuracy:", str(true_count_rate))
    print("Random Baseline Accuracy:", str(rand_true_count_rate))
    print("\n\n")
                                





















