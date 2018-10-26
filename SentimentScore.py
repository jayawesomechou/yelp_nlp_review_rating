# Sentimental Analysis

# Author: Boyang Lu, Zhangyang Wei, Jie Zhou


import pickle
from textblob import TextBlob


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
    print(bussid)
    bussiness_score=0
    
    #count how many predicted score is withhin 0.5 from the true review star.
    true_count=0
    for rv_id in business_review[bussid]:
        text=id_content[rv_id]['text']
        stars=id_content[rv_id]['stars']
        
        testimonial=TextBlob(text)
        polar=testimonial.sentiment.polarity
        subject=testimonial.sentiment.subjectivity
        
        #rv_score=polar*2.5*(3*subject)+2.5
        #rv_score=polar*2*(1+3*subject)+3
        #rv_score=polar*2.5*(1+3*subject)+2.5
        rv_score=polar*1.5*(1+subject)+3.5
        
        
        bussiness_score+=rv_score

        if stars-0.5<rv_score<stars+0.5:
            true_count+=1

        
    # Get the average result of the bussiness score
    bussiness_score=bussiness_score/len(business_review[bussid])
    bussiness_score_LS.append(bussiness_score)
    true_count_rate=true_count/len(business_review[bussid])

    print(bussiness_score)
    print(true_count_rate)
    print("\n\n")
                                





















