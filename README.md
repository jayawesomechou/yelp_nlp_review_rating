# NLPYelpReview


This is a course final project of processing restaurant review data from Yelp.

There are mainly two function of this project:

    1. Find the food-related keyword from Yelp review or from a specific restaurant's review

    2. Generate a review based rating to each restaurant using Sentimental analysis and compare with the true rating.



# Task 1: KeywordExtraction.py

    Part 0: Tokenize the reviews
    Part 1: Use nltk to find the unigram, bigram and trigram collocations
    Part 2: Use tf-idf to find the collocations that are specific to each document (business)
    Part 3: Find food-related collocations
        Method 1: wordnet
        Method 2: word2vec
        
        
# Task 2: SentimentScore.py

    Using textblob sentiment function and generate the sentimental score of reviews based on polarity and subjectivity.
    
