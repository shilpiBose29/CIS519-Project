#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:32:38 2017

@author: arranzignacio
"""

import pandas as pd
from textblob import TextBlob

filename = "../../reviews_with_languages_detected.csv"

df = pd.read_csv(filename, sep=',')


#df2 = df.groupby(['language'])['id'].count()


reviews = df['comments']
languages = df['language']

test_set = reviews[:]
language_set = languages[:]

review = []
all_reviews = []
c = 0
for i in test_set:
    if c%100 ==0:
        print "STEP ",c
    if languages[c] == 'en':
        i = i.decode('utf-8','replace')
        sentence_blob = TextBlob(i)
        
        sentence_string = i
        a = sentence_blob.sentiment
        review = [a.polarity, a.subjectivity]
    else:
        review = ["not available","not available"]
    
    all_reviews.append(review)
    c = c+1
    
sentiment_df = pd.DataFrame(all_reviews)
sentiment_df.columns = ['Polarity', 'Subjectivity']

listings_with_sentiment = pd.concat([df, sentiment_df], axis=1, join_axes=[df.index])

print listings_with_sentiment

#file_name = 'listings_with_score.csv'

#listings_with_sentiment.to_csv(file_name, sep=',', encoding='utf-8')

print "done"
    



