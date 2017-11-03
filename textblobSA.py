import numpy as np 
import pandas as pd
from textblob import TextBlob

reviewFile='datasets/Asheville-reviews.csv'
data = pd.read_csv(reviewFile, sep=',', names=['listing_id', 'id', 'date', 'reviewer_id', 'reviewer_name', 'comments'])
#print data.shape
reviews = data['comments']
#print reviews[:10]
test_set = reviews[:10]
print test_set
for i in range(10):
	sentence = TextBlob(test_set[i])
	print sentence
	print sentence.sentiment 
	# print sentence.detect_language	
	print '\n'