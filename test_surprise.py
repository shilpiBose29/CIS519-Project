#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:38:53 2017

@author: lmy
"""

from surprise import SVD
from surprise import Dataset, Reader
from surprise import evaluate, print_perf

import pandas as pd

#remove_non_numerical_rows_in_column = lambda df, col: df[df[col].apply(lambda x: x.isnumeric())]

# Read data:
user_ratings_df = pd.read_csv('getting_scores/listings_with_score.csv', usecols=["reviewer_id", "listing_id", "Polarity"])

# remove rows with non-numerical values in the column "Polarity":
user_ratings_df = user_ratings_df[pd.to_numeric(user_ratings_df['Polarity'], errors='coerce').notnull()]

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(-1, 1))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(user_ratings_df[['reviewer_id', 'listing_id', 'Polarity']], reader)

# split it into 3 folds for cross-validation.
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)

trainset = data.build_full_trainset()
algo.train(trainset)


all_listings = user_ratings_df['listing_id'].unique()

enduserID = str(19634955)

results = pd.DataFrame({'listingID': all_listings, 
                        'rating': [algo.predict(enduserID, str(i)).est for i in all_listings]}).sort_values('rating', ascending=False)

print results.head()
