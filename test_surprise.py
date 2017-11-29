#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:38:53 2017

@author: lmy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from surprise import SVD, GridSearch
from surprise import Dataset, Reader
from surprise import evaluate, print_perf
from pprint import pprint as pp
import pandas as pd
from tabulate import tabulate

#%% Configurations

use_mini = False
verbose  = True
random_state = 42 # Using 42, because it's the Answer to the Ultimate Question of Life, the Universe, and Everything.


#%% my print function

import datetime
from pprint import pformat as pf
dt = lambda: datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
filename = datetime.datetime.now().strftime("Output %Y-%m-%d %H-%M-%S.txt")
def p(*argv):
    if not verbose: return
    string = dt()
    for i in argv:
        if type(i) == str:
            string += ' '+i
        else:
            string += ' '+pf(i)
    with open(filename, 'a') as f:
        print(string, file = f)
    print(string)


#%% Development -- optimize the classifier!

#remove_non_numerical_rows_in_column = lambda df, col: df[df[col].apply(lambda x: x.isnumeric())]

# Read data:
user_ratings_df = pd.read_csv('getting_scores/listings_with_score.csv', usecols=["reviewer_id", "listing_id", "Polarity"])

# remove rows with non-numerical values in the column "Polarity":
user_ratings_df = user_ratings_df[pd.to_numeric(user_ratings_df['Polarity'], errors='coerce').notnull()]

# A reader is still needed but only the rating_scale param is required.
reader = Reader(rating_scale=(-1, 1))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(user_ratings_df[['reviewer_id', 'listing_id', 'Polarity']], reader)

# split it into 3 folds for cross-validation.
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

param_grid = {'n_epochs': [5, 10],
	      'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}

grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])#, 'FCP'])

# Evaluate performances of our algorithm on the dataset.
#perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

#print_perf(perf)

# Evaluate performances of our algorithm on the dataset.
perf = grid_search.evaluate(data)
#print_perf(perf)

#pp(dict(grid_search.best_score))
#pp(dict(grid_search.best_params))
#pp(dict(grid_search.best_estimator))x

#%% Production!

grid_search.best_score['RMSE']

trainset = data.build_full_trainset()
algo.train(trainset)

all_listings = user_ratings_df['listing_id'].unique()

enduserID = str(19634955)
print('Hello, User #'+enduserID+'.')
results = pd.DataFrame({'Listing ID': all_listings, 
                        'Predicted Rating From You': [algo.predict(enduserID, str(i)).est for i in all_listings]})\
         .sort_values('Predicted Rating From You', ascending=False)
print('Based on your previous ratings, here are our suggestions for your future stay:')
print(tabulate(results.head(), headers='keys', showindex=False, tablefmt="fancy_grid"))
print('(root-mean-square error (RMSE): ',grid_search.best_score['RMSE'], ', better smaller.)')