#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:51:52 2017

@author: lmy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
from surprise import evaluate, print_perf


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


#%% Configurations

use_mini = False
verbose  = True
random_state = 42 # Using 42, because it's the Answer to the Ultimate Question of Life, the Universe, and Everything.
csv_file = 'datasets/Paris/Paris_reviews_with_sentiment_5K-15K.csv'


#%% Import data:

p('Reading dataset from', csv_file, '...')
df = pd.read_csv(csv_file, header = 0,  usecols = ['reviewer_id', 'listing_id', 'Polarity'])

from surprise import Dataset, Reader

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(-1, 1))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df, reader)

# Retrieve the trainset.
trainset = data.build_full_trainset()


#%% Initialize the algorithm:

from surprise import SVD
algo = SVD()
algo.train(trainset)
evaluate(algo, data)

#%%


uid = str(50345378)  # raw user id (as in the ratings file). They are **strings**!
iid = str(4596040)  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, verbose=True)

p('Demo prediction: User #{0} should rate {1:.3f} on Listing #{2}.'.format(pred.uid, pred.est, pred.iid))






