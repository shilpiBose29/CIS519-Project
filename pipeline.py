#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:33:42 2017

@author: lmy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd

#%% Configurations

use_mini = False
verbose  = True
random_state = 42 # Using 42, because it's the Answer to the Ultimate Question of Life, the Universe, and Everything.

# path here
listings_ratings_csv = ''
enduser_ratings_csv  = ''

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


#%% Read datasets:

user_ratings_df = pd.read_csv(user_ratings_csv)


#%% Train a ratings-to-clusters classifier for listings:

clusters_ratings_to_groups_clsfr = KMeans(n_clusters=8).fit(user_ratings_df[1:]) 
user_groups_df = clusters_ratings_to_groups_clsfr.predict(user_ratings_df[1:]) # ['userID', 'groupID']



#%% simulate enduser input:

p('Please enter a list of listings and your ratings of them:')
enduser_listings_ratings_df = pd.read_csv(enduser_listings_ratings_csv)
p(enduser_ratings_df)

p('Based on your input, we think that you have the following ratings to these groups:')
# create new column "clusterID" in enduser_listings_ratings_df, value of which being the 
# clusterID in listings_ratings_df where enduser_listings_ratings_df.listingID == listings_ratings_df.listingID
enduser_listings_ratings_df['clusterID'] = ..........
# Create a dictionary of all listings_clusterIDs to the enduser_ratings of them, taking zero for unrated clusters:
enduser_clusters_ratings_df = ......
p(enduser_clusters_ratings_df.head())

# Identify which group this enduser belongs to:
p('Hence, we think that you are one of these people:')
enduser_group = clusters_ratings_to_groups_clsfr.predict(enduser_clusters_ratings_df)
p(user_groups_df[ user_groups_df['groupID'] == enduser_group ])



