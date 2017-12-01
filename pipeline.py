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
destin_city = 'Paris'

# Paths here
your_ratings_to_listings_csv  = 'datasets/recommender_system/enduser_listings_ratings.csv'
eigenmatrix_csv = 'getting_scores/user_cluster_rating_per_listing_cluster'
user_groups_csv = 'getting_scores/reviewers_rating_per_listing_cluster.csv'
listing_clusters_csv = 'datasets/All_listings/clusterized_listings_with_AMN.csv'
listings_details_csv = 'datasets/All_listings/sample_listings.csv'
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

user_groups_df  = pd.read_csv(user_groups_csv)
eigenmatrix_df  = pd.read_csv(eigenmatrix_csv)
your_ratings_to_listings_df = pd.read_csv(your_ratings_to_listings_csv)
listing_clusters_df         = pd.read_csv(listing_clusters_csv)
listings_details_df         = pd.read_csv(listings_details_csv)
#%% Derive intermediate variables:

num_groups, num_clusters    = eigenmatrix_df.shape()
average_ratings_for_cluster = eigenmatrix_df.mean(axis=1).tolist()

###%% Train a ratings-to-clusters classifier for listings:
##clusters_ratings_to_groups_clsfr = KMeans(n_clusters=8).fit(user_ratings_df[1:]) 
##user_groups_df = clusters_ratings_to_groups_clsfr.predict(user_ratings_df[1:]) # ['userID', 'groupID']

#%% simulate enduser input:

p('Please enter a list of listings and your ratings of them:')
your_ratings_to_listings_df = pd.read_csv(your_ratings_to_listings_csv)
p(your_ratings_to_listings_df)

p('(For debug only) Based on your input, we think that you have the following ratings to these types of places:')
your_ratings_to_clusters_df = listings_clusterer(your_ratings_to_listings_df)
p(your_ratings_to_clusters_df)

# Identify which group this enduser belongs to:
clusters_you_rated = your_ratings_to_clusters_df[['clustersID']]
eigenmatrix_reduced_df = eigenmatrix_df[[clusters_you_rated]]
from sklearn.neighbors import KNeighborsClassifier as KNN
users_grouper = KNN(n_neighbors=1).fit(X = eigenmatrix_reduced_df, y = eigenmatrix_reduced_df.index)
your_groupID = users_grouper.transform(clusters_you_rated)
p('Hence, we think that you are one of the group', your_groupID, '. Your peers include:')
your_peerIDs = user_groups_df[ user_groups_df[['groupID']] == your_groupID ].index
p(your_peerIDs.head())

# Sort clusters by ratings from this group:
your_clusterID = eigenmatrix_df[your_groupID].idxmax()

def resort_according_to_peers(peerIDs, listingIDs):
    '''Returns the list of listingIDs sorted to their ratings as provided by users specified peerIDs.'''
    
    users_in_this_group        = user_groups_df[user_groups_df.groupID == groupID]
    .............
    pass

listingIDs_in_this_cluster = resort_according_to_peers(
    peerIDs = your_peerIDs,
    listingIDs = listing_clusters_df[listing_clusters_df.clusterID == clusterID].index)

picked_listings_details_df = listings_details_df[listingIDs_in_this_cluster]
# Take only the listings in your destination city:
picked_listings_details_df = picked_listings_details_df[picked_listings_details_df.city == destin_city]

p('May I suggest:')
p(picked_listings_details_df[['name']].head())