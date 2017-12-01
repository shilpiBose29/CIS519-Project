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
import pickle
from tabulate import tabulate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier as KNN
from scipy.sparse import load_npz, save_npz

#%% Configurations

verbose  = True
random_state = 42 # Using 42, because it's the Answer to the Ultimate Question of Life, the Universe, and Everything.
destin_city = 'Paris'

# Paths here
your_ratings_to_listings_csv  = 'datasets/recommender_system/enduser_listings_ratings.csv'
eigenmatrix_csv = 'getting_scores/user_cluster_rating_per_listing_cluster.csv'
user_groups_csv = 'getting_scores/reviewers_rating_per_listing_cluster.csv'
listing_clusters_csv = 'datasets/All_listings/clusterized_listings_with_AMN.csv'
listings_details_csv = 'datasets/All_listings/sample_listings.csv'
all_listings_csv = 'datasets/All_listings/all_listings.csv'
sample_listings_amenities_csv = 'datasets/All_listings/sample_listings_amenities.csv'
#sample_listings_amenities_expanded_csv = 'datasets/All_listings/sample_listings_amenities_expanded.csv'
#sample_listings_amenities_expanded_npz = 'datasets/All_listings/sample_listings_amenities_expanded.npz'
PCA_loadings_csv = 'datasets/All_listings/sample_AMN_PCA_U.csv'

amenity_vectorizer_pkl = 'amenity_vectorizer.pkl'

#%% my print function

import datetime
from pprint import pformat as pf
dt = lambda: datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
filename = datetime.datetime.now().strftime("Output %Y-%m-%d %H-%M-%S.txt")
def p(*argv):
    if not verbose: return
    string = dt()
    for i in argv:
        if type(i) == str: string += ' '+i
        else: string += ' '+pf(i)
    with open(filename, 'a') as f: print(string, file = f)
    print(string)

#%% Read datasets:

user_groups_df  = pd.read_csv(user_groups_csv, index_col = 'reviewer_id').rename(columns = {'clusters': 'groupID'})
eigenmatrix_df  = pd.read_csv(eigenmatrix_csv).drop('user_cluster', axis=1)
your_ratings_to_listings_df = pd.read_csv(your_ratings_to_listings_csv)
listing_clusters_df         = pd.read_csv(listing_clusters_csv)
listings_details_df         = pd.read_csv(listings_details_csv)
PCA_loadings_df = pd.read_csv(PCA_loadings_csv)
amn_all_col = pd.read_csv(all_listings_csv, usecols=['id','amenities'], index_col='id')['amenities'].dropna().str[1:-1]


kmeans = pickle.load(open("datasets/All_listings/kmeans_model.pkl", 'rb'))

#%% Derive intermediate variables:

num_groups, num_clusters    = eigenmatrix_df.shape
average_ratings_for_cluster = eigenmatrix_df.mean(axis=1).tolist()


#%% Preapre Amenity Vectorizer:
try:
    vectorizer = pickle.load(open(amenity_vectorizer_pkl, 'rb'))
except IOError:
    amn_smp_col = pd.read_csv(sample_listings_amenities_csv).dropna()['amenities'].str[1:-1]
    p('Training Amenity Vectorizer...')
    vectorizer = CountVectorizer(token_pattern = ur'(?!,|$)(.+?)(?=,|$)',decode_error='ignore')
    vectorizer.fit(amn_smp_col)
    p('Success.')
    pickle.dump(vectorizer, open(amenity_vectorizer_pkl, 'w'))
else:
    labels = vectorizer.get_feature_names()
    p('Successfully loaded Amenity Vectorizer.')
#%%


all_columns = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20"]

def listing_clusterer(your_ratings_to_listings):
    #MERGE ENDUSER PREFERENCES WITH AMENITIES
    enduser_listings_with_AMN = amn_all_col.loc[your_ratings_to_listings.index]
    enduser_listings_with_AMN_binary = pd.DataFrame(
                                            vectorizer.transform(enduser_listings_with_AMN).todense(), 
                                            columns = ['AMN_'+label for label in labels]
                                        ).set_index(enduser_listings_with_AMN.index)
    enduser_listings_PCA = pd.DataFrame(np.matmul(enduser_listings_with_AMN_binary,PCA_loadings_df),columns=all_columns)*100
    enduser_cluster_labels = pd.DataFrame({'clusterID': kmeans.predict(enduser_listings_PCA)}).set_index(enduser_listings_with_AMN.index)
    enduser_PCAs_with_cluster = pd.concat([enduser_cluster_labels,your_ratings_to_listings],axis=1)
    # now group by clusterID:
    enduser_preference_profile = pd.DataFrame(enduser_PCAs_with_cluster.groupby(['clusterID'])['ratings'].mean())
    return enduser_preference_profile['ratings']

#%% simulate enduser input:

p('Please enter a list of listings and your ratings of them:')
your_ratings_to_listings = pd.read_csv(your_ratings_to_listings_csv, index_col = 'listing_id').ratings
p('\n', your_ratings_to_listings)

p('(For debug only) Based on your input, we think that you have the following ratings to these types of places:')
your_ratings_to_clusters = listing_clusterer(your_ratings_to_listings)
p('\n', your_ratings_to_clusters)


#%% Identify which group this enduser belongs to:
clusters_you_rated = your_ratings_to_clusters.index
eigenmatrix_reduced_df = eigenmatrix_df.iloc[:,clusters_you_rated]
users_grouper = KNN(n_neighbors=1).fit(X = eigenmatrix_reduced_df, y = eigenmatrix_reduced_df.index)
your_groupID = users_grouper.predict(np.array(clusters_you_rated).reshape(1, -1))[0]
p('Hence, we think that you are one of the group', your_groupID, '.')
your_peerIDs = user_groups_df[ user_groups_df.groupID == your_groupID ].index
p('You have', len(your_peerIDs), 'peers.')
your_clusterID = int(eigenmatrix_df.loc[your_groupID].idxmax())
p('You people love listings in cluster', your_clusterID, '.')

#%% Sort clusters by ratings from this group:


def resort_according_to_peers(peerIDs, listingIDs):
    '''Returns the list of listingIDs sorted to their ratings as provided by users specified peerIDs.'''
    
    users_in_this_group        = user_groups_df[user_groups_df.groupID == groupID]
    #.............
    pass

listingIDs_in_this_cluster = resort_according_to_peers(
    peerIDs = your_peerIDs,
    listingIDs = listing_clusters_df[listing_clusters_df.index == clusterID].index)

picked_listings_details_df = listings_details_df[listingIDs_in_this_cluster]
# Take only the listings in your destination city:
picked_listings_details_df = picked_listings_details_df[picked_listings_details_df.city == destin_city]

p('May I suggest:')
p(picked_listings_details_df[['name']].head())