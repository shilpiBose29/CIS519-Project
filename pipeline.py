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
listings_with_score_csv = 'getting_scores/listings_with_score.csv'

amenity_vectorizer_pkl = 'amenity_vectorizer.pkl'

#%% my print function

p = print

#%% Read datasets:

user_groups_df  = pd.read_csv(user_groups_csv, encoding='utf-8', index_col = 'reviewer_id').rename(columns = {'clusters': 'groupID'})
eigenmatrix_df  = pd.read_csv(eigenmatrix_csv, encoding='utf-8').drop('user_cluster', axis=1)
your_ratings_to_listings_df = pd.read_csv(your_ratings_to_listings_csv, encoding='utf-8')
listing_clusters_df         = pd.read_csv(listing_clusters_csv, encoding='utf-8')
listing_details_df          = pd.read_csv(listings_details_csv, encoding='utf-8', index_col = 'id', usecols = ['id', 'name', 'city'])
PCA_loadings_df = pd.read_csv(PCA_loadings_csv, encoding='utf-8')
amn_all_col = pd.read_csv(all_listings_csv, usecols=['id','amenities'], index_col='id')['amenities'].dropna().str[1:-1]
listings_with_score_df = pd.read_csv(listings_with_score_csv, encoding='utf-8', header=0, 
                                     usecols=['listing_id','reviewer_id','Polarity'], 
                                     na_values = {'listing_id': 'listing_id', 
                                                  'reviewer_id': 'reviewer_id', 
                                                  'Polarity': 'not available'}).dropna()

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
p('-'*30, '\n', your_ratings_to_listings, '\n', '-'*30)

p('Based on your input, we think that you have the following ratings to these types of places:')
your_ratings_to_clusters = listing_clusterer(your_ratings_to_listings)
p(tabulate(pd.DataFrame(your_ratings_to_clusters), headers=['Cluster ID', 'Estimated Rating From You'], tablefmt="fancy_grid"))


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

def get_ranking(your_groupID,your_clusterID):
    '''Returns the list of listingIDs sorted to their ratings as provided by users specified peerIDs.'''
    peersIDs = user_groups_df[user_groups_df.groupID == your_groupID].index
    # get a mapping of all listings to their clusters:
    listings = listing_clusters_df[['id','listing_AMN_cluster']]\
                .astype(int)\
                .rename(columns = {
                        'id':'listing_id', 
                        'listing_AMN_cluster': 'cluster_id'})
    # get a list of listings in your favorite cluster:
    listings = listings[listings.cluster_id==your_clusterID].listing_id.tolist()
    # narrow down listings by clusters:
    listings_with_score = listings_with_score_df[listings_with_score_df.listing_id.isin(listings)]
    # narrow down listings by peers:
    listings_with_score = listings_with_score[listings_with_score.reviewer_id.isin(peersIDs)]
    # average_ratings_of_suggested_listings_by_peers :
    return listings_with_score.groupby(['listing_id'])['Polarity'].mean().sort_values(ascending=False) 
# Sort listings in this cluster by ratings according to peers in this group:
listings_with_score = get_ranking(your_clusterID, your_groupID)
# Get a unique list of all suggested listings' IDs:
listingIDs_in_this_cluster = pd.unique(listings_with_score.index)
# Get details of these listings:
picked_listings_details_df = listing_details_df.loc[listingIDs_in_this_cluster]
# Take only the listings in your destination city:
picked_listings_details_df = picked_listings_details_df[picked_listings_details_df.city == destin_city].drop('city', axis=1)
# Append the column of ratings:
picked_listings_details_df['rating'] = listings_with_score[picked_listings_details_df.index]
# Output:
p('For your coming stay in', destin_city, 'May I suggest:')
p(tabulate(picked_listings_details_df.head(), headers=['Name', 'Average Rating From Your Peers'], showindex=False, tablefmt="fancy_grid"))
