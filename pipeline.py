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
from random import choice
# for plotting
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

#%% Configurations

verbose  = True
random_state = 42 # Using 42, because it's the Answer to the Ultimate Question of Life, the Universe, and Everything.

# Paths here
your_ratings_to_listings_csv    = 'datasets/recommender_system/enduser_listings_ratings.csv'
eigenmatrix_csv                 = 'getting_scores/user_cluster_rating_per_listing_cluster.csv'
user_groups_csv                 = 'getting_scores/reviewers_rating_per_listing_cluster.csv'
listing_clusters_csv            = 'datasets/All_listings/clusterized_listings_with_AMN.csv'
listings_details_csv            = 'datasets/All_listings/sample_listings.csv'
all_listings_csv                = 'datasets/All_listings/all_listings.csv'
sample_listings_amenities_csv   = 'datasets/All_listings/sample_listings_amenities.csv'
#sample_listings_amenities_expanded_csv = 'datasets/All_listings/sample_listings_amenities_expanded.csv'
#sample_listings_amenities_expanded_npz = 'datasets/All_listings/sample_listings_amenities_expanded.npz'
PCA_loadings_csv                = 'datasets/All_listings/sample_AMN_PCA_U.csv'
listings_with_score_csv         = 'getting_scores/listings_with_score.csv'

amenity_vectorizer_pkl          = 'amenity_vectorizer.pkl'

#%% my print function

# =============================================================================
# import datetime
# from pprint import pformat as pf
# dt = lambda: datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
# filename = datetime.datetime.now().strftime("Output %Y-%m-%d %H-%M-%S.txt")
# def p(*argv):
#     if not verbose: return
#     string = dt()
#     for i in argv:
#         if type(i) == str:
#             string += ' '+i
#         else:
#             string += ' '+pf(i)
#     with open(filename, 'a') as f:
#         print(string, file = f)
#     print(string)
# 
# =============================================================================

p = print
#%% Read datasets:

user_groups_df  = pd.read_csv(user_groups_csv, encoding='utf-8', index_col = 'reviewer_id').rename(columns = {'clusters': 'groupID'})
eigenmatrix_df  = pd.read_csv(eigenmatrix_csv, encoding='utf-8').drop('user_cluster', axis=1)
eigenmatrix_df.index = pd.to_numeric(eigenmatrix_df.index)
your_ratings_to_listings_df = pd.read_csv(your_ratings_to_listings_csv, encoding='utf-8')
listing_clusters_df         = pd.read_csv(listing_clusters_csv, encoding='utf-8')
listing_details_df          = pd.read_csv(listings_details_csv, encoding='utf-8', index_col = 'id', usecols = ['id', 'name', 'city', 'thumbnail_url', 'listing_url', 'latitude', 'longitude'])
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
    #global enduser_preference_profile, enduser_cluster_labels
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
    enduser_cluster_labels.clusterID.value_counts().plot.pie(label='Cluster ID', figsize=(2, 2),autopct='%d%%').set_title('Listings You Rated Grouped By Cluster')
    centre_circle = plt.Circle((0,0),0.75,color='white', fc='white',linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    return enduser_preference_profile['ratings']

def autolabel(rects, ax, c):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                 "%.2f" % (height),
                ha='center', va='top', color = c)

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

#%% simulate enduser input:

p('Please enter a list of listings and your ratings of them:')
random_enduser = True
if random_enduser:
    num_listings_enduser_rated = np.random.randint(5,15)
    your_ratings_to_listings = pd.DataFrame(data={'ratings':np.random.random(num_listings_enduser_rated), 
                                                  'listing_id': amn_all_col.index.to_series().sample(num_listings_enduser_rated).astype(float)}).ratings
else:
    your_ratings_to_listings = pd.read_csv(your_ratings_to_listings_csv, index_col = 'listing_id').ratings
p('-'*30, '\n', your_ratings_to_listings, '\n', '-'*30)
p('Listings you visited, shown as proportions of clusters:')
your_ratings_to_clusters = listing_clusterer(your_ratings_to_listings)
#p('Based on your input, we think that you have the following ratings to these types of places:')
#p(tabulate(pd.DataFrame(your_ratings_to_clusters), headers=['Cluster ID', 'Estimated Rating From You'], tablefmt="fancy_grid"))


# Identify which group this enduser belongs to:
clusters_you_rated = your_ratings_to_clusters.index
eigenmatrix_reduced_df = eigenmatrix_df.iloc[:,clusters_you_rated]
users_grouper = KNN(n_neighbors=1).fit(X = eigenmatrix_reduced_df, y = eigenmatrix_reduced_df.index)
your_groupID = users_grouper.predict(np.array(clusters_you_rated).reshape(1, -1))[0]
p('Hence, we think that you are one of the group', your_groupID, '.')
your_peerIDs = user_groups_df[ user_groups_df.groupID == your_groupID ].index
p('You have', len(your_peerIDs), 'peers.')
your_clusterID = int(eigenmatrix_df.loc[your_groupID].idxmax())
ratings_from_peers = eigenmatrix_df.loc[your_groupID]

# draw ratings to clusters:

cluster_ratings_from_all_groups = eigenmatrix_df.mean()
clusterIDs = pd.to_numeric(cluster_ratings_from_all_groups.index)
num_clusters = len(clusterIDs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
colormap = plt.get_cmap('terrain')
bar1 = ax1.bar(clusterIDs, ratings_from_peers, width=.8, label='Your Peers', 
               facecolor=[1,0,0,.5])
bar2 = ax1.bar(your_ratings_to_clusters.index, your_ratings_to_clusters, width=.9, label='You', 
               facecolor=[0,1,0,.5])
bar3 = ax1.bar(clusterIDs, cluster_ratings_from_all_groups, width=.7, label='All Users', 
               facecolor=[0,0,1,.5])
ax1.legend((bar1, bar2, bar3), ('Your Peers', 'You', 'All Users'), loc='best')
autolabel(bar1, ax=ax1, c='white')
autolabel(bar2, ax=ax1, c='white')
autolabel(bar3, ax=ax1, c='white')
ax1.set_xlabel('Listing Cluster ID')
ax1.set_ylabel('Rating')
ax1.set_ylim(eigenmatrix_df.min().min(), eigenmatrix_df.max().max())
ax1.set_title('Ratings From Your User Group To Each Listing Cluster')

hm = sns.heatmap(eigenmatrix_df, annot=True, fmt=".2f", square=True)
hm.add_patch(Rectangle((0, your_groupID), len(ratings_from_peers), 1, fill=False, edgecolor='blue', lw=3))
ax2.set_xlabel('Listing Cluster ID')
ax2.set_ylabel('User Group ID')
ax2.set_title('Ratings From Each User Group To Each Listing Cluster')
plt.show()

p('You people love listings in cluster', your_clusterID, '.')

# Sort clusters by ratings from this group:

destin_city = choice(['New York', 'London', 'Paris'])

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
picked_listings_details_df['visual'] = ['|'*int(10*i) for i in picked_listings_details_df['rating']]
# Output:
p('For your coming stay in', destin_city, ', may I suggest:')
p(tabulate(picked_listings_details_df[['name', 'rating', 'visual']].head(), headers=['Name', '', 'Average Rating From Your Peers'], showindex=False, tablefmt="fancy_grid"))
#first_5_images = picked_listings_details_df.thumbnail_url.fillna('http://via.placeholder.com/216x144').head()
#for url in first_5_images:
#    Image(url.split('?')[0]).show()
p(picked_listings_details_df.listing_url.head())
