#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:11:03 2017

@author: arranzignacio
"""
import pandas as pd

def get_ranking(user_cluster,listing_cluster):
    
    filename = "reviewers_rating_per_listing_cluster.csv"
    reviewers = pd.read_csv(filename, sep=',')
    reviewers = reviewers[['reviewer_id','clusters']]
    
    reviewers = reviewers[reviewers.clusters.isin([user_cluster])]
    reviewers = list(reviewers.reviewer_id)
    reviewers = [str(float(i)) for i in reviewers]
    
    
    
    filename = "../datasets/All_listings/clusterized_listings_with_AMN.csv"
    listings = pd.read_csv(filename, sep=',')
    listings = listings[['id','listing_AMN_cluster']]
    
    listings=listings.rename(columns = {'id':'listing_id'})
    listings = listings[listings.listing_AMN_cluster.isin([float(listing_cluster)])]
    listings = list(listings.listing_id)
    listings = [str(i) for i in listings]
    
    
    filename = "listings_with_score.csv"
    listings_with_score = pd.read_csv(filename, sep=',')
    listings_with_score = listings_with_score[['listing_id','reviewer_id','Polarity']]
    
    listings_with_score = listings_with_score[listings_with_score.listing_id.isin(listings)]
    listings_with_score = listings_with_score[listings_with_score.reviewer_id.isin(reviewers)]
    
    listings_with_score = listings_with_score[listings_with_score['Polarity'] != 'not available']

    
    listings_with_score = listings_with_score.sort_values(by='Polarity', ascending=False)
    
    return listings_with_score




#--------------------------------------------------------------------
    
a = get_ranking(5,3)

print a
    
    