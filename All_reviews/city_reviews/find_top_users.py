#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:56:46 2017

@author: arranzignacio
"""

import pandas as pd

filename = 'all_reviews.csv'
all_cities = pd.read_csv(filename, sep=',', header=0)

#unique_listings = all_cities.listing_id.unique()
#unique_reviewers = all_cities.reviewer_id.unique()
#unique_reviews = all_cities.id.unique()

reviewers = all_cities.groupby(['reviewer_id'])['id'].count().reset_index()

#create files with repeat users

repetitions = [2,3,4,7,10,15]

indice = [1,2,3,6,9,14]

for i in range(len(indice)):
    top_users = reviewers.loc[reviewers['id'] > indice[i]]
    
    users_list = top_users.reviewer_id
    
    reviews_reduced = all_cities[all_cities.reviewer_id.isin(users_list)]
    
    file_name = "reviews_by_top_users_"+str(repetitions[i])+"_reviews.csv"

    reviews_reduced.to_csv(file_name, sep=',', encoding='utf-8')
    print(i)

print('done')

