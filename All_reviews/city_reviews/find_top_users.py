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

folder = 'top_users/'

for i in range(len(indice)):
    top_users = reviewers.loc[reviewers['id'] > indice[i]]
    file_name = "top_users_"+str(repetitions[i])+"_reviews.csv"
    
    path = folder+file_name
    
    top_users.to_csv(path, sep=',', encoding='utf-8')
    print(i)

print('done')