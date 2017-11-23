#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:05:01 2017

@author: arranzignacio
"""


import pandas as pd


def concatenate_all_reviews(cities):
    
    reviews = []
    
    for i in range(len(cities)):
        review = cities[i]+"-reviews.csv"
        reviews.append(review)
    
    header = ['listing_id', 'id', 'date', 'reviewer_id', 'reviewer_name', 'comments']
    reviews_dictionary = {}
    
    for i in range(len(reviews)):
        df = pd.read_csv(reviews[i], sep=',', names=header)
        reviews_dictionary[cities[i]] = df
    
    all_reviews = pd.concat(reviews_dictionary.values(), ignore_index=True)
    
    return all_reviews

###########################################################################################################

if __name__ == '__main__':
    
    cities1 = ['Paris','London','Rome','NewYork']
    cities2 = ['Madrid','Amsterdam','Athens','Barcelona']
    cities3 = ['Berlin','Boston','Brussels','Chicago']
    cities4 = ['Copenhagen','Dublin','Edinburgh','Geneva']
    cities5 = ['HongKong','LosAngeles']
    
    cities = cities1 + cities2 + cities3 + cities4 + cities5
    
    all_reviews = concatenate_all_reviews(cities)
    print all_reviews.shape
    
    file_name = 'all_reviews.csv'
    all_reviews.to_csv(file_name, sep=',', encoding='utf-8')

    

