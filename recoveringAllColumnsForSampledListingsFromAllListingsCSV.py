#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:48:54 2017

@author: lmy
"""

import pandas as pd

all_df = pd.read_csv('/Volumes/GoogleDrive/My Drive/CIS 519 Project/Data/All_listings/all_cities_listings.csv')
samples_df = pd.read_csv('/Volumes/GoogleDrive/My Drive/CIS 519 Project/Data/All_listings/sample_listings_amenities.csv', index_col='id')

result_df = all_df[all_df['id'].isin(samples_df['id'].tolist())]

result_df.set_index('id').to_csv('datasets/All_listings/sample_listings.csv')
