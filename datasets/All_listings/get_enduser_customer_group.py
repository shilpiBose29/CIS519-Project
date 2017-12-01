#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:06:32 2017

@author: arranzignacio
"""
import pandas as pd
import numpy as np

filename = "../../getting_scores/user_cluster_rating_per_listing_cluster.csv"
user_groups = pd.read_csv(filename, sep=',')


filename = "enduser_preference_profile.csv"
enduser_customer_profile = pd.read_csv(filename, sep=',')


enduser_customer_profile.set_index(['listing_cluster'])


print enduser_customer_profile

selected_listings = list(enduser_customer_profile.listing_cluster)

selected_listings = [str(i) for i in selected_listings]

user_groups_modified = user_groups[selected_listings]

print user_groups_modified

#subtraction = user_groups

