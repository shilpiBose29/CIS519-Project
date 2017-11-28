#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:00:30 2017

@author: arranzignacio
"""

import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
#from sparsesvd import sparsesvd


filename1 = "listings_with_score.csv"
df1 = pd.read_csv(filename1, sep=',')

filename2 = "../clusterized_listings.csv"
df2 = pd.read_csv(filename2, sep=',')

df1 = df1.drop(df1.index[df1.Polarity == 'not available'])

df2=df2.rename(columns = {'id':'listing_id'})

df2["listing_id"] = df2["listing_id"].astype(int)
df2["listing_id"] = df2["listing_id"].astype(str)

df3 = pd.merge(df1, df2, on='listing_id')

df3 = df3.drop('Subjectivity',1)

df3["Polarity"] = df3["Polarity"].astype(float)

#df4 = df3.groupby(['reviewer_id','cluster'])['Polarity'].mean()
#Sparse_matrix = df4.reset_index()

Sparse_matrix = df3

u, s, vt = svds(Sparse_matrix, k=2)

print u


#Average_cluster = Sparse_matrix.groupby(['cluster'])['Polarity'].mean()
#print Average_cluster
#Average_user = Sparse_matrix.groupby(['reviewer_id'])['Polarity'].mean()
#print Average_user[0:10]

