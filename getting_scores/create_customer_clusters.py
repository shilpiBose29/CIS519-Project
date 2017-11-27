#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:51:08 2017

@author: arranzignacio
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
#from sklearn.decomposition import SparsePCA

def normalize(X):
    X_scaled = preprocessing.scale(X)
    return X_scaled

def assign_missing_values(X):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    X_transformed = imp.transform(X)
    return X_transformed

def check_PCA(X):
    r,c = X.shape
    pca = PCA(n_components=c)
    pca.fit(X)
    return pca.explained_variance_ratio_

def check_truncated_SVD(X):
    r,c = X.shape
    svd = TruncatedSVD(n_components=(c-1))
    svd.fit(X)
    return svd.explained_variance_ratio_

def check_randomized_PCA(X):
    r,c = X.shape
    pca = RandomizedPCA(n_components=c)
    pca.fit(X)
    return pca.explained_variance_ratio_

def check_sparse_PCA(X):
    r,c = X.shape
    pca = SparsePCA(alpha = 1)
    pca.fit(X)
    return pca.components_ 

def apply_PCA(X,comps):
    pca = PCA(n_components=comps)
    pca.fit(X)
    coefficients = pca.components_
    X_transformed = pca.transform(X)
    return coefficients, X_transformed

def apply_randomized_PCA(X,comps):
    pca = RandomizedPCA(n_components=comps)
    pca.fit(X)
    coefficients = pca.components_
    X_transformed = pca.transform(X)
    return coefficients, X_transformed

def merge_data_cluster(data,clusters):
    

    print data.shape
    #r = len(clusters)
    #clusters = clusters.reshape((r,1))
    clusters_df = pd.DataFrame({'clusters':clusters})
    print clusters_df.shape
    
    final = pd.concat([data.reset_index(),clusters_df.reset_index()],axis=1)
    
    print final.shape

    return final

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

#print df3[0:10]

df4 = df3.groupby(['reviewer_id','cluster'])['Polarity'].mean()
df4 = df4.reset_index()

df5 = df4.pivot(index='reviewer_id', columns='cluster', values='Polarity')

#print df5[0:10]

#NOT A FAN OF THIS, BUT THIS ASSIGNS MISSING VALUE
df6 = df5.fillna(0)
#df6 = assign_missing_values(df5)

kmeans = KMeans(n_clusters=8)
kmeans.fit(df6)
cluster_labels = kmeans.predict(df6)
#print cluster_labels

finale = merge_data_cluster(df6,cluster_labels)

print finale[0:10]

finale2 = pd.melt(finale, id_vars=['reviewer_id', 'clusters'], 
            value_vars=list(finale.columns[1:9]), # list of days of the week
            var_name='user_cluster', 
            value_name='Polarity')

finale2['Polarity'] = finale2['Polarity'].replace(0.000000,np.nan)


finale3 = finale2.groupby(['user_cluster','clusters'])['Polarity'].mean()
finale3 = finale3.reset_index()

finale4 = finale3.pivot(index='user_cluster', columns='clusters', values='Polarity')

print finale4
#coefficients3, X_transformed3 = apply_PCA(data,3)

