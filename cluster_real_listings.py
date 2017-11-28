#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 09:49:59 2017

@author: arranzignacio
"""

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import csv

from sklearn.decomposition import PCA

def check_PCA(X):
    r,c = X.shape
    pca = PCA(n_components=c)
    pca.fit(X)
    return pca.explained_variance_ratio_

def apply_PCA(X,comps):
    pca = PCA(n_components=comps)
    pca.fit(X)
    coefficients = pca.components_
    X_transformed = pca.transform(X)
    return coefficients, X_transformed

def normalize(X):
    X_scaled = preprocessing.scale(X)
    return X_scaled


def clean_data(X):
    #removes accomodates
    attributes1 = ['id','review_scores_rating']
    attributes2 = ['review_scores_accuracy','review_scores_cleanliness','review_scores_checkin']
    attributes3 = ['review_scores_communication','review_scores_location','review_scores_value']
    attributes4 = ['percentile_price']
    attributes = attributes1+attributes2+attributes3+attributes4
    
    X = X[attributes]
    X = X.dropna(axis=0)
    return X


def plot_pretty_graph(X_transformed,kmeans,index1,index2):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    x_min, x_max = X_transformed[:, index1].min() - 1, X_transformed[:, index1].max() + 1
    y_min, y_max = X_transformed[:, index2].min() - 1, X_transformed[:, index2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    plt.plot(X_transformed[:, index1], X_transformed[:, index2], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering with listing reviews'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.label(Z)
    plt.show()
    

def plot_pretty_3Dgraph(X_transformed,cluster_labels):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    print X_transformed.shape
    print len(cluster_labels)

    
    colors = ['b','g','r','c','m','y','k','w']
    
    for i in range(2000):
        xs = X_transformed[i][0]
        ys = X_transformed[i][1]
        zs = X_transformed[i][2]
        indice = cluster_labels[i]
        ax.scatter(xs, ys, zs, c=colors[indice], marker='o')
        
    plt.show()
    

def merge_ids_PCS_cluster(ids,PCs,clusters):
    ids = ids.as_matrix()
    r, c = ids.shape
    clusters = clusters.reshape((r,1))
    
    final = np.hstack((ids,PCs,clusters))
    
    return final
    
    
    


X = pd.read_csv('all_listings_reviewed_by_top_reviewers.csv', header = 0)

X = clean_data(X)

ids = X[['id']]

data = X.drop(['id'],axis=1)

data = normalize(data)

#CHECKED - ALL VALUES ARE NUMERIC, WE'RE GOLDEN - and we dropped NAs


variance_ratio = check_PCA(data)

var1=np.cumsum(np.round(variance_ratio, decimals=4)*100)

print var1

#plt.plot(var1)

#FIRST TWO EXPLAIN 70%

coefficients3, X_transformed3 = apply_PCA(data,3)
coefficients2, X_transformed2 = apply_PCA(data,2)

coefficients_nice3=np.round(coefficients3, decimals=4)*100

print coefficients_nice3


x = X_transformed2[:,0]
y = X_transformed2[:,1]


#DEFINE THE CLUSTERING FUNCTION
kmeans = KMeans(n_clusters=8)

kmeans.fit(X_transformed3)

cluster_labels = kmeans.predict(X_transformed3)


plot_pretty_3Dgraph(X_transformed3,cluster_labels)


output = merge_ids_PCS_cluster(ids,X_transformed3,cluster_labels)

print output[0]
'''
header = ["id","PC1","PC2","PC3","cluster"]

with open('clusterized_listings.csv', 'a') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(header)
    for i in range(len(output)):
        writer.writerow(output[i])
        
'''
