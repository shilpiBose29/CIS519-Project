#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:54:19 2017

@author: arranzignacio
"""
from __future__ import unicode_literals
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import csv
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer



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
    

def merge_ids_PCS_cluster(ids,clusters,PCs):
    ids = ids.as_matrix()
    r, c = ids.shape
    #clusters = clusters.as_matrix()
    #print clusters.shape
    clusters = clusters.reshape((r,1))
    
    final = np.hstack((ids,clusters,PCs))
    
    return final

def save_to_csv(matrix,filename):

    header1 = ["id","listing_AMN_cluster"]
    header2 = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10"]
    header3 = ["PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20"]
    
    header = header1 + header2 + header3
    
    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        for i in range(len(matrix)):
            writer.writerow(matrix[i])


def get_amenities_listings(original,dataframe):
    vectorizer = CountVectorizer(token_pattern = ur'(?!,|$)(.+?)(?=,|$)',encoding='utf-8',decode_error='ignore')
    # first, get rid of all the '\"'s:
    dataframe = dataframe.str[1:-1].replace('\"', '')
    # Now actually transform the data:
    X = vectorizer.fit_transform(dataframe)
    labels = vectorizer.get_feature_names()
    
    expanded_df = pd.DataFrame(X.todense(), columns = ['AMN_'+label for label in labels])
    expanded_df.set_index(original.listing_id, inplace=True)
    
    return expanded_df

def get_amenities_listings_original(original,dataframe):
    amn_df = pd.read_csv('sample_listings_amenities.csv').dropna()
    amenities_col = amn_df['amenities']
    vectorizer = CountVectorizer(token_pattern = ur'(?!,|$)(.+?)(?=,|$)',decode_error='ignore')
    amenities_col = amenities_col.str[1:-1].replace('\"', '')
    X1 = vectorizer.fit_transform(amenities_col)
    
    labels = vectorizer.get_feature_names()
    
    X2 = vectorizer.transform(dataframe.amenities)
    expanded_df = pd.DataFrame(X2.todense(), columns = ['AMN_'+label for label in labels])
    expanded_df.set_index(original.listing_id, inplace=True)
    
    return expanded_df


    
    
    

#----------------MAIN---------------------------------------------------


filename = "sample_AMN_PCAed.csv"
listings_with_AMN_PCAs = pd.read_csv(filename, sep=',')
ids = listings_with_AMN_PCAs[['ID']]
listings_with_AMN_PCAs = listings_with_AMN_PCAs.drop(['ID'],axis=1)

kmeans = KMeans(n_clusters=8)

kmeans.fit(listings_with_AMN_PCAs)
cluster_labels = kmeans.predict(listings_with_AMN_PCAs)
listings_clusters_and_PCAs = merge_ids_PCS_cluster(ids,cluster_labels,listings_with_AMN_PCAs)



#NOW GETTING INFO FROM ENDUSER
enduser_filename_path = "../recommender_system/enduser_listings_ratings.csv" 
enduser_preferences = pd.read_csv(enduser_filename_path, sep=',')

ratings = enduser_preferences.ratings



#NOW GETTING ALL LISTINGS
all_listings_file = "../../../all_listings.csv"
all_listings = pd.read_csv(all_listings_file, sep=',')
all_listings=all_listings.rename(columns = {'id':'listing_id'})


#MERGE ENDUSER PREFERENCES WITH AMENITIES
enduser_listings_with_AMN = enduser_preferences.merge(all_listings, on='listing_id')
enduser_listings_with_AMN = enduser_listings_with_AMN[['listing_id','amenities']]
amenities_col = enduser_listings_with_AMN['amenities']


enduser_listings_with_AMN_binary = get_amenities_listings_original(enduser_preferences,enduser_listings_with_AMN)



#GET THE DOCUMENT THAT HAS THE LOADINGS FOR THE DIFFERENT AMENITIES
filename = "sample_AMN_PCA_U.csv"
PCA_loadings = pd.read_csv(filename, sep=',')


columns1 = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10"]
columns2 = ["PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20"]
all_columns = columns1 + columns2

enduser_listings_PCA = pd.DataFrame(np.matmul(enduser_listings_with_AMN_binary,PCA_loadings),columns=all_columns)
enduser_listings_PCA.set_index(enduser_listings_with_AMN.listing_id, inplace=True)

enduser_listings_PCA = enduser_listings_PCA*100 

#print enduser_listings_PCA

enduser_cluster_labels = kmeans.predict(enduser_listings_PCA)

enduser_cluster_labels = pd.DataFrame(enduser_cluster_labels.reshape((len(enduser_cluster_labels),1)),columns =['listing_cluster'])
enduser_cluster_labels.set_index(enduser_listings_with_AMN.listing_id, inplace=True)
ratings = pd.DataFrame(ratings,columns=['ratings'])
ratings.set_index(enduser_listings_with_AMN.listing_id, inplace=True)

enduser_PCAs_with_cluster = pd.concat([enduser_listings_PCA,enduser_cluster_labels,ratings],axis=1)

#need to add the rating

enduser_preference_profile = pd.DataFrame(enduser_PCAs_with_cluster.groupby(['listing_cluster'])['ratings'].mean())

#enduser_preference_profile['listing_cluster'] = enduser_preference_profile.index

return enduser_preference_profile




