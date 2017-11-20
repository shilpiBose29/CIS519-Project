#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:20:55 2017

@author: arranzignacio
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import numpy as np

X = pd.read_csv('datasets/Asheville/amn_PCAed.csv', header = None)

# # XMeans only gave 2 clusters -- don't use.
# from xmeans import XMeans
# xmeans = XMeans(random_state = 1).fit(X) 


kmeans = KMeans()#n_clusters=3)

searcher = GridSearchCV(kmeans, {'n_clusters': np.arange(1,600)})#, verbose = 100)

searcher.fit(X)

#searcher.cv_results_
searcher.best_params_

#categorization1 = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_
print centroids

print(np.shape(centroids))



