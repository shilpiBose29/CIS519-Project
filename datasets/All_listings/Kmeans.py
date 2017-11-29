#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:20:55 2017

@author: arranzignacio
"""

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np



d = {'col1': [1, 2,4,2,4,5,7], 'col2': [3, 4,5,4,2,7,6], 'col3': [3, 3,2,4,5,2,1]}


X = pd.DataFrame(data=d)




kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
categorization1 = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_
print centroids

print(np.shape(centroids))