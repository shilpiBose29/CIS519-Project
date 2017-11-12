#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:51:52 2017

@author: lmy
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd

#%% my print function

import datetime
from pprint import pformat as pf
dt = lambda: datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
filename = datetime.datetime.now().strftime("Output %Y-%m-%d %H-%M-%S.txt")
def p(*argv):
    if not verbose: return
    string = dt()
    for i in argv:
        if type(i) == str:
            string += ' '+i
        else:
            string += ' '+pf(i)
    with open(filename, 'a') as f:
        print(string, file = f)
    print(string)


#%% Configurations

use_mini = False
verbose  = True
random_state = 42 # Using 42, because it's the Answer to the Ultimate Question of Life, the Universe, and Everything.
csv_file = 'datasets/Asheville/Asheville-processed2.csv'


#%% Import data:

p('Reading dataset from', csv_file, '...')
datasets = pd.read_csv(csv_file).groupby('city')

for city_name, indices in datasets.groups.items():
    p('\t', len(indices), '\tlistings in', city_name, '.')

#%%
    
data_Asheville = datasets.get_group('Asheville')
