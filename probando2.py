#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 20:46:42 2017

@author: kevin
"""
# Required Python Machine learning Packages
import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.cross_validation import train_test_split
# To model the Gaussian Navie Bayes classifier
from sklearn.naive_bayes import GaussianNB
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score

adult_df = pd.read_csv('prueba.csv',
                       header = None, delimiter=' *, *', engine='python')

#agregar los nombres de las caracteristicas
adult_df.columns = ['movieid','userid','rating',
                    'gender','age','occupation','zipcode',
                    'namewords','namepar','year','action',
                    'adventure','animation','childrens',
                    'comedy','crime','documentary','drama',
                    'fantasy','filmnoir','horror','musical',
                    'mystery','romance',
                    'scifi','thriller','war','western']

adult_df_rev = adult_df

#estadisticas de los datos
#print(adult_df_rev.describe(include= 'all'))

for value in ['movieid','userid','rating',
              'gender','age','occupation','zipcode',
              'namewords','namepar','year','action',
              'adventure','animation','childrens',
              'comedy','crime','documentary','drama',
              'fantasy','filmnoir','horror','musical',
              'mystery','romance',
              'scifi','thriller','war','western']:
    adult_df_rev[value].replace(['?'], [adult_df_rev.describe(include='all')[value][2]],
                                inplace='True')
print(adult_df_rev)