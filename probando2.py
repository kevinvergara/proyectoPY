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
from sklearn.preprocessing import StandardScaler
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
    
#estadisticas basicas de cada caracteristica
#print(adult_df_rev.describe(include= 'all'))

le = preprocessing.LabelEncoder()
movieid_cat = le.fit_transform(adult_df.movieid)
userid_cat = le.fit_transform(adult_df.userid)
rating_cat = le.fit_transform(adult_df.rating)
gender_cat = le.fit_transform(adult_df.gender)
age_cat = le.fit_transform(adult_df.age)
occupation_cat = le.fit_transform(adult_df.occupation)
zipcode_cat = le.fit_transform(adult_df.zipcode)
namewords_cat = le.fit_transform(adult_df.namewords)
namepar_cat = le.fit_transform(adult_df.namepar)
year_cat = le.fit_transform(adult_df.year)
action_cat = le.fit_transform(adult_df.action)
adventure_cat = le.fit_transform(adult_df.adventure)
animation_cat = le.fit_transform(adult_df.animation)
childrens_cat = le.fit_transform(adult_df.childrens)
comedy_cat = le.fit_transform(adult_df.comedy)
crime_cat = le.fit_transform(adult_df.crime)
documentary_cat = le.fit_transform(adult_df.documentary)
drama_cat = le.fit_transform(adult_df.drama)
fantasy_cat = le.fit_transform(adult_df.fantasy)
filmnoir_cat = le.fit_transform(adult_df.filmnoir)
horror_cat = le.fit_transform(adult_df.horror)
musical_cat = le.fit_transform(adult_df.musical)
mystery_cat = le.fit_transform(adult_df.mystery)
romance_cat = le.fit_transform(adult_df.romance)
scifi_cat = le.fit_transform(adult_df.scifi)
thriller_cat = le.fit_transform(adult_df.thriller)
war_cat = le.fit_transform(adult_df.war)
western_cat = le.fit_transform(adult_df.western)

adult_df_rev['movieid_cat'] = movieid_cat
adult_df_rev['userid_cat'] = userid_cat
adult_df_rev['rating_cat'] = rating_cat
adult_df_rev['gender_cat'] = gender_cat
adult_df_rev['age_cat'] = age_cat
adult_df_rev['occupation_cat'] = occupation_cat
adult_df_rev['zipcode_cat'] = zipcode_cat
adult_df_rev['namewords_cat'] = namewords_cat
adult_df_rev['namepar_cat'] = namepar_cat
adult_df_rev['year_cat'] = year_cat
adult_df_rev['action_cat'] = action_cat
adult_df_rev['adventure_cat'] = adventure_cat
adult_df_rev['animation_cat'] = animation_cat
adult_df_rev['childrens_cat'] = childrens_cat
adult_df_rev['comedy_cat'] = comedy_cat
adult_df_rev['crime_cat'] = crime_cat
adult_df_rev['documentary_cat'] = documentary_cat
adult_df_rev['drama_cat'] = drama_cat
adult_df_rev['fantasy_cat'] = fantasy_cat
adult_df_rev['filmnoir_cat'] = filmnoir_cat
adult_df_rev['horror_cat'] = horror_cat
adult_df_rev['musical_cat'] = musical_cat
adult_df_rev['mystery_cat'] = mystery_cat
adult_df_rev['romance_cat'] = romance_cat
adult_df_rev['scifi_cat'] = scifi_cat
adult_df_rev['thriller_cat'] = thriller_cat
adult_df_rev['war_cat'] = war_cat
adult_df_rev['western_cat'] = western_cat

#Eliminar las columnas categóricas antiguas del marco de datos
dummy_fields = ['movieid','userid','rating',
              'gender','age','occupation','zipcode',
              'namewords','namepar','year','action',
              'adventure','animation','childrens',
              'comedy','crime','documentary','drama',
              'fantasy','filmnoir','horror','musical',
              'mystery','romance',
              'scifi','thriller','war','western']
adult_df_rev = adult_df_rev.drop(dummy_fields, axis = 1)
'''
adult_df_rev = adult_df_rev.reindex_axis(['movieid','userid','rating',
              'gender','age','occupation','zipcode',
              'namewords','namepar','year','action',
              'adventure','animation','childrens',
              'comedy','crime','documentary','drama',
              'fantasy','filmnoir','horror','musical',
              'mystery','romance',
              'scifi','thriller','war','western'], axis= 1)
'''
#print(adult_df_rev)

num_features = ['movieid_cat','userid_cat','rating_cat',
                'gender_cat','age_cat','occupation_cat',
                'zipcode_cat','namewords_cat','namepar_cat',
                'year_cat','action_cat','adventure_cat',
                'animation_cat','childrens_cat','comedy_cat',
                'crime_cat','documentary_cat','drama_cat',
                'fantasy_cat','filmnoir_cat','horror_cat',
                'musical_cat','mystery_cat','romance_cat',
                'scifi_cat','thriller_cat','war_cat','western_cat']

scaled_features = {}

for each in num_features:
    mean, std = adult_df_rev[each].mean(), adult_df_rev[each].std()
    scaled_features[each] = [mean, std]
    adult_df_rev.loc[:, each] = (adult_df_rev[each] - mean)/std


features = adult_df_rev.values[:,:27]
target = adult_df_rev.values[:,27]
features_train, features_test, target_train, target_test = train_test_split(features,
                                                                            target, test_size = 0.33, random_state = 10)
x_scalar = StandardScaler()
features_train = x_scalar.transform(features_train)
target_test = x_scalar.transform(features_test)


clf = GaussianNB()
clf.fit(features_train, target_test)
target_pred = clf.predict(features_test)

print(target_pred)

