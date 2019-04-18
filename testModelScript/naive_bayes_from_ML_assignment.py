# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:02:02 2019

@author: IlhamDCP
"""

import pandas as pd
from utils.naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score

# load training data
data = pd.read_csv('../dataset/parsed_data.csv')
X = data.content
y = data.sentiment

# initialize model
naive_bayes = NaiveBayes()
NaiveBayes()
nb = NaiveBayes()
nb.fit(X,y)

# load testing data
data_test = pd.read_csv('../dataset/test_parsed_data.csv')
data_test.head()

X_test = data_test.content
y_test = data_test.sentiment
result = nb.predict(X_test)

# measure accuracy
print(accuracy_score(y_test, result))