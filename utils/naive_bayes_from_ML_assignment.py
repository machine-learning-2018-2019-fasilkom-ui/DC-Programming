# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:02:02 2019

@author: IlhamDCP
"""

import pandas as pd
from naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# import numpy as np
# load training data
data = pd.read_csv('../dataset/parsed_data.csv')
X = data.content
y = data.sentiment

# initialize model
naive_bayes = NaiveBayes()
NaiveBayes()
nb = NaiveBayes()
nb.fit(X,y)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X)
nb_library = MultinomialNB()
nb_library.fit(X_train_tfidf, y)

# load testing data
data_test = pd.read_csv('../dataset/test_parsed_data.csv')
data_test.head()

X_test = data_test.content
y_test = data_test.sentiment
result = nb.predict(X_test)

X_test_tfidf = vectorizer.transform(X_test)
result_library = nb_library.predict(X_test_tfidf)

# measure accuracy
print("Single label accuracy score :",accuracy_score(y_test, result))
print("Single label accuracy score with library:",accuracy_score(y_test,result_library))

labels = []
for n in range(len(X_test)):
    items = []
    for item in nb.calc_prob(X_test[n]):
        items.append(item[0])
    labels.append(items)

for x in range(13):
    count = 0
    for y in range(len(y_test)):
        cust_labels = labels[y][0:x+1]
        if y_test[y] in cust_labels:
            count += 1
    score = count/len(y_test)
    print("Multi labels ({}) accuracy score : {}".format(x+1, score))

#T adalah testnya
#test=vectorizer.transform(T)
#probs = naive_bayes_library.predict_proba(test)
#best_n = np.argsort(probs, axis=1)[:,-3:]
#for a in range(best_n.size):
#    print(cust_labels[0][a-1])


# result
"""
Single label accuracy score : 0.2277580071174377
Multi labels (1) accuracy score : 0.2277580071174377
Multi labels (2) accuracy score : 0.385008896797153
Multi labels (3) accuracy score : 0.5269128113879004
Multi labels (4) accuracy score : 0.6383451957295374
Multi labels (5) accuracy score : 0.7333185053380783
Multi labels (6) accuracy score : 0.8040480427046264
Multi labels (7) accuracy score : 0.8580960854092526
Multi labels (8) accuracy score : 0.9036921708185054
Multi labels (9) accuracy score : 0.9403914590747331
Multi labels (10) accuracy score : 0.9715302491103203
Multi labels (11) accuracy score : 0.9933274021352313
Multi labels (12) accuracy score : 0.9986654804270463
Multi labels (13) accuracy score : 1.0
"""
