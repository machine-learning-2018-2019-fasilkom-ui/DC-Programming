# -*- coding: utf-8 -*-

import pandas as pd
from naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score
from preprocessor import preprocess

print("Showcasing Multinomial Naive Bayes with 1-gram BoW features")

data_train = pd.read_csv('../dataset/parsed_data.csv')
X_train = data_train.content
y_train = data_train.sentiment

lst = []
for x in X_train:
    lst.append(preprocess(x, gram=3))
X_train = lst

data_test = pd.read_csv('../dataset/test_parsed_data.csv')
X_test = data_test.content
y_test = data_test.sentiment

lst = []
for x in X_test:
    lst.append(preprocess(x, gram=3))
X_test = lst

nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print("Single label accuracy score with project-made model :", accuracy_score(y_test, y_pred))

labels = []
for n in range(len(X_test)):
    items = []
    for item in nb.calc_prob(X_test[n]):
        items.append(item[0])
    labels.append(items)

print("Multi label(s) accuracy score : ")
for x in range(13):
    count = 0
    for y in range(len(y_test)):
        cust_labels = labels[y][0:x+1]
        if y_test[y] in cust_labels:
            count += 1
    score = count/len(y_test)
    print("({}) label(s) : {}".format(x+1, score))
    

"""
Single label accuracy score with project-made model : 0.7153024911032029
Multi label(s) accuracy score : 
(1) label(s) : 0.7153024911032029
(2) label(s) : 0.8078291814946619
(3) label(s) : 0.8347419928825622
(4) label(s) : 0.8939056939501779
(5) label(s) : 0.9370551601423488
(6) label(s) : 0.9481761565836299
(7) label(s) : 0.9532918149466192
(8) label(s) : 0.9657473309608541
(9) label(s) : 0.9837633451957295
(10) label(s) : 0.9919928825622776
(11) label(s) : 0.9991103202846975
(12) label(s) : 1.0
(13) label(s) : 1.0
"""