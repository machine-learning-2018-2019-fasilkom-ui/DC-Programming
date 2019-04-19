import pandas as pd
from naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score
from preprocessor import preprocess

print("Showcasing Multinomial Naive Bayes with 2-gram BoW features")

data_train = pd.read_csv('../dataset/parsed_data.csv')
X_train = data_train.content
y_train = data_train.sentiment

lst = []
for x in X_train:
    lst.append(preprocess(x, 2))
X_train = lst

data_test = pd.read_csv('../dataset/test_parsed_data.csv')
X_test = data_test.content
y_test = data_test.sentiment

lst = []
for x in X_test:
    lst.append(preprocess(x, 2))
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

""" Output
Showcasing Multinomial Naive Bayes with 2-gram BoW features
Single label accuracy score with project-made model : 0.7513345195729537
Multi label(s) accuracy score :
(1) label(s) : 0.7513345195729537
(2) label(s) : 0.8358540925266904
(3) label(s) : 0.8692170818505338
(4) label(s) : 0.9145907473309609
(5) label(s) : 0.9432829181494662
(6) label(s) : 0.958185053380783
(7) label(s) : 0.9673042704626335
(8) label(s) : 0.9753113879003559
(9) label(s) : 0.9868772241992882
(10) label(s) : 0.994661921708185
(11) label(s) : 0.9993327402135231
(12) label(s) : 1.0
(13) label(s) : 1.0
"""