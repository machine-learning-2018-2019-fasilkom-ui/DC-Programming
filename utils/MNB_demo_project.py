import pandas as pd
from naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score
from preprocessor import preprocess

data_train = pd.read_csv('../dataset/parsed_data.csv')
X_train = data_train.content
y_train = data_train.sentiment

lst = []
for x in X_train:
    lst.append(preprocess(x))
X_train = lst

data_test = pd.read_csv('../dataset/test_parsed_data.csv')
X_test = data_test.content
y_test = data_test.sentiment

lst = []
for x in X_test:
    lst.append(preprocess(x))
X_test = lst

nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print("Single label accuracy score with project-made model :", accuracy_score(y_test, y_pred))
print("=====================================================================\n")

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

print(len(X_train))
print(len(X_train[0]))
print(X_train[0])
