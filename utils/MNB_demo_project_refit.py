import pandas as pd
from naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score
from preprocessor import preprocess

print("Showcasing Multinomial Naive Bayes with Re-fit procedure")

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

split_num = len(X_train)//3

nb = NaiveBayes()
nb.fit(X_train[:split_num], y_train[:split_num])
y_pred = nb.predict(X_test)
print("Single label accuracy score with 1/3 train set :", accuracy_score(y_test, y_pred))

# =================================================================

nb.refit(list(X_train[split_num:]), list(y_train[split_num:]))
y_pred = nb.predict(X_test)
print("Single label accuracy score with additional 2/3 train set :", accuracy_score(y_test, y_pred))

""" Output
Single label accuracy score with 1/3 train set : 0.3398576512455516
Single label accuracy score with additional 2/3 train set : 0.43994661921708184
"""