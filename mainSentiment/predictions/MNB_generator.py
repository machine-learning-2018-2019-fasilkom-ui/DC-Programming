import pandas as pd
from naive_bayes import NaiveBayes
from preprocessor import preprocess

data_train = pd.read_csv('../../dataset/parsed_data.csv')
X_train = data_train.content
y_train = data_train.sentiment

lst = []
for x in X_train:
    lst.append(preprocess(x, 2))
X_train = lst

nb = NaiveBayes()
nb.fit(X_train, y_train)

nb.export_vars()
