import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from preprocessor import preprocess

# Loading data
data = pd.read_csv('../dataset/parsed_data.csv')
X = data.content
X = [" ".join(preprocess(x)) for x in X]

y = data.sentiment

# Transforming dataset
vectorizer = CountVectorizer()
X_train_tfidf = vectorizer.fit_transform(X)
naive_bayes_library = MultinomialNB()
naive_bayes_library.fit(X_train_tfidf, y)

data_test = pd.read_csv('../dataset/test_parsed_data.csv')
data_test.head()

X_test = data_test.content
X_test = [" ".join(preprocess(x)) for x in X_test]
y_test = data_test.sentiment

X_test_tfidf = vectorizer.transform(X_test)
result_library = naive_bayes_library.predict(X_test_tfidf)

print("Single label accuracy score with library:", accuracy_score(y_test, result_library))

""" Output
Single label accuracy score with library: 0.4415035587188612
"""
