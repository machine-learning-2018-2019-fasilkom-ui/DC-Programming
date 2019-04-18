import numpy as np
from sklearn.naive_bayes import MultinomialNB
def naiveBayes(X,y,input):
    clf = MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf.fit(X_train, y_train)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    return clf.predict(input)
