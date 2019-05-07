import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Loading data
data = pd.read_csv('../dataset/parsed_data.csv')
factor = pd.factorize(data['sentiment'])
definition = factor[1]
data.sentiment = factor[0]

from Sastrawi.StopWordRemover.StopWordRemoverFactory import  StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
stem = StemmerFactory()
stemmer = stem.create_stemmer()

def prep(text):
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return " ".join(text.split())

import re
def pre_processing(tweet_list):
    data_clean = []
    for tw in tweet_list:
        clean_str = tw.lower() #lowercase
        clean_str = re.sub(r"(?:\@|https?\://)\S+", " ", clean_str)
        #eliminating username and url
        clean_str = re.sub(r'[^\w\s]',' ',clean_str) #removing punctuation
        rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE) #regex for repeating characters
        clean_str = re.sub('\s+', ' ', clean_str) # removing extra space
        clean_str = clean_str.strip() #trimming
        clean_str = prep(clean_str) #removing stowords and stemming
        data_clean.append(clean_str)
    return data_clean

raw = data['content']
clean = pre_processing(raw)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
#splitting data
train_x, test_x, train_y, test_y = train_test_split(clean, data['sentiment'], test_size = 0.30, random_state = 7)
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer='word', ngram_range=(2,2))
count_vect.fit(train_x)
train_x_feat = count_vect.transform(train_x)
test_x_feat = count_vect.transform(test_x)

from sklearn import naive_bayes
classifier = naive_bayes.MultinomialNB()
classifier.fit(train_x_feat, train_y)
pred_y = classifier.predict(test_x_feat)
reversefactor = dict(zip(range(3), definition))
test_y = np.vectorize(reversefactor.get)(test_y)
pred_y = np.vectorize(reversefactor.get)(pred_y)
accuracy = accuracy_score(test_y, pred_y)
print(accuracy)