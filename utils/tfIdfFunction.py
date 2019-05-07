# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:27:25 2019

@author: Luthfi Dzaky
"""
import pandas as pd
import math
data = pd.read_csv('../dataset/parsed_data.csv')

def tf(word, doc):
    frequency = doc.count(word)
    total = len(doc.split())
    return frequency/total

def idf(list_of_docs):
    dict_idf = {}
    for document in list_of_docs:
        for word in document.split():
            if(word in dict_idf):
                continue
            else:
                count = 0
                for docs in list_of_docs:
                    if word in docs.split():
                        count += 1
                dict_idf[word]=math.log(len(list_of_docs)/ count)
                print('word :'+ word + ' tfidfnya' + dict_idf[word])
    return dict_idf

dict_idf = idf(data['content'])

def tf_idf(word, doc):
    return (tf(word, doc) * dict_idf[word])
