from django.shortcuts import render
from mainSentiment.predictions import naive_bayes, preprocessor
import json
import math

def generate_prob(sentence):
    nb = naive_bayes.NaiveBayes()
    nb.import_vars()
    sentence = preprocessor.preprocess(sentence,2)
    pred = nb.calc_prob(sentence)
    result = []
    sum_lst = [x[1] for x in pred]
    sum_val = sum(sum_lst)
    for res in pred:
        dct = {}
        dct['y'] = res[1] * 100 / sum_val
        dct['label'] = res[0]
        result.append(dct)
    print(result)
    return result

def index(request):
    response = {}
    if request.method == 'POST':
        message_data = request.POST['text-sentiment']
        response['chart_data'] = json.dumps(generate_prob(message_data))
        return render(request, 'index.html', response)
    else:
        return render(request, 'index.html', {'chart_data': []})
