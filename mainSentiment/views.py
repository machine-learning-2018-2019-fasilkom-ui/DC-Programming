from django.shortcuts import render
from predictions import naive_bayes, preprocessor
import json

def check():
    nb = naive_bayes.NaiveBayes()
    nb.import_vars()
    tes = "why is everyones tweets about britains got talent?! i feel left out"
    tes = " ".join(preprocessor.preprocess(tes,2))
    print(tes)
    print(nb.calc_prob(tes))

def index(request):
    response = {}
    if request.method == 'POST':
        response['chart_data'] = json.dumps([
            {'y': 9, 'label': "sleep time"},
            {'y': 30, 'label': "work time"},
        ])
        return render(request, 'index.html', response)
    else:
        return render(request, 'index.html', {'chart_data': []})

check()