from django.shortcuts import render
import json


# Create your views here.
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
