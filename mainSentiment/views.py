from django.http import HttpResponseRedirect
from django.shortcuts import render

# Create your views here.
def index(request):
    response = {}
    if request.method == 'POST':
        return HttpResponseRedirect('/')
    return render(request, 'index.html', response)
