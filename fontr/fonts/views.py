from django.shortcuts import render
from django.http import HttpResponseRedirect

from .models import Document
from .forms import DocumentForm

import datetime

import json

x = ""
def index(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()
            global x
            x = "/media/images/" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            return HttpResponseRedirect('/fonts/result')
    else:
        form = DocumentForm()
    documents = Document.objects.all()
    return render(request, 'fonts/index.html', {'documents': documents, 'form': form},)


def result(request):
    # json data = blackbox(x)

    fonts = []
    for i in xrange(10):
        fonts.append((data["data"][i]["font"],(int)(100*(data["data"][i]["popularity"]))))

    return render(request, 'fonts/result.html', {'fonts':fonts})

