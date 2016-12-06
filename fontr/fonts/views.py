import json
from django.shortcuts import render
from django.http import HttpResponseRedirect

from .models import Document
from .forms import DocumentForm

import datetime

import json

# from svm import SVM
# import Multiclass_Char import classify_text

import sys
sys.path.insert(0, 'fonts/')
import black_box as bb

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
    with open('output.txt', 'w') as f:
        json.dump(data, f, ensure_ascii=False)

    return render(request, 'fonts/result.html')   # this may be incorrect because we no longer pass an HTTPResponse Object,
  # look here if things are on fire

