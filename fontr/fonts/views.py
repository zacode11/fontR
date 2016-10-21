from django.shortcuts import render
from django.http import HttpResponseRedirect

from .models import Document
from .forms import DocumentForm


def index(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()
            # put the call to the blackbox here
            return HttpResponseRedirect('/fonts/result')
    else:
        form = DocumentForm()
    documents = Document.objects.all()
    return render(request, 'fonts/index.html', {'documents': documents, 'form': form},)


def result(request):
    return render(request, 'fonts/result.html')