from django.db import models
import datetime
import os
# Create your models here.


def file_path(instance, filename):
    path = 'images/'
    format = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    return os.path.join(path, format)


class Document(models.Model):
    docfile = models.FileField(upload_to=file_path)
