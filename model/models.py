from django.db import models

# Create your models here.

class StudentModel(models.Model):
    name = models.CharField(max_length=255)
    # id = models.CharField(max_length=255, primary_key=True)
    mail = models.EmailField(max_length=255)
    image = models.FileField(upload_to='images/student/')

class ProfessorModel(models.Model):
    name = models.CharField(max_length=255)
    # id = models.CharField(max_length=255, primary_key=True)
    mail = models.EmailField(max_length=255)
    image = models.FileField(upload_to='images/professor/')
