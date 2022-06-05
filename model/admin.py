from django.contrib import admin
from model.models import *

class StudentModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'mail', 'image')
    search_fields = ('name',)

# Register your models here.
admin.site.register(StudentModel, StudentModelAdmin)