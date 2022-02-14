# *_coding=utf-8_*
# yuxi   当前系统用户
# 11/2/22   当前系统日期
# 17:42   当前系统时间
# PyCharm   创建文件的IDE名称
from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse_lazy
from django.views.generic import FormView
from rest_framework.decorators import api_view

from backend import forms
from backend.models import User, Course
from django.contrib.auth import authenticate, login


def show(request):
    return HttpResponse("hello world")


@api_view(['POST'])
def login(request):
    data = request.data
    passwd = data['passwd']
    phonenumber = data['phonenumber']
    user = authenticate(phone_number=phonenumber, password=passwd)
    if user:
        return HttpResponse(f'{user.id}')
    return HttpResponse(f'{user}')
