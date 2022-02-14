# *_coding=utf-8_*
# yuxi   当前系统用户
# 11/2/22   当前系统日期
# 17:42   当前系统时间
# PyCharm   创建文件的IDE名称
from django.http import HttpResponse
from rest_framework.decorators import api_view
from backend.models import User, Course
from django.contrib.auth import authenticate, login


def show(request):
    return HttpResponse("hello world")


@api_view((['POST']))
def login(request):
    data = request.data
    phonenumber = data['phonenumber']
    passwd = data['passwd']
    user = authenticate(phonenumber=phonenumber, password=passwd)
    return HttpResponse(f'{user}')
    pass
