# *_coding=utf-8_*
# yuxi   当前系统用户
# 11/2/22   当前系统日期
# 17:41   当前系统时间
# PyCharm   创建文件的IDE名称

from django.urls import path

from api import views

urlpatterns = [
    path('', views.show),
]