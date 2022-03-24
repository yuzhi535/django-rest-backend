"""Django_rest_backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import api
from django.contrib import admin
from django.urls import path
from api import views

import backend
from backend import views

urlpatterns = [
    path('', api.views.show),
    path('api/login', api.views.login),
    path('api/register', api.views.register),
    path('api/example/login', api.views.CustomAuthToken.as_view())
]

# 自动生成路由信息[和视图集一起使用]
from rest_framework.routers import SimpleRouter

# 1. 实例化路由类
router = SimpleRouter()
# 2. 给路由注册视图集
router.register("demo", backend.views.UserViewSet, basename="demo")
print(router.urls)
# 3. 把生成的路由列表 和 urlpatterns进行拼接
urlpatterns += router.urls

router = SimpleRouter()
router.register("course", views.CourseViewSet, basename="course")
print(router.urls)
urlpatterns += router.urls
