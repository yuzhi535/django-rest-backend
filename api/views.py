# *_coding=utf-8_*
# yuxi   当前系统用户
# 11/2/22   当前系统日期
# 17:42   当前系统时间
# PyCharm   创建文件的IDE名称
from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.urls import reverse_lazy
from django.views.generic import FormView
from rest_framework import views
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.decorators import api_view
from rest_framework.parsers import FileUploadParser, MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.authtoken.models import Token

from backend import forms
from backend.models import User, Course, CustomUser
from django.contrib.auth import authenticate, login

from backend.serializers import UserModelSerializer


def show(request):
    return HttpResponse("hello world")


@api_view(['POST'])
def login(request):
    data = request.data
    passwd = data['passwd']
    phonenumber = data['phonenumber']
    user = authenticate(phone_number=phonenumber, password=passwd)
    if user:
        ret = {'id': user.id, 'status': 200}
        return JsonResponse(ret)
    return JsonResponse({'status': 404})


@api_view(['POST'])
def register(request):
    phone_number = request.data.get('phone_number')
    # 获取用户第一次输入的密码
    password1 = request.data.get('password1')
    # 获取用户输入的第二次密码
    password2 = request.data.get('password2')
    name = request.data.get('name')
    gender = request.data.get('gender')
    height = request.data.get('height')
    weight = request.data.get('weight')
    avatar = request.data.get('avatar')
    birthday = request.data.get('birthday')
    idcard_number = request.data.get('idcard_number')
    hobbies = request.data.get('hoobies')

    user = User.objects.filter(phone_number=phone_number)
    if user:
        return Response({'msg': '该用户名存在了', 'code': 400})
    else:
        if password1 == password2:
            user_dict = {'phone_number': phone_number, 'password': password1, 'name': name,
                         'gender': gender, 'height': height, 'weight': weight, 'avatar': avatar,
                         'birthday': birthday, 'idcard_number': idcard_number, 'hobbies': hobbies}
            user_serializer = UserModelSerializer(data=user_dict)
            # 进行数据校验，保存
            if user_serializer.is_valid():
                user_serializer.save()
                return Response({'msg': '注册成功', 'code': 200})
            else:
                return Response({'msg': user_serializer.errors, 'code': 400})


class FileUploadView(views.APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, filename, format=None):
        file_obj = request.data['file']
        with open(filename, 'wb') as f:
            for chunk in file_obj.chunks():
                f.write(chunk)
        return JsonResponse({'status': 204})


class CustomAuthToken(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data,
                                           context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'token': token.key,
            'user_id': user.pk,
        })
