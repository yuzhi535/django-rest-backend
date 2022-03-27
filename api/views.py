# *_coding=utf-8_*
# yuxi   当前系统用户
# 11/2/22   当前系统日期
# 17:42   当前系统时间
# PyCharm   创建文件的IDE名称
from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.urls import reverse_lazy
from django.views.generic import FormView
from django.contrib.auth.hashers import make_password
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
    data = request.POST
    passwd = data['passwd']
    phonenumber = data['phonenumber']
    user = authenticate(phone_number=phonenumber, password=passwd)
    if user:
        ret = {'id': user.id, 'status': 200}
        return JsonResponse(ret)
    return JsonResponse({'status': 404})


@api_view(['POST'])
def register(request):
    phone_number = request.POST.get('phone_number')
    # 获取用户第一次输入的密码
    password1 = request.POST.get('password1')
    # 获取用户输入的第二次密码
    password2 = request.POST.get('password2')
    name = request.POST.get('name')
    gender = request.POST.get('gender')
    height = request.POST.get('height')
    weight = request.POST.get('weight')
    avatar = request.FILES.get('avatar')
    birthday = request.POST.get('birthday')
    idcard_number = request.POST.get('idcard_number')
    hobbies = request.POST.get('hobbies')

    user = CustomUser.objects.filter(phone_number=phone_number)
    if user:
        return JsonResponse({'msg': '该用户已注册', 'status': 400})
    else:
        if password1 == password2:
            password = make_password(password1)
            instance1 = CustomUser.objects.create(
                phone_number=phone_number,
                password=password,
                gender=gender,
            )
            instance2 = User.objects.create(
                user=instance1,
                name=name,
                height=height,
                weight=weight,
                birthday=birthday,
                hobbies=hobbies,
                idcard_number=idcard_number,
                avatar=avatar,
            )
            
            # return HttpResponse(content=f'avatar={avatar}fucku')
            
            return JsonResponse(data={
                "phone_number": instance1.phone_number,
                "password": instance1.password,
                "gender": instance1.gender,
                "name": instance2.name,
                "height": instance2.height,
                "weight": instance2.weight,
                "birthday": instance2.birthday,
                "hobbies": instance2.hobbies,
                "idcard_number": instance2.idcard_number,
                "avatar": f'{instance2.avatar}' if instance2.avatar else 'none'
            }, status=200)


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
