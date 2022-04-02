# *_coding=utf-8_*
# yuxi   当前系统用户
# 11/2/22   当前系统日期
# 17:42   当前系统时间
# PyCharm   创建文件的IDE名称
import os

from django.contrib import messages
from django.db import IntegrityError
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

import paddlehub as hub


def show(request):
    return HttpResponse("hello world")


@api_view(['POST'])
def login(request):
    password = request.POST.get('password')
    phone_number = request.POST.get('phone_number')
    user = authenticate(phone_number=phone_number, password=password)
    if not user:
        try:
            if not CustomUser.objects.get(phone_number=phone_number):
                return JsonResponse({'status': 'A404'})  # 用户不存在
            else:
                return JsonResponse({'status': 'B404'})  # 密码输入错误
        except CustomUser.DoesNotExist:
            return JsonResponse({'status': 'A404'})
    else:
        return JsonResponse({'id': user.id, 'status': 200})


@api_view(['POST'])
def register(request):
    phone_number = request.POST.get('phone_number')
    try:
        password1 = request.POST.get('password1')
        password1 = password_long(password1)
    except Exception as e:
        return HttpResponse(content=f'{format(e)}')
    password2 = request.POST.get('password2')
    name = request.POST.get('name')
    gender = request.POST.get('gender')
    try:
        height = request.POST.get('height')
        height = more_than(height)
    except Exception as e:
        return HttpResponse(content=f'{format(e)}')
    try:
        weight = request.POST.get('weight')
        weight = more_than(weight)
    except Exception as e:
        return HttpResponse(content=f'{format(e)}')
    avatar = request.FILES.get('avatar')
    birthday = request.POST.get('birthday')
    idcard_number = request.POST.get('idcard_number')
    hobbies = request.POST.get('hobbies')

    user = CustomUser.objects.filter(phone_number=phone_number)
    if user:
        return JsonResponse({'status': 'C400'})
    else:
        if password1 == password2:
            password = make_password(password1)
            try:
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
            except(TypeError, ValueError, IntegrityError):
                CustomUser.objects.get(phone_number=phone_number).delete()
                return JsonResponse({'status': 'A400'})
            else:
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
        else:
            return JsonResponse({'status': 'B400'})


# 密码小于6位异常
def password_long(pwd):
    if len(pwd) >= 6:
        return pwd
    else:
        pwd_error = Exception('密码长度不能小于6位')
        raise pwd_error


# 身高体重必须大于0异常
def more_than(hw):
    if hw:
        if int(hw) > 0:
            return hw
        else:
            zero_error = Exception('身高或体重不可以是负数')
            raise zero_error


class FileUploadView(views.APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, filename, format=None):
        file_obj = request.data['file']
        course = request.data.get('course')
        userID = request.data.get('userID')
        assert course is not None  # have to do this!!!
        assert userID is not None

        # @todo 这里根据用户id，每个用户id创建一个目录，并且在用户目录里面根据课程创建一个目录，视频存到课程目录里面。如果考虑次数，则可能需要对视频重命名。
        # 不够先别这么做吧。

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


# @TODO predict and process
class Predict(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = hub.Module(name='openpose_body_estimation')

    def post(self, requests):
        return requests

    def predict(self, img):
        '''
        @ todo just predict and return candidate points
        '''
        pass

    def split_video(self, vid):
        '''
        @ todo split a video to frames. return a list of a quantity of images
        @ note  use opencv
        '''
        pass

    def process(self, ):
        '''
        @ todo we mainly use process
        '''
        pass

    def dis(self, pos1, pos2):
        '''
        @ todo calculate the distance between two points in an image
        '''
        pass

    def body_dis(self, body1, body2):
        '''
        @ todo calculate the total distance between two bodies of a frame
        '''
        pass

    def dynamic_time_warping(self, series1, series2):
        '''
        @ todo DTW
        '''
        pass
