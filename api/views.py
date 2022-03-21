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
    data = request.data
    if not data:
        return HttpResponse(f'{data}')
    else:
        return HttpResponse(f'fuck u')


class FileUploadView(views.APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, filename, format=None):
        file_obj = request.data['file']
        with open(filename, 'wb') as f:
            for chunk in file_obj.chunks():
                f.write(chunk)
        return Response(status=204)


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
