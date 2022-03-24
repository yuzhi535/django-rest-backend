from django.shortcuts import render

# Create your views here.
from rest_framework.viewsets import ModelViewSet

from .models import User, Course
from .serializers import UserModelSerializer, CourseSerializer


class UserViewSet(ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserModelSerializer


class CourseViewSet(ModelViewSet):
    queryset = Course.objects.all()
    serializer_class = CourseSerializer
