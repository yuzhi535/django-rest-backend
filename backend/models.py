from django.db import models
from django.contrib.auth.base_user import AbstractBaseUser

'''
用户鉴权
from django.contrib.auth import authenticate
user = authenticate(username='john', password='secret')

from django.contrib.auth import authenticate, login

user login
https://docs.djangoproject.com/en/4.0/topics/auth/default/#how-to-log-a-user-in
'''


# Create your models here.
class CustomUser(AbstractBaseUser):
    phone_number = models.CharField(max_length=11, verbose_name='your phone number')  # 手机号
    option = [('男', '男性'), ('女', '女性')]
    gender = models.CharField(choices=option, default='男', max_length=10)  # 性别
    REQUIRED_FIELDS = ['sex']

    def __str__(self):
        return self.get_username()


class User(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, verbose_name='account')
    avatar = models.FileField(verbose_name='avatar')
    height = models.IntegerField(verbose_name='your height')
    weight = models.IntegerField(verbose_name='your weight')
    birthday = models.DateField(verbose_name='your birthday')
    idcard_number = models.CharField(max_length=18, verbose_name='ID card', blank=True)
    hobby_options = [('太极', 'taiji'), ('瑜伽', "yoga")]
    hobbies = models.CharField(choices=hobby_options, verbose_name='your hobbies', max_length=20)


class Course(models.Model):
    option = [('太极', 'taiji'), ('瑜伽', "yujia")]
    name = models.CharField(max_length=255, choices=option, verbose_name='course name')
    rating = models.IntegerField(verbose_name='course rating')
    assessment = models.FloatField(verbose_name='course assessment')
    during_time = models.IntegerField(verbose_name='during time')


class Comment(models.Model):
    context = models.CharField(max_length=255, verbose_name='comment context')
    likes = models.IntegerField(verbose_name='the like')
    following = []


class User2Course(models.Model):
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    course = models.ForeignKey(Course, on_delete=models.DO_NOTHING)
