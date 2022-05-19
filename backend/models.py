from django.db import models
from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import PermissionsMixin
from django.utils.translation import gettext_lazy as _

'''
用户鉴权
from django.contrib.auth import authenticate
user = authenticate(username='john', password='secret')

from django.contrib.auth import authenticate, login

user login
https://docs.djangoproject.com/en/4.0/topics/auth/default/#how-to-log-a-user-in
'''

import hashlib


def encrypt_password(password):
    # 使用md5算法对密码进行加密
    md5 = hashlib.md5()
    sign_str = password + '#@%^&*'
    sign_bytes_utf8 = sign_str.encode(encoding='utf-8')

    md5.update(sign_bytes_utf8)
    encrypted_password = md5.hexdigest()

    return encrypted_password


class CustomUserManager(BaseUserManager):
    """custom suer manager"""
    use_in_migration = True

    def _create_user(self, phone_number, passwd, **extra_fields):
        if not phone_number:
            raise ValueError('phone number is essential!')
        user = self.model(phone_number=phone_number, **extra_fields)
        user.set_password(passwd)
        user.save(using=self._db)
        return user

    def create_user(self, phone_number, passwd, **extra_field):
        extra_field.setdefault('is_superuser', False)
        return self._create_user(phone_number, passwd, **extra_field)

    def create_superuser(self, phone_number, password, **extra_fields):
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self._create_user(phone_number, password, **extra_fields)


# Create your models here.
class CustomUser(AbstractBaseUser, PermissionsMixin):
    """自定义用户"""
    phone_number = models.CharField(_('phone number'),
                                    unique=True, max_length=11)  # 手机号
    option = [('男', '男性'), ('女', '女性')]
    gender = models.CharField(choices=option, default='男', max_length=10)  # 性别

    password = models.CharField(max_length=256, default='000000')

    is_superadmin = models.BooleanField(_('is_superadmin'), default=False)
    is_active = models.BooleanField(_('is_active'), default=True)
    is_staff = models.BooleanField(default=True)

    REQUIRED_FIELDS = ['gender']
    USERNAME_FIELD = 'phone_number'

    objects = CustomUserManager()

    class Meta:
        verbose_name = _('user')
        verbose_name_plural = _('users')

    def __str__(self):
        return self.phone_number

    def set_password(self, password):
        self.password = encrypt_password(password)

    def verify_password(self, password):
        return self.password == encrypt_password(password)


class User(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='no_essential',
                                verbose_name='account')
    name = models.CharField(_('user name'), max_length=30)
    avatar = models.FileField(verbose_name='avatar', blank=True, null=True)
    height = models.IntegerField(verbose_name='your height', blank=True)
    weight = models.IntegerField(verbose_name='your weight', blank=True)
    birthday = models.DateField(verbose_name='your birthday', blank=True, null=True)
    idcard_number = models.CharField(max_length=18, verbose_name='ID card', blank=True, null=True)
    hobby_options = [('太极', 'taiji'), ('瑜伽', "yoga")]
    hobbies = models.CharField(choices=hobby_options, verbose_name='your hobbies', max_length=20, blank=True, null=True)

    def __str__(self):
        return self.user.phone_number

    @property
    def phone_number(self):
        return self.phone_number

    @property
    def password(self):
        return self.password

    @property
    def gender(self):
        return self.gender


class Course(models.Model):
    option = [('太极', 'taiji'), ('瑜伽', "yujia")]
    name = models.CharField(max_length=255, choices=option, verbose_name='course name')
    rating = models.IntegerField(verbose_name='course rating')
    assessment = models.FloatField(verbose_name='course assessment')
    during_time = models.IntegerField(verbose_name='during time')
    
    def __name__(self):
        return self.name


class Comment(models.Model):
    context = models.CharField(max_length=255, verbose_name='comment context')
    likes = models.IntegerField(verbose_name='the like')
    following = []


class User2Course(models.Model):
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    course = models.ForeignKey(Course, on_delete=models.DO_NOTHING)
