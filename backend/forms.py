# *_coding=utf-8_*
# yuxi   当前系统用户
# 14/2/22   当前系统日期
# 23:01   当前系统时间
# PyCharm   创建文件的IDE名称

from django import forms
from django.contrib.auth import get_user_model


class SignupForm(forms.ModelForm):
    """user signup form"""
    password = forms.CharField(widget=forms.PasswordInput())

    class Meta:
        model = get_user_model()
        fields = ('phone_number', 'password',)


class LoginForm(forms.Form):
    """user login form"""
    phonenumber = forms.IntegerField
    password = forms.CharField(widget=forms.PasswordInput())