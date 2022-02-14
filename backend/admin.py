from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from backend.models import CustomUser, User, Course, User2Course

#
# class UserModel(UserAdmin):
#     model = CustomUser
#
#
# # Register your models here.
# admin.site.register(CustomUser, UserModel)


admin.site.register(User)
admin.site.register(Course)
admin.site.register(User2Course)
