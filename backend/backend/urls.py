from django.urls import path, include
from django.contrib.auth.models import User
from django.contrib import admin
from rest_framework import routers, serializers, viewsets


# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("api.urls", namespace="rest_framework")),
]
