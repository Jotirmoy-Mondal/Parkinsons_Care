# voice_test/urls.py

from django.urls import path
from voice_test import views

urlpatterns = [
    path("upload/", views.upload_voice_test, name="upload_voice_test"),
]