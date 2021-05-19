from django.urls import path
from .views import HateSpeechApi

app_name = "hatespeechapi"

urlpatterns = [
    path('hatespeech/', HateSpeechApi.as_view(), name="hatespeech")
]
