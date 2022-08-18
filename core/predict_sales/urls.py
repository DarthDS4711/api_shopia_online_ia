from django.urls import path

from .views import PredictedNumberSalesApi

urlpatterns = [
    path('prediction/', PredictedNumberSalesApi.as_view()),
]