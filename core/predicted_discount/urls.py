from django.urls import path

from .views import PredictedDiscountProductsApi

urlpatterns = [
    path('prediction-discount/', PredictedDiscountProductsApi.as_view()),
]