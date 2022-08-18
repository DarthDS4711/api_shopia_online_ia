from django.urls import path

from .views.predicted_discount.views import TestClassApi

urlpatterns = [
    path('api/', TestClassApi.as_view())
]
