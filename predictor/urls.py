from django.urls import path
from .views import predict_crop,predict_fertilizer

urlpatterns = [
    path('crop-predict/', predict_crop, name='predict_crop'),
    path('fert-predict/', predict_fertilizer, name='predict_fertilizer'),

]
