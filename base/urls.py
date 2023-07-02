from django.urls import path
from base import views

urlpatterns = [
    path("", views.home, name="home"),
    path("create/", views.CreateEmployeeView.as_view(), name="create"),
    path("token/", views.GetAuthTokenAPIView.as_view(), name="token"),
    path("video/", views.LiveStreamAPIView.as_view(), name="live")
]
# http://1273.0.0.7/camera/create/ [POST]
