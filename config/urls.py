from django.contrib import admin
from django.urls import path, include
from base.views import home
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
                  path("", home, name="home"),
                  path('admin/', admin.site.urls),
                  path("cam/", include("base.urls"))
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
