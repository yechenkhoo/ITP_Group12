"""
URL configuration for ITP_WEBAPP project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from .views import login, register, forgot_pass
from django.views.generic.base import RedirectView
from django.contrib.staticfiles.storage import staticfiles_storage
from django.conf.urls import handler404, handler500, handler403, handler400
from ITP_WEBAPP.views import error_404, error_500, error_403, error_400
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

handler404 = error_404
handler500 = error_500
handler403 = error_403
handler400 = error_400

urlpatterns = [
    path('admin/', admin.site.urls),
    path("__reload__/", include("django_browser_reload.urls")),
    
    path("", login, name="login"),
    path("register/", register, name="register"),
    path("forgot_password/", forgot_pass, name="forgot_password"),
    
    path("home/", include("dashboard.urls")),
    
    path('favicon.ico', RedirectView.as_view(url=staticfiles_storage.url('images/favicon.ico'))),
]

if not settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)