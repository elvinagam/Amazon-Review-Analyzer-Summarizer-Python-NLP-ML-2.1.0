from django.urls import path, include
from . import views
from . views import (home,about)
from django.conf.urls import url

urlpatterns = [
    path('',views.home, name='user-home'),
    path('about/',views.about, name='user-about'),
    # path('result/',views.analyze, name='user-result'),

]
