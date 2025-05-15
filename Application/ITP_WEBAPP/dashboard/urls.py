from django.urls import include, path
from .views import dashboard_results, dashboard_videoFeed, logout, home, dashboard_dataSpace, create_account,live_stream, start_recording, upload_from_pi, admin_model


urlpatterns = [
    path("video_feed/", dashboard_videoFeed, name="dashboard_videoFeed"),
    path("logout/", logout, name="logout"),
    
    path('dataSpace/<str:id>/', dashboard_dataSpace, name='dashboard_dataSpace'),
    path("dataSpace/<str:id>/results/<str:VideoId>/", dashboard_results, name="results"),
    path("create_account/", create_account, name="create_account"),
    path("uploadModel/", admin_model, name="admin_model"),
    

    path("", home, name="home"),

    path("live_stream/", live_stream, name="live_stream"),
    path('start_recording/', start_recording, name='start_recording'),
 
    path('upload_from_pi/', upload_from_pi, name='upload_from_pi'),
]
