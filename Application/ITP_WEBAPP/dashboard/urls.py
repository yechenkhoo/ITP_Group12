from django.urls import include, path
from .views import dashboard_results, dashboard_videoFeed, logout, home, dashboard_dataSpace, create_account,live_stream, start_recording, upload_from_pi, admin_model, golf_video_feed, golf_status, golf_start_recording, golf_toggle_auto_recording, golf_toggle_pose_detection, golf_reload_models, golf_health


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

    path("golf/video-feed/", golf_video_feed, name="golf_video_feed"),
    path("golf/status/", golf_status, name="golf_status"),
    path("golf/start-recording/", golf_start_recording, name="golf_start_recording"),
    path("golf/toggle-auto/", golf_toggle_auto_recording, name="golf_toggle_auto"),
    path("golf/toggle-pose/", golf_toggle_pose_detection, name="golf_toggle_pose"),
    path("golf/reload-models/", golf_reload_models, name="golf_reload_models"),
    path("golf/health/", golf_health, name="golf_health"),
]
