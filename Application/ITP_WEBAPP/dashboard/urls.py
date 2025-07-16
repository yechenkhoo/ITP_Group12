from django.urls import include, path
from .views import dashboard_results, dashboard_videoFeed, logout, home, dashboard_dataSpace, create_account,live_stream, start_recording, upload_from_pi, admin_model, golf_video_feed, golf_status, golf_start_recording, golf_toggle_auto_recording, golf_toggle_pose_detection, golf_reload_models, golf_health, golf_set_user_context, api_my_students

from .views import (
    dashboard_results, dashboard_videoFeed, logout, home, dashboard_dataSpace,
    create_account, live_stream, start_recording, upload_from_pi, admin_model,
    dashboard_compareSwings, add_video_comment_ajax, update_comment_position_ajax,
    delete_video_comment_ajax,
    edit_video_comment_ajax,
    add_video_reply_ajax,
    edit_video_reply_ajax,
    delete_video_reply_ajax,
    mark_comment_read,
    check_video_status_ajax
)

urlpatterns = [
    path("video_feed/", dashboard_videoFeed, name="dashboard_videoFeed"),
    path("logout/", logout, name="logout"),

    path('dataSpace/<str:id>/', dashboard_dataSpace, name='dashboard_dataSpace'),
    path("dataSpace/<str:id>/results/<str:VideoId>/", dashboard_results, name="results"),
    path("create_account/", create_account, name="create_account"),
    path("uploadModel/", admin_model, name="admin_model"),
    path('dataSpace/<str:id>/compare-swings/', dashboard_compareSwings, name='dashboard_compareSwings'),

    # AJAX endpoints for comments
    path('dataSpace/<str:id>/results/<str:VideoId>/add_comment_ajax/', add_video_comment_ajax, name='add_video_comment_ajax'),
    path('dataSpace/<str:id>/results/<str:VideoId>/update_comment_position_ajax/', update_comment_position_ajax, name='update_comment_position_ajax'),
    path('dataSpace/<str:id>/results/<str:VideoId>/delete_comment_ajax/', delete_video_comment_ajax, name='delete_video_comment_ajax'),
    path('dataSpace/<str:id>/results/<str:VideoId>/edit_video_comment_ajax/', edit_video_comment_ajax, name='edit_video_comment_ajax'),
    path('dataSpace/<str:id>/results/<str:VideoId>/add_reply_ajax/', add_video_reply_ajax, name='add_video_reply_ajax'),
    path('dataSpace/<str:id>/results/<str:VideoId>/edit_reply_ajax/', edit_video_reply_ajax, name='edit_video_reply_ajax'),
    path('dataSpace/<str:id>/results/<str:VideoId>/delete_reply_ajax/', delete_video_reply_ajax, name='delete_video_reply_ajax'),
    path('dataSpace/<str:id>/results/<str:VideoId>/mark_read_ajax/', mark_comment_read, name='mark_comment_read'),

    # New AJAX endpoint for video statuses
    path('dataSpace/<str:id>/check_video_status_ajax/<str:video_id>/', check_video_status_ajax, name='check_video_status_ajax'),

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
    path('golf/set-user-context/', golf_set_user_context, name='golf_set_user_context'),
    path('api/my-students/', api_my_students, name='api_my_students'),
]