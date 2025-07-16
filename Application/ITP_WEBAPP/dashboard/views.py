from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect, JsonResponse, StreamingHttpResponse, HttpResponse
from django.http import HttpResponseRedirect, JsonResponse, StreamingHttpResponse, HttpResponse
from ITP_WEBAPP.models import User
from .models import Coach, Video, Comment
from ITP_WEBAPP.views import is_logged_in
from bson import ObjectId
import requests
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import os
from datetime import datetime
import json
from django.utils import timezone
import pandas as pd
from io import StringIO

# Helper Functions
def isCoach(request):
    return request.session.get('Role') in ['coach']

def isStudent(request):
    return request.session.get('Role') in ['student']

def isAdmin(request):
    return request.session.get('Role') in ['admin']

def fetch_all_students(coach_id):
    return Coach.fetch_all_students(coach_id)

# =============================================================================
# 🏌️ CAMERA CONFIGURATION
# =============================================================================
# Define the Raspberry Pi URLs
RASPBERRY_PI_URL = 'http://172.20.10.5:5000'  # Old camera system
GOLF_CAMERA_URL = 'http://172.20.10.5:5000'     # New golf camera system

# =============================================================================
# 📱 MAIN DASHBOARD VIEWS
# =============================================================================

def home(request):
    if not is_logged_in(request):
        return redirect('login')

    if isCoach(request):
        return dashboard_Coach(request)
    if isStudent(request):
        user_id = request.session.get('Id')
        return redirect(reverse('dashboard_dataSpace', kwargs={'id': user_id}))
    if isAdmin(request):
        return dashboard_admin(request)
    return redirect('home')

def dashboard_dataSpace(request, id):
    if not is_logged_in(request):
        return redirect('login')

    if isCoach(request) and not Coach.verify_coach_student_relationship(request.session['Id'], id):
        return redirect('home')

    view = request.GET.get('view', 'list')
    sort = request.GET.get('sort', 'earliest')

    user = User.find_user_by_id(ObjectId(request.session['Id']))
    student = user if not isCoach(request) else User.find_user_by_id(ObjectId(id))

    student_id = student['_id']
    video_list = Video.get_all_videos(student_id)
    user_id_str = str(user['_id'])

    def parse_date(s):
        # matches "HH:MM Mon DD, YYYY"
        return datetime.strptime(s, "%H:%M %b %d, %Y")

    if sort == 'az':
        video_list.sort(key=lambda v: v['Title'].lower())
    elif sort == 'za':
        video_list.sort(key=lambda v: v['Title'].lower(), reverse=True)
    elif sort == 'latest':
        video_list.sort(key=lambda v: parse_date(v['DateUploaded']), reverse=True)
    else:
        video_list.sort(key=lambda v: parse_date(v['DateUploaded']))

    if request.method == 'POST':
        upload_video(request, student_id)
        base = reverse('dashboard_dataSpace', args=[id])
        query = f'?view={view}&sort={sort}'
        query = f'?view={view}&sort={sort}'
        return HttpResponseRedirect(base + query)

    video_processing = [video for video in video_list if video.get('Status') == 'Processing']
    video_completed = [video for video in video_list if video.get('Status') == 'Completed']

    # Paginate each list
    per_page = 4 if view == 'list' else 9

    videos_page = Paginator(video_list, per_page).get_page(page_num)
    completed_page = Paginator(video_completed, per_page).get_page(page_num)
    processing_page = Paginator(video_processing, per_page).get_page(page_num)

    page_map = {
        'tab1': videos_page,
        'tab2': completed_page,
        'tab3': processing_page,
    }
    page_obj = page_map.get(tab, videos_page)

    return render(request, 'dashboard_dataSpace.html', {
        'Role': user['Role'],
        'Name': user['Name'],
        'user_id': user_id_str,
        'studentID': student_id,
        'studentName': student['Name'],
        'videos': video_list,
        'processing_video': video_processing,
        'completed_video': video_completed,
        'videos': video_list,
        'processing_video': video_processing,
        'completed_video': video_completed,
        'sort': sort,
        'view': view,
    })

def dashboard_videoFeed(request):
    if not is_logged_in(request):
        return redirect('login')
    
    user = User.find_user_by_id(ObjectId(request.session['Id']))
    user_id_str = str(user['_id'])

    return render(request, 'dashboard_videoFeed.html', {
        'Role': user['Role'], 
        'Name': user['Name'],
        'user_id': user_id_str
    })

def dashboard_results(request, id, VideoId):
    if not is_logged_in(request):
        return redirect('login')

    if isCoach(request) and not Coach.verify_coach_student_relationship(request.session['Id'], id):
        return redirect('home')

    user = User.find_user_by_id(ObjectId(request.session['Id']))
    student = user if not isCoach(request) else User.find_user_by_id(ObjectId(id))
    
    video_url = Video.get_video_url(VideoId)
    csv_url = Video.get_csv_url(VideoId)
    
    # Fetch and process the CSV
    response = requests.get(csv_url)
    column_status_mapping = {}

    if response.status_code == 200:
        csv_data = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))
        all_columns = df.columns.tolist()
        
        # Select only the first 3 columns for display
        display_columns = all_columns[:3]
        
        # All the csv data
        full_data = df.to_dict('records')
        
        for column in all_columns:
            if "Status" in column:
                corresponding_column = column.replace(" Status", "")
                column_status_mapping[corresponding_column] = column
    else:
        display_columns = []
        full_data = []

    if request.method == 'POST':
        feedback = request.POST['feedback']
        Comment.add_comment(request.session['Id'], VideoId, feedback)
        return HttpResponseRedirect(reverse('results', args=[id, VideoId]))

    # Fetch video comments
    comments = Video.get_all_video_comments(VideoId)
    return render(request, 'dashboard_results.html', {
        'Role': user['Role'],
        'Name': user['Name'],
        'studentID': student['_id'],
        'studentID': student['_id'],
        'videoId': VideoId,
        'comments': comments,
        'video_url': video_url,
        'columns': display_columns,
        'full_data': full_data,
        'column_status_mapping': column_status_mapping,
    })

def dashboard_Coach(request):
    """Displays the Coach dashboard with associated students."""
    if not is_logged_in(request):
        return redirect('login')

    if not isCoach(request):
        return redirect('home')

    view = request.GET.get('view', 'list')
    user = User.find_user_by_id(ObjectId(request.session['Id']))
    students = fetch_all_students(request.session['Id'])

    if request.method == 'POST':
        upload_video(request)
        return HttpResponseRedirect(reverse('home'))

    return render(request, 'dashboard_coach.html', {
        'Role': user['Role'],
        'Name': user['Name'],
        'students': students,
        'view': view,
    })

def dashboard_admin(request):
    """Displays the Admin dashboard."""
    if not is_logged_in(request):
        return redirect('login')

    if not isAdmin(request):
        return redirect('home')

    user = User.find_user_by_id(ObjectId(request.session['Id']))

    if request.method == 'POST':
        create_account(request)
        return HttpResponseRedirect(reverse('home'))

    return render(request, 'dashboard_admin.html', {
        'Role': user['Role'],
        'Name': user['Name'],
    })

def admin_model(request):
    """Displays the model upload dashboard."""
    if not is_logged_in(request):
        return redirect('login')

    if not isAdmin(request):
        return redirect('home')

    user = User.find_user_by_id(ObjectId(request.session['Id']))

    if request.method == 'POST':
        return HttpResponseRedirect(reverse('home'))

    return render(request, 'dashboard_model.html', {
        'Role': user['Role'],
        'Name': user['Name'],
    })

def create_account(request):
    """Handles account creation."""
    if not is_logged_in(request):
        return redirect('login')

    # Check if the user is not a Coach or Admin and redirect to 'home' if neither
    if not (isCoach(request) or isAdmin(request)):
        return redirect('home')

    # Render specific dashboard pages based on role and request method
    if request.method != 'POST':
        if isCoach(request):
            return render(request, 'dashboard_coach.html')
        elif isAdmin(request):
            return render(request, 'dashboard_admin.html')

    # Handle account creation logic
    email = request.POST.get('email')
    password = request.POST.get('password')
    name = request.POST.get('name')
    session_id = request.session.get('Id')

    if isCoach(request):
        Coach.create_user(email, password, "student", name, session_id)
        Coach.update_student_array(email, session_id)
        view = request.GET.get('view', 'list')
        return HttpResponseRedirect(f"{reverse('home')}?view={view}")

    if isAdmin(request):
        Coach.create_user(email, password, "coach", name, session_id)
        return HttpResponseRedirect(reverse('home'))

    return redirect('home')

def upload_video(request, student_id=None):
    """Handles video upload."""
    user_role = request.session.get('Role')
    if user_role not in ['student', 'coach']:
        return redirect('home')

    if user_role == 'student':
        video_file = request.FILES.get('videoDBFile')
        video_type = request.POST.get('videoType')
        if video_file and video_type:
            Video.upload_video(request.session['Id'], request.session['Id'], video_file.name, video_type, video_file)
        return redirect('home')

    if user_role == 'coach':
        video_file = request.FILES.get('videoDBFile')
        video_type = request.POST.get('videoType')
        if '/home/dataSpace/' in request.path and video_file and video_type:
            Video.upload_video(request.session['Id'], student_id, video_file.name, video_type, video_file)
            return redirect('home')

        student_id = request.POST.get('student_id')
        video_name = request.POST.get('fileValue')
        video_type = request.POST.get('videoType')
        video_file = request.FILES.get('videoFile')
        if student_id and video_name and video_type and video_file:
            Video.upload_video(request.session['Id'], student_id, video_name, video_type, video_file)
        return redirect('home')

def logout(request):
    """Logs out the user and clears session data."""
    request.session.flush()
    return redirect('login')

# =============================================================================
# OLD CAMERA SYSTEM (Keep for backward compatibility)
# =============================================================================

def live_stream(request):
    """Streaming response for old camera system live feed"""
    try:
        response = requests.get(f'{GOLF_CAMERA_URL}/start_live_cam', stream=True)
        if response.status_code == 200:
            return StreamingHttpResponse(response.iter_content(chunk_size=1024),
                                         content_type='multipart/x-mixed-replace; boundary=frame')
        else:
            return HttpResponse("Failed to connect to Raspberry Pi camera.", status=500)
    except requests.exceptions.RequestException as e:
        return HttpResponse(f"Error: {e}", status=500)

@csrf_exempt
def start_recording(request):
    """Start recording on the old camera system"""
    try:
        response = requests.post(f'{GOLF_CAMERA_URL}/start_recording')
        return JsonResponse(response.json() if response.ok else {"message": "Failed to start recording"}, status=response.status_code)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"message": f"Error: {str(e)}"}, status=500)

@csrf_exempt
def upload_from_pi(request):
    """Handle video upload from old camera system (backward compatibility)"""
    if request.method == 'POST':
        # Directory where videos are saved
        save_directory = os.path.join(settings.BASE_DIR, "dashboard/pi_video")
        os.makedirs(save_directory, exist_ok=True)
        
        base_filename = "video"
        extension = ".mp4"
        counter = 1

        # Generate the next available filename
        while os.path.exists(os.path.join(save_directory, f"{base_filename}{counter}{extension}")):
            counter += 1

        # Save the uploaded file with the generated filename
        filename = f"{base_filename}{counter}{extension}"
        file_path = os.path.join(save_directory, filename)
        uploaded_file = request.FILES['file']
        
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        return JsonResponse({"message": f"File uploaded successfully as {filename}"}, status=200)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)

@csrf_exempt
def predict(request):
    """Placeholder predict endpoint"""
    if request.method == 'POST':
        return JsonResponse({'message': 'Predict endpoint is a placeholder and does nothing.'}, status=200)
    else:
        return JsonResponse({'error': 'Invalid request method. Only POST is allowed.'}, status=405)

# =============================================================================
# GOLF CAMERA ENDPOINTS
# =============================================================================

def golf_video_feed(request):
    try:
        response = requests.get(f'{GOLF_CAMERA_URL}/video_feed', stream=True)
        if response.status_code == 200:
            return StreamingHttpResponse(
                response.iter_content(chunk_size=1024),
                content_type='multipart/x-mixed-replace; boundary=frame'
            )
        else:
            return HttpResponse("Golf camera feed unavailable", status=503)
    except requests.exceptions.RequestException as e:
        return HttpResponse(f"Golf camera connection error: {e}", status=503)

def golf_status(request):
    try:
        response = requests.get(f'{GOLF_CAMERA_URL}/recording_status', timeout=5)
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({
                'error': 'Golf camera offline',
                'is_recording': False,
                'auto_recording_enabled': False,
                'pose_detection_enabled': False,
                'predicted_class': 'Unknown',
                'pose_stage': 'disconnected',
                'p1_confidence': 0,
                'p10_confidence': 0
            }, status=503)
    except requests.exceptions.RequestException as e:
        return JsonResponse({
            'error': f'Golf camera connection error: {str(e)}',
            'is_recording': False,
            'auto_recording_enabled': False,
            'pose_detection_enabled': False,
            'predicted_class': 'Unknown',
            'pose_stage': 'disconnected',
            'p1_confidence': 0,
            'p10_confidence': 0
        }, status=503)

@csrf_exempt
@require_http_methods(["POST"])
def golf_start_recording(request):
    """Enhanced recording with assignee support"""
    try:
        data_from_browser = json.loads(request.body)
        
        # Get current user info
        user = User.find_user_by_id(ObjectId(request.session['Id']))
        user_id_str = str(user['_id'])
        
        # Prepare payload with enhanced data
        payload = {
            'user_id': user_id_str,
            'role': user['Role'],
            'duration': data_from_browser.get('duration', 10),
            # Add callback URL so RPi knows where to upload
            'upload_callback_url': request.build_absolute_uri('/home/upload_from_camera_system/')
        }
        
        # If assignee_id provided, include it
        assignee_id = data_from_browser.get('assignee_id')
        if assignee_id:
            payload['assignee_id'] = assignee_id
        
        # Forward to RPi
        response = requests.post(
            f'{GOLF_CAMERA_URL}/start_recording',
            json=payload,
            timeout=5
        )
        
        if response.ok:
            return JsonResponse(response.json())
        else:
            return JsonResponse({'error': 'Failed to start golf recording on RPi'}, status=response.status_code)
            
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def golf_toggle_auto_recording(request):
    """Toggle auto recording on golf camera"""
    try:
        response = requests.post(f'{GOLF_CAMERA_URL}/toggle_auto_recording', timeout=5)
        
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({'error': 'Failed to toggle auto recording'}, status=503)
            
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': f'Golf camera error: {str(e)}'}, status=503)

@csrf_exempt
@require_http_methods(["POST"])
def golf_toggle_pose_detection(request):
    """Toggle pose detection on golf camera"""
    try:
        response = requests.post(f'{GOLF_CAMERA_URL}/toggle_pose_detection', timeout=5)
        
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({'error': 'Failed to toggle pose detection'}, status=503)
            
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': f'Golf camera error: {str(e)}'}, status=503)

@csrf_exempt
@require_http_methods(["POST"])
def golf_reload_models(request):
    """Reload AI models on golf camera"""
    try:
        response = requests.post(f'{GOLF_CAMERA_URL}/reload_models', timeout=30)
        
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({'error': 'Failed to reload models'}, status=503)
            
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': f'Golf camera error: {str(e)}'}, status=503)
    
@csrf_exempt
@require_http_methods(["POST"])
def golf_set_user_context(request):
    """Enhanced user context setting with assignee support"""
    try:
        # Get user from session
        user = User.find_user_by_id(ObjectId(request.session['Id']))
        user_id_str = str(user['_id'])
        user_role = user['Role']
        
        # Get request data
        data = json.loads(request.body) if request.body else {}
        assignee_id = data.get('assignee_id')  # Optional: who the video is for
        
        # Prepare payload for RPi
        payload = {
            'user_id': user_id_str,
            'role': user_role
        }
        
        # If assignee_id provided, include it
        if assignee_id:
            payload['assignee_id'] = assignee_id
        
        # Send to RPi
        response = requests.post(
            f'{GOLF_CAMERA_URL}/set_user_context',
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            response_data = response.json()
            return JsonResponse({
                'status': 'success', 
                'operator_id': user_id_str,
                'assignee_id': response_data.get('assignee_id', user_id_str),
                'role': user_role
            })
        else:
            return JsonResponse({'error': 'Failed to set user context on RPi'}, status=503)
            
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': f'RPi connection error: {str(e)}'}, status=503)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)

def api_my_students(request):
    """API endpoint to get coach's students"""
    if not is_logged_in(request) or not isCoach(request):
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    
    try:
        students = Coach.fetch_all_students(request.session['Id'])
        student_list = [
            {'id': student['id'], 'name': student['Name']} 
            for student in students
        ]
        return JsonResponse(student_list, safe=False)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def golf_health(request):
    """Check golf camera health"""
    try:
        response = requests.get(f'{GOLF_CAMERA_URL}/recording_status', timeout=3)
        
        if response.status_code == 200:
            return JsonResponse({
                'status': 'connected',
                'golf_camera_ip': '172.20.10.5',
                'golf_camera_url': GOLF_CAMERA_URL,
                'timestamp': timezone.now().isoformat(),
                'camera_data': response.json()
            })
        else:
            return JsonResponse({
                'status': 'error',
                'golf_camera_ip': '172.20.10.5',
                'golf_camera_url': GOLF_CAMERA_URL,
                'error': f'HTTP {response.status_code}',
                'timestamp': timezone.now().isoformat()
            })
            
    except requests.exceptions.RequestException as e:
        return JsonResponse({
            'status': 'disconnected',
            'golf_camera_ip': '172.20.10.5',
            'golf_camera_url': GOLF_CAMERA_URL,
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        })