from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect, JsonResponse, StreamingHttpResponse, HttpResponse, HttpResponseBadRequest
from ITP_WEBAPP.models import User
from .models import Coach, Video, Comment
from ITP_WEBAPP.views import is_logged_in
from bson import ObjectId
import requests
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import os
from datetime import datetime
from django.core.paginator import Paginator
import json

# Helper Functions
def isCoach(request):
    """Checks if the user is an admin or coach."""
    return request.session.get('Role') in ['coach']

def isStudent(request):
    """Checks if the user is a student."""
    return request.session.get('Role') in ['student']

def isAdmin(request):
    """Checks if the user is an admin."""
    return request.session.get('Role') in ['admin']

def fetch_all_students(coach_id):
    """Fetches all students associated with a coach."""
    return Coach.fetch_all_students(coach_id)


# Views
def home(request):
    """Redirects to the appropriate dashboard based on user role."""
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
    """Displays the data space dashboard."""
    if not is_logged_in(request):
        return redirect('login')

    # Verify coach-student relationship if role is coach
    if isCoach(request) and not Coach.verify_coach_student_relationship(request.session['Id'], id):
        return redirect('home')

    view = request.GET.get('view', 'list')
    sort = request.GET.get('sort', 'earliest')
    tab  = request.GET.get('tab',  'tab1')
    page_num = request.GET.get('page', 1)

    # Fetch user and student details
    user = User.find_user_by_id(ObjectId(request.session['Id']))
    student = user if not isCoach(request) else User.find_user_by_id(ObjectId(id))

    # Fetch videos
    student_id = student['_id']
    video_list = Video.get_all_videos(student_id)

    if request.method == "POST" and "video_ids" in request.POST:
        ids = request.POST["video_ids"].split(",")
        # delete from DB:
        Video.objects.filter(id__in=ids).delete()

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
        query = f'?tab={tab}&view={view}&sort={sort}'
        return HttpResponseRedirect(base + query)

    # Separate videos by status
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

    # Render template
    return render(request, 'dashboard_dataSpace.html', {
        'Role': user['Role'],
        'Name': user['Name'],
        'studentID': student_id,
        'studentName': student['Name'],
        'videos': videos_page,
        'processing_video': processing_page,
        'completed_video': completed_page,
        'sort': sort,
        'view': view,
        'tab': tab,
        "page_obj": page_obj,
    })


def dashboard_videoFeed(request):
    """Displays the video feed dashboard."""
    if not is_logged_in(request):
        return redirect('login')
    
    # Fetch user details
    user = User.find_user_by_id(ObjectId(request.session['Id']))

    return render(request, 'dashboard_videoFeed.html', {'Role': user['Role'], 'Name': user['Name']})


import requests
import pandas as pd
from io import StringIO

def dashboard_results(request, id, VideoId):
    """Displays the results dashboard."""
    if not is_logged_in(request):
        return redirect('login')

    # Verify coach-student relationship if role is coach
    if isCoach(request) and not Coach.verify_coach_student_relationship(request.session['Id'], id):
        return redirect('home')

    # Fetch user and student details
    user = User.find_user_by_id(ObjectId(request.session['Id']))
    student = user if not isCoach(request) else User.find_user_by_id(ObjectId(id))

    # Fetch title of the raw video
    all_videos = Video.get_all_videos(student['_id'])
    video_item = next((v for v in all_videos if v['id'] == VideoId), None)
    video_title = video_item.get('Title', 'Untitled Video') if video_item else 'Untitled Video'
    
    video_url = Video.get_video_url(VideoId)
    csv_url = Video.get_csv_url(VideoId)
    
    # Fetch and process the CSV
    response = requests.get(csv_url)

    column_status_mapping = {}

    if response.status_code == 200:
        csv_data = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))
        
        # Get all column names
        all_columns = df.columns.tolist()
        
        # Select only the first 3 columns for display
        display_columns = all_columns
        
        # All the csv data
        full_data = df.to_dict('records')
        
        #column_status_mapping = {}
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
        'studentID': id,
        'videoId': VideoId,
        'comments': comments,
        'video_title': video_title,
        'video_url': video_url,
        'columns': display_columns,  # Filtered columns for display
        'full_data': full_data,  # Full data for other purposes
        'column_status_mapping': column_status_mapping,
    })

# New view for comparing two swings
def dashboard_compareSwings(request, id):
    """Displays a comparison of two selected swing videos."""
    if not is_logged_in(request):
        return redirect('login')

    # Verify coach-student relationship if role is coach
    if request.session.get('Role') == 'coach' and not Coach.verify_coach_student_relationship(request.session['Id'], id):
        return redirect('home')

    # Grab the two video IDs from the query string
    video_ids_param = request.GET.get('video_ids')
    if not video_ids_param:
        # Use HttpResponseBadRequest for client-side errors
        return HttpResponseBadRequest("Missing video_ids parameter. Please select two videos to compare.")

    video_ids = video_ids_param.split(',')
    if len(video_ids) != 2:
        return HttpResponseBadRequest("Exactly two video IDs are required for comparison.")

    video1_id, video2_id = video_ids

    # Fetch all of this student's videos once
    all_videos = Video.get_all_videos(id)

    # Look up metadata dicts by matching the 'id' field
    video1_item = next((v for v in all_videos if v['id'] == video1_id), None)
    video2_item = next((v for v in all_videos if v['id'] == video2_id), None)

    if not video1_item or not video2_item:
        return HttpResponseBadRequest("One or more selected videos not found.")

    # Titles and URLs
    video1_title = video1_item.get('Title', 'Untitled Video 1')
    video2_title = video2_item.get('Title', 'Untitled Video 2')
    video1_url   = Video.get_video_url(video1_id)
    video2_url   = Video.get_video_url(video2_id)

    # CSV data for each - MODIFIED TO USE JSON.DUMPS()
    def fetch_csv_data_json(vid): # Renamed for clarity
        url = Video.get_csv_url(vid)
        resp = requests.get(url)
        if resp.status_code == 200:
            df = pd.read_csv(StringIO(resp.content.decode('utf-8')))
            return json.dumps(df.to_dict('records')) # <-- IMPORTANT: Convert to JSON string here
        return json.dumps([]) # <-- IMPORTANT: Return empty JSON array string

    video1_full_data_json = fetch_csv_data_json(video1_id) # Store as JSON string
    video2_full_data_json = fetch_csv_data_json(video2_id) # Store as JSON string

    # Current user (for base template)
    user = User.find_user_by_id(ObjectId(request.session['Id']))

    context = {
        'Role': user['Role'],
        'Name': user['Name'],
        'studentID': id,
        'video1_id': video1_id,
        'video1_title': video1_title,
        'video1_url': video1_url,
        'video1_full_data': video1_full_data_json, # Pass the JSON string
        'video2_id': video2_id,
        'video2_title': video2_title,
        'video2_url': video2_url,
        'video2_full_data': video2_full_data_json, # Pass the JSON string
    }
    return render(request, 'dashboard_compareSwings.html', context)

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
    """Displays the Coach dashboard with associated students."""
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
    """Displays the Coach dashboard with associated students."""
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

#=====================================================yitong======================================================

    
# Define the Raspberry Pi URL once here
#RASPBERRY_PI_URL = 'http://192.168.1.224:5000'
RASPBERRY_PI_URL = 'http://192.168.93.15:5000'

# Streaming response for live camera feed
def live_stream(request):
    try:
        response = requests.get(f'{RASPBERRY_PI_URL}/start_live_cam', stream=True)
        if response.status_code == 200:
            return StreamingHttpResponse(response.iter_content(chunk_size=1024),
                                         content_type='multipart/x-mixed-replace; boundary=frame')
        else:
            return HttpResponse("Failed to connect to Raspberry Pi camera.", status=500)
    except requests.exceptions.RequestException as e:
        return HttpResponse(f"Error: {e}", status=500)

@csrf_exempt
# Start recording on the Raspberry Pi
def start_recording(request):
    try:
        response = requests.post(f'{RASPBERRY_PI_URL}/start_recording')
        return JsonResponse(response.json() if response.ok else {"message": "Failed to start recording"}, status=response.status_code)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"message": f"Error: {str(e)}"}, status=500)
    
@csrf_exempt
def upload_from_pi(request):
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
#Temp predict to be edit    
def predict(request):
    if request.method == 'POST':
        # Placeholder: Does nothing meaningful
        return JsonResponse({'message': 'Predict endpoint is a placeholder and does nothing.'}, status=200)
    else:
        return JsonResponse({'error': 'Invalid request method. Only POST is allowed.'}, status=405)