from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect, JsonResponse, StreamingHttpResponse, HttpResponse
from ITP_WEBAPP.models import User
from .models import Coach, Video, Comment
from ITP_WEBAPP.views import is_logged_in
from bson import ObjectId
import requests
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import os

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

    # Fetch user and student details
    user = User.find_user_by_id(ObjectId(request.session['Id']))
    student = user if not isCoach(request) else User.find_user_by_id(ObjectId(id))

    # Fetch videos
    student_id = student['_id']
    video_list = Video.get_all_videos(student_id)

    if request.method == 'POST':
        upload_video(request, student_id)
        
        return HttpResponseRedirect(reverse('dashboard_dataSpace', args=[id]))

    # Separate videos by status
    video_processing = [video for video in video_list if video.get('Status') == 'Processing']
    video_completed = [video for video in video_list if video.get('Status') == 'Completed']

    # Render template
    return render(request, 'dashboard_dataSpace.html', {
        'Role': user['Role'],
        'Name': user['Name'],
        'studentID': student_id,
        'studentName': student['Name'],
        'videos': video_list,
        'processing_video': video_processing,
        'completed_video': video_completed,
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
    
    video_url = Video.get_video_url(VideoId)
    csv_url = Video.get_csv_url(VideoId)
    
    # Fetch and process the CSV
    response = requests.get(csv_url)
    if response.status_code == 200:
        csv_data = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))
        
        # Get all column names
        all_columns = df.columns.tolist()
        
        # Select only the first 3 columns for display
        display_columns = all_columns[:3]
        
        # All the csv data
        full_data = df.to_dict('records')
        
        column_status_mapping = {}
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
        'videoId': VideoId,
        'comments': comments,
        'video_url': video_url,
        'columns': display_columns,  # Filtered columns for display
        'full_data': full_data,  # Full data for other purposes
        'column_status_mapping': column_status_mapping,
    })


def dashboard_Coach(request):
    """Displays the Coach dashboard with associated students."""
    if not is_logged_in(request):
        return redirect('login')

    if not isCoach(request):
        return redirect('home')

    user = User.find_user_by_id(ObjectId(request.session['Id']))
    students = fetch_all_students(request.session['Id'])

    if request.method == 'POST':
        upload_video(request)
        return HttpResponseRedirect(reverse('home'))

    return render(request, 'dashboard_coach.html', {
        'Role': user['Role'],
        'Name': user['Name'],
        'students': students,
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
        return redirect('home')

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