# dashboard/views.py

from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect, JsonResponse, StreamingHttpResponse, HttpResponse, HttpResponseBadRequest
from ITP_WEBAPP.models import User
from django.views.decorators.http import require_POST
from .models import Coach, Video, Comment, Users_Collection # Make sure Comment is imported
from ITP_WEBAPP.views import is_logged_in
from bson import ObjectId
import requests
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt, csrf_protect
import os
from datetime import datetime
from django.core.paginator import Paginator
import json
import pandas as pd
from io import StringIO
import traceback

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

# NEW HELPER FUNCTION to recursively convert ObjectIds to strings
def convert_objectids_to_str(data):
    if isinstance(data, list):
        return [convert_objectids_to_str(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_objectids_to_str(v) for k, v in data.items()}
    elif isinstance(data, ObjectId):
        return str(data)
    # You might also need to handle datetime objects if your comments_data
    # contains them and they are not already formatted to strings by your model.
    # For example:
    # elif isinstance(data, datetime):
    #     return data.strftime("%Y-%m-%d %H:%M:%S") # Or any desired format
    return data


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

    # Convert ObjectIds in video_list if necessary (though your current error is in comments_data)
    # It's good practice to ensure all data passed to templates is JSON serializable.
    video_list = convert_objectids_to_str(video_list)

    if request.method == "POST" and "video_ids" in request.POST:
        ids = request.POST["video_ids"].split(",")
        # NOTE: Your current `Video.objects.filter(id__in=ids).delete()` line
        # appears to be Django ORM syntax, but your models.py uses direct PyMongo.
        # This line might cause an error or not function as expected for MongoDB.
        # If video deletion is needed, you'll need a corresponding method in your Video class.
        pass # Placeholder - revisit video deletion if it's not working with current setup

    def parse_date(s):
        # matches "HH:MM Mon DD,YYYY"
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

    # Determine current_user_name to pass to the template
    # Prioritize username from session, fallback to user's 'Name' field
    current_user_name = request.session.get('Username', user.get('Name', ''))

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
        
        # Display all columns from CSV
        display_columns = all_columns
        
        # All the csv data (converted to list of dicts for JSON serialization)
        full_data = df.to_dict('records')
        
        for column in all_columns:
            if "Status" in column:
                corresponding_column = column.replace(" Status", "")
                column_status_mapping[corresponding_column] = column
    else:
        display_columns = []
        full_data = []

    # Fetch video comments with hierarchical structure using the updated Video.get_all_video_comments
    # This method now returns a list of top-level comments, each with a 'replies' list,
    # and all necessary formatting (id conversion, date formatting, user name, and x_pos/y_pos for top-level)
    comments_data = Video.get_all_video_comments(VideoId)
    
    # --- IMPORTANT FIX: Convert all ObjectIds in comments_data to strings ---
    processed_comments_data = convert_objectids_to_str(comments_data)

    print(f"DEBUG: comments_data (from model, structured): {comments_data}")
    print(f"DEBUG: Type of comments_data: {type(comments_data)}")
    print(f"DEBUG: processed_comments_data (after conversion): {processed_comments_data}")


    return render(request, 'dashboard_results.html', {
        'Role': user['Role'],
        'Name': user['Name'],
        'studentID': id,
        'videoId': VideoId,
        'comments': processed_comments_data, # Pass the processed data to the template
        'video_title': video_title,
        'video_url': video_url,
        'columns': display_columns,  # Filtered columns for display
        'full_data': full_data,  # Full data for other purposes
        'column_status_mapping': column_status_mapping,
        'current_user_name': current_user_name, # ADDED THIS LINE
    })

# Add these new AJAX views to handle comments
@csrf_exempt # Use csrf_exempt for simplicity in development, consider proper CSRF token handling in production
def add_video_comment_ajax(request, id, VideoId):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            comment_text = data.get('comment')
            x_pos = data.get('x_pos') # This might be None/null if from table
            y_pos = data.get('y_pos') # This might be None/null if from table

            if not comment_text:
                return JsonResponse({'status': 'error', 'message': 'Comment text is required.'}, status=400)

            current_user_id = request.session.get('Id')
            if not current_user_id:
                return JsonResponse({'status': 'error', 'message': 'User not logged in.'}, status=401)

            # Call add_comment which now returns the ObjectId of the new comment
            # Ensure Comment.add_comment can handle x_pos and y_pos as None
            new_comment_obj_id = Comment.add_comment(current_user_id, VideoId, comment_text, x_pos, y_pos)

            if new_comment_obj_id:
                # Fetch the full, processed comment data using the new helper method
                new_comment_data = Comment.get_comment_by_id(str(new_comment_obj_id))

                if new_comment_data:
                    # --- IMPORTANT FIX: Convert all ObjectIds in new_comment_data to strings ---
                    processed_new_comment_data = convert_objectids_to_str(new_comment_data)

                    return JsonResponse({
                        'status': 'success',
                        'message': 'Comment added successfully.',
                        'comment': processed_new_comment_data # Return the full processed comment object
                    })
                else:
                    return JsonResponse({'status': 'error', 'message': 'Comment added but failed to retrieve details.'}, status=500)
            else:
                return JsonResponse({'status': 'error', 'message': 'Failed to add comment to database.'}, status=500)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON.'}, status=400)
        except Exception as e:
            traceback.print_exc() # Print full traceback to console for debugging
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)


# NEW AJAX VIEW for adding replies
@csrf_exempt
@require_POST
def add_video_reply_ajax(request, id, VideoId): # `id` is studentID, `VideoId` is video ID
    try:
        data = json.loads(request.body)
        parent_comment_id = data.get('parent_comment_id')
        reply_text = data.get('reply_text')

        if not parent_comment_id or not reply_text:
            return JsonResponse({'success': False, 'message': 'Missing parent comment ID or reply text'}, status=400)

        current_user_id = request.session.get('Id') # This should be the ObjectId of the current user as a string
        if not current_user_id:
            return JsonResponse({'success': False, 'message': 'User not authenticated or user_id missing'}, status=401)

        # Call the Comment class's add_reply method
        new_reply_obj_id = Comment.add_reply(
            current_user_id=current_user_id,
            video_id=VideoId, # Pass video_id to the reply for context in DB
            parent_comment_id=parent_comment_id,
            reply_text=reply_text
        )

        if new_reply_obj_id:
            # Fetch the newly created reply's full details (structured as expected by frontend)
            new_reply_data = Comment.get_comment_by_id(str(new_reply_obj_id))

            if new_reply_data:
                # --- IMPORTANT FIX: Convert all ObjectIds in new_reply_data to strings ---
                processed_new_reply_data = convert_objectids_to_str(new_reply_data)

                return JsonResponse({
                    'success': True,
                    'message': 'Reply added successfully',
                    'reply': processed_new_reply_data # Pass the processed data
                }, status=201)
            else:
                return JsonResponse({'success': False, 'message': 'Reply added but failed to retrieve full details.'}, status=500)
        else:
            return JsonResponse({'success': False, 'message': 'Failed to add reply to database.'}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'message': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        traceback.print_exc() # For debugging
        return JsonResponse({'success': False, 'message': f'An error occurred: {str(e)}'}, status=500)

@csrf_exempt
def update_comment_position_ajax(request, id, VideoId):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            comment_id = data.get('comment_id')
            x_pos = data.get('x_pos')
            y_pos = data.get('y_pos')

            if not all([comment_id, x_pos is not None, y_pos is not None]):
                return JsonResponse({'status': 'error', 'message': 'Missing comment ID or position data.'}, status=400)

            # Ensure x_pos and y_pos are numbers before passing to model
            try:
                x_pos = float(x_pos)
                y_pos = float(y_pos)
            except (ValueError, TypeError):
                return JsonResponse({'status': 'error', 'message': 'Invalid x_pos or y_pos format.'}, status=400)


            success = Comment.update_comment_position(comment_id, x_pos, y_pos)

            if success:
                return JsonResponse({'status': 'success', 'message': 'Comment position updated successfully.'})
            else:
                return JsonResponse({'status': 'error', 'message': 'Failed to update comment position.'}, status=500)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON.'}, status=400)
        except Exception as e:
            traceback.print_exc()
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)

@csrf_exempt
def delete_video_comment_ajax(request, id, VideoId):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            comment_id = data.get('comment_id')

            if not comment_id:
                return JsonResponse({'status': 'error', 'message': 'Comment ID is required.'}, status=400)

            # Authorization check: Ensure the current user is allowed to delete this comment.
            current_user_id = request.session.get('Id')
            if not current_user_id:
                return JsonResponse({'status': 'error', 'message': 'User not authenticated.'}, status=401)

            # Assuming your Comment.delete_comment method handles authorization internally
            # or needs the user ID to verify ownership/permissions.
            success = Comment.delete_comment(comment_id, current_user_id)

            if success:
                return JsonResponse({'status': 'success', 'message': 'Comment deleted successfully.'})
            else:
                return JsonResponse({'status': 'error', 'message': 'Failed to delete comment or unauthorized.'}, status=500)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON.'}, status=400)
        except Exception as e:
            traceback.print_exc()
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)

# New AJAX View for editing a comment
@csrf_exempt
@require_POST
def edit_video_comment_ajax(request, id, VideoId):
    if not is_logged_in(request):
        return JsonResponse({'status': 'error', 'message': 'User not logged in.'}, status=401)
    
    try:
        data = json.loads(request.body)
        comment_id = data.get('comment_id')
        new_text = data.get('comment_text') # Changed parameter name to 'comment'

        if not comment_id or not new_text:
            return JsonResponse({'status': 'error', 'message': 'Missing comment ID or new text.'}, status=400)

        # Get the current user's ID from the session
        current_user_id = request.session.get('Id')

        # Call the Comment model method to update the comment
        # You'll need to implement this method in your Comment model
        success = Comment.edit_comment(comment_id, new_text, current_user_id)

        if success:
            return JsonResponse({'status': 'success', 'message': 'Comment updated successfully.'})
        else:
            # You might want more specific error messages here (e.g., "Comment not found", "Unauthorized")
            return JsonResponse({'status': 'error', 'message': 'Failed to update comment or unauthorized.'}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON.'}, status=400)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

# NEW AJAX View for editing a reply
@csrf_exempt
@require_POST
def edit_video_reply_ajax(request, id, VideoId):
    if not is_logged_in(request):
        return JsonResponse({'success': False, 'message': 'User not logged in.'}, status=401)

    try:
        data = json.loads(request.body)
        reply_id = data.get('reply_id')
        new_reply_text = data.get('reply_text')

        if not reply_id or not new_reply_text:
            return JsonResponse({'success': False, 'message': 'Missing reply ID or new text.'}, status=400)

        current_user_id = request.session.get('Id')

        # Call your Comment model's method to edit the reply
        # You need to implement `Comment.edit_reply(reply_id, new_reply_text, current_user_id)`
        success = Comment.edit_reply(reply_id, new_reply_text, current_user_id)

        if success:
            return JsonResponse({'success': True, 'message': 'Reply updated successfully.'})
        else:
            return JsonResponse({'success': False, 'message': 'Failed to update reply or unauthorized.'}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'message': 'Invalid JSON.'}, status=400)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'success': False, 'message': str(e)}, status=500)

# NEW AJAX View for deleting a reply
@csrf_exempt
@require_POST
def delete_video_reply_ajax(request, id, VideoId):
    if not is_logged_in(request):
        return JsonResponse({'success': False, 'message': 'User not logged in.'}, status=401)

    try:
        data = json.loads(request.body)
        reply_id = data.get('reply_id')

        if not reply_id:
            return JsonResponse({'success': False, 'message': 'Reply ID is required.'}, status=400)

        current_user_id = request.session.get('Id')

        # Call your Comment model's method to delete the reply
        # You need to implement `Comment.delete_reply(reply_id, current_user_id)`
        success = Comment.delete_reply(reply_id, current_user_id)

        if success:
            return JsonResponse({'success': True, 'message': 'Reply deleted successfully.'})
        else:
            return JsonResponse({'success': False, 'message': 'Failed to delete reply or unauthorized.'}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'message': 'Invalid JSON.'}, status=400)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'success': False, 'message': str(e)}, status=500)


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

    # Convert ObjectId in user object if it's directly passed
    processed_user = convert_objectids_to_str(user)

    context = {
        'Role': processed_user['Role'], # Use processed user
        'Name': processed_user['Name'], # Use processed user
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

    # Convert ObjectIds in user and students data
    processed_user = convert_objectids_to_str(user)
    processed_students = convert_objectids_to_str(students)


    if request.method == 'POST':
        upload_video(request)
        return HttpResponseRedirect(reverse('home'))

    return render(request, 'dashboard_coach.html', {
        'Role': processed_user['Role'],
        'Name': processed_user['Name'],
        'students': processed_students,
        'view': view,
    })
    
    
def dashboard_admin(request):
    """Displays the Coach dashboard with associated students."""
    if not is_logged_in(request):
        return redirect('login')

    if not isAdmin(request):
        return redirect('home')

    user = User.find_user_by_id(ObjectId(request.session['Id']))
    processed_user = convert_objectids_to_str(user)

    if request.method == 'POST':
        create_account(request)
        return HttpResponseRedirect(reverse('home'))

    return render(request, 'dashboard_admin.html', {
        'Role': processed_user['Role'],
        'Name': processed_user['Name'],
    })
    
def admin_model(request):
    """Displays the Coach dashboard with associated students."""
    if not is_logged_in(request):
        return redirect('login')

    if not isAdmin(request):
        return redirect('home')

    user = User.find_user_by_id(ObjectId(request.session['Id']))
    processed_user = convert_objectids_to_str(user)

    if request.method == 'POST':
        return HttpResponseRedirect(reverse('home'))

    return render(request, 'dashboard_model.html', {
        'Role': processed_user['Role'],
        'Name': processed_user['Name'],
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