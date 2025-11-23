# dashboard/views.py

from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponseRedirect, JsonResponse, StreamingHttpResponse, HttpResponse, HttpResponseBadRequest
from ITP_WEBAPP.models import User
from django.views.decorators.http import require_POST
from .models import Coach, Video, Comment, Users_Collection, Videos_Collection  # Make sure Comment is imported
from ITP_WEBAPP.views import is_logged_in
from bson import ObjectId
import requests
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.views.decorators.http import require_http_methods
import os
from datetime import datetime
import json
from django.utils import timezone
import pandas as pd
from io import StringIO
from django.core.paginator import Paginator
import traceback
import re

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

# --- [FIX] ADDED THE MISSING process_csv_data FUNCTION ---
def process_csv_data(csv_url):
    """
    Fetches CSV data from a URL, processes it into a list of dictionaries,
    and returns the data and column names.
    """
    if not csv_url:
        return [], []
    
    try:
        response = requests.get(csv_url)
        if response.status_code == 200:
            # Use StringIO to treat the string content as a file
            csv_data = StringIO(response.content.decode('utf-8'))
            df = pd.read_csv(csv_data)
            
            # Handle empty CSV
            if df.empty:
                return [], []
            
            # --- MODIFICATION: Round all numeric data values to 2 decimal points ---
            df = df.round(2)
            
            # --- [FIX] NEW MODIFICATION: Replace all nan/NaN values with the string 'None' for display ---
            df = df.fillna('None')
            # --- END [FIX] NEW MODIFICATION ---
                
            # Get column names
            display_columns = list(df.columns)
            
            # Convert DataFrame to list of dictionaries
            full_data = df.to_dict('records')
            
            return full_data, display_columns
        else:
            print(f"Failed to fetch CSV data from {csv_url}. Status code: {response.status_code}")
            return [], []
    except Exception as e:
        print(f"Error processing CSV data from {csv_url}: {e}")
        traceback.print_exc()
        return [], []
    
def correlate_fo_to_dtl_data(fo_full_data, dtl_full_data):
    """
    Correlates Down-The-Line (DTL) data to Face-On (FO) timestamps,
    returning a dictionary of correlated DTL rows with all available columns.
    """
    if not fo_full_data or not dtl_full_data:
        return [], []

    try:
        # 1. Convert lists of dicts to DataFrames
        fo_df = pd.DataFrame(fo_full_data)
        dtl_df = pd.DataFrame(dtl_full_data)
        
        # Ensure Time columns are numeric for comparison
        # Note: 'None' strings from process_csv_data will become NaN here, which is fine
        fo_df['Time Frame'] = pd.to_numeric(fo_df['Time Frame'], errors='coerce')
        dtl_df['Time Frame'] = pd.to_numeric(dtl_df['Time Frame'], errors='coerce')

        # Drop rows with invalid time data
        fo_df.dropna(subset=['Time Frame', 'Pose Class'], inplace=True)
        dtl_df.dropna(subset=['Time Frame'], inplace=True)
        
        # 2. Extract unique Pose and Time from FO data (P1-P10)
        pose_map = fo_df[['Pose Class', 'Time Frame']].copy()
        # Filter for P1 through P10 only
        pose_map = pose_map[pose_map['Pose Class'].str.match(r'P(10|[1-9])$', na=False)]
        
        correlated_dtl_rows = []

        # 3. For each FO Time Frame, find the closest DTL entry
        for index, fo_row in pose_map.iterrows():
            fo_time = fo_row['Time Frame']
            pose_class = fo_row['Pose Class']
            
            # Calculate time difference and find the index of the minimum difference
            dtl_df['time_diff'] = (dtl_df['Time Frame'] - fo_time).abs()
            
            # Find the row with the minimum time difference
            # Keep all DTL data columns for merging later in dashboard_results
            closest_dtl_row_series = dtl_df.loc[dtl_df['time_diff'].idxmin()].drop(
                labels=['time_diff', 'Frame', 'Predicted Class', 'Confidence'], 
                errors='ignore'
            )
            closest_dtl_row = closest_dtl_row_series.to_dict()
            
            # Add the FO Pose Class and Time Frame to the DTL result for context
            closest_dtl_row['Reference Pose (FO)'] = pose_class
            closest_dtl_row['Time (FO)'] = fo_time

            correlated_dtl_rows.append(closest_dtl_row)

        if not correlated_dtl_rows:
            return [], []

        # 4. Convert back to list of dictionaries and set columns for display
        final_df = pd.DataFrame(correlated_dtl_rows)
        
        # Define columns to exclude from the returned DTL data
        cols_to_exclude = ['Overall Status'] # Explicitly remove Overall Status as requested
        
        # Define a consistent order for the DTL-specific columns for stability
        final_display_cols_base = [
            'Reference Pose (FO)', 
            'Time (FO)', 
            'Time Frame',
        ]
        
        # Collect all other non-excluded columns and append them
        other_cols = [col for col in final_df.columns 
                      if col not in final_display_cols_base and col not in cols_to_exclude]
        
        final_display_cols = final_display_cols_base + sorted(other_cols)
        
        # Ensure only columns that exist are included
        final_display_cols = [col for col in final_display_cols if col in final_df.columns]
        
        final_df = final_df[final_display_cols]
        
        # --- [FIX] NEW MODIFICATION: Replace all nan/NaN values with the string 'None' for display ---
        final_df = final_df.fillna('None')
        # --- END [FIX] NEW MODIFICATION ---
        
        final_data = final_df.to_dict('records')
        
        return final_data, final_display_cols
        
    except Exception as e:
        print(f"Error correlating FO and DTL data: {e}")
        traceback.print_exc()
        return [], []

def fetch_all_students(coach_id):
    """Fetches all students associated with a coach."""
    return Coach.fetch_all_students(coach_id)

def upload_single_video_helper(request, current_user_id, assignee_id):
    """
    Handles a single-file video upload and initiates asynchronous processing.
    """
    # Check for all possible file field names from the different forms
    file = (request.FILES.get('videoDBFile_face') or
            request.FILES.get('videoDBFile_dtl') or
            request.FILES.get('videoFile_face') or
            request.FILES.get('videoFile_dtl'))

    if request.method == 'POST' and file:
        title = file.name
        video_type = request.POST.get('videoType', 'face-on') # Default to face-on

        try:
            Video.upload_video(
                current_user_id=current_user_id,
                assignee_id=str(assignee_id),
                title=title,
                video_type=video_type,
                file=file,
                upload_source="manual"
            )
            return True, "Video upload initiated successfully."
        except Exception as e:
            traceback.print_exc()
            return False, f"Upload error: {str(e)}"
    
    return False, "No file or invalid request."

# =============================================================================
# 🏌️ CAMERA CONFIGURATION
# =============================================================================
# Define the Raspberry Pi URLs
RASPBERRY_PI_URL = 'http://172.20.10.5:5000'  # Old camera system
GOLF_CAMERA_URL = 'http://172.20.10.5:5000'     # New golf camera system
DTL_CAMERA_URL = 'http://172.20.10.10:5001'      # Down-the-line golf camera system

# =============================================================================
# 📱 MAIN DASHBOARD VIEWS
# =============================================================================
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

    if isCoach(request) and not Coach.verify_coach_student_relationship(request.session['Id'], id):
        return redirect('home')

    view = request.GET.get('view', 'list')
    sort = request.GET.get('sort', 'earliest')
    tab = request.GET.get('tab', 'tab1')
    
    # --- [NEW] Get active video type filters from URL ---
    active_filters = request.GET.getlist('filter_type') # Gets a list of 'filter_type' values

    # --- MODIFIED: HANDLE POST REQUESTS (Deletion or Upload) ---
    if request.method == 'POST':
        if 'video_ids' in request.POST:
            video_ids_to_delete = [vid for vid in request.POST.get("video_ids", "").split(",") if vid]
            if video_ids_to_delete:
                Video.delete_videos(video_ids_to_delete)
            query_params = request.GET.urlencode()
            redirect_url = reverse('dashboard_dataSpace', args=[id])
            return HttpResponseRedirect(f'{redirect_url}?{query_params if query_params else ""}')

        # --- Handle video uploads (single or dual) ---
        else:
            video_type = request.POST.get('videoType')
            current_user_id = request.session.get('Id')
            student_doc = User.find_user_by_id(ObjectId(id))
            assignee_id = str(student_doc['_id'])

            if video_type == 'both':
                # [MODIFIED] Use the correct names from dashboard_popupUploadModal.js
                face_on_file = request.FILES.get('face_on_file')
                dtl_file = request.FILES.get('dtl_file')
                
                if face_on_file and dtl_file:
                    Video.upload_dual_videos(
                        current_user_id=current_user_id,
                        assignee_id=assignee_id,
                        face_on_file=face_on_file,
                        dtl_file=dtl_file
                    )
                    print("Dual video upload initiated successfully.")
                else:
                    print("Dual upload error: one or both files were missing.")
            else:
                success, message = upload_single_video_helper(request, current_user_id, assignee_id)
                print(f"Single upload status: {message}")

            base = reverse('dashboard_dataSpace', args=[id])
            query = f'?tab={tab}&view={view}&sort={sort}'
            return HttpResponseRedirect(base + query)

    # --- HANDLE GET REQUESTS (Page Load) ---
    page_num = request.GET.get('page', 1)
    user = User.find_user_by_id(ObjectId(request.session['Id']))
    student = user if not isCoach(request) else User.find_user_by_id(ObjectId(id))
    student_id = student['_id']
    video_list = convert_objectids_to_str(Video.get_all_videos(student_id))
    
    def parse_date(s):
        return datetime.strptime(s, "%H:%M %b %d, %Y")

    def clean_title(title):
        """Removes file extension and common type identifiers from a title string."""
        if not isinstance(title, str):
            return ""
        # 1. Remove file extension (e.g., .mp4, .mov)
        title_no_ext = re.sub(r'\.[a-z0-9]+$', '', title, flags=re.IGNORECASE)
        # 2. Remove common type identifiers
        title_cleaned = title_no_ext.replace('_face-on', '').replace('_down-the-line', '').replace('face-on', '').replace('down-the-line', '')
        # 3. Strip leading/trailing spaces
        return title_cleaned.strip()
    
    # --- [NEW] Group videos by session_id ---
    # [ ... existing grouping logic ... ]
    grouped_video_list = []
    processed_session_ids = set()

    # Pass 1: Find all videos with session_ids and group them
    session_videos = {} # {session_id: [video_doc, ...]}
    for video in video_list:
        session_id = video.get('session_id')
        if session_id:
            if session_id not in session_videos:
                session_videos[session_id] = []
            session_videos[session_id].append(video)

    # Pass 2: Create combined entries for groups
    for session_id, videos_in_group in session_videos.items():
        if session_id in processed_session_ids:
            continue
        
        # We only group if there's a pair (face-on and dtl)
        if len(videos_in_group) >= 2:
            face_on_video = next((v for v in videos_in_group if v.get('Type') == 'face-on'), None)
            dtl_video = next((v for v in videos_in_group if v.get('Type') == 'down-the-line'), None)

            # If we have both, create a combined entry
            if face_on_video and dtl_video:
                # Determine combined status
                status = "Completed"
                if face_on_video.get('Status') == 'Processing' or dtl_video.get('Status') == 'Processing':
                    status = "Processing"
                elif face_on_video.get('Status') == 'Failed' or dtl_video.get('Status') == 'Failed':
                    status = "Failed"

                # Use face-on video as the primary
                combined_video = face_on_video.copy() # Start with face-on video's data
                combined_video['Type'] = 'both'
                combined_video['Status'] = status
                # Use the *earlier* date
                try:
                    face_on_date = parse_date(face_on_video['DateUploaded'])
                    dtl_date = parse_date(dtl_video['DateUploaded'])
                    combined_video['DateUploaded'] = min(face_on_date, dtl_date).strftime("%H:%M %b %d, %Y")
                except Exception:
                    pass # Keep face-on date if parsing fails
                
                # Store the DTL video's ID for the results page
                combined_video['dtl_video_id'] = dtl_video.get('id')
                combined_video['dtl_rawVideoLink'] = dtl_video.get('rawVideoLink')
                
                # Use the face-on video's ID as the main 'id'
                combined_video['id'] = face_on_video.get('id')
                
                # --- START MODIFIED TITLE LOGIC ---
                fo_name = face_on_video.get('Title', 'FaceOn')
                dtl_name = dtl_video.get('Title', 'DTL')
                
                # Apply the cleaning helper
                fo_name_cleaned = clean_title(fo_name)
                dtl_name_cleaned = clean_title(dtl_name)
                
                # Format the new title: "face on name & down the line name_session_id"
                custom_title = f"{fo_name_cleaned} & {dtl_name_cleaned}_{session_id}"
                combined_video['Title'] = custom_title
                # --- END MODIFIED TITLE LOGIC ---
                
                grouped_video_list.append(combined_video)
                processed_session_ids.add(session_id)
            else:
                # Not a valid pair, add them individually
                for v in videos_in_group:
                    grouped_video_list.append(v)
                    processed_session_ids.add(session_id) # Mark as 'processed' to avoid re-adding
        else:
            # Only one video with this session_id, treat as individual
            for v in videos_in_group:
                grouped_video_list.append(v)
                processed_session_ids.add(session_id)

    # Pass 3: Add all non-session videos (session_id is None or "")
    for video in video_list:
        if not video.get('session_id'):
            grouped_video_list.append(video)
            
    # Now, `grouped_video_list` replaces `video_list` for sorting and pagination
    video_list = grouped_video_list # Overwrite video_list with our new grouped list
    # --- [END NEW] ---

    # --- [NEW] Apply Video Type Filtering from URL params ---
    if active_filters: # Only filter if filter_type params exist
        filtered_video_list = []
        for video in video_list:
            # Check if the video type is in the active filters
            # Note: 'both' type from grouping logic becomes 'Face On & Down The Line' in template
            # Let's check the 'Type' field which is 'both', 'face-on', 'down-the-line'
            video_type = video.get('Type')
            
            # Map the 'Type' field to the filter values
            type_to_filter_map = {
                'face-on': 'Face On',
                'down-the-line': 'Down The Line',
                'both': 'Face On & Down The Line'
            }
            
            if type_to_filter_map.get(video_type) in active_filters:
                 filtered_video_list.append(video)
        video_list = filtered_video_list # Overwrite video_list with the filtered list

    # ... (Sorting and filtering logic now operates on the grouped and filtered list) ...
    if sort == 'az':
        video_list.sort(key=lambda v: v['Title'].lower())
    elif sort == 'za':
        video_list.sort(key=lambda v: v['Title'].lower(), reverse=True)
    elif sort == 'latest':
        video_list.sort(key=lambda v: parse_date(v['DateUploaded']), reverse=True)
    else:
        video_list.sort(key=lambda v: parse_date(v['DateUploaded']))

    video_processing = [video for video in video_list if video.get('Status') == 'Processing']
    video_completed = [video for video in video_list if video.get('Status') == 'Completed']
    video_failed = [video for video in video_list if video.get('Status') == 'Failed']

    per_page = 4 if view == 'list' else 10
    paginator_map = {
        'tab1': Paginator(video_list, per_page),
        'tab2': Paginator(video_completed, per_page),
        'tab3': Paginator(video_processing, per_page),
        'tab4': Paginator(video_failed, per_page)
    }
    page_obj = paginator_map.get(tab, Paginator(video_list, per_page)).get_page(page_num)

    # --- [NEW] Pass filter params to template for pagination links ---
    filter_query_string = ""
    if active_filters:
        # We must URL-encode the filter values which may contain '&'
        from django.utils.http import urlencode
        filter_query_string = "&" + urlencode({'filter_type': active_filters}, doseq=True)

    return render(request, 'dashboard_dataSpace.html', {
        'Role': user['Role'], 'Name': user['Name'], 'user_id': str(user['_id']),
        'studentID': student_id, 'studentName': student['Name'], 'videos': page_obj,
        'processing_video': paginator_map['tab3'].get_page(page_num),
        'completed_video': paginator_map['tab2'].get_page(page_num),
        'failed_video': paginator_map['tab4'].get_page(page_num),
        'sort': sort, 'view': view, 'tab': tab, "page_obj": page_obj,
        'filter_query_string': filter_query_string # Pass the query string
    })

def dashboard_videoFeed(request):
    """Displays the video feed dashboard with both old and new camera systems."""
    if not is_logged_in(request):
        return redirect('login')
    
    # Fetch user details
    user = User.find_user_by_id(ObjectId(request.session['Id']))

    user_id_str = str(user['_id'])

    return render(request, 'dashboard_videoFeed.html', {
        'Role': user['Role'], 
        'Name': user['Name'],
        'user_id': user_id_str
    })

def dashboard_results(request, id, VideoId):
    """
    Displays the results dashboard. Modified to handle conditional single/dual video display
    and combined/correlated data tables for dual uploads.
    """
    if not is_logged_in(request):
        return redirect('login')

    # Verify coach-student relationship if role is coach
    if isCoach(request) and not Coach.verify_coach_student_relationship(request.session['Id'], id):
        return redirect('home')

    # Fetch user and student details
    user = User.find_user_by_id(ObjectId(request.session['Id']))
    student = user if not isCoach(request) else User.find_user_by_id(ObjectId(id))
    current_user_name = request.session.get('Username', user.get('Name', ''))

    all_videos = Video.get_all_videos(student['_id'])
    video_item = next((v for v in all_videos if v['id'] == VideoId), None)
    
    if not video_item:
        return redirect('home')

    # --- Video 1: Primary Video (Face On or Single) ---
    video1_id = VideoId
    video1_title = video_item.get('Title', 'Face On Video')
    video1_processed_url = Video.get_video_url(video1_id)
    video1_raw_url = video_item.get('rawVideoLink')
    video1_url = video1_processed_url if video1_processed_url else video1_raw_url
    video1_csv_url = Video.get_csv_url(video1_id)
    
    # --- Video 2: Secondary Video (DTL) ---
    video2_title = None
    video2_url = None
    video2_csv_url = None
    is_dual_upload = False
    
    # Check for DTL pair using session_id and Type
    session_id = video_item.get('session_id')
    if session_id and (video_item.get('Type') == 'face-on' or video_item.get('Type') == 'both'):
        dtl_video_item = next((
            v for v in all_videos 
            if v['id'] != video1_id
            and v.get('session_id') == session_id 
            and v.get('Type') == 'down-the-line'
        ), None)
        
        if dtl_video_item:
            is_dual_upload = True
            video2_id = dtl_video_item['id']
            video2_csv_url = Video.get_output_csv_url(video2_id)
            video2_title = dtl_video_item.get('Title', 'Down The Line Video')

            # fallback to raw link if no processed URL
            processed_url = Video.get_video_url(video2_id)
            raw_url = dtl_video_item.get('rawVideoLink')
            video2_url = processed_url if processed_url else raw_url

    # --- Process CSV Data ---
    # MODIFIED: Capture columns from video 1
    video1_full_data, video1_columns = process_csv_data(video1_csv_url) # FO or single data
    video2_full_data, _ = process_csv_data(video2_csv_url) if is_dual_upload else ([], [])
    
    # --- [START MODIFIED LOGIC: Combine FO and DTL data into a single table] ---
    
    # 1. Define the mapping logic for data source
    def get_pose_num(pose_class_str):
        match = re.search(r'P(\d+)', pose_class_str)
        return int(match.group(1)) if match else None

    def get_data_source(column, pose_class_str):
        p_num = get_pose_num(pose_class_str)
        if not p_num: return 'FO'

        # Shoulder Tilt: P1, P6 (FO); Others (P2, P3, P4, P5, P7, P8, P9, P10) (DTL)
        if column == 'Shoulder Tilt':
            return 'FO' if p_num in [1, 2, 6, 7, 8] else 'DTL'
        
        # Hip Tilt: 
        # P1, P2, P3, P4, P5, P7, P8 are FO values. 
        # Rest (P6, P9, P10) are DTL.
        if column == 'Hip Tilt':
            return 'FO' if p_num in [1, 2, 3, 4, 5, 7, 8] else 'DTL'

        # Shoulder Rotation, Hip Rotation, Lead Arm Angle: Always FO (if value exists)
        if column in ['Shoulder Rotation', 'Hip Rotation', 'Lead Arm Angle']:
            return 'FO'
            
        # Forward Tilt, Knee Bend: Always DTL (no status)
        if column in ['Forward Tilt', 'Knee Bend']:
            return 'DTL'
            
        return 'FO' # Default to FO

    # 2. Correlate FO data (video1) to DTL data (video2)
    dtl_correlated_data, _ = (
        correlate_fo_to_dtl_data(video1_full_data, video2_full_data) 
        if is_dual_upload else ([], [])
    )
    
    # Convert DTL correlated data to a dictionary keyed by Pose Class for quick lookup
    # Note: dtl_correlated_data already has 'None' strings instead of nan
    dtl_data_map = {
        row.get('Reference Pose (FO)'): row 
        for row in dtl_correlated_data 
        if row.get('Reference Pose (FO)')
    }

    # Columns to handle (Angle and Status pairs)
    angle_status_columns = [
        'Shoulder Tilt', 'Hip Tilt', 'Shoulder Rotation', 
        'Hip Rotation', 'Lead Arm Angle'
    ]
    # DTL-only columns (no Status)
    dtl_only_columns = ['Forward Tilt', 'Knee Bend']
    
    # 3. Create the final combined table data
    p_class_pattern = re.compile(r'P(10|[1-9])$')
    final_combined_table_data = []
    
    # Filter FO data to only include P1 to P10 pose classes as the base structure
    # Note: video1_full_data already has 'None' strings instead of nan
    fo_base_rows = [
        row.copy() for row in video1_full_data
        if p_class_pattern.match(row.get('Pose Class', ''))
    ]

    for fo_row in fo_base_rows:
        pose_class = fo_row.get('Pose Class', '')
        dtl_row = dtl_data_map.get(pose_class, {})
        new_row = {'camera_view': 'Combined'}

        # Copy over base FO columns and DTL time reference
        new_row['Time Frame'] = fo_row.get('Time Frame')
        new_row['Pose Class'] = pose_class
        new_row['Time Frame'] = dtl_row.get('Time Frame', '')

        # Iterate through the columns that need sourcing
        for col_name in angle_status_columns:
            status_col_name = f'{col_name} Status'
            source = get_data_source(col_name, pose_class)
            
            value = None
            status = None

            if col_name == 'Lead Arm Angle' and pose_class == 'P10':
                # Special case: P10 Lead Arm Angle is '-' and has no status
                value = '-'
                status = ''
                # (Removed 'continue', will be assigned below)

            elif source == 'FO':
                # Use value and status from FO data
                value = fo_row.get(col_name)
                status = fo_row.get(status_col_name)
            elif source == 'DTL':
                # Use value and status from DTL correlated data
                value = dtl_row.get(col_name)
                status = dtl_row.get(status_col_name)
            else:
                value = ''
                status = ''
                
            # --- [FIX ADDED] ---
            # If the final calculated value is '-', ensure its status is blank
            # so it doesn't render a badge.
            if value == '-':
                status = ''
            # --- [END FIX] ---

            # Since data was cleaned, default get() to 'None' if key missing
            new_row[col_name] = value if value is not None else 'None'
            new_row[status_col_name] = status if status is not None else ''
                
        # Handle DTL-only columns (Forward Tilt and Knee Bend - NO STATUS)
        for col_name in dtl_only_columns:
            # Default get() to 'None' if key missing
            new_row[col_name] = dtl_row.get(col_name, 'None')
            
        final_combined_table_data.append(new_row)

    # 4. Final column definition and status mapping for the template
    display_columns_combined = [
        'Time Frame', 'Pose Class', 'Time Frame', 
        'Shoulder Tilt', 'Shoulder Tilt Status', 
        'Hip Tilt', 'Hip Tilt Status', 
        'Shoulder Rotation', 'Shoulder Rotation Status', 
        'Hip Rotation', 'Hip Rotation Status', 
        'Lead Arm Angle', 'Lead Arm Angle Status', 
        'Forward Tilt', 'Knee Bend'
    ]
    
    column_status_mapping = {}
    # Use the combined columns list to build the status mapping
    for col in display_columns_combined:
        if col.endswith(' Status'):
            angle_col = col.replace(" Status", "")
            column_status_mapping[angle_col] = col
            
    # --- [START NEW LOGIC: Set template data based on upload type] ---
    if is_dual_upload:
        final_data_for_template = final_combined_table_data
        final_columns_for_template = display_columns_combined # The combined columns
    else:
        final_data_for_template = video1_full_data
        final_columns_for_template = video1_columns # The original columns from video 1
        
        # Build status mapping from single video columns if not dual upload
        # (This ensures the badge coloring still works for single uploads)
        for col in final_columns_for_template:
            if col.endswith(' Status'):
                angle_col = col.replace(" Status", "")
                column_status_mapping[angle_col] = col
    # --- [END NEW LOGIC] ---

    # Fetch pose classification image URL (from primary video)
    video_status_info = Video.get_video_status(video1_id)
    pose_class_images_url = video_status_info.get('poseClassImagesLink') if video_status_info else None

    # Fetch and process comments
    comments_data = Video.get_all_video_comments(video1_id, request.session['Id'])
    processed_comments_data = convert_objectids_to_str(comments_data)

    # Final Context
    return render(request, 'dashboard_results.html', {
        'Role': user['Role'],
        'Name': user['Name'],
        'studentID': id,
        'videoId': video1_id,
        'comments': processed_comments_data,
        'current_user_name': current_user_name,
        
        # --- Conditional View Data ---
        'is_dual_upload': is_dual_upload, # Flag for template switch
        
        # Video 1 Data (Primary Source / Left Video)
        'video1_title': video1_title,
        'video1_url': video1_url,
        # Dump the *original* FO data for JS charts/processing
        'video1_full_data': json.dumps(video1_full_data), 
        
        # Video 2 Data (DTL / Right Video)
        'video2_title': video2_title,
        'video2_url': video2_url,
        # Dump the *original* DTL data for JS charts/processing
        'video2_full_data': json.dumps(video2_full_data), 
        
        # Table/Chart Data
        'columns': final_columns_for_template, # Columns (Combined or Single)
        'column_status_mapping': column_status_mapping, 
        'full_data': final_data_for_template, # Data (Combined or Single)
        
        # Single Video Fallback Data (for the 'else' branch)
        'video_title': video1_title, 
        'video_url': video1_url, 
        'pose_class_images_url': pose_class_images_url,
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
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=400)

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

# Add immediately after delete_video_reply_ajax

@csrf_exempt
@require_POST
def mark_comment_read(request, id, VideoId):
    """
    AJAX endpoint: mark a comment and all its direct replies as read
    by the current user (adds them to the comment's `readBy` array).
    """
    try:
        payload = json.loads(request.body)
        comment_id = payload.get('commentId')
        if not comment_id:
            return JsonResponse({'success': False, 'message': 'Missing commentId'}, status=400)

        user_id = request.session.get('Id')
        if not user_id:
            return JsonResponse({'success': False, 'message': 'User not authenticated'}, status=401)

        user_oid = ObjectId(user_id)
        cid = ObjectId(comment_id)

        # Mark the top-level comment read
        Comments_Collection.update_one(
            {'_id': cid},
            {'$addToSet': {'readBy': user_oid}}
        )
        # Also mark all direct replies read
        Comments_Collection.update_many(
            {'parent_comment_id': cid},
            {'$addToSet': {'readBy': user_oid}}
        )

        return JsonResponse({'success': True})
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'message': 'Invalid JSON'}, status=400)
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'success': False, 'message': str(e)}, status=500)

# New AJAX endpoint for checking a single video status
@csrf_exempt # Use this decorator if you're not sending CSRF tokens with the AJAX request
def check_video_status_ajax(request, id, video_id):
    """
    Checks the current processing status of a specific video.
    Expected URL: /dataSpace/<student_id>/check_video_status_ajax/<video_id>/
    """
    if request.method == 'GET':
        try:
            # Convert the video_id string from the URL to a MongoDB ObjectId
            video_obj_id = ObjectId(video_id)
            
            # Find the video document in the Videos_Collection
            video_doc = Videos_Collection.find_one({"_id": video_obj_id})
            
            if video_doc:
                # Get the 'Status' field from the video document, default to 'Unknown' if not found
                status = video_doc.get('Status', 'Unknown')
                raw_video_link = None
                processed_video_link = None

                # If the video is completed, include the raw and processed video links
                if status == 'Completed':
                    # Assuming 'rawVideoLink' and 'processedVideoLink' fields exist in your MongoDB document
                    # and store the URLs directly. Adjust field names if they are different.
                    raw_video_link = video_doc.get('rawVideoLink') 
                    processed_video_link = video_doc.get('processedVideoLink') # Include if you use it on frontend

                return JsonResponse({
                    'status': 'success', # Indicate overall success of the AJAX call
                    'video_status': status, # The actual video processing status
                    'rawVideoLink': raw_video_link,
                    'processedVideoLink': processed_video_link # Include if your frontend needs it
                })
            else:
                # Return a 404 error if the video is not found
                return JsonResponse({'status': 'error', 'message': 'Video not found'}, status=404)
        except Exception as e:
            # Log the full traceback for debugging purposes
            traceback.print_exc()
            # Return a 500 server error if something goes wrong
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    # Return a 405 error if the request method is not GET
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

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

    # Titles and URLs for student videos
    video1_title = video1_item.get('Title', 'Untitled Video 1')
    video2_title = video2_item.get('Title', 'Untitled Video 2')
    video1_url   = Video.get_video_url(video1_id)
    video2_url   = Video.get_video_url(video2_id)

    # Helper function to fetch CSV data and return as JSON string
    def fetch_csv_data_json(vid):
        url = Video.get_csv_url(vid)
        resp = requests.get(url)
        if resp.status_code == 200:
            df = pd.read_csv(StringIO(resp.content.decode('utf-8')))
            return json.dumps(df.to_dict('records'))
        return json.dumps([])

    video1_full_data_json = fetch_csv_data_json(video1_id)
    video2_full_data_json = fetch_csv_data_json(video2_id)

    # --- Fetch Pro Golfer Video Data ---
    pro_video_title = "Pro Golfer Swing"
    pro_video_url = "" # Placeholder, will be updated
    pro_video_full_data_json = json.dumps([]) # Default to empty array

    try:
        # Assuming you have a `Video` class or similar for handling GCS
        # And a method to get a random pro video URL and its corresponding CSV URL
        pro_video_blob_name = Video.get_random_pro_video_blob_name() # You'll need to implement this
        if pro_video_blob_name:
            pro_video_url = Video.get_pro_video_url_from_blob(pro_video_blob_name) # Implement this
            pro_csv_url = Video.get_pro_csv_url_from_blob(pro_video_blob_name) # Implement this

            # Fetch CSV data for the pro video
            resp_pro = requests.get(pro_csv_url)
            if resp_pro.status_code == 200:
                df_pro = pd.read_csv(StringIO(resp_pro.content.decode('utf-8')))
                pro_video_full_data_json = json.dumps(df_pro.to_dict('records'))
            else:
                print(f"Warning: Could not fetch CSV for pro video from {pro_csv_url}. Status code: {resp_pro.status_code}")
        else:
            print("Warning: No random pro video blob name found.")

    except Exception as e:
        print(f"Error fetching pro golfer video data: {e}")
        # In a production environment, you might want to log this error
        # and provide a user-friendly fallback.

    # Current user (for base template)
    user = User.find_user_by_id(ObjectId(request.session['Id']))
    processed_user = convert_objectids_to_str(user)

    context = {
        'Role': processed_user['Role'],
        'Name': processed_user['Name'],
        'studentID': id,
        'video1_id': video1_id,
        'video1_title': video1_title,
        'video1_url': video1_url,
        'video1_full_data': video1_full_data_json,
        'video2_id': video2_id,
        'video2_title': video2_title,
        'video2_url': video2_url,
        'video2_full_data': video2_full_data_json,
        'pro_video_title': pro_video_title, # Pass pro video title
        'pro_video_url': pro_video_url,     # Pass pro video URL
        'pro_video_full_data': pro_video_full_data_json, # Pass pro video data
    }
    return render(request, 'dashboard_compareSwings.html', context)

def dashboard_Coach(request):
    """Displays the Coach dashboard with associated students."""
    if not is_logged_in(request) or not isCoach(request):
        return redirect('home')

    view = request.GET.get('view', 'list')
    user = convert_objectids_to_str(User.find_user_by_id(ObjectId(request.session['Id'])))
    students = convert_objectids_to_str(fetch_all_students(request.session['Id']))

    # --- MODIFIED: Handle POST requests for uploads from the coach dashboard ---
    if request.method == 'POST':
        video_type = request.POST.get('videoType')
        current_user_id = request.session.get('Id')
        assignee_id = request.POST.get('student_id')

        if not assignee_id:
            print("Coach upload error: No student was selected.")
            return HttpResponseRedirect(reverse('home'))

        if video_type == 'both':
            face_on_file = request.FILES.get('face_on_file')
            dtl_file = request.FILES.get('dtl_file')
            if face_on_file and dtl_file:
                Video.upload_dual_videos(
                    current_user_id=current_user_id,
                    assignee_id=assignee_id,
                    face_on_file=face_on_file,
                    dtl_file=dtl_file
                )
                print(f"Dual video upload initiated by coach {current_user_id} for student {assignee_id}.")
            else:
                print("Coach dual upload error: one or both files were missing.")
        else:
            success, message = upload_single_video_helper(request, current_user_id, assignee_id)
            print(f"Coach single upload status: {message}")

        return HttpResponseRedirect(reverse('home'))

    return render(request, 'dashboard_coach.html', {
        'Role': user['Role'], 'Name': user['Name'], 'students': students, 'view': view,
    })

def dashboard_admin(request):
    """Displays the Admin dashboard."""
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
    """Displays the model upload dashboard."""
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
    """Handles video upload - UPDATED with upload_source parameter."""
    user_role = request.session.get('Role')
    if user_role not in ['student', 'coach']:
        return redirect('home')

    if user_role == 'student':
        video_file = request.FILES.get('videoDBFile')
        video_type = request.POST.get('videoType')
        if video_file and video_type:
            # UPDATED: Added upload_source="manual"
            Video.upload_video(
                request.session['Id'], 
                request.session['Id'], 
                video_file.name, 
                video_type, 
                video_file, 
                upload_source="manual"
            )
        return redirect('home')

    if user_role == 'coach':
        video_file = request.FILES.get('videoDBFile')
        video_type = request.POST.get('videoType')
        if '/home/dataSpace/' in request.path and video_file and video_type:
            # UPDATED: Added upload_source="manual"
            Video.upload_video(
                request.session['Id'], 
                student_id, 
                video_file.name, 
                video_type, 
                video_file, 
                upload_source="manual"
            )
            return redirect('home')

        student_id = request.POST.get('student_id')
        video_name = request.POST.get('fileValue')
        video_type = request.POST.get('videoType')
        video_file = request.FILES.get('videoFile')
        if student_id and video_name and video_type and video_file:
            # UPDATED: Added upload_source="manual"
            Video.upload_video(
                request.session['Id'], 
                student_id, 
                video_name, 
                video_type, 
                video_file, 
                upload_source="manual"
            )
        return redirect('home')

def logout(request):
    """Logs out the user and clears session data."""
    request.session.flush()
    return redirect('login')

# =============================================================================
# 📹 OLD CAMERA SYSTEM (Existing RPi Integration)
# =============================================================================

def live_stream(request):
    """Streaming response for old camera system live feed"""
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
def start_recording(request):
    """Start recording on the old camera system"""
    try:
        response = requests.post(f'{RASPBERRY_PI_URL}/start_recording')
        return JsonResponse(response.json() if response.ok else {"message": "Failed to start recording"}, status=response.status_code)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"message": f"Error: {str(e)}"}, status=500)

@csrf_exempt
def upload_from_pi(request):
    """Handle video upload from RPi camera system - UPDATED to use Video.upload_video."""
    if request.method == 'POST':
        try:
            # Get uploaded file
            uploaded_file = request.FILES.get('file')
            if not uploaded_file:
                return JsonResponse({"error": "No file provided"}, status=400)

            # Extract user context from request - RPi should send these
            operator_id = request.POST.get('operator_id')  # Who recorded it
            assignee_id = request.POST.get('assignee_id')  # Who it's for
            video_type = request.POST.get('video_type', 'face-on')
            
            if not operator_id:
                return JsonResponse({"error": "Missing operator_id"}, status=400)
            
            # Default assignee to operator if not specified
            if not assignee_id:
                assignee_id = operator_id
            
            # UPDATED: Use Video.upload_video for RPI uploads
            result = Video.upload_video(
                current_user_id=operator_id,
                assignee_id=assignee_id,
                title=uploaded_file.name,
                video_type=video_type,
                file=uploaded_file,
                upload_source="rpi"  # ← SPECIFY RPI SOURCE
            )
            
            return JsonResponse({
                "message": f"RPi video upload initiated: {uploaded_file.name}",
                "result": result
            }, status=200)
            
        except Exception as e:
            print(f"Error in RPi upload: {e}")
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)

# =============================================================================
# 🏌️ NEW GOLF CAMERA SYSTEM (Advanced AI Integration)
# =============================================================================

def golf_video_feed(request):
    """Proxy the live video feed from golf camera"""
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
    """Get golf camera status"""
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
            'duration': data_from_browser.get('duration', 10)
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
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=503)

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
    
    # =============================================================================
# 🏌️ DOWN-THE-LINE GOLF CAMERA SYSTEM (Additional Angle)
# =============================================================================

def dtl_video_feed(request):
    """Proxy the live video feed from down-the-line golf camera"""
    try:
        response = requests.get(f'{DTL_CAMERA_URL}/video_feed', stream=True)
        if response.status_code == 200:
            return StreamingHttpResponse(
                response.iter_content(chunk_size=1024),
                content_type='multipart/x-mixed-replace; boundary=frame'
            )
        else:
            return HttpResponse("Down-the-line camera feed unavailable", status=503)
    except requests.exceptions.RequestException as e:
        return HttpResponse(f"Down-the-line camera connection error: {e}", status=503)

def dtl_status(request):
    """Get down-the-line camera status"""
    try:
        response = requests.get(f'{DTL_CAMERA_URL}/recording_status', timeout=5)
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({
                'error': 'Down-the-line camera offline',
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
            'error': f'Down-the-line camera connection error: {str(e)}',
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
def dtl_start_recording(request):
    """Enhanced recording with assignee support for down-the-line camera"""
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
            'angle': 'down-the-line'  # Specify angle for this camera
        }
        
        # If assignee_id provided, include it
        assignee_id = data_from_browser.get('assignee_id')
        if assignee_id:
            payload['assignee_id'] = assignee_id
        
        # Forward to RPi
        response = requests.post(
            f'{DTL_CAMERA_URL}/start_recording',
            json=payload,
            timeout=5
        )
        
        if response.ok:
            return JsonResponse(response.json())
        else:
            return JsonResponse({'error': 'Failed to start down-the-line recording on RPi'}, status=response.status_code)
            
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def dtl_toggle_auto_recording(request):
    """Toggle auto recording on down-the-line camera"""
    try:
        response = requests.post(f'{DTL_CAMERA_URL}/toggle_auto_recording', timeout=5)
        
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({'error': 'Failed to toggle auto recording'}, status=503)
            
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': f'Down-the-line camera error: {str(e)}'}, status=503)

@csrf_exempt
@require_http_methods(["POST"])
def dtl_toggle_pose_detection(request):
    """Toggle pose detection on down-the-line camera"""
    try:
        response = requests.post(f'{DTL_CAMERA_URL}/toggle_pose_detection', timeout=5)
        
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({'error': 'Failed to toggle pose detection'}, status=503)
            
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': f'Down-the-line camera error: {str(e)}'}, status=503)

@csrf_exempt
@require_http_methods(["POST"])
def dtl_reload_models(request):
    """Reload AI models on down-the-line camera"""
    try:
        response = requests.post(f'{DTL_CAMERA_URL}/reload_models', timeout=30)
        
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({'error': 'Failed to reload models'}, status=503)
            
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': f'Down-the-line camera error: {str(e)}'}, status=503)
    
@csrf_exempt
@require_http_methods(["POST"])
def dtl_set_user_context(request):
    """Enhanced user context setting with assignee support for down-the-line camera"""
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
            'role': user_role,
            'angle': 'down-the-line'  # Specify angle
        }
        
        # If assignee_id provided, include it
        if assignee_id:
            payload['assignee_id'] = assignee_id
        
        # Send to RPi
        response = requests.post(
            f'{DTL_CAMERA_URL}/set_user_context',
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            response_data = response.json()
            return JsonResponse({
                'status': 'success', 
                'operator_id': user_id_str,
                'assignee_id': response_data.get('assignee_id', user_id_str),
                'role': user_role,
                'angle': 'down-the-line'
            })
        else:
            return JsonResponse({'error': 'Failed to set user context on down-the-line RPi'}, status=503)
            
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': f'Down-the-line RPi connection error: {str(e)}'}, status=503)
    except Exception as e:
        return JsonResponse({'error': f'Server error: {str(e)}'}, status=500)

def dtl_health(request):
    """Check down-the-line camera health"""
    try:
        response = requests.get(f'{DTL_CAMERA_URL}/recording_status', timeout=3)
        
        if response.status_code == 200:
            return JsonResponse({
                'status': 'connected',
                'dtl_camera_ip': '172.20.10.5',
                'dtl_camera_url': DTL_CAMERA_URL,
                'timestamp': timezone.now().isoformat(),
                'camera_data': response.json()
            })
        else:
            return JsonResponse({
                'status': 'error',
                'dtl_camera_ip': '172.20.10.5',
                'dtl_camera_url': DTL_CAMERA_URL,
                'error': f'HTTP {response.status_code}',
                'timestamp': timezone.now().isoformat()
            })
            
    except requests.exceptions.RequestException as e:
        return JsonResponse({
            'status': 'disconnected',
            'dtl_camera_ip': '172.20.10.5',
            'dtl_camera_url': DTL_CAMERA_URL,
            'error': str(e),
            'timestamp': timezone.now().isoformat()
        })