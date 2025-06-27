import functions_framework
import requests
import json
from google.cloud import storage
import os
from datetime import datetime

@functions_framework.cloud_event
def process_uploaded_video(cloud_event):
    """
    Cloud Function triggered when a new video is uploaded to the specified bucket.
    """
    
    # Extract event data
    data = cloud_event.data
    bucket_name = data['bucket']
    file_name = data['name']
    
    # Only process files in the golf_videos folder
    if not file_name.startswith('golf_videos/'):
        print(f"Ignoring file {file_name} - not in golf_videos folder")
        return
    
    # Only process video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    if not any(file_name.lower().endswith(ext) for ext in video_extensions):
        print(f"Ignoring file {file_name} - not a video file")
        return
    
    print(f"Processing new video: {file_name} in bucket: {bucket_name}")
    
    # Extract just the filename without the folder path
    video_filename = os.path.basename(file_name)
    video_id = os.path.splitext(video_filename)[0]  # Remove extension for ID
    
    # Generate output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = f"processed/{video_id}_output_{timestamp}.mp4"
    output_csv_path = f"processed/{video_id}_output_{timestamp}.csv"
    output_angle_csv_path = f"processed/{video_id}_angles_{timestamp}.csv"
    
    # Prepare the API request
    api_url = "https://ml-model-api-1067172605110.asia-southeast1.run.app/process-video"
    
    payload = {
        "video_id": video_id,
        "video_path": file_name,  # Use just the filename
        "classification_model": "basemodel.keras",
        "output_video_path": output_video_path,
        "output_csv_path": output_csv_path,
        "output_angle_csv_path": output_angle_csv_path
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Make the API request
        print(f"Calling API with payload: {json.dumps(payload, indent=2)}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=300)
        
        if response.status_code == 200:
            print(f"Successfully processed video {video_filename}")
            print(f"Response: {response.text}")
        else:
            print(f"API request failed with status {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    
    return "OK"