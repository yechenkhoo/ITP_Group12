import functions_framework
import requests
import json
from google.cloud import storage
import os
from datetime import datetime
from pymongo import MongoClient

# MongoDB setup
MONGO_URI = "mongodb+srv://ITP_AUTH:SXgJ9MGaEVBKZmN@itpteam13.wtajb.mongodb.net/"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["ITP_GROUP_13"]
collection = db["videos"]  # or the correct collection you're using

@functions_framework.cloud_event
def process_uploaded_video(cloud_event):
    data = cloud_event.data
    bucket_name = data['bucket']
    file_name = data['name']

    if not file_name.startswith('golf_videos/'):
        print(f"Ignoring file {file_name} - not in golf_videos folder")
        return

    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    if not any(file_name.lower().endswith(ext) for ext in video_extensions):
        print(f"Ignoring file {file_name} - not a video file")
        return

    print(f"Processing new video: {file_name} in bucket: {bucket_name}")

    video_filename = os.path.basename(file_name)
    video_id = os.path.splitext(video_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_video_path = f"processed/{video_id}_output_{timestamp}.mp4"
    output_csv_path = f"processed/{video_id}_output_{timestamp}.csv"
    output_angle_csv_path = f"processed/{video_id}_angles_{timestamp}.csv"

    api_url = "https://ml-model-api-1067172605110.asia-southeast1.run.app/process-video"

    payload = {
        "video_id": video_id,
        "video_path": file_name,
        "classification_model": "basemodel.keras",
        "output_video_path": output_video_path,
        "output_csv_path": output_csv_path,
        "output_angle_csv_path": output_angle_csv_path
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        print(f"Calling API with payload: {json.dumps(payload, indent=2)}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=300)

        if response.status_code == 200:
            print(f"Successfully processed video {video_filename}")
            response_data = response.json()

            # You need a way to identify the user. For now assume it's embedded in video_id like user123_video_2025...
            user_id = video_id.split("_")[0]  # adjust as needed

            # Store into MongoDB
            document = {
                "user_id": user_id,
                "video_id": video_id,
                "uploaded_filename": video_filename,
                "processed_video_url": response_data.get("output_video"),
                "processed_csv_url": response_data.get("output_csv"),
                "processed_angle_csv_url": response_data.get("output_angle_csv"),
                "timestamp": datetime.now()
            }
            collection.insert_one(document)
            print("Saved processed metadata to MongoDB")

        else:
            print(f"API request failed with status {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

    return "OK"
