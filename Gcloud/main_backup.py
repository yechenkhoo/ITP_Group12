import functions_framework
import requests
import json
from google.cloud import storage
import os
from datetime import datetime
from pymongo import MongoClient
import traceback

# MongoDB setup
MONGO_URI = "mongodb+srv://ITP_AUTH:SXgJ9MGaEVBKZmN@itpteam13.wtajb.mongodb.net/"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["ITP"]
collection = db["Videos"]
users_collection = db["Users"]

@functions_framework.cloud_event
def process_uploaded_video(cloud_event):
    try:
        data = cloud_event.data
        bucket_name = data['bucket']
        file_name = data['name']

        print(f"Processing file: {file_name} in bucket: {bucket_name}")

        if not file_name.startswith('golf_videos/'):
            print(f"Ignoring file {file_name} - not in golf_videos folder")
            return "OK"

        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        if not any(file_name.lower().endswith(ext) for ext in video_extensions):
            print(f"Ignoring file {file_name} - not a video file")
            return "OK"

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

        print(f"Calling API with payload: {json.dumps(payload, indent=2)}")
        
        # API Call
        response = requests.post(api_url, headers=headers, json=payload, timeout=300)

        if response.status_code == 200:
            print(f"Successfully processed video {video_filename}")
            response_data = response.json()
            print(f"API Response: {json.dumps(response_data, indent=2)}")

            angle_link = response_data.get("output_angle_csv")
            frame_csv_link = response_data.get("output_csv")
            processed_video = response_data.get("output_video")

            # Enhanced user lookup with better error handling
            user_name = None
            uploaded_by = None
            assignee = None
            
            try:
                # Extract user ID from filename (e.g., nick_video1.mp4 → "nick")
                if "_" in video_id:
                    user_name = video_id.split("_")[0]
                    print(f"Extracted user name: {user_name}")
                    
                    # Try to find user - check what field names actually exist
                    user = users_collection.find_one({"Name": user_name})
                    if not user:
                        # Try alternative field names
                        user = users_collection.find_one({"name": user_name})
                    if not user:
                        user = users_collection.find_one({"username": user_name})
                    
                    if user:
                        print(f"Found user: {user}")
                        uploaded_by = user.get("_id")
                        assignee = user.get("CreatedBy")
                    else:
                        print(f"No user found with name: {user_name}")
                        # Let's see what users actually exist
                        sample_users = list(users_collection.find().limit(3))
                        print(f"Sample users in database: {sample_users}")
                else:
                    print(f"No underscore in video_id: {video_id}, cannot extract user name")
                    
            except Exception as user_error:
                print(f"Error during user lookup: {str(user_error)}")
                print(f"Traceback: {traceback.format_exc()}")

            # MongoDB operations with error handling
            try:
                # Check if video already exists
                existing = collection.find_one({"Title": video_filename})
                print(f"Checking for existing video with title: {video_filename}")

                if existing:
                    print(f"Found existing record: {existing['_id']}")
                    update_result = collection.update_one(
                        {"_id": existing["_id"]},
                        {
                            "$set": {
                                "angleCsvLink": angle_link,
                                "frameByFrameCsvLink": frame_csv_link,
                                "processedVideoLink": processed_video,
                                "Status": "Completed",
                                "LastUpdated": datetime.now().strftime("%H:%M %b %d, %Y")
                            }
                        }
                    )
                    print(f"Update result: {update_result.modified_count} documents modified")
                    print(f"Updated existing record for {video_filename}")
                else:
                    print("No existing record found, creating new document")
                    new_doc = {
                        "Title": video_filename,
                        "Type": "face-on",
                        "DateUploaded": datetime.now().strftime("%H:%M %b %d, %Y"),
                        "Status": "Completed",
                        "UploadedBy": uploaded_by,
                        "Assignee": assignee,
                        "angleCsvLink": angle_link,
                        "frameByFrameCsvLink": frame_csv_link,
                        "processedVideoLink": processed_video
                    }
                    print(f"Inserting document: {json.dumps(new_doc, indent=2, default=str)}")
                    
                    insert_result = collection.insert_one(new_doc)
                    print(f"Insert result: {insert_result.inserted_id}")
                    print(f"Successfully inserted new record for {video_filename}")

            except Exception as mongo_error:
                print(f"MongoDB operation failed: {str(mongo_error)}")
                print(f"Traceback: {traceback.format_exc()}")
                # Continue execution even if MongoDB fails
                
        else:
            print(f"API request failed with status {response.status_code}: {response.text}")
            
            # Still try to create a record with failed status
            try:
                new_doc = {
                    "Title": video_filename,
                    "Type": "face-on",
                    "DateUploaded": datetime.now().strftime("%H:%M %b %d, %Y"),
                    "Status": "Failed",
                    "Error": f"API failed with status {response.status_code}",
                    "UploadedBy": None,
                    "Assignee": None
                }
                collection.insert_one(new_doc)
                print(f"Inserted failed record for {video_filename}")
            except Exception as mongo_error:
                print(f"Failed to insert error record: {str(mongo_error)}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

    return "OK"
