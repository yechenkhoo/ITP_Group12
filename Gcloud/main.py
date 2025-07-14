import functions_framework
import requests
import json
from google.cloud import storage
import os
from datetime import datetime
from pymongo import MongoClient
import traceback
from bson import ObjectId

# MongoDB setup
MONGO_URI = "mongodb+srv://ITP_AUTH:SXgJ9MGaEVBKZmN@itpteam13.wtajb.mongodb.net/"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["ITP"]
collection = db["Videos"]
users_collection = db["Users"]

def find_user_by_objectid(user_id_str):
    """Find user by ObjectId extracted from filename"""
    if not user_id_str:
        return None
        
    try:
        # Convert the string from the filename into a real ObjectId
        uploaded_by_oid = ObjectId(user_id_str)
        
        # Search the database for a document where the '_id' field matches
        user_doc = users_collection.find_one({"_id": uploaded_by_oid})
        
        if user_doc:
            print(f"Successfully found user '{user_doc.get('Name')}' by ID: {user_id_str}")
            return user_doc
        else:
            print(f"User with ID '{user_id_str}' not found in the database.")
            return None
            
    except Exception as e:
        print(f"Error looking up user by ID in MongoDB: {e}")
        return None

@functions_framework.cloud_event
def process_uploaded_video(cloud_event):
    try:
        data = cloud_event.data
        bucket_name = data['bucket']
        file_name = data['name']

        print(f"Processing file: {file_name} in bucket: {bucket_name}")

        # Check if file is in the golf_videos folder (old structure)
        if not file_name.startswith('golf_videos/'):
            print(f"Ignoring file {file_name} - not in golf_videos folder")
            return "OK"

        # Check if it's a video file
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        if not any(file_name.lower().endswith(ext) for ext in video_extensions):
            print(f"Ignoring file {file_name} - not a video file")
            return "OK"

        print(f"Processing video: {file_name} in bucket: {bucket_name}")

        # Extract filename and video ID
        video_filename = os.path.basename(file_name)
        video_id = os.path.splitext(video_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- PARSE USER ID FROM FILENAME (using ObjectId approach from main(5).py) ---
        user_id_str = None
        uploaded_by = None
        assignee = None
        
        if "_" in video_filename:
            user_id_str = video_filename.split("_")[0]
            print(f"Extracted user ID string: {user_id_str}")
            
            user_doc = find_user_by_objectid(user_id_str)
            if user_doc:
                uploaded_by = user_doc.get("_id")
                assignee = user_doc.get("CreatedBy")
                print(f"Found user ID: {uploaded_by}, Assignee (Coach) ID: {assignee}")
            else:
                print(f"User not found with ID: {user_id_str}")
        else:
            print(f"No underscore in video_filename: {video_filename}, cannot extract user ID")

        # Generate output paths
        output_video_path = f"processed/{video_id}_output_{timestamp}.mp4"
        output_csv_path = f"processed/{video_id}_output_{timestamp}.csv"
        output_angle_csv_path = f"processed/{video_id}_angles_{timestamp}.csv"

        # API payload
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

            # MongoDB operations
            try:
                # Look for existing video record
                query_filter = {"Title": video_filename}
                
                # Add UploadedBy to query only if we have a valid user
                if uploaded_by is not None:
                    query_filter["UploadedBy"] = uploaded_by
                    print(f"Searching for existing video with title: {video_filename} and UploadedBy: {uploaded_by}")
                else:
                    print(f"Searching for existing video with title: {video_filename} (no user found)")
                
                existing = collection.find_one(query_filter)

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
                                "LastUpdated": datetime.now().strftime("%H:%M %b %d, %Y"),
                                "originalVideoPath": file_name,
                                "processedTimestamp": datetime.now().isoformat()
                            }
                        }
                    )
                    print(f"Update result: {update_result.modified_count} documents modified")
                    print(f"Updated existing record for {video_filename}")
                else:
                    print("No existing record found, creating new document")
                    
                    # Check if there's any video with same title (different user)
                    any_existing = collection.find_one({"Title": video_filename})
                    if any_existing:
                        print(f"Warning: Found video with same title but different user: {any_existing.get('UploadedBy')}")
                    
                    new_doc = {
                        "Title": video_filename,
                        "Type": "face-on",  # Default type for old structure
                        "DateUploaded": datetime.now().strftime("%H:%M %b %d, %Y"),
                        "Status": "Completed",
                        "UploadedBy": uploaded_by,
                        "Assignee": assignee,
                        "angleCsvLink": angle_link,
                        "frameByFrameCsvLink": frame_csv_link,
                        "processedVideoLink": processed_video,
                        "originalVideoPath": file_name,
                        "processedTimestamp": datetime.now().isoformat()
                    }
                    print(f"Inserting document: {json.dumps(new_doc, indent=2, default=str)}")
                    
                    insert_result = collection.insert_one(new_doc)
                    print(f"Insert result: {insert_result.inserted_id}")
                    print(f"Successfully inserted new record for {video_filename}")

            except Exception as mongo_error:
                print(f"MongoDB operation failed: {str(mongo_error)}")
                print(f"Traceback: {traceback.format_exc()}")

                
        else:
            print(f"API request failed with status {response.status_code}: {response.text}")
            
            # Handle failed API calls
            try:
                # Look for existing record to update with error
                query_filter = {"Title": video_filename}
                if uploaded_by is not None:
                    query_filter["UploadedBy"] = uploaded_by

                existing = collection.find_one(query_filter)
                
                if existing:
                    # Update existing record with error status
                    collection.update_one(
                        {"_id": existing["_id"]},
                        {
                            "$set": {
                                "Status": "Failed",
                                "Error": f"API failed with status {response.status_code}",
                                "LastUpdated": datetime.now().strftime("%H:%M %b %d, %Y"),
                                "ErrorDetails": response.text[:500] if response.text else "Unknown error",
                                "originalVideoPath": file_name
                            }
                        }
                    )
                    print(f"Updated existing record with error status for {video_filename}")
                else:
                    # Create new record with failed status
                    new_doc = {
                        "Title": video_filename,
                        "Type": "face-on",
                        "DateUploaded": datetime.now().strftime("%H:%M %b %d, %Y"),
                        "Status": "Failed",
                        "Error": f"API failed with status {response.status_code}",
                        "ErrorDetails": response.text[:500] if response.text else "Unknown error",
                        "UploadedBy": uploaded_by,
                        "Assignee": assignee,
                        "originalVideoPath": file_name
                    }
                    collection.insert_one(new_doc)
                    print(f"Inserted failed record for {video_filename}")
                    
            except Exception as mongo_error:
                print(f"Failed to handle error record: {str(mongo_error)}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Try to create/update record with network error status
        try:
            video_filename = os.path.basename(file_name)
            
            existing = collection.find_one({"Title": video_filename})
            if existing:
                collection.update_one(
                    {"_id": existing["_id"]},
                    {
                        "$set": {
                            "Status": "Failed",
                            "Error": "Network error calling ML API",
                            "ErrorDetails": str(e),
                            "LastUpdated": datetime.now().strftime("%H:%M %b %d, %Y")
                        }
                    }
                )
                print(f"Updated record with network error for {video_filename}")
        except Exception as final_error:
            print(f"Final error handling failed: {str(final_error)}")
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

    return "OK"
