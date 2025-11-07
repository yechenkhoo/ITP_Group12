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
        user_oid = ObjectId(user_id_str)
        
        # Search the database for a document where the '_id' field matches
        user_doc = users_collection.find_one({"_id": user_oid})
        
        if user_doc:
            print(f"Found user '{user_doc.get('Name')}' (Role: {user_doc.get('Role')}) by ID: {user_id_str}")
            return user_doc
        else:
            print(f"User with ID '{user_id_str}' not found in the database.")
            return None
            
    except Exception as e:
        print(f"Error looking up user by ID in MongoDB: {e}")
        return None

def generate_raw_video_link(file_path, bucket_name):
    """Generate the public URL for the raw video in Google Cloud Storage"""
    try:
        # Create the public URL format for Google Cloud Storage
        raw_video_url = f"https://storage.googleapis.com/{bucket_name}/{file_path}"
        print(f"Generated raw video link: {raw_video_url}")
        return raw_video_url
    except Exception as e:
        print(f"Error generating raw video link: {e}")
        return None

@functions_framework.cloud_event
def process_uploaded_video(cloud_event):
    try:
        data = cloud_event.data
        bucket_name = data['bucket']
        file_name = data['name']

        print(f"Processing file: {file_name} in bucket: {bucket_name}")

        # Check if file is in a valid video folder (support both structures)
        valid_folders = ['golf_videos/', 'dtl_videos/']
        if not any(file_name.startswith(folder) for folder in valid_folders):
            print(f"Ignoring file {file_name} - not in a valid video folder: {valid_folders}")
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

        print(f"Video filename: {video_filename}")
        print(f"Video ID: {video_id}")

        # Generate raw video link
        raw_video_link = generate_raw_video_link(file_name, bucket_name)

        # Determine video type based on folder
        video_type = "down-the-line" if file_name.startswith('dtl_videos/') else "face-on"
        print(f"Video type: {video_type}")

        # Parse both operator and assignee from filename
        user_id_str = None
        assignee_id_str = None
        uploaded_by = None
        assignee = None
        
        # Parse filename to extract user info
        filename_parts = video_filename.split("_")
        print(f"Parsing filename: {video_filename}")
        print(f"Filename parts: {filename_parts}")

        # Extract session_id from filename
        # Format: ...swing_{date}_{time}.mp4 -> session_id = {date}_{time}
        session_id = None
        try:
            # Find the index of "swing" in filename_parts
            if "swing" in filename_parts:
                swing_index = filename_parts.index("swing")
                # Session ID is the parts after "swing" (date and time), excluding the file extension
                if len(filename_parts) > swing_index + 2:
                    # Get date and time parts, remove .mp4 extension from the last part
                    date_part = filename_parts[swing_index + 1]
                    time_part = filename_parts[swing_index + 2].split('.')[0]  # Remove file extension
                    session_id = f"{date_part}_{time_part}"
                    print(f"Extracted session_id: {session_id}")
        except Exception as e:
            print(f"Could not extract session_id: {e}")

        # Determine if this has camera_id in the filename
        if len(filename_parts) >= 5 and filename_parts[3] == "swing":
            # NEW FORMAT with different operator/assignee: 
            # {operator_id}_{assignee_id}_{camera_id}_swing_{timestamp}.mp4
            user_id_str = filename_parts[0]      # Operator (who recorded)
            assignee_id_str = filename_parts[1]  # Assignee (who it's for)
            camera_id = filename_parts[2]        # Camera type (face-on or down-line)
            
            print(f"Enhanced format detected (5 parts):")
            print(f"   Operator ID: {user_id_str}")
            print(f"   Assignee ID: {assignee_id_str}")
            print(f"   Camera ID: {camera_id}")
            
            # Get operator info
            operator_doc = find_user_by_objectid(user_id_str)
            if operator_doc:
                uploaded_by = operator_doc.get("_id")
                print(f"   Operator found: {operator_doc.get('Name')} ({operator_doc.get('Role')})")
            else:
                print(f"   Operator not found: {user_id_str}")
                
            # Get assignee info
            assignee_doc = find_user_by_objectid(assignee_id_str)
            if assignee_doc:
                assignee = assignee_doc.get("_id")
                print(f"   Assignee found: {assignee_doc.get('Name')} ({assignee_doc.get('Role')})")
            else:
                print(f"   Assignee not found: {assignee_id_str}")
                
        elif len(filename_parts) >= 4 and filename_parts[2] == "swing":
            # NEW FORMAT with same operator/assignee: 
            # {user_id}_{camera_id}_swing_{timestamp}.mp4
            user_id_str = filename_parts[0]      # User ID
            camera_id = filename_parts[1]        # Camera type (face-on or down-line)
            
            print(f"Enhanced format detected (4 parts - same user):")
            print(f"   User ID: {user_id_str}")
            print(f"   Camera ID: {camera_id}")
            
            user_doc = find_user_by_objectid(user_id_str)
            if user_doc:
                uploaded_by = user_doc.get("_id")
                assignee = user_doc.get("_id")  # Same person
                print(f"   User found: {user_doc.get('Name')} (recording for themselves)")
            else:
                print(f"   User not found: {user_id_str}")
                
        elif len(filename_parts) >= 3 and filename_parts[1] == "swing":
            # OLD FORMAT (backward compatibility): {user_id}_swing_{timestamp}.mp4
            user_id_str = filename_parts[0]
            print(f"Old format detected - User ID: {user_id_str}")
            
            user_doc = find_user_by_objectid(user_id_str)
            if user_doc:
                uploaded_by = user_doc.get("_id")
                assignee = user_doc.get("_id")  # Same person
                print(f"   User found: {user_doc.get('Name')} (old format)")
            else:
                print(f"   User not found: {user_id_str}")
        else:
            print(f"Unknown filename format: {video_filename}")
            print(f"   Expected formats:")
            print(f"     1. {{user_id}}_{{camera_id}}_swing_{{timestamp}}.mp4")
            print(f"     2. {{operator_id}}_{{assignee_id}}_{{camera_id}}_swing_{{timestamp}}.mp4")
            print(f"     3. {{user_id}}_swing_{{timestamp}}.mp4 (old format)")

        # Generate output paths
        output_video_path = f"processed/{video_id}_output_{timestamp}.mp4"
        output_csv_path = f"processed/{video_id}_output_{timestamp}.csv"
        output_angle_csv_path = f"processed/{video_id}_angles_{timestamp}.csv"

        # API payload
        api_url = "https://ml-model-api-1067172605110.asia-southeast1.run.app/process-video"

        payload = {
            "video_id": video_id,
            "bucket_name": bucket_name,
            "video_path": file_name,
            "classification_model": "best_model.keras",
            "output_video_path": output_video_path,
            "output_csv_path": output_csv_path,
            "output_angle_csv_path": output_angle_csv_path
        }

        headers = {
            "Content-Type": "application/json"
        }

        print(f"Calling ML API with payload:")
        print(json.dumps(payload, indent=2))
        
        # API Call
        response = requests.post(api_url, headers=headers, json=payload, timeout=300)

        if response.status_code == 200:
            print(f"Successfully processed video {video_filename}")
            response_data = response.json()
            print(f"API Response:")
            print(json.dumps(response_data, indent=2))

            angle_link = response_data.get("output_angle_csv")
            frame_csv_link = response_data.get("output_csv")
            processed_video = response_data.get("output_video")

            # MongoDB operations
            try:
                print(f"Starting MongoDB operations...")
                
                # [MODIFIED] Query by title first
                query_filter = {"Title": video_filename}
                existing = collection.find_one(query_filter)
                
                if not existing and uploaded_by is not None:
                    query_filter["UploadedBy"] = uploaded_by
                    print(f"Searching for existing video:")
                    print(f"   Title: {video_filename}")
                    print(f"   UploadedBy: {uploaded_by}")
                    existing = collection.find_one(query_filter)
                elif not existing:
                    print(f"Searching for existing video:")
                    print(f"   Title: {video_filename}")
                    print(f"   No UploadedBy filter (user not found)")
                

                if existing:
                    print(f"Found existing record: {existing['_id']}")
                    print(f"   Current Status: {existing.get('Status', 'Unknown')}")
                    
                    # Update with raw video link
                    update_fields = {
                        "angleCsvLink": angle_link,
                        "frameByFrameCsvLink": frame_csv_link,
                        "processedVideoLink": processed_video,
                        "rawVideoLink": raw_video_link,
                        "Status": "Completed",
                        "LastUpdated": datetime.now().strftime("%H:%M %b %d, %Y"),
                        "originalVideoPath": file_name,
                        "processedTimestamp": datetime.now().isoformat(),
                        # "session_id": session_id <-- [REMOVED]
                    }
                    
                    # [NEW] Only update session_id if it was found in the filename
                    if session_id is not None:
                        update_fields["session_id"] = session_id

                    # Add assignee if we have new info
                    if assignee is not None:
                        update_fields["Assignee"] = assignee
                    
                    update_result = collection.update_one(
                        {"_id": existing["_id"]},
                        {"$set": update_fields}
                    )
                    print(f"Update result: {update_result.modified_count} documents modified")
                    print(f"Updated existing record for {video_filename}")
                    
                else:
                    print(f"No existing record found, creating new document")
                    
                    # Check if there's any video with same title (different user)
                    any_existing = collection.find_one({"Title": video_filename})
                    if any_existing:
                        print(f"Warning: Found video with same title but different user:")
                        print(f"   Existing UploadedBy: {any_existing.get('UploadedBy')}")
                        print(f"   Current UploadedBy: {uploaded_by}")
                    
                    # Include raw video link in new document
                    new_doc = {
                        "Title": video_filename,
                        "Type": video_type,
                        "DateUploaded": datetime.now().strftime("%H:%M %b %d, %Y"),
                        "Status": "Completed",
                        "UploadedBy": uploaded_by,
                        "Assignee": assignee,
                        "angleCsvLink": angle_link,
                        "frameByFrameCsvLink": frame_csv_link,
                        "processedVideoLink": processed_video,
                        "rawVideoLink": raw_video_link,
                        "originalVideoPath": file_name,
                        "processedTimestamp": datetime.now().isoformat(),
                        "session_id": session_id # This is fine (will be None)
                    }
                    
                    print(f"Inserting new document:")
                    print(json.dumps(new_doc, indent=2, default=str))
                    
                    insert_result = collection.insert_one(new_doc)
                    print(f"Insert result: {insert_result.inserted_id}")
                    print(f"Successfully inserted new record for {video_filename}")

                print(f"MongoDB operations completed successfully!")

            except Exception as mongo_error:
                print(f"MongoDB operation failed: {str(mongo_error)}")
                print(f"Traceback:")
                print(traceback.format_exc())

                
        else:
            print(f"API request failed with status {response.status_code}")
            print(f"Response text: {response.text}")
            
            # Handle failed API calls
            try:
                print(f"Handling failed API call...")
                
                # [MODIFIED] Query by title first
                query_filter = {"Title": video_filename}
                existing = collection.find_one(query_filter)
                if not existing and uploaded_by is not None:
                    query_filter["UploadedBy"] = uploaded_by
                    existing = collection.find_one(query_filter)

                
                if existing:
                    # Include raw video link even in failed records
                    update_fields = {
                        "Status": "Failed",
                        "Error": f"API failed with status {response.status_code}",
                        "LastUpdated": datetime.now().strftime("%H:%M %b %d, %Y"),
                        "ErrorDetails": response.text[:500] if response.text else "Unknown error",
                        "originalVideoPath": file_name,
                        "rawVideoLink": raw_video_link,
                        # "session_id": session_id <-- [REMOVED]
                    }
                    
                    # [NEW] Only update session_id if it was found in the filename
                    if session_id is not None:
                        update_fields["session_id"] = session_id
                        
                    collection.update_one(
                        {"_id": existing["_id"]},
                        {"$set": update_fields}
                    )
                    print(f"Updated existing record with error status for {video_filename}")
                else:
                    # Include raw video link in new failed record
                    new_doc = {
                        "Title": video_filename,
                        "Type": video_type,
                        "DateUploaded": datetime.now().strftime("%H:%M %b %d, %Y"),
                        "Status": "Failed",
                        "Error": f"API failed with status {response.status_code}",
                        "ErrorDetails": response.text[:500] if response.text else "Unknown error",
                        "UploadedBy": uploaded_by,
                        "Assignee": assignee,
                        "originalVideoPath": file_name,
                        "rawVideoLink": raw_video_link,
                        "session_id": session_id # This is fine (will be None)
                    }
                    collection.insert_one(new_doc)
                    print(f"Inserted failed record for {video_filename}")
                    
            except Exception as mongo_error:
                print(f"Failed to handle error record: {str(mongo_error)}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {str(e)}")
        print(f"Traceback:")
        print(traceback.format_exc())
        
        # Try to create/update record with network error status
        try:
            video_filename = os.path.basename(file_name)
            raw_video_link = generate_raw_video_link(file_name, bucket_name)
            video_type = "down-the-line" if file_name.startswith('dtl_videos/') else "face-on"
            print(f"Creating network error record for {video_filename}")
            
            # Extract session_id for network error records
            network_session_id = None
            try:
                filename_parts = video_filename.split("_")
                if "swing" in filename_parts:
                    swing_index = filename_parts.index("swing")
                    if len(filename_parts) > swing_index + 2:
                        date_part = filename_parts[swing_index + 1]
                        time_part = filename_parts[swing_index + 2].split('.')[0]
                        network_session_id = f"{date_part}_{time_part}"
            except Exception:
                pass
            
            # [MODIFIED] Query by title first
            existing = collection.find_one({"Title": video_filename})
            
            if existing:
                update_fields = {
                    "Status": "Failed",
                    "Error": "Network error calling ML API",
                    "ErrorDetails": str(e),
                    "LastUpdated": datetime.now().strftime("%H:%M %b %d, %Y"),
                    "rawVideoLink": raw_video_link,
                    # "session_id": network_session_id <-- [REMOVED]
                }
                
                # [NEW] Only update session_id if found
                if network_session_id is not None:
                    update_fields["session_id"] = network_session_id

                collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": update_fields}
                )
                print(f"Updated record with network error for {video_filename}")
            else:
                # Include raw video link in network error record
                new_doc = {
                    "Title": video_filename,
                    "Type": video_type,
                    "DateUploaded": datetime.now().strftime("%H:%M %b %d, %Y"),
                    "Status": "Failed",
                    "Error": "Network error calling ML API",
                    "ErrorDetails": str(e),
                    "originalVideoPath": file_name,
                    "rawVideoLink": raw_video_link,
                    "session_id": network_session_id # This is fine
                }
                collection.insert_one(new_doc)
                print(f"Inserted network error record for {video_filename}")
                
        except Exception as final_error:
            print(f"Final error handling failed: {str(final_error)}")
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Traceback:")
        print(traceback.format_exc())

    print(f"Function execution completed")
    return "OK"
