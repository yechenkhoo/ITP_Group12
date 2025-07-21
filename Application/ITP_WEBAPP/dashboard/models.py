# dashboard/models.py

from django.db import models
from db_connection import MONGO_CLIENT
from bson import ObjectId
from django.conf import settings
from google.cloud import storage
import os
from datetime import datetime
from django.http import JsonResponse
import uuid
import requests
from dashboard.google_cloud import get_google_cloud_storage_client
import time
from concurrent.futures import ThreadPoolExecutor
import io
import traceback # Import traceback for detailed error logging
import random
import json # Ensure json module is imported for JSONDecodeError
from urllib.parse import urlparse

# MongoDB collections
Users_Collection = MONGO_CLIENT['Users']
Videos_Collection = MONGO_CLIENT['Videos']
Comments_Collection = MONGO_CLIENT['Comments']


class Coach:
    """Handles operations related to staff members."""

    @staticmethod
    def create_user(email, password, role, name, created_by):
        """Creates a new user account in MongoDB."""
        if Users_Collection.find_one({'Email': email}):
            return False

        formatted_date = datetime.now().strftime("%H:%M %b %d, %Y")

        user_data = {
            'Email': email,
            'Name': name,
            'Password': password,
            'Role': role,
            'CreatedBy': ObjectId(created_by),
            'DateCreated': formatted_date,
        }
        try:
            Users_Collection.insert_one(user_data)
            return True
        except Exception as e:
            print(f"Error creating user: {e}")
            return False

    @staticmethod
    def update_student_array(student_email, coach_id):
        """Adds a student to a coach's student list."""
        try:
            # Ensure coach_id is a valid ObjectId
            coach_object_id = ObjectId(coach_id)

            # Find the student by email
            student = Users_Collection.find_one({'Email': student_email})
            if student:
                # Add the student's ObjectId to the coach's student list
                Users_Collection.update_one(
                    {'_id': coach_object_id},
                    {'$push': {'Students': student['_id']}}
                )
                return True
            return False
        except Exception as e:
            print(f"Error updating student array: {e}")
            return False


    @staticmethod
    def fetch_all_students(coach_id):
        """Fetches all students assigned to a specific coach."""
        coach = Users_Collection.find_one({'_id': ObjectId(coach_id)})
        if not coach or 'Students' not in coach:
            return []

        students = []
        for student_id in coach['Students']:
            student = Users_Collection.find_one({'_id': student_id})
            if not student:
                continue

            # --- backfill DateCreated if the field is missing ---
            if 'DateCreated' not in student:
                # ObjectId.generation_time is a datetime in UTC
                ts = student['_id'].generation_time
                student['DateCreated'] = ts.strftime("%H:%M %b %d, %Y")

            # Replace _id with string id
            oid = student.pop('_id')
            student['id'] = str(oid)

            students.append(student)
        return students

    @staticmethod
    def verify_coach_student_relationship(coach_id, student_id):
        """Verifies if a student is assigned to a specific coach."""
        coach = Users_Collection.find_one({'_id': ObjectId(coach_id)})
        if coach and 'Students' in coach:
            return ObjectId(student_id) in coach['Students']
        return False


class Video:
    """Handles operations related to videos."""

    executor = ThreadPoolExecutor(max_workers=5)

    @staticmethod
    def upload_video(current_user_id, assignee_id, title, video_type, file, upload_source="manual"):
        """
        Trigger asynchronous video upload to GCP.
        upload_source: "manual" for user uploads, "rpi" for RPi recordings
        """
        formatted_date = datetime.now().strftime("%H:%M %b %d, %Y")
        
        video_document = {
            'UploadedBy': ObjectId(current_user_id),
            'Assignee': ObjectId(assignee_id),
            'Title': title,
            'Type': video_type,
            'DateUploaded': formatted_date,
            'Status': 'Processing',
        }

        result = Videos_Collection.insert_one(video_document)
        
        try:
            # Read file into memory
            file_data = file.read()   # Read file data as bytes
            file_name = file.name   # Preserve the file name
            content_type = file.content_type   # Preserve the content type

            # Submit the task to the executor
            future = Video.executor.submit(
                Video._async_upload_video_task,
                current_user_id,
                assignee_id,
                title,
                video_type,
                file_data,
                file_name,
                content_type,
                result,
                upload_source  # Pass upload source
            )

            # Optional: Add a callback to handle post-upload logic
            future.add_done_callback(Video._upload_callback)

            return {"message": "Upload started in the background."}
        except Exception as e:
            print(f"Error starting async upload: {e}")
            return {"error": "Failed to start the upload process."}

    @staticmethod
    def _async_upload_video_task(current_user_id, assignee_id, title, video_type, file_data, file_name, content_type, result, upload_source="manual"):
        """
        Perform the actual upload to GCP in the background.
        upload_source: "manual" for user uploads, "rpi" for RPi recordings
        """
        try:
            print("uploading")
            bucket_name = 'golf-swing-models'

            # Initialize GCP storage client
            storage_client = get_google_cloud_storage_client()

            # Get the bucket
            bucket = storage_client.bucket(bucket_name)

            video_id = str(result.inserted_id)

            # FIXED: Generate blob name based on upload source - all go to golf_videos/
            if upload_source == "rpi":
                # RPi uploads: Enhanced filename with user context
                unique_id = uuid.uuid4().hex
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                blob_name = f'golf_videos/{current_user_id}_{assignee_id}_swing_{timestamp}_{unique_id}.mp4'
                print(f"RPi upload: {blob_name}")
            else:
                # Manual uploads: Use original filename
                blob_name = f'golf_videos/{file_name}'
                print(f"Manual upload: {blob_name}")

            # Create a file-like object from the in-memory data
            file_stream = io.BytesIO(file_data)

            # Upload file to GCP Storage with metadata
            blob = bucket.blob(blob_name)
            
            # Add metadata to the blob for additional context
            blob.metadata = {
                'uploaded_by': current_user_id,
                'assignee': assignee_id,
                'video_id': video_id,
                'video_type': video_type,
                'title': title,
                'upload_source': upload_source,
                'upload_timestamp': datetime.now().isoformat()
            }
            
            blob.upload_from_file(file_stream, content_type=content_type)

            # --- NEW: Construct the public URL for the raw video ---
            raw_video_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"

            if not blob.exists():
                print("Error: File does not exist in GCS.")
                time.sleep(2)
            
            # --- NEW: Update the document with the raw video URL ---
            Videos_Collection.update_one(
                {'_id': result.inserted_id},
                {'$set': {'rawVideoLink': raw_video_url}}
            )

            response = Video.process_video(blob_name, video_id, current_user_id, assignee_id)

            if response.get("status") == "Processing complete":
                print(f"Processing succeeded; keeping raw upload at {blob_name}")
            else:
                print(f"Processing error, raw upload kept: {response.get('error')}")

            return response

        except Exception as e:
            print(f"Error uploading video: {e}")
            return {"error": "An error occurred during video upload."}

    @staticmethod
    def _upload_callback(future):
        """
        Handle post-upload completion logic.
        """
        try:
            result = future.result()   # Retrieve the result of the background task
            print("Upload task completed")
        except Exception as e:
            print("Error in upload callback:", e)
        
    @staticmethod
    def get_video_url(video_id):
        """Fetches the URL of a video from Google Cloud Storage."""
        video = Videos_Collection.find_one({'_id': ObjectId(video_id)})
        return video['processedVideoLink'] 
        
    @staticmethod
    def get_csv_url(video_id):
        """Fetches the URL of a video from Google Cloud Storage."""
        video = Videos_Collection.find_one({'_id': ObjectId(video_id)})
        return video['angleCsvLink']

    @staticmethod
    def delete_videos(video_ids):
        """
        Deletes videos and all associated data from MongoDB and Google Cloud Storage.
        Handles raw video, processed video, CSVs, and comments.

        :param video_ids: A list of video ID strings to be deleted.
        """
        if not isinstance(video_ids, list):
            video_ids = [video_ids]  # Ensure it's a list

        storage_client = get_google_cloud_storage_client()
        bucket_name = 'golf-swing-models'  # Your GCS bucket name
        bucket = storage_client.bucket(bucket_name)
        object_ids = [ObjectId(vid) for vid in video_ids if vid]

        # Find all video documents to gather their associated file URLs and comment lists
        videos_to_delete = list(Videos_Collection.find({'_id': {'$in': object_ids}}))

        if not videos_to_delete:
            print("No videos found for the given IDs.")
            return {'status': 'error', 'message': 'No videos found.'}

        all_comment_ids_to_delete = []
        blobs_to_delete = []

        def get_blob_name_from_url(url):
            """Helper function to extract the GCS blob name from a public URL."""
            if not url or not url.startswith(f'https://storage.googleapis.com/{bucket_name}/'):
                return None
            parsed_url = urlparse(url)
            # The path is /bucket-name/blob/path/file.ext, so we strip the bucket name
            return parsed_url.path.replace(f'/{bucket_name}/', '', 1)

        for video in videos_to_delete:
            # 1. Gather all associated GCS blobs to delete
            urls = [
                video.get('rawVideoLink'),
                video.get('processedVideoLink'),
                video.get('frameByFrameCsvLink'),
                video.get('angleCsvLink'),
                video.get('poseClassImagesLink')
            ]
            for url in urls:
                blob_name = get_blob_name_from_url(url)
                if blob_name:
                    blobs_to_delete.append(blob_name)

            # 2. Gather all associated comment IDs to delete
            video_id = video['_id']
            comments_cursor = Comments_Collection.find({'video_id': video_id}, {'_id': 1})
            for comment in comments_cursor:
                all_comment_ids_to_delete.append(comment['_id'])

        # --- Perform Deletions ---

        # A. Delete GCS blobs in parallel for efficiency
        def delete_blob(blob_name):
            try:
                blob = bucket.blob(blob_name)
                if blob.exists():
                    blob.delete()
                    print(f"Successfully deleted GCS blob: {blob_name}")
                    return True
            except Exception as e:
                print(f"Error deleting GCS blob {blob_name}: {e}")
            return False

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(delete_blob, blobs_to_delete)

        # B. Delete all associated comments from MongoDB in a single operation
        if all_comment_ids_to_delete:
            Comments_Collection.delete_many({'_id': {'$in': all_comment_ids_to_delete}})
            print(f"Deleted {len(all_comment_ids_to_delete)} comments from MongoDB.")

        # C. Delete video documents from MongoDB in a single operation
        result_videos = Videos_Collection.delete_many({'_id': {'$in': object_ids}})
        deleted_videos_count = result_videos.deleted_count
        print(f"Deleted {deleted_videos_count} video documents from MongoDB.")

        return {'status': 'success', 'deleted_videos': deleted_videos_count}
        
    @staticmethod
    def process_video(file_path, video_id, uploader_id, assignee_id):
        """Process video with user context included in paths"""
        print(f"Processing video: {file_path}")
        try:
            # Define the URL for the GCP function
            gcp_function_url = "https://ml-model-api-1067172605110.asia-southeast1.run.app/process-video"

            # Extract filename from path for output naming
            filename = file_path.split('/')[-1]
            
            # Create organized output paths that maintain user context
            base_output_path = f"processed/{uploader_id}/{assignee_id}"
            
            # Prepare the request payload with user context
            payload = {
                "classification_model": "best_model.keras",
                "video_id": video_id,
                "uploader_id": str(uploader_id),
                "assignee_id": str(assignee_id),
                "video_path": file_path,
                "output_video_path": f"{base_output_path}/{filename}",
                "output_csv_path": f"{base_output_path}/{filename}.csv",
                "output_angle_csv_path": f"{base_output_path}/{filename}_angles.csv"
            }

            # Send the POST request to the GCP function
            headers = {"Content-Type": "application/json"}
            response = requests.post(gcp_function_url, json=payload, headers=headers)

            if response.status_code == 200:
                # Parse the JSON response from the GCP function
                response_data = response.json()
                output_video_url = response_data.get('output_video')
                output_csv_url = response_data.get('output_csv')
                output_angle_csv_url = response_data.get('output_angle_csv')

                # Update the MongoDB document with the returned URLs and status
                Videos_Collection.update_one(
                    {'_id': ObjectId(video_id)},  # Find the document by ID
                    {
                        '$set': {
                            'Status': 'Completed',
                            'frameByFrameCsvLink': output_csv_url,
                            'angleCsvLink': output_angle_csv_url,
                            'processedVideoLink': output_video_url,
                            'originalVideoPath': file_path,  # Store original path for reference
                            'LastUpdated': datetime.now().strftime("%H:%M %b %d, %Y")
                        }
                    }
                )
                return response_data
            else:
                print(f"Error in video processing: {response.text}")
                return {"error": "An error occurred during video processing."}

        except Exception as e:
            print(f"Error during video processing: {e}")
            return {"error": "An error occurred during video processing."}

    @staticmethod
    def get_all_videos(assignee_id):
        """Fetches all videos assigned to a specific user."""
        return [
            {**video, 'id': str(video.pop('_id'))}
            for video in Videos_Collection.find({'Assignee': ObjectId(assignee_id)})
        ]

    @staticmethod
    def get_video_status(video_id):
        """Fetches the status of a specific video."""
        video = Videos_Collection.find_one({'_id': ObjectId(video_id)}, {'Status': 1, 'rawVideoLink': 1, 'processedVideoLink': 1, 'poseClassImagesLink': 1}) # Added 'poseClassImagesLink'
        if video:
            return {
                'status': video.get('Status'),
                'rawVideoLink': video.get('rawVideoLink'),
                'processedVideoLink': video.get('processedVideoLink'),
                'poseClassImagesLink': video.get('poseClassImagesLink') # Return poseClassImagesLink
            }
        return None

    @staticmethod
    def get_random_pro_video_blob_name():
        """
        Retrieves a random pro video blob name (e.g., 'videoModels/pro_swing_1.mp4')
        from Google Cloud Storage under the 'videoModels/' prefix.
        """
        try:
            bucket_name = 'golf-swing-models'
            storage_client = get_google_cloud_storage_client()
            bucket = storage_client.get_bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix='modelVideos/'))
            video_blobs = [b.name for b in blobs if b.name.lower().endswith('.mp4')]
            if not video_blobs:
                print(f"No pro videos found under 'modelVideos/' in bucket '{bucket_name}'")
                return None
            return random.choice(video_blobs)
        except Exception as e:
            print(f"Error listing GCS blobs for pro videos: {e}")
            return None

    @staticmethod
    def get_pro_video_url_from_blob(blob_name):
        """
        Given a pro video blob name under 'modelVideos/', returns its public URL
        if the blob exists, else None.
        """
        if not blob_name:
            return None
        bucket_name = 'golf-swing-models'
        storage_client = get_google_cloud_storage_client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if blob.exists():
            return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
        print(f"Pro video blob not found: {blob_name}")
        return None

    @staticmethod
    def get_pro_csv_url_from_blob(blob_name):
        """
        Given a pro video blob name (e.g., 'modelVideos/pro_swing_1.mp4'),
        derives the CSV blob name (e.g., 'modelVideos/pro_swing_1.csv')
        and returns its public URL if it exists.
        """
        if not blob_name:
            return None
        bucket_name = 'golf-swing-models'
        # Replace extension with .csv
        base, _ = blob_name.rsplit('.', 1)
        csv_blob_name = f"{base}_angles.csv"
        storage_client = get_google_cloud_storage_client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(csv_blob_name)
        if blob.exists():
            return f"https://storage.googleapis.com/{bucket_name}/{csv_blob_name}"
        print(f"Pro CSV blob not found: {csv_blob_name}")
        return None

    @staticmethod
    def get_all_video_comments(video_id, current_user_id):
        """
        Fetches all comments and their replies for a specific video,
        sorted by DateCommented, and structured hierarchically.
        """
        video_obj_id = ObjectId(video_id)
        
        # Get all comment _ids from the video document (these are explicitly top-level comments)
        video = Videos_Collection.find_one({'_id': video_obj_id}, {'Comments': 1})
        top_level_comment_ids = video.get('Comments', []) if video else []

        # Find all comments related to this video. This includes top-level comments
        # and all replies, which should have a 'video_id' field.
        all_related_comments_cursor = Comments_Collection.find(
            {'$or': [
                {'_id': {'$in': top_level_comment_ids}},
                {'video_id': video_obj_id} # This covers replies that link to the video directly
            ]},
            sort=[("DateCommented", 1)] # Sort ascending to build hierarchy easily
        )
        all_related_comments = list(all_related_comments_cursor)

        # Dictionary to store comments by their string ID for easy lookup
        comments_map = {}
        for comment in all_related_comments:
            # Enrich comment with user details
            user = Users_Collection.find_one({'_id': ObjectId(comment['CommentedBy'])})
            comment['CommentedBy'] = user['Name'] if user else 'Unknown User'
            comment['FormattedDate'] = comment['DateCommented'].strftime("%H:%M %b %d, %Y")
            comment['id'] = str(comment['_id']) # Convert _id to 'id' string for frontend

            comments_map[comment['id']] = comment
            comment['replies'] = [] # Initialize replies list for all comments

        # List to hold final structured top-level comments
        structured_comments = []

        # Populate replies and identify top-level comments
        for comment_id_str, comment in comments_map.items():
            if 'parent_comment_id' in comment and comment['parent_comment_id'] is not None:
                parent_id_str = str(comment['parent_comment_id'])
                if parent_id_str in comments_map:
                    comments_map[parent_id_str]['replies'].append(comment)
                else:
                    # Handle orphaned replies (e.g., if parent was deleted but reply still exists)
                    print(f"Warning: Orphaned reply {comment['id']} for non-existent parent {parent_id_str}")
            else:
                # This is a top-level comment (no parent_comment_id)
                # Ensure it's truly a top-level comment by checking if its original _id was in the video's list
                if ObjectId(comment['id']) in top_level_comment_ids:
                    structured_comments.append(comment)
                # If a comment has no parent_comment_id but its _id is NOT in the video's 'Comments' array,
                # it's an anomaly or a standalone comment not properly linked. We'll ignore it for this video.


        # Sort replies within each comment by DateCommented
        for comment in structured_comments:
            comment['replies'].sort(key=lambda r: r['DateCommented'])

        # Sort top-level comments by DateCommented (descending for latest first)
        structured_comments.sort(key=lambda c: c['DateCommented'], reverse=True)

        user_oid = ObjectId(current_user_id)
        for c in structured_comments:
            c['unread'] = user_oid not in c.get('readBy', [])
            for r in c['replies']:
                r['unread'] = user_oid not in r.get('readBy', [])

        return structured_comments

        


class Comment:
    """Handles operations related to comments."""

    @staticmethod
    def add_comment(current_user_id, video_id, comment_text, x_pos=None, y_pos=None):
        """
        Adds a top-level comment to a video.
        Includes optional position data for free-moving comments.
        Returns the inserted comment's _id.
        """
        try:
            # Create comment document
            comment_document = {
                'Comment': comment_text,
                'CommentedBy': ObjectId(current_user_id),
                'DateCommented': datetime.now(),
                'x_pos': x_pos, # Store x-coordinate
                'y_pos': y_pos, # Store y-coordinate
                'video_id': ObjectId(video_id), # Link top-level comment to video as well for easier lookup
                'parent_comment_id': None, # Explicitly mark as top-level
                'readBy': [ ObjectId(current_user_id) ],
            }
            inserted_comment = Comments_Collection.insert_one(comment_document)

            # Link the top-level comment to the video's 'Comments' array
            Videos_Collection.update_one(
                {'_id': ObjectId(video_id)},
                {'$push': {'Comments': inserted_comment.inserted_id}}
            )
            return inserted_comment.inserted_id
        except Exception as e:
            print(f"Error adding comment: {e}")
            traceback.print_exc() # Added traceback for debugging
            return None # Indicate failure by returning None

    @staticmethod
    def add_reply(current_user_id, video_id, parent_comment_id, reply_text):
        """
        Adds a reply to an existing comment.
        Replies do not have x_pos/y_pos and are linked via parent_comment_id.
        They are NOT added to the video's 'Comments' array.
        """
        try:
            # Ensure parent_comment_id is a valid ObjectId
            parent_obj_id = ObjectId(parent_comment_id)

            # Check if parent comment exists
            if not Comments_Collection.find_one({'_id': parent_obj_id}):
                print(f"Parent comment with ID {parent_comment_id} not found.")
                return None

            reply_document = {
                'Comment': reply_text,
                'CommentedBy': ObjectId(current_user_id),
                'DateCommented': datetime.now(),
                'video_id': ObjectId(video_id), # Store video_id for replies for context/easier lookup
                'parent_comment_id': parent_obj_id, # Link to the parent comment
                'x_pos': None, # Replies do not have explicit positions
                'y_pos': None, # Replies do not have explicit positions
                'readBy': [ ObjectId(current_user_id) ], 
            }
            inserted_reply = Comments_Collection.insert_one(reply_document)
            return inserted_reply.inserted_id
        except Exception as e:
            print(f"Error adding reply: {e}")
            traceback.print_exc() # Added traceback for debugging
            return None

    @staticmethod
    def get_comment_by_id(comment_id):
        """
        Fetches a single comment by its ID and processes it for frontend display.
        This will fetch either a top-level comment or a reply.
        """
        try:
            comment = Comments_Collection.find_one({'_id': ObjectId(comment_id)})
            if not comment:
                return None

            # Enrich comment with user details
            user = Users_Collection.find_one({'_id': ObjectId(comment['CommentedBy'])})
            comment['CommentedBy'] = user['Name'] if user else 'Unknown User'

            # Format DateCommented for returning
            comment['FormattedDate'] = comment['DateCommented'].strftime("%H:%M %b %d, %Y")
            
            # If DateEdited exists, format it too
            if 'DateEdited' in comment:
                comment['FormattedDateEdited'] = comment['DateEdited'].strftime("%H:%M %b %d, %Y")

            # Convert _id to 'id' string for frontend consumption
            comment['id'] = str(comment.pop('_id'))
            
            # Add 'replies' key for consistency if it's a top-level comment and you plan to expand it
            if 'parent_comment_id' not in comment or comment['parent_comment_id'] is None:
                comment['replies'] = [] # This ensures a consistent structure
                # You might want to fetch and populate direct replies here if this method is used in isolation
                # For now, get_all_video_comments handles full tree, so this is just for single lookup.

            return comment
        except Exception as e:
            print(f"Error in get_comment_by_id: {e}")
            traceback.print_exc() # Added traceback for debugging
            return None

    @staticmethod
    def update_comment_position(comment_id, x_pos, y_pos):
        """Updates the position of an existing comment. Only applies to top-level comments."""
        try:
            result = Comments_Collection.update_one(
                {'_id': ObjectId(comment_id), 'parent_comment_id': None}, # Only update if it's a top-level comment
                {'$set': {'x_pos': x_pos, 'y_pos': y_pos}}
            )
            return result.matched_count > 0
        except Exception as e:
            print(f"Error updating comment position: {e}")
            traceback.print_exc() # Added traceback for debugging
            return False

    @staticmethod
    def delete_comment(comment_id, current_user_id): # Added current_user_id parameter
        """
        Deletes a top-level comment and all its direct replies,
        after verifying the current user is the author of the comment.
        Also removes the top-level comment's reference from the video.
        """
        try:
            obj_comment_id = ObjectId(comment_id)
            obj_user_id = ObjectId(current_user_id) # Convert current_user_id to ObjectId

            # Find the comment to delete
            comment_to_delete = Comments_Collection.find_one({"_id": obj_comment_id})

            if not comment_to_delete:
                print(f"Comment with ID {comment_id} not found for deletion.")
                return False

            # Authorization Check: Ensure the current user is the author of the comment
            if comment_to_delete.get('CommentedBy') != obj_user_id:
                print(f"User {current_user_id} is not authorized to delete comment {comment_id}.")
                return False # Not authorized

            if comment_to_delete.get('parent_comment_id') is not None:
                # It's a reply: for a top-level comment deletion, we only care about top-level comments.
                # This branch implies an attempt to delete a reply using delete_comment, which is not intended.
                print(f"Comment with ID {comment_id} is a reply. Use delete_reply instead for specific reply deletion.")
                return False
            else:
                # It's a top-level comment:
                # 1. Delete all replies that reference this comment as their parent
                Comments_Collection.delete_many({'parent_comment_id': obj_comment_id})
                
                # 2. Remove this comment's ID from the associated video's 'Comments' array
                # Ensure 'video_id' exists in the comment_to_delete document
                if 'video_id' in comment_to_delete:
                    Videos_Collection.update_one(
                        {'_id': comment_to_delete['video_id']},
                        {'$pull': {'Comments': obj_comment_id}}
                    )
                else:
                    print(f"Comment {comment_id} does not have an associated video_id.")

            # Finally, delete the top-level comment itself
            delete_result = Comments_Collection.delete_one({'_id': obj_comment_id})

            return delete_result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting comment {comment_id}: {e}")
            traceback.print_exc() # Added traceback for debugging
            return False

    @staticmethod
    def delete_reply(reply_id, current_user_id):
        """
        Deletes a specific reply after verifying the current user is the author.
        """
        try:
            obj_reply_id = ObjectId(reply_id)
            obj_user_id = ObjectId(current_user_id)

            # Find the reply to delete
            reply_to_delete = Comments_Collection.find_one({"_id": obj_reply_id})

            if not reply_to_delete:
                print(f"Reply with ID {reply_id} not found for deletion.")
                return False

            # Ensure it's actually a reply (has a parent_comment_id) and not a top-level comment
            if reply_to_delete.get('parent_comment_id') is None:
                print(f"Comment with ID {reply_id} is a top-level comment, not a reply. Use delete_comment instead.")
                return False

            # Check if the current user is the author of the reply
            if reply_to_delete.get('CommentedBy') != obj_user_id:
                print(f"User {current_user_id} is not authorized to delete reply {reply_id}.")
                return False # Not authorized

            # Delete the reply document
            delete_result = Comments_Collection.delete_one({'_id': obj_reply_id})

            return delete_result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting reply {reply_id}: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def edit_comment(comment_id, new_text, current_user_id):
        """
        Edits the text of an existing comment.
        Only allows the original author of the comment to edit it.
        Adds/updates a 'DateEdited' field.
        """
        try:
            comment_obj_id = ObjectId(comment_id)
            user_obj_id = ObjectId(current_user_id)

            # Find the comment and check if the current_user_id matches the author's CommentedBy
            comment = Comments_Collection.find_one({"_id": comment_obj_id})
            
            if not comment:
                print(f"Comment with ID {comment_id} not found.")
                return False
            
            # Ensure the user attempting to edit is the author of the comment
            if comment.get('CommentedBy') != user_obj_id:
                print(f"User {current_user_id} is not authorized to edit comment {comment_id}.")
                return False # Not authorized to edit this comment

            # Update the comment text and set/update DateEdited
            result = Comments_Collection.update_one(
                {'_id': comment_obj_id},
                {'$set': {'Comment': new_text, 'DateEdited': datetime.now()}}
            )
            return result.matched_count > 0
        except Exception as e:
            print(f"Error editing comment: {e}")
            traceback.print_exc() # Added traceback for debugging
            return False

    @staticmethod
    def edit_reply(reply_id, new_text, current_user_id):
        """
        Edits the text of an existing reply.
        Only allows the original author of the reply to edit it.
        Adds/updates a 'DateEdited' field.
        """
        try:
            reply_obj_id = ObjectId(reply_id)
            user_obj_id = ObjectId(current_user_id)

            # Find the reply and check if the current_user_id matches the author's CommentedBy
            reply = Comments_Collection.find_one({"_id": reply_obj_id})
            
            if not reply:
                print(f"Reply with ID {reply_id} not found.")
                return False
            
            # Ensure the user attempting to edit is the author of the reply
            if reply.get('CommentedBy') != user_obj_id:
                print(f"User {current_user_id} is not authorized to edit reply {reply_id}.")
                return False # Not authorized to edit this reply

            # Update the reply text and set/update DateEdited
            result = Comments_Collection.update_one(
                {'_id': reply_obj_id},
                {'$set': {'Comment': new_text, 'DateEdited': datetime.now()}}
            )
            return result.matched_count > 0
        except Exception as e:
            print(f"Error editing reply: {e}")
            traceback.print_exc() # Added traceback for debugging
            return False