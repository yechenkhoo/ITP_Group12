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
import traceback
import random
import json

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
            coach_object_id = ObjectId(coach_id)
            student = Users_Collection.find_one({'Email': student_email})
            if student:
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
        """Fetches all students assigned to a specific coach,
           and backfills DateCreated from ObjectId if missing."""
        """Fetches all students assigned to a specific coach,
           and backfills DateCreated from ObjectId if missing."""
        coach = Users_Collection.find_one({'_id': ObjectId(coach_id)})
        if not coach or 'Students' not in coach:
            return []

        students = []
        for student_id in coach['Students']:
            student = Users_Collection.find_one({'_id': student_id})
            if not student:
                continue

            if 'DateCreated' not in student:
                ts = student['_id'].generation_time
                student['DateCreated'] = ts.strftime("%H:%M %b %d, %Y")

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
    """Unified video handling for both frontend uploads and camera recordings."""

    executor = ThreadPoolExecutor(max_workers=5)

    @staticmethod
    def upload_video(current_user_id, assignee_id, title, video_type, file, source="frontend"):
        """
        FIXED: Simplified video upload with proper user tracking and simple file structure.
        """
        formatted_date = datetime.now().strftime("%H:%M %b %d, %Y")
        
        video_document = {
            'UploadedBy': ObjectId(current_user_id),
            'Assignee': ObjectId(assignee_id),
            'Title': title,  # Keep original title
            'Type': video_type,
            'DateUploaded': formatted_date,
            'Status': 'Processing',
            'Source': source,
        }

        result = Videos_Collection.insert_one(video_document)
        video_id = str(result.inserted_id)
        
        try:
            # Read file into memory
            file_data = file.read()  # Read file data as bytes
            file_name = file.name  # Preserve the file name
            content_type = file.content_type  # Preserve the content type

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
            )

            # Optional: Add a callback to handle post-upload logic
            future.add_done_callback(Video._upload_callback)

            return {"message": "Upload started in the background."}
            if source == "frontend":
                # Handle frontend file upload - simplified structure
                file_data = file.read()
                file_name = file.name
                content_type = file.content_type
                
                # Submit async upload task
                future = Video.executor.submit(
                    Video._async_upload_video_task,
                    current_user_id, assignee_id, title, video_type,
                    file_data, file_name, content_type, result, source
                )
                future.add_done_callback(Video._upload_callback)
                
                return {"message": "Frontend upload started in the background.", "video_id": video_id}
                
            elif source == "camera":
                # Handle camera recording - file is already in GCS
                response = Video.process_video(
                    file_path=file,  # file is GCS path for camera
                    video_id=video_id,
                    uploader_id=current_user_id,
                    assignee_id=assignee_id,
                    source=source
                )
                return {"message": "Camera recording processing started.", "video_id": video_id}
                
        except Exception as e:
            print(f"Error starting upload (source: {source}): {e}")
            # Update status to Failed with error
            Videos_Collection.update_one(
                {'_id': result.inserted_id},
                {'$set': {'Status': 'Failed', 'Error': str(e)}}
            )
            return {"error": f"Failed to start {source} upload process.", "video_id": video_id}

    @staticmethod
    def _async_upload_video_task(current_user_id, assignee_id, title, video_type, 
                                file_data, file_name, content_type, result, source):
        """FIXED: Simplified upload task with better error handling."""
        try:
            print(f"Starting {source} upload for video {result.inserted_id}")
            bucket_name = 'golf-swing-models'
            
            storage_client = get_google_cloud_storage_client()
            bucket = storage_client.bucket(bucket_name)
            
            # SIMPLIFIED: Just store in golf_videos/ with original filename
            video_id = str(result.inserted_id)
            blob_name = f'golf_videos/{file_name}'  # Simple structure
            
            # Upload to GCS
            file_stream = io.BytesIO(file_data)
            blob = bucket.blob(blob_name)
            blob.metadata = {
                'uploaded_by': str(current_user_id),  # Convert to string to avoid ObjectId issues
                'assignee': str(assignee_id),
                'video_id': video_id,
                'video_type': video_type,
                'title': title,
                'source': source,
                'upload_timestamp': datetime.now().isoformat()
            }
            blob.upload_from_file(file_stream, content_type=content_type)

            if not blob.exists():
                print("Error: File does not exist in GCS.")
                time.sleep(2)

            response = Video.process_video(blob_name, video_id, current_user_id, assignee_id)

            if response.get("status") == "Processing complete":
                print(f"Deleting original video: {blob_name}")
                blob.delete()  # Delete the video file
            else:
                print(f"Skipping deletion due to processing error: {response.get('error')}")

            
            # Store raw video URL
            raw_video_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
            Videos_Collection.update_one(
                {'_id': result.inserted_id},
                {'$set': {'rawVideoLink': raw_video_url}}
            )
            
            # Process the video
            response = Video.process_video(
                file_path=blob_name,
                video_id=video_id,
                uploader_id=str(current_user_id),  # Convert to string
                assignee_id=str(assignee_id),      # Convert to string
                source=source
            )
            
            return response
            
        except Exception as e:
            print(f"Error in {source} upload task: {e}")
            traceback.print_exc()
            # Update status to Failed
            Videos_Collection.update_one(
                {'_id': result.inserted_id},
                {'$set': {'Status': 'Failed', 'Error': str(e)}}
            )
            return {"error": f"An error occurred during {source} video upload."}

    @staticmethod
    def _upload_callback(future):
        """Handle post-upload completion logic."""
        try:
            result = future.result()  # Retrieve the result of the background task
            print("Upload task completed")
            result = future.result()
            print("Upload task completed:", result.get('status', 'unknown'))
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
                "classification_model": "models/basemodel.keras",
                "video_id": video_id,
                "uploader_id": uploader_id,
                "assignee_id": assignee_id,
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
    def get_csv_url(video_id):
        """Fetches the URL of a video from Google Cloud Storage."""
        video = Videos_Collection.find_one({'_id': ObjectId(video_id)})
        return video['angleCsvLink']
        
    @staticmethod
    def process_video(file_path,video_id):
        print(file_path)
        try:
            # Define the URL for the GCP function
            gcp_function_url = "https://ml-model-api-1067172605110.asia-southeast1.run.app/process-video"

            # Prepare the request payload
            payload = {
                
                "classification_model": "models/best_model.keras",
                "video_id":video_id,
                "video_path": file_path,
                "output_video_path": f"processed/{file_path.split('/')[-1]}",
                "output_csv_path": f"processed/{file_path.split('/')[-1]}.csv",
                "output_angle_csv_path": f"processed/{file_path.split('/')[-1]}_angles.csv"
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
                #thumbnail_url = response_data.get('output_thumbnail')

                # Update the MongoDB document with the returned URLs and status
                Videos_Collection.update_one(
                    {'_id': ObjectId(video_id)},  # Find the document by ID
                    {
                        '$set': {
                            'Status': 'Completed',
                            'frameByFrameCsvLink': output_csv_url,
                            'angleCsvLink': output_angle_csv_url,
                            'processedVideoLink': output_video_url
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
    def get_all_video_comments(video_id, current_user_id):
        """Fetches all comments and their replies for a specific video."""
        video_obj_id = ObjectId(video_id)
        video = Videos_Collection.find_one({'_id': video_obj_id}, {'Comments': 1})
        top_level_comment_ids = video.get('Comments', []) if video else []

        all_related_comments_cursor = Comments_Collection.find(
            {'$or': [
                {'_id': {'$in': top_level_comment_ids}},
                {'video_id': video_obj_id}
            ]},
            sort=[("DateCommented", 1)]
        )
        all_related_comments = list(all_related_comments_cursor)
    def get_all_video_comments(video_id):
        """Fetches all comments for a specific video, sorted by the latest DateCommented."""
        video = Videos_Collection.find_one({'_id': ObjectId(video_id)})
        if not video or 'Comments' not in video:
            return []

        comment_ids = video['Comments']
        # Fetch all comments in one query, sorted by DateCommented in descending order
        comments = list(Comments_Collection.find(
            {'_id': {'$in': [ObjectId(comment_id) for comment_id in comment_ids]}},
            sort=[("DateCommented", -1)]  # Sort by DateCommented descending
        ))

        # Enrich comments with user details
        for comment in comments:
        comments_map = {}
        for comment in all_related_comments:
            user = Users_Collection.find_one({'_id': ObjectId(comment['CommentedBy'])})
            comment['CommentedBy'] = user['Name'] if user else 'Unknown User'

            # Format DateCommented for returning

            # Format DateCommented for returning
            comment['FormattedDate'] = comment['DateCommented'].strftime("%H:%M %b %d, %Y")

        return comments
            comment['id'] = str(comment['_id'])
            comments_map[comment['id']] = comment
            comment['replies'] = []

        structured_comments = []
        for comment_id_str, comment in comments_map.items():
            if 'parent_comment_id' in comment and comment['parent_comment_id'] is not None:
                parent_id_str = str(comment['parent_comment_id'])
                if parent_id_str in comments_map:
                    comments_map[parent_id_str]['replies'].append(comment)
            else:
                if ObjectId(comment['id']) in top_level_comment_ids:
                    structured_comments.append(comment)

        for comment in structured_comments:
            comment['replies'].sort(key=lambda r: r['DateCommented'])

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
    def add_comment(current_user_id, video_id, comment_text):
        """
        Adds a comment to a video and associates it with the commenting user.
        """

    def add_comment(current_user_id, video_id, comment_text, x_pos=None, y_pos=None):
        """Adds a top-level comment to a video."""
        try:
            # Create comment document
            
            comment_document = {
                'Comment': comment_text,
                'CommentedBy': ObjectId(current_user_id),
                'DateCommented': datetime.now(),
                'x_pos': x_pos,
                'y_pos': y_pos,
                'video_id': ObjectId(video_id),
                'parent_comment_id': None,
                'readBy': [ObjectId(current_user_id)],
            }
            Comments_Collection.insert_one(comment_document)
            Comments_Collection.insert_one(comment_document)

            # Link the comment to the video
            Videos_Collection.update_one(
                {'_id': ObjectId(video_id)},
                {'$push': {'Comments': comment_document['_id']}}
                {'$push': {'Comments': comment_document['_id']}}
            )
        except Exception as e:
            print(f"Error adding comment: {e}")

            traceback.print_exc()
            return None

    @staticmethod
    def add_reply(current_user_id, video_id, parent_comment_id, reply_text):
        """Adds a reply to an existing comment."""
        try:
            parent_obj_id = ObjectId(parent_comment_id)
            if not Comments_Collection.find_one({'_id': parent_obj_id}):
                print(f"Parent comment with ID {parent_comment_id} not found.")
                return None

            reply_document = {
                'Comment': reply_text,
                'CommentedBy': ObjectId(current_user_id),
                'DateCommented': datetime.now(),
                'video_id': ObjectId(video_id),
                'parent_comment_id': parent_obj_id,
                'x_pos': None,
                'y_pos': None,
                'readBy': [ObjectId(current_user_id)],
            }
            inserted_reply = Comments_Collection.insert_one(reply_document)
            return inserted_reply.inserted_id
        except Exception as e:
            print(f"Error adding reply: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def get_comment_by_id(comment_id):
        """Fetches a single comment by its ID and processes it for frontend display."""
        try:
            comment = Comments_Collection.find_one({'_id': ObjectId(comment_id)})
            if not comment:
                return None

            user = Users_Collection.find_one({'_id': ObjectId(comment['CommentedBy'])})
            comment['CommentedBy'] = user['Name'] if user else 'Unknown User'
            comment['FormattedDate'] = comment['DateCommented'].strftime("%H:%M %b %d, %Y")
            
            if 'DateEdited' in comment:
                comment['FormattedDateEdited'] = comment['DateEdited'].strftime("%H:%M %b %d, %Y")

            comment['id'] = str(comment.pop('_id'))
            
            if 'parent_comment_id' not in comment or comment['parent_comment_id'] is None:
                comment['replies'] = []

            return comment
        except Exception as e:
            print(f"Error in get_comment_by_id: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def update_comment_position(comment_id, x_pos, y_pos):
        """Updates the position of an existing comment."""
        try:
            result = Comments_Collection.update_one(
                {'_id': ObjectId(comment_id), 'parent_comment_id': None},
                {'$set': {'x_pos': x_pos, 'y_pos': y_pos}}
            )
            return result.matched_count > 0
        except Exception as e:
            print(f"Error updating comment position: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def delete_comment(comment_id, current_user_id):
        """Deletes a top-level comment and all its direct replies."""
        try:
            obj_comment_id = ObjectId(comment_id)
            obj_user_id = ObjectId(current_user_id)

            comment_to_delete = Comments_Collection.find_one({"_id": obj_comment_id})

            if not comment_to_delete:
                print(f"Comment with ID {comment_id} not found for deletion.")
                return False

            if comment_to_delete.get('CommentedBy') != obj_user_id:
                print(f"User {current_user_id} is not authorized to delete comment {comment_id}.")
                return False

            if comment_to_delete.get('parent_comment_id') is not None:
                print(f"Comment with ID {comment_id} is a reply. Use delete_reply instead.")
                return False
            else:
                # Delete all replies
                Comments_Collection.delete_many({'parent_comment_id': obj_comment_id})
                
                # Remove from video's Comments array
                if 'video_id' in comment_to_delete:
                    Videos_Collection.update_one(
                        {'_id': comment_to_delete['video_id']},
                        {'$pull': {'Comments': obj_comment_id}}
                    )

            # Delete the comment itself
            delete_result = Comments_Collection.delete_one({'_id': obj_comment_id})
            return delete_result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting comment {comment_id}: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def delete_reply(reply_id, current_user_id):
        """Deletes a specific reply after verifying the current user is the author."""
        try:
            obj_reply_id = ObjectId(reply_id)
            obj_user_id = ObjectId(current_user_id)

            reply_to_delete = Comments_Collection.find_one({"_id": obj_reply_id})

            if not reply_to_delete:
                print(f"Reply with ID {reply_id} not found for deletion.")
                return False

            if reply_to_delete.get('parent_comment_id') is None:
                print(f"Comment with ID {reply_id} is a top-level comment, not a reply.")
                return False

            if reply_to_delete.get('CommentedBy') != obj_user_id:
                print(f"User {current_user_id} is not authorized to delete reply {reply_id}.")
                return False

            delete_result = Comments_Collection.delete_one({'_id': obj_reply_id})
            return delete_result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting reply {reply_id}: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def edit_comment(comment_id, new_text, current_user_id):
        """Edits the text of an existing comment."""
        try:
            comment_obj_id = ObjectId(comment_id)
            user_obj_id = ObjectId(current_user_id)

            comment = Comments_Collection.find_one({"_id": comment_obj_id})
            
            if not comment:
                print(f"Comment with ID {comment_id} not found.")
                return False
            
            if comment.get('CommentedBy') != user_obj_id:
                print(f"User {current_user_id} is not authorized to edit comment {comment_id}.")
                return False

            result = Comments_Collection.update_one(
                {'_id': comment_obj_id},
                {'$set': {'Comment': new_text, 'DateEdited': datetime.now()}}
            )
            return result.matched_count > 0
        except Exception as e:
            print(f"Error editing comment: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def edit_reply(reply_id, new_text, current_user_id):
        """Edits the text of an existing reply."""
        try:
            reply_obj_id = ObjectId(reply_id)
            user_obj_id = ObjectId(current_user_id)

            reply = Comments_Collection.find_one({"_id": reply_obj_id})
            
            if not reply:
                print(f"Reply with ID {reply_id} not found.")
                return False
            
            if reply.get('CommentedBy') != user_obj_id:
                print(f"User {current_user_id} is not authorized to edit reply {reply_id}.")
                return False

            result = Comments_Collection.update_one(
                {'_id': reply_obj_id},
                {'$set': {'Comment': new_text, 'DateEdited': datetime.now()}}
            )
            return result.matched_count > 0
        except Exception as e:
            print(f"Error editing reply: {e}")
            traceback.print_exc()
            return False
