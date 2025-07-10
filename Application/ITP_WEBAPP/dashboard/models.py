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
    def upload_video(current_user_id, assignee_id, title, video_type, file):
        """
        Trigger asynchronous video upload to GCP.
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
        except Exception as e:
            print(f"Error starting async upload: {e}")
            return {"error": "Failed to start the upload process."}

    @staticmethod
    def _async_upload_video_task(current_user_id, assignee_id, title, video_type, file_data, file_name, content_type, result):
        """
        Perform the actual upload to GCP in the background.
        """
        try:
            print("uploading")
            bucket_name = 'golf-swing-models'

            # Initialize GCP storage client
            storage_client = get_google_cloud_storage_client()

            # Get the bucket
            bucket = storage_client.bucket(bucket_name)

            # Generate a unique blob name
            unique_id = uuid.uuid4().hex  # Generate a unique ID
            blob_name = f'golf_videos/{file_name}'

            # Create a file-like object from the in-memory data
            file_stream = io.BytesIO(file_data)

            # Upload file to GCP Storage
            blob = bucket.blob(blob_name)
            blob.upload_from_file(file_stream, content_type=content_type)

            if not blob.exists():
                print("Error: File does not exist in GCS.")
                time.sleep(2)
            
            video_id = str(result.inserted_id)  # Get the ID as a string

            response = Video.process_video(blob_name, video_id)

            if response.get("status") == "Processing complete":
                print(f"Processing succeeded; keeping raw upload at {blob_name}")
                # raw golf_videos/… blob is left in place
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
            result = future.result()  # Retrieve the result of the background task
            print("Upload task completed")
        except Exception as e:
            print("Error in upload callback:", e)
        
    @staticmethod
    def get_video_url(video_id):
        """Fetches the URL of a video from Google Cloud Storage."""
        video = Videos_Collection.find_one({'_id': ObjectId(video_id)})
        return video['processedVideoLink'] 
        

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
                
                "classification_model": "basemodel.keras",
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
    def get_all_video_comments(video_id):
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
            }
            inserted_reply = Comments_Collection.insert_one(reply_document)
            return inserted_reply.inserted_id
        except Exception as e:
            print(f"Error adding reply: {e}")
            return None

    @staticmethod
    def get_comment_by_id(comment_id):
        """
        Fetches a single comment by its ID and processes it for frontend display.
        This will fetch either a top-level comment or a reply.
        """
        comment = Comments_Collection.find_one({'_id': ObjectId(comment_id)})
        if not comment:
            return None

        # Enrich comment with user details
        user = Users_Collection.find_one({'_id': ObjectId(comment['CommentedBy'])})
        comment['CommentedBy'] = user['Name'] if user else 'Unknown User'

        # Format DateCommented for returning
        comment['FormattedDate'] = comment['DateCommented'].strftime("%H:%M %b %d, %Y")

        # Convert _id to 'id' string for frontend consumption
        comment['id'] = str(comment.pop('_id'))
        
        # Add 'replies' key for consistency if it's a top-level comment and you plan to expand it
        if 'parent_comment_id' not in comment or comment['parent_comment_id'] is None:
            comment['replies'] = [] # This ensures a consistent structure
            # You might want to fetch and populate direct replies here if this method is used in isolation
            # For now, get_all_video_comments handles full tree, so this is just for single lookup.

        return comment

    @staticmethod
    def update_comment_position(comment_id, x_pos, y_pos):
        """Updates the position of an existing comment. Only applies to top-level comments."""
        try:
            Comments_Collection.update_one(
                {'_id': ObjectId(comment_id), 'parent_comment_id': None}, # Only update if it's a top-level comment
                {'$set': {'x_pos': x_pos, 'y_pos': y_pos}}
            )
            return True
        except Exception as e:
            print(f"Error updating comment position: {e}")
            return False

    @staticmethod
    def delete_comment(comment_id):
        """
        Deletes a comment and all its direct replies.
        Also removes the top-level comment's reference from the video.
        """
        try:
            obj_comment_id = ObjectId(comment_id)

            # First, delete all replies whose parent_comment_id is the comment being deleted
            Comments_Collection.delete_many({'parent_comment_id': obj_comment_id})

            # Then, delete the top-level comment itself
            delete_result = Comments_Collection.delete_one({'_id': obj_comment_id})

            if delete_result.deleted_count == 0:
                print(f"Comment with ID {comment_id} not found for deletion.")
                return False

            # Finally, remove the top-level comment's ObjectId from any video that references it
            Videos_Collection.update_many(
                {'Comments': obj_comment_id},
                {'$pull': {'Comments': obj_comment_id}}
            )
            return True
        except Exception as e:
            print(f"Error deleting comment {comment_id}: {e}")
            return False