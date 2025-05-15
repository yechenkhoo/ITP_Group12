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

        user_data = {
            'Email': email,
            'Name': name,
            'Password': password,
            'Role': role,
            'CreatedBy': ObjectId(created_by),
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
            if student:
                student['id'] = str(student.pop('_id'))  # Replace _id with id as a string
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
            bucket_name = 'itp-se-team13'

            # Initialize GCP storage client
            storage_client = get_google_cloud_storage_client()

            # Get the bucket
            bucket = storage_client.bucket(bucket_name)

            # Generate a unique blob name
            unique_id = uuid.uuid4().hex  # Generate a unique ID
            blob_name = f'videos/{unique_id}_{file_name}'

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
                print(f"Deleting original video: {blob_name}")
                blob.delete()  # Delete the video file
            else:
                print(f"Skipping deletion due to processing error: {response.get('error')}")

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
            gcp_function_url = "https://itp-se-team13-357970185934.asia-southeast1.run.app/process-video"

            # Prepare the request payload
            payload = {
                
                "classification_model": "models/frontalV2.keras",
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
            user = Users_Collection.find_one({'_id': ObjectId(comment['CommentedBy'])})
            comment['CommentedBy'] = user['Name'] if user else 'Unknown User'

            # Format DateCommented for returning
            comment['FormattedDate'] = comment['DateCommented'].strftime("%H:%M %b %d, %Y")

        return comments


class Comment:
    """Handles operations related to comments."""

    @staticmethod
    def add_comment(current_user_id, video_id, comment_text):
        """
        Adds a comment to a video and associates it with the commenting user.
        """

        try:
            # Create comment document
            
            comment_document = {
                'Comment': comment_text,
                'CommentedBy': ObjectId(current_user_id),
                'DateCommented': datetime.now(),
            }
            Comments_Collection.insert_one(comment_document)

            # Link the comment to the video
            Videos_Collection.update_one(
                {'_id': ObjectId(video_id)},
                {'$push': {'Comments': comment_document['_id']}}
            )
        except Exception as e:
            print(f"Error adding comment: {e}")
