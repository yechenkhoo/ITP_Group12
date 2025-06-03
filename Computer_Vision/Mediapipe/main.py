from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import pandas as pd
import math
from tensorflow.keras.models import load_model
import mediapipe as mp
from google.cloud import storage
import tempfile
import csv
from angle_utils import calculate_and_draw_shoulder_tilt, calculate_and_draw_hip_tilt
from db_connection import Videos_Collection
from bson import ObjectId
import ffmpeg




app = Flask(__name__)

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Class names
class_names = ['P1', 'P10', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']

# Ideal angles
ideal_shoulder_tilt = {'P1': 8, 'P2': 24, 'P3': 35, 'P4': 37, 'P5': 33, 'P6': 12, 'P7': 30, 'P8': 38, 'P9': 45, 'P10': 6}
ideal_hip_tilt = {'P1': 1, 'P2': 4, 'P3': 8, 'P4': 9, 'P5': 7, 'P6': 8, 'P7': 11, 'P8': 12, 'P9': 14, 'P10': 5}

previous_class_index = -1


# Function to calculate custom standard deviation bounds
def calculate_custom_bounds(ideal_angle):
    stddev = ideal_angle  # Use the ideal angle as the standard deviation
    lower_bound = max(0, ideal_angle - stddev)  # Ensure the lower bound is not negative
    upper_bound = ideal_angle + stddev
    return lower_bound, upper_bound
def calculate_stddev(angles, ideal_angle):
    if len(angles) < 2:
        return float('inf')  # Return a large value to ensure no angle is marked as "Good" when there's insufficient data
    squared_diff = [(angle - ideal_angle) ** 2 for angle in angles]
    variance = sum(squared_diff) / len(angles)
    return math.sqrt(variance)

# Function to determine the status of tilt angles using the custom bounds
def get_tilt_status(current_angle, ideal_angle):
    lower_bound, upper_bound = calculate_custom_bounds(ideal_angle)

    if lower_bound <= current_angle <= upper_bound:
        return 'Good'
    elif abs(current_angle - ideal_angle) <= 2 * ideal_angle:  # Within two standard deviations
        return 'Bad'
    else:
        return 'Very Bad'

def evaluate_overall_status(statuses):
    overall = 0
    for status in statuses:
        if status == 'Good':
            overall += 2
        elif status == 'Bad':
            overall +=1
        else:
            overall +=0
    
    avg = overall/len(statuses)
    if(avg ==2):
        return 'Good'
    elif(avg >=1):
        return 'Bad'
    else:
        return 'Very Bad'

def extract_thumbnail(video_path, output_path):
    print("Extracting thumbnail...")
    try:
        (
            ffmpeg
            .input(video_path, ss="00:00:01")  # Seek to 1 second into the video
            .output(output_path, vframes=1, format='image2')  # Save a single frame
            .overwrite_output()  # Overwrite if the file exists
            .run()
        )
        print(f"Thumbnail saved to {output_path}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e}")

def convert_to_h264(input_path, output_path):
    try:
        ffmpeg.input(input_path).output(output_path, vcodec='libx264', preset='fast', crf=23).run()
        print(f"Converted {input_path} to {output_path}")
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    # Initialize a Cloud Storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a new blob in the bucket
    blob = bucket.blob(destination_blob_name)

    # Upload the local file to the blob
    blob.upload_from_filename(source_file_name)

    # Make the blob publicly accessible (if UBLA is disabled)
    # blob.make_public()

    # Return the direct public URL
    public_url = f"https://storage.googleapis.com/{bucket_name}/processed/{destination_blob_name}"
    print(f"Public URL: {public_url}")

    return blob.public_url

def upload_video_blob(bucket_name, source_file_name, destination_blob_name):
    # Initialize a Cloud Storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Create a new blob in the bucket
    blob = bucket.blob(destination_blob_name)

    # Upload the local file to the blob
    blob.upload_from_filename(source_file_name, content_type="video/mp4")

    # Make the blob publicly accessible (if UBLA is disabled)
    # blob.make_public()

    # Return the direct public URL
    public_url = f"https://storage.googleapis.com/{bucket_name}/processed/{destination_blob_name}"
    print(f"Public URL: {public_url}")

    return blob.public_url

# Google Cloud Storage download function
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")

# Function to draw bounding box
def draw_bounding_box(image, lm_list):
    min_x = min([lm.x for lm in lm_list])
    min_y = min([lm.y for lm in lm_list])
    max_x = max([lm.x for lm in lm_list])
    max_y = max([lm.y for lm in lm_list])

    height, width, _ = image.shape
    top_left = (int(min_x * width), int(min_y * height))
    bottom_right = (int(max_x * width), int(max_y * height))

    image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    return image

# Function to draw landmarks
def draw_landmarks(image, lm_list, connections, point_radius=2, line_thickness=1):
    height, width, _ = image.shape
    for lm in lm_list:
        center = (int(lm.x * width), int(lm.y * height))
        image = cv2.circle(image, center, point_radius, (0, 0, 255), -1)
    for connection in connections:
        start_idx, end_idx = connection
        start_point = (int(lm_list[start_idx].x * width), int(lm_list[start_idx].y * height))
        end_point = (int(lm_list[end_idx].x * width), int(lm_list[end_idx].y * height))
        image = cv2.line(image, start_point, end_point, (0, 255, 0), line_thickness)
    return image

previous_class_index = -1  # Index of the previous class in class_names
pose_class_angles = {
    pose: {"shoulder_tilt": [], "hip_tilt": [], "time_frame": [],
           "shoulder_tilt_status": [],"hip_tilt_status": [], "overall_status":[]} for pose in class_names
}

first_instance_added = {pose: False for pose in class_names}  # Track first instance of each pose class

valid_transitions = {
    'P1': ['P1','P2'],
    'P10': ['P10'],
    'P2': ['P2', 'P3'],
    'P3': ['P3', 'P4'],
    'P4': ['P4','P5'],
    'P5': ['P5', 'P6'],
    'P6': ['P6', 'P7'],
    'P7': ['P7', 'P8'],
    'P8': ['P8', 'P9'],
    'P9': ['P9', 'P10']
}

def is_next_class_valid(current_class_index, previous_class_index):
    # Checks if the next class is valid
    previous_class = class_names[previous_class_index]
    current_class = class_names[current_class_index]
    if current_class in valid_transitions[previous_class]:
        return True
    return False

# Pose processing route
@app.route('/process-video', methods=['POST'])
def process_video():
    print("DEBUG: /process-video route accessed")
    try:
        # Get the request data
        data = request.json
        print(f"Received request: {data}")

         # Extract MongoDB update fields from the request
        video_id = data['video_id']  # MongoDB document ID for the video
        video_path = data['video_path']  # Path in Google Cloud Storage
        classification_model = data['classification_model']
        output_video_path_gcs = data.get('output_video_path', None)
        output_csv_path_gcs = data.get('output_csv_path', None)
        output_angle_csv_path_gcs = data.get('output_angle_csv_path', None)
        # Bucket name where the models and video are stored
        bucket_name = 'golf-swing-video'

        # Download the required files from GCS to /tmp/
        download_blob(bucket_name, classification_model, '/tmp/' + os.path.basename(classification_model))
        download_blob(bucket_name, video_path, '/tmp/' + os.path.basename(video_path))

        # Update the paths to use the downloaded files in /tmp/
        classification_model = '/tmp/' + os.path.basename(classification_model)
        video_local_path = '/tmp/' + os.path.basename(video_path)

        # Load the classification model
        model = load_model(classification_model, compile=True)

        # Open the video file
        cap = cv2.VideoCapture(video_local_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open the video file'}), 400

        # Prepare video writer for output video
        output_video_path = '/tmp/processed_video'+video_id+'.mp4'
        h264_video_path = '/tmp/processed_video_h264'+video_id+'.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if unknown
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Prepare CSV file for storing predictions
        output_csv_path = '/tmp/predictions.csv'
        with open(output_csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Frame', 'Predicted Class', 'Confidence', 'Video Time(s)', 'Shoulder Tilt','Hip Tilt', 'Shoulder Tilt Status', 'Hip Tilt Status', 'Overall Status'])  # CSV header

            predictions = []
            frame_count = 0
            global previous_class_index

            # Process video frame by frame
            while True:
                success, img = cap.read()
                if not success:
                    break  # End of video
                status_list = []
                frame_count += 1
                video_time = frame_count / 30
                # Convert the frame to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = pose.process(img_rgb)

                if result.pose_landmarks:
                    lm_list = result.pose_landmarks.landmark

                    # Draw landmarks and bounding box
                    img = draw_bounding_box(img, lm_list)
                    img = draw_landmarks(img, lm_list, mp_pose.POSE_CONNECTIONS)

                    # Normalize landmarks for prediction
                    center_x = (lm_list[mp_pose.PoseLandmark.RIGHT_HIP].x +
                                lm_list[mp_pose.PoseLandmark.LEFT_HIP].x) / 2
                    center_y = (lm_list[mp_pose.PoseLandmark.RIGHT_HIP].y +
                                lm_list[mp_pose.PoseLandmark.LEFT_HIP].y) / 2
                    max_distance = max([
                        math.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2) for lm in lm_list
                    ])

                    pose_landmarks = np.array([
                        [(landmark.x - center_x) / max_distance,
                         (landmark.y - center_y) / max_distance,
                         landmark.z / max_distance,
                         landmark.visibility] for landmark in lm_list
                    ]).flatten()

                    # Predict with the model
                    pose_landmarks = np.expand_dims(pose_landmarks, axis=0)
                    prediction = model.predict(pose_landmarks)
                    current_class_index = np.argmax(prediction)

                    # Check if the predicted class is valid
                    pose_class = class_names[current_class_index]
                    confidence = np.max(prediction)
                    shoulder_tilt_angle = calculate_and_draw_shoulder_tilt(img, lm_list, pose_class)
                   
                # Calculate hip tilt
                    hip_tilt_angle = calculate_and_draw_hip_tilt(img, lm_list, pose_class)

                   # Use the revised function for checking tilt status
                    # Use the custom function for checking tilt status
                    shoulder_tilt_status = get_tilt_status(shoulder_tilt_angle, ideal_shoulder_tilt[pose_class])
                    status_list.append(shoulder_tilt_status)
                    hip_tilt_status = get_tilt_status(hip_tilt_angle, ideal_hip_tilt[pose_class])
                    status_list.append(hip_tilt_status)
                    overall_status = evaluate_overall_status(status_list)
                    # Updated logic to handle class index validity
                    if previous_class_index == -1 or is_next_class_valid(current_class_index, previous_class_index):
                        if not first_instance_added[pose_class]:
                            pose_class_angles[pose_class]["shoulder_tilt"].append(shoulder_tilt_angle)
                            pose_class_angles[pose_class]["hip_tilt"].append(hip_tilt_angle)
                            pose_class_angles[pose_class]["time_frame"].append(video_time)
                            pose_class_angles[pose_class]["shoulder_tilt_status"].append(shoulder_tilt_status)
                            pose_class_angles[pose_class]["hip_tilt_status"].append(hip_tilt_status)
                            pose_class_angles[pose_class]['overall_status'].append(overall_status)
                            # pose_class_angles[pose_class]["time_frame"].append(video_time)
                            first_instance_added[pose_class] = True
                        previous_class_index = current_class_index
                    else:
                        previous_class = class_names[previous_class_index]
                        current_class = class_names[current_class_index]
                        print(f"Invalid transition from {previous_class} to {current_class}")
                        current_class_index = -1  # Reset to -1 for invalid transition
                    
                    pose_class_text = class_names[current_class_index] if current_class_index != -1 else 'Unknown Pose'
                    
                    # Annotate the frame with the prediction
                    cv2.putText(img, f"{pose_class} ({confidence:.2f})", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    # Append prediction to CSV
                    csv_writer.writerow([frame_count, pose_class, confidence, video_time, shoulder_tilt_angle, hip_tilt_angle, shoulder_tilt_status, hip_tilt_status, overall_status])

                    # Append prediction to the JSON response
                    predictions.append({
                        'frame': frame_count,
                        'predicted_class': pose_class,
                        'confidence': float(confidence),
                        'video_time': video_time,
                        'shoulder_tilt_angle': shoulder_tilt_angle,
                        'hip_tilt_angle':hip_tilt_angle,
                        'shoulder_tilt_status': shoulder_tilt_status,
                        'hip_tilt_status': hip_tilt_status,
                        'overall_status': overall_status
                    })

                # Write the annotated frame to the output video
                out.write(img)

        cap.release()
        out.release()  # Close video writer
        # Write angles to CSV
        convert_to_h264(output_video_path, h264_video_path)
        output_angles_csv_path = '/tmp/angles'+video_id+'.csv'
        base_name = os.path.splitext(video_path.split('/')[-1])[0]
        thumbnail_path = 'tmp/'+ base_name + '.jpg'
        thumbnail_path_gcp = 'thumbnails/'+ base_name + '.jpg'
        #extract_thumbnail(h264_video_path, thumbnail_path)

        with open(output_angles_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Pose Class', 'Shoulder Tilt', 'Hip Tilt', 'Time Frame', 'Shoulder Tilt Status', 'Hip Tilt Status', 'Overall Status'])
            # Specify the desired order
            ordered_classes = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
                # Iterate over pose_class_angles in the specified order
            for pose_class in ordered_classes:
                angles = pose_class_angles[pose_class]
                shoulder_tilt = ', '.join(map(str, angles["shoulder_tilt"])) if angles["shoulder_tilt"] else 'No data'
                hip_tilt = ', '.join(map(str, angles["hip_tilt"])) if angles["hip_tilt"] else 'No data'
                time_frame = ', '.join(map(str, angles["time_frame"])) if angles["time_frame"] else 'No data'
                shoulder_tilt_status = ', '.join(map(str, angles["shoulder_tilt_status"])) if angles["shoulder_tilt_status"] else 'No data'
                hip_tilt_status = ', '.join(map(str, angles["hip_tilt_status"])) if angles["hip_tilt_status"] else 'No data'
                overall_status = ', '.join(map(str, angles["overall_status"])) if angles["overall_status"] else 'No data'
                writer.writerow([pose_class, shoulder_tilt, hip_tilt, time_frame, shoulder_tilt_status, hip_tilt_status, overall_status])

        # Upload the thumbnail to GCS if `thumbnail_path_gcp` is specified
        #if thumbnail_path_gcp:
        #    output_thumbnail_url = upload_blob(bucket_name, thumbnail_path, thumbnail_path_gcp)
        #    print(f"Uploaded output thumbnail to GCS: {output_thumbnail_url}")
        #else:
        #    output_thumbnail_url = None

        # Upload the output video to GCS if `output_video_path` is specified
        if output_video_path_gcs:
            output_video_url = upload_video_blob(bucket_name, h264_video_path, output_video_path_gcs)
            print(f"Uploaded output video to GCS: {output_video_url}")
        else:
            output_video_url = None
            

        # Upload the CSV file to GCS if `output_csv_path` is specified
        if output_csv_path_gcs:
            output_csv_url = upload_blob(bucket_name, output_csv_path, output_csv_path_gcs)
            print(f"Uploaded predictions CSV to GCS: {output_csv_url}")
            output_angle_csv_url = upload_blob(bucket_name, output_angles_csv_path, output_angle_csv_path_gcs)
            print(f"Uploaded angles CSV to GCS: {output_angle_csv_url}")
        else:
            output_csv_url = None


         # Clean up temporary files
        temp_files = [
            classification_model,
            video_local_path,
            output_video_path,
            h264_video_path,
            output_csv_path,
            output_angles_csv_path,
            thumbnail_path
        ]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return jsonify({
            'status': 'Processing complete',
            'predictions': predictions,
            'output_video': output_video_url,
            'output_csv': output_csv_url,
            'output_angle_csv': output_angle_csv_url,
            #'output_thumbnail' : output_thumbnail_url
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500 

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "API is up and running"}), 200

@app.route('/list_routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "rule": str(rule)
        })
    return jsonify(routes)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Default to 8080 if not set
    app.run(host="0.0.0.0", port=port)