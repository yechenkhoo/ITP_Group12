# main.py

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
from angle_utils import (
    calculate_and_draw_shoulder_tilt,
    calculate_and_draw_hip_tilt,
    calculate_and_draw_shoulder_rotation,
    calculate_and_draw_hip_rotation,
    calculate_and_draw_forward_tilt_dtl,
    calculate_and_draw_lead_arm_angle,
    calculate_and_draw_knee_bend
)
from db_connection import Videos_Collection
from bson import ObjectId
import ffmpeg
import traceback




app = Flask(__name__)

# Mediapipe setup
mp_pose = mp.solutions.pose

# Class names
class_names = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']

# Ideal tilt angles
ideal_shoulder_tilt = {'P1': 8, 'P2': 24, 'P3': 35, 'P4': 37, 'P5': 33, 'P6': 12, 'P7': 30, 'P8': 38, 'P9': 45, 'P10': 6}
ideal_hip_tilt = {'P1': 1, 'P2': 4, 'P3': 8, 'P4': 9, 'P5': 7, 'P6': 8, 'P7': 11, 'P8': 12, 'P9': 14, 'P10': 5}

# Ideal rotation angles
ideal_shoulder_rotation = {'P1': 8, 'P2': 42, 'P3': 77, 'P4': 88, 'P5': 65, 'P6': 1, 'P7': 31, 'P8': 51, 'P9': 71, 'P10': 138}
ideal_hip_rotation = {'P1': 4, 'P2': 18, 'P3': 36, 'P4': 44, 'P5': 7, 'P6': 32, 'P7': 43, 'P8': 53, 'P9': 63, 'P10': 109}

previous_class_index = -1

csv_field_order = [
    'shoulder_tilt', 'hip_tilt', 'shoulder_rotation', 'hip_rotation',
    'forward_tilt', 'lead_arm_angle', 'knee_bend', 'time_frame',
    'shoulder_tilt_status', 'hip_tilt_status', 'shoulder_rotation_status',
    'hip_rotation_status', 'lead_arm_angle_status', 'overall_status'
]

names_order_list = [
    'Frame', 'Predicted Class', 'Confidence', 'Video Time(s)', 
    'Shoulder Tilt', 'Hip Tilt', 'Shoulder Rotation', 'Hip Rotation', 
    'Forward Tilt', 'Lead Arm Angle', 'Knee Bend',
    'Shoulder Tilt Status', 'Hip Tilt Status', 'Shoulder Rotation Status', 
    'Hip Rotation Status', 'Lead Arm Angle Status', 'Overall Status'
]

# Function to determine the status of tilt angles using the custom bounds
def get_tilt_status(current_angle, ideal_angle):
    """
    Evaluate angle deviation from ideal using 5-point scale.
    
    Thresholds:
    - Very Good (Excellent): ≤5° deviation
    - Good: ≤10° deviation
    - Average (OK): ≤15° deviation
    - Bad (Needs Work): ≤20° deviation
    - Very Bad (Critical): >20° deviation
    """
    deviation = abs(current_angle - ideal_angle)
    
    if deviation <= 5:
        return 'Very Good'
    elif deviation <= 10:
        return 'Good'
    elif deviation <= 15:
        return 'Average'
    elif deviation <= 20:
        return 'Bad'
    else:
        return 'Very Bad'
    

def get_lead_arm_status(current_angle):
    """
    Evaluate lead arm angle using custom thresholds.
    Only applicable for P1-P9 (P10 excluded).
    
    Thresholds:
    - Very Good: 170-180°
    - Good: 160-170°
    - Average: 150-160°
    - Bad: 140-150°
    - Very Bad: <140°
    """
    if 170 <= current_angle <= 180:
        return 'Very Good'
    elif 160 <= current_angle < 170:
        return 'Good'
    elif 150 <= current_angle < 160:
        return 'Average'
    elif 140 <= current_angle < 150:
        return 'Bad'
    else:  # < 140
        return 'Very Bad'
    

def evaluate_overall_status(statuses):
    """
    Calculate overall status from individual angle statuses.
    
    Scoring (5-point scale):
    - Very Good: 4 points
    - Good: 3 points
    - Average: 2 points
    - Bad: 1 point
    - Very Bad: 0 points
    """
    score_map = {
        'Very Good': 4,
        'Good': 3,
        'Average': 2,
        'Bad': 1,
        'Very Bad': 0
    }
    
    total = sum(score_map.get(status, 0) for status in statuses)
    avg = total / len(statuses)
    
    # Thresholds for overall status
    if avg >= 3.5:
        return 'Very Good'
    elif avg >= 2.5:
        return 'Good'
    elif avg >= 1.5:
        return 'Average'
    elif avg >= 0.5:
        return 'Bad'
    else:
        return 'Very Bad'

def convert_to_h264(input_path, output_path):
    try:
        ffmpeg.input(input_path).output(output_path, vcodec='libx264', preset='fast', crf=23).run()
        print(f"Converted {input_path} to {output_path}")
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")

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

valid_transitions = {
    'P1': ['P1','P2'],
    'P2': ['P2', 'P3'],
    'P3': ['P3', 'P4'],
    'P4': ['P4','P5'],
    'P5': ['P5', 'P6'],
    'P6': ['P6', 'P7'],
    'P7': ['P7', 'P8'],
    'P8': ['P8', 'P9'],
    'P9': ['P9', 'P10'],
    'P10': ['P10']
}

def is_next_class_valid(current_class_index, previous_class_index):
    # Checks if the next class is valid
    previous_class = class_names[previous_class_index]
    current_class = class_names[current_class_index]
    
    if current_class in valid_transitions[previous_class]:
        return True
    return False

# Pose processing route
@app.route('/process-video-local', methods=['POST'])
def process_video():
    print("DEBUG: /process-video route accessed")
    try:
        # CREATE NEW POSE INSTANCE FOR THIS REQUEST
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Get the request data
        data = request.json
        print(f"Received request: {data}")

        # Updated with new body angles
        pose_class_angles = {
            pose: {field: [] for field in csv_field_order}
            for pose in class_names
        }

        # first_instance_added = {pose: False for pose in class_names}
        best_pose_frames = {
            pose: {'confidence': 0, 'frame': None} for pose in class_names
        }

        previous_class_index = -1

        video_filename = data['video_path']  # e.g., "tom_2.mp4"
        model_filename = data['classification_model']  # e.g., "best_model.keras"

        # Set up local paths
        video_local_path = os.path.join('FO', 'input', video_filename)
        classification_model = os.path.join('FO', 'input', model_filename)
        classification_model = model_filename

        # Generate a unique video_id based on filename and timestamp
        video_id = "TEST"

        camera_is_face_on = True
        if "dtl" in video_filename.lower():
            camera_is_face_on = False
        
        print(f"Processing video: {video_local_path}")
        print(f"Using model: {classification_model}")

        # Verify files exist
        if not os.path.exists(video_local_path):
            return jsonify({'error': f'Video file not found: {video_local_path}'}), 400
        if not os.path.exists(classification_model):
            return jsonify({'error': f'Model file not found: {classification_model}'}), 400

        # Load the classification model
        model = load_model(classification_model, compile=True)

        # Open the video file
        cap = cv2.VideoCapture(video_local_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open the video file'}), 400

        # Prepare output directories
        os.makedirs('FO/output', exist_ok=True)
        os.makedirs('FO/output/videos', exist_ok=True)
        os.makedirs('FO/output/csv', exist_ok=True)
        os.makedirs('FO/output/thumbnails', exist_ok=True)

        # Prepare video writer for output video
        output_video_path = f'FO/output/processed_{video_id}.mp4'
        h264_video_path = f'FO/output/processed_{video_id}_h264.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if unknown
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Prepare CSV file for storing predictions
        output_csv_path = f'FO/output/predictions_{video_id}.csv'
        
        # Dictionary to store local paths of pose class thumbnails
        pose_thumbnails = {}

        with open(output_csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(names_order_list)

            predictions = []
            frame_count = 0

            consecutive_class_buffer = []

            # Process video frame by frame
            while True:
                success, img = cap.read()
                if not success:
                    break  # End of video
                
                # VALIDATE FRAME
                if img is None or img.size == 0:
                    print(f"Warning: Invalid frame at {frame_count}")
                    continue
                status_list = []
                frame_count += 1
                video_time = frame_count / 30
                # Convert the frame to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Image processing for better detection
                img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=15)  # enhance contrast
                img_rgb = cv2.GaussianBlur(img_rgb, (3,3), 0)               # smooth noise
                gamma = 1.3  # >1 brightens image
                img_rgb = np.power(img_rgb / 255.0, 1.0 / gamma)
                img_rgb = np.uint8(img_rgb * 255)

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

                    # --- Stabilisation logic: accept if 3 consecutive predictions ---
                    consecutive_class_buffer.append(current_class_index)
                    if len(consecutive_class_buffer) > 3:
                        consecutive_class_buffer.pop(0)
                    # if len(consecutive_class_buffer) == 3 and all(idx == current_class_index for idx in consecutive_class_buffer):
                    #     accept_transition = True
                    # else:
                    #     accept_transition = (previous_class_index == -1 or is_next_class_valid(current_class_index, previous_class_index))
                    if len(consecutive_class_buffer) == 3 and all(idx == current_class_index for idx in consecutive_class_buffer):
                        accept_transition = is_next_class_valid(current_class_index, previous_class_index)
                    else:
                        accept_transition = (previous_class_index == -1 or is_next_class_valid(current_class_index, previous_class_index))

                    # Check if the predicted class is valid (with stabilization)
                    pose_class = class_names[current_class_index]
                    confidence = np.max(prediction)

                    # Calculate tilts
                    shoulder_tilt_angle = calculate_and_draw_shoulder_tilt(img, lm_list, pose_class)
                    hip_tilt_angle = calculate_and_draw_hip_tilt(img, lm_list, pose_class)

                    # Use the custom function for checking tilt status
                    shoulder_tilt_status = get_tilt_status(shoulder_tilt_angle, ideal_shoulder_tilt[pose_class])
                    status_list.append(shoulder_tilt_status)
                    hip_tilt_status = get_tilt_status(hip_tilt_angle, ideal_hip_tilt[pose_class])
                    status_list.append(hip_tilt_status)

                    # New body angles (Rotations, forward tilt, lead arm, knee bend)
                    if camera_is_face_on:
                        shoulder_rotation_angle = calculate_and_draw_shoulder_rotation(img, lm_list, pose_class)
                        hip_rotation_angle = calculate_and_draw_hip_rotation(img, lm_list, pose_class)
                        forward_tilt_angle = None
                        knee_bend_angle = None
                        
                        # Calculate statuses for new angles
                        shoulder_rotation_status = get_tilt_status(shoulder_rotation_angle, ideal_shoulder_rotation[pose_class])
                        status_list.append(shoulder_rotation_status)
                        hip_rotation_status = get_tilt_status(hip_rotation_angle, ideal_hip_rotation[pose_class])
                        status_list.append(hip_rotation_status)

                        # Lead arm angle status only for P1-P9 (exclude P10)
                        if pose_class != 'P10':
                            lead_arm_angle = calculate_and_draw_lead_arm_angle(img, lm_list, pose_class)
                            lead_arm_angle_status = get_lead_arm_status(lead_arm_angle)
                            status_list.append(lead_arm_angle_status)
                        else:
                            lead_arm_angle = None
                            lead_arm_angle_status = '-' # (Not applicable for P10)

                    else:
                        shoulder_rotation_angle = None
                        shoulder_rotation_status = None
                        hip_rotation_angle = None
                        hip_rotation_status = None
                        lead_arm_angle = None
                        lead_arm_angle_status = None
                        forward_tilt_angle = calculate_and_draw_forward_tilt_dtl(img, lm_list, pose_class)
                        knee_bend_angle = calculate_and_draw_knee_bend(img, lm_list, pose_class)

                    overall_status = evaluate_overall_status(status_list)
                    # Updated logic to handle class index validity
                    # if previous_class_index == -1 or is_next_class_valid(current_class_index, previous_class_index):
                    # if accept_transition:
                    print("Confidence is ", confidence)
                    if confidence > best_pose_frames[pose_class]['confidence']:
                        best_pose_frames[pose_class]['confidence'] = confidence
                        best_pose_frames[pose_class]['frame'] = img.copy()
                        best_pose_frames[pose_class]['data'] = {
                            "shoulder_tilt": shoulder_tilt_angle,
                            "hip_tilt": hip_tilt_angle,
                            "shoulder_rotation": shoulder_rotation_angle,
                            "hip_rotation": hip_rotation_angle,
                            "forward_tilt": forward_tilt_angle,
                            "lead_arm_angle": lead_arm_angle,
                            "knee_bend": knee_bend_angle,
                            "time_frame": video_time,
                            "shoulder_tilt_status": shoulder_tilt_status,
                            "hip_tilt_status": hip_tilt_status,
                            "shoulder_rotation_status": shoulder_rotation_status,
                            "hip_rotation_status": hip_rotation_status,
                            "lead_arm_angle_status": lead_arm_angle_status,
                            "overall_status": overall_status
                        }

                    # previous_class_index = current_class_index
                    # else:
                    #     previous_class = class_names[previous_class_index]
                    #     current_class = class_names[current_class_index]
                    #     print(f"Invalid transition from {previous_class} to {current_class}")
                    #     current_class_index = -1  # Reset to -1 for invalid transition
                    
                    pose_class_text = class_names[current_class_index] if current_class_index != -1 else 'Unknown Pose'
                    
                    # Annotate the frame with the prediction
                    if camera_is_face_on:
                        cv2.putText(img, f"{pose_class} ({confidence:.2f})", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    # Append prediction to CSV
                    csv_writer.writerow([
                        frame_count, pose_class, confidence, video_time, 
                        shoulder_tilt_angle, hip_tilt_angle, shoulder_rotation_angle, 
                        hip_rotation_angle, forward_tilt_angle, lead_arm_angle, knee_bend_angle,
                        shoulder_tilt_status, hip_tilt_status, shoulder_rotation_status, 
                        hip_rotation_status, lead_arm_angle_status, overall_status
                    ])

                    # Append prediction to the JSON response
                    predictions.append({
                        'frame': frame_count,
                        'predicted_class': pose_class,
                        'confidence': float(confidence),
                        'video_time': video_time,
                        'shoulder_tilt_angle': shoulder_tilt_angle,
                        'hip_tilt_angle': hip_tilt_angle,
                        'shoulder_rotation_angle': shoulder_rotation_angle,
                        'hip_rotation_angle': hip_rotation_angle,
                        'forward_tilt_angle': forward_tilt_angle,
                        'lead_arm_angle': lead_arm_angle,
                        'knee_bend_angle': knee_bend_angle,
                        'shoulder_tilt_status': shoulder_tilt_status,
                        'hip_tilt_status': hip_tilt_status,
                        'shoulder_rotation_status': shoulder_rotation_status,
                        'hip_rotation_status': hip_rotation_status,
                        'lead_arm_angle_status': lead_arm_angle_status,
                        'overall_status': overall_status
                    })

                # Write the annotated frame to the output video
                out.write(img)

        cap.release()
        out.release()  # Close video writer

        for pose_class, data in best_pose_frames.items():
            if data['frame'] is not None:
                try:
                    local_thumbnail_path = f"FO/output/thumbnails/thumb_{video_id}_{pose_class}.jpg"
                    cv2.imwrite(local_thumbnail_path, data['frame'])
                    pose_thumbnails[pose_class] = local_thumbnail_path

                    for field in csv_field_order:
                        pose_class_angles[pose_class][field].append(data['data'][field])
    
                except Exception as e:
                    print(f"Error saving thumbnail for {pose_class}: {e}")
        
        # Convert to H264 if function exists
        try:
            convert_to_h264(output_video_path, h264_video_path)
            final_video_path = h264_video_path
        except:
            print("H264 conversion not available, using original video")
            final_video_path = output_video_path
            
        output_angles_csv_path = f'FO/output/angles_{video_id}.csv'

        # Write angles and thumbnail URLs to CSV
        with open(output_angles_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Used list of attributes (csv_field_order) instead of hardcoding text
            csv_header = ['Pose Class'] + [field.replace('_', ' ').title() for field in csv_field_order] + ['Pose Thumbnail URL']
            writer.writerow(csv_header)

            ordered_classes = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']

            for pose_class in ordered_classes:
                if pose_class not in pose_class_angles:
                    continue

                angles = pose_class_angles[pose_class]

                has_data = any(angles.get(field) for field in csv_field_order)
                if not has_data:
                    print(f"Skipping {pose_class}: no angle data.")
                    continue
                
                row_data = [pose_class]
                
                for field in csv_field_order:
                    value = ', '.join(map(str, angles[field])) if angles[field] else ''
                    row_data.append(value)
                
                # Add thumbnail URL
                thumbnail_url = pose_thumbnails.get(pose_class, "")
                row_data.append(thumbnail_url)
                
                writer.writerow(row_data)

        print(f"DEBUG: pose_thumbnails content: {pose_thumbnails}")
        
        return jsonify({
            'status': 'Processing complete',
            'predictions': predictions,
            'output_video': final_video_path,
            'output_csv': output_csv_path,
            'output_angle_csv': output_angles_csv_path,
            'output_pose_images': pose_thumbnails,
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        # CLEANUP: Always close the pose instance
        try:
            pose.close()
            print("Pose instance closed successfully")
        except Exception as cleanup_error:
            print(f"Error closing pose instance: {cleanup_error}")

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