import os
from keras.models import load_model  # type: ignore
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import argparse
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
from angle_utils import calculate_and_draw_shoulder_tilt, calculate_and_draw_hip_tilt

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to saved .h5 model, eg: dir/model.h5")
ap.add_argument("-c", "--conf", type=float, required=True,
                help="min prediction conf to detect pose class (0<conf<1)")
ap.add_argument("-i", "--source", type=str, required=True,
                help="path to sample image")
ap.add_argument("--save", action='store_true',
                help="Save video")

args = vars(ap.parse_args())
source = args["source"]
path_saved_model = args["model"]
threshold = args["conf"]
save = args['save']

# Constants
torso_size_multiplier = 2.5
n_landmarks = 33
n_dimensions = 3
landmark_names = [
    'nose',
    'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky_1', 'right_pinky_1',
    'left_index_1', 'right_index_1',
    'left_thumb_2', 'right_thumb_2',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]
class_names = [
    'P1', 'P10', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'
]
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),     # Face
    (0, 4), (4, 5), (5, 6), (6, 8),     # Face
    (9, 10),                            # Shoulders
    (11, 12), (11, 13), (13, 15),       # Left Arm
    (12, 14), (14, 16),                 # Right Arm
    (11, 23), (12, 24),                 # Torso
    (23, 24), (23, 25), (24, 26),       # Hips
    (25, 27), (27, 29), (29, 31),       # Left Leg
    (26, 28), (28, 30), (30, 32)        # Right Leg
]

# Initialize MediaPipe Pose and Keras model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
pose_drawer = mp.solutions.drawing_utils
model = load_model(path_saved_model, compile=True)

col_names = []
for i in range(n_landmarks):
    name = mp_pose.PoseLandmark(i).name
    name_x = name + '_X'
    name_y = name + '_Y'
    name_z = name + '_Z'
    name_v = name + '_V'
    col_names.append(name_x)
    col_names.append(name_y)
    col_names.append(name_z)
    col_names.append(name_v)

def adjust_brightness_contrast(frame, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(frame, alpha_b, frame, 0, gamma_b)
    else:
        buf = frame.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def reduce_noise(frame):
    return cv2.medianBlur(frame, 5)

# Draw Bounding Box
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
    pose: {"shoulder_tilt": [], "hip_tilt": []} for pose in class_names
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

if source.endswith(('.jpg', '.jpeg', '.png')):
    path_to_img = source
    # Load sample Image
    img = cv2.imread(path_to_img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply preprocessing steps
    img_processed = adjust_brightness_contrast(img_rgb, brightness=-10, contrast=20)
    img_processed = reduce_noise(img_processed)

    result = pose.process(img_processed)
    if result.pose_landmarks:
        lm_list = result.pose_landmarks.landmark
        img = draw_bounding_box(img, lm_list)
        img = draw_landmarks(img, lm_list, POSE_CONNECTIONS)

        # Preprocessing and prediction code...

        # Get landmarks and scale it to the same size as the input image
        pose_landmarks = np.array(
            [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in lm_list],
            dtype=np.float32)

        max_distance = 0
        center_x = (lm_list[landmark_names.index('right_hip')].x +
                    lm_list[landmark_names.index('left_hip')].x) * 0.5
        
        
        center_y = (lm_list[landmark_names.index('right_hip')].y +
                    lm_list[landmark_names.index('left_hip')].y) * 0.5

        shoulders_x = (lm_list[landmark_names.index('right_shoulder')].x +
                        lm_list[landmark_names.index('left_shoulder')].x) * 0.5
        shoulders_y = (lm_list[landmark_names.index('right_shoulder')].y +
                        lm_list[landmark_names.index('left_shoulder')].y) * 0.5

        for lm in lm_list:
            distance = math.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2)
            if distance > max_distance:
                max_distance = distance
        torso_size = math.sqrt((shoulders_x - center_x) ** 2 + (shoulders_y - center_y) ** 2)
        max_distance = max(torso_size * torso_size_multiplier, max_distance)

        pre_lm = list(np.array([[(landmark.x-center_x)/max_distance, (landmark.y-center_y)/max_distance,
                                    landmark.z/max_distance, landmark.visibility] for landmark in lm_list]).flatten())
        data = pd.DataFrame([pre_lm], columns=col_names)
        predict = model.predict(data)[0]

        if max(predict) > threshold:
            pose_class = class_names[predict.argmax()]
            print('predictions: ', predict)
            print('predicted Pose Class: ', pose_class)
        else:
            pose_class = 'Unknown Pose'
            print('[INFO] Predictions is below given Confidence!!')

        # Calculate shoulder tilt
        if all(lm_list[i].visibility > 0.1 for i in 
                [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value]):
            shoulder_tilt_angle = calculate_and_draw_shoulder_tilt(img, lm_list, pose_class)
                        # Append shoulder tilt angle
            if pose_class != "Unknown Pose":
                if not first_instance_added[pose_class]["shoulder_tilt"]:
                    pose_class_angles[pose_class]["shoulder_tilt"].append(shoulder_tilt_angle)
                    first_instance_added[pose_class]["shoulder_tilt"] = True


        # Calculate hip tilt
        if all(lm_list[i].visibility > 0.1 for i in 
                [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]):
            hip_tilt_angle = calculate_and_draw_hip_tilt(img, lm_list, pose_class)
            if pose_class != "Unknown Pose":
                if not first_instance_added[pose_class]["hip_tilt"]:
                    pose_class_angles[pose_class]["hip_tilt"].append(hip_tilt_angle)
                    first_instance_added[pose_class]["hip_tilt"] = True
    # Show Result
    img = cv2.putText(
        img, f'{class_names[predict.argmax()]}',
        (40, 50), cv2.FONT_HERSHEY_PLAIN,
        2, (255, 0, 255), 2
    )

    if save:
        os.makedirs('ImageOutput', exist_ok=True)
        img_full_name = os.path.split(path_to_img)[1]
        img_name = os.path.splitext(img_full_name)[0]
        path_to_save_img = f'ImageOutput/{img_name}.jpg'
        cv2.imwrite(f'{path_to_save_img}', img)
        print(f'[INFO] Output Image Saved in {path_to_save_img}')

    cv2.imshow('Output Image', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    print('[INFO] Inference on Test Image is Ended...')

else:
    if source.isnumeric():
        source = int(source)

    cap = cv2.VideoCapture(source)
    source_width = int(cap.get(3))
    source_height = int(cap.get(4))

    # Prepare Video Writer
    if save:
        os.makedirs('VideoOutput', exist_ok=True)
        out_video = cv2.VideoWriter(
            'VideoOutput/output.mp4',  
            cv2.VideoWriter_fourcc(*'mp4v'),  
            30, (source_width, source_height)
        )

    while True:
        success, img = cap.read()
        if not success:
            print('[ERROR] Failed to Read Video feed')
            break

        adjusted_frame = adjust_brightness_contrast(img, brightness=-10, contrast=20)
        noisy_frame = reduce_noise(adjusted_frame)
        img_rgb = cv2.cvtColor(noisy_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            lm_list = result.pose_landmarks.landmark
            img = draw_bounding_box(img, lm_list)
            img = draw_landmarks(img, lm_list, POSE_CONNECTIONS)

            # Preprocessing and prediction code...

            # Get landmarks and scale it to the same size as the input image
            pose_landmarks = np.array(
                [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in lm_list],
                dtype=np.float32)
            
            max_distance = 0

            # Find center of hips and shoulders
            center_x = (lm_list[landmark_names.index('right_hip')].x +
                        lm_list[landmark_names.index('left_hip')].x) * 0.5
            center_y = (lm_list[landmark_names.index('right_hip')].y +
                        lm_list[landmark_names.index('left_hip')].y) * 0.5

            shoulders_x = (lm_list[landmark_names.index('right_shoulder')].x +
                            lm_list[landmark_names.index('left_shoulder')].x) * 0.5
            shoulders_y = (lm_list[landmark_names.index('right_shoulder')].y +
                            lm_list[landmark_names.index('left_shoulder')].y) * 0.5

            # Calculate max_distance from center and torso_size
            for lm in lm_list:
                distance = math.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2)
                if distance > max_distance:
                    max_distance = distance
            torso_size = math.sqrt((shoulders_x - center_x) ** 2 + (shoulders_y - center_y) ** 2)
            max_distance = max(torso_size * torso_size_multiplier, max_distance)

            # Preprocess landmarks
            pre_lm = list(np.array([[(landmark.x-center_x)/max_distance, (landmark.y-center_y)/max_distance,
                                    landmark.z/max_distance, landmark.visibility] for landmark in lm_list]).flatten())
            data = pd.DataFrame([pre_lm], columns=col_names)
            predict = model.predict(data)[0]

            if max(predict) > threshold:
                pose_class = class_names[predict.argmax()]
                current_class_index = predict.argmax()
                print('predictions: ', predict)
                print('predicted Pose Class: ', class_names[current_class_index])
                # Calculate shoulder tilt
                shoulder_tilt_angle = calculate_and_draw_shoulder_tilt(img, lm_list, pose_class)
                
                # Calculate hip tilt
                hip_tilt_angle = calculate_and_draw_hip_tilt(img, lm_list, pose_class)
                # Updated logic to handle class index validity
                if previous_class_index == -1 or is_next_class_valid(current_class_index, previous_class_index):
                    if not first_instance_added[pose_class]:
                        pose_class_angles[pose_class]["shoulder_tilt"].append(shoulder_tilt_angle)
                        pose_class_angles[pose_class]["hip_tilt"].append(hip_tilt_angle)
                        first_instance_added[pose_class] = True
                    previous_class_index = current_class_index
                else:
                    print(f"Invalid transition from {previous_class_index} to {current_class_index}")
                    current_class_index = -1  # Reset to -1 for invalid transition
            else:
                pose_class = 'Unknown Pose'
                print('[INFO] Predictions is below given Confidence!!')
                current_class_index = -1

            if current_class_index != -1:
                print(f'[INFO] Using predicted Pose Class: {class_names[current_class_index]}')
            else:
                print('[INFO] No valid pose class detected.')

            # Get Coordinates of Landmarks and draw angle
            if all(lm_list[i].visibility > 0.1 for i in 
                    [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value]):
                 # Calculate shoulder tilt
                shoulder_tilt_angle = calculate_and_draw_shoulder_tilt(img, lm_list, pose_class)
                
                # Calculate hip tilt
                hip_tilt_angle = calculate_and_draw_hip_tilt(img, lm_list, pose_class)

                
                if pose_class != "Unknown Pose" and not first_instance_added[pose_class]:
                    pose_class_angles[pose_class]["shoulder_tilt"].append(shoulder_tilt_angle)
                    pose_class_angles[pose_class]["hip_tilt"].append(hip_tilt_angle)
                    first_instance_added[pose_class] = True

            # Display Result
            pose_class_text = class_names[current_class_index] if current_class_index != -1 else 'Unknown Pose'
            text_color = (0, 0, 255) if max(predict) < threshold else (0, 255, 0) # Red for low confidence, Green for high confidence
            img = cv2.putText(
                img, f'Classified as: {pose_class_text}',
                (40, 80), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2
            )
            
            img = cv2.putText(
                img, f'Predicted: {pose_class}: {max(predict):.2f}',
                (40, 110), cv2.FONT_HERSHEY_PLAIN,
                2, text_color, 2
            )

        # Write Frame to Video
        if save:
            out_video.write(img)

        cv2.imshow('Output Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ideal_shoulder_tilt = {'P1': 8, 'P2': 24, 'P3': 35, 'P4': 37, 'P5': 33, 'P6': 12, 'P7': 30, 'P8': 38, 'P9': 45, 'P10': 6 }
    ideal_hip_tilt = { 'P1': 1, 'P2':4,'P3':8,'P4':9,'P5':7,'P6':8,'P7':11,'P8':12,'P9':14, 'P10':5}
    # Calculate and print average shoulder tilt for each pose class
    good_postures = 0
    bad_postures = 0

    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()
    # Create figures for shoulder and hip tilt distributions
    fig_shoulder, axes_shoulder = plt.subplots(5, 2, figsize=(15, 20))
    axes_shoulder = axes_shoulder.flatten()
    fig_hip, axes_hip = plt.subplots(5, 2, figsize=(15, 20))
    axes_hip = axes_hip.flatten()

     # Iterate through each pose class and plot both distributions
    for i, (pose_class, angles) in enumerate(pose_class_angles.items()):
        # Shoulder Tilt Plot
        if angles["shoulder_tilt"]:
            ideal_angle = ideal_shoulder_tilt[pose_class]
            average_angle = sum(angles["shoulder_tilt"]) / len(angles["shoulder_tilt"])

            # Calculate the standard deviation
            squared_diff = [(angle - ideal_angle) ** 2 for angle in angles["shoulder_tilt"]]
            variance = sum(squared_diff) / len(angles["shoulder_tilt"])
            stddev = math.sqrt(variance)

            print(f"Standard deviation of shoulder tilt for {pose_class}: {stddev:.2f} degrees")

            # Plot the normal distribution curve
            x = np.linspace(ideal_angle - 4 * stddev, ideal_angle + 4 * stddev, 100)
            y = stats.norm.pdf(x, ideal_angle, stddev)
            axes_shoulder[i].plot(x, y, color='green', label=f'Normal Curve, std: {stddev:.2f}')

            # Fill the area under the curve
            axes_shoulder[i].fill_between(x, y, color='green', alpha=0.2)

            # Plot the actual angles
            for angle in angles["shoulder_tilt"]:
                angle = math.floor(angle)
                axes_shoulder[i].axvline(x=angle, color='red', label=f'Predicted: {angle}')

            # Add line for ideal value
            axes_shoulder[i].axvline(x=ideal_angle, color='blue', label=f'Ideal: {ideal_angle}')

            # Add lines for ±1 standard deviation from the ideal value
            axes_shoulder[i].axvline(x=ideal_angle - stddev, color='orange', label=f'Ideal - 1 std: {ideal_angle - stddev:.2f}')
            axes_shoulder[i].axvline(x=ideal_angle + stddev, color='orange', label=f'Ideal + 1 std: {ideal_angle + stddev:.2f}')
        else:
            axes_shoulder[i].text(0.5, 0.5, f'No detections found for {pose_class}',
                                horizontalalignment='center', verticalalignment='center',
                                transform=axes_shoulder[i].transAxes, fontsize=10, color='red')

        axes_shoulder[i].legend()
        axes_shoulder[i].set_title(f'Shoulder Tilt: {pose_class}')
        axes_shoulder[i].set_xlabel('Angle (degrees)')
        axes_shoulder[i].set_ylabel('Probability Density')
        axes_shoulder[i].grid()

        # Hip Tilt Plot
        if angles["hip_tilt"]:
            ideal_angle = ideal_hip_tilt[pose_class]
            average_angle = sum(angles["hip_tilt"]) / len(angles["hip_tilt"])

            # Calculate the standard deviation
            squared_diff = [(angle - ideal_angle) ** 2 for angle in angles["hip_tilt"]]
            variance = sum(squared_diff) / len(angles["hip_tilt"])
            stddev = math.sqrt(variance)

            print(f"Standard deviation of hip tilt for {pose_class}: {stddev:.2f} degrees")

            # Plot the normal distribution curve
            x = np.linspace(ideal_angle - 4 * stddev, ideal_angle + 4 * stddev, 100)
            y = stats.norm.pdf(x, ideal_angle, stddev)
            axes_hip[i].plot(x, y, color='green', label=f'Normal Curve, std: {stddev:.2f}')

            # Fill the area under the curve
            axes_hip[i].fill_between(x, y, color='green', alpha=0.2)

            # Plot the actual angles
            for angle in angles["hip_tilt"]:
                angle = math.floor(angle)
                axes_hip[i].axvline(x=angle, color='red', label=f'Predicted: {angle}')

            # Add line for ideal value
            axes_hip[i].axvline(x=ideal_angle, color='blue', label=f'Ideal: {ideal_angle}')

            # Add lines for ±1 standard deviation from the ideal value
            axes_hip[i].axvline(x=ideal_angle - stddev, color='orange', label=f'Ideal - 1 std: {ideal_angle - stddev:.2f}')
            axes_hip[i].axvline(x=ideal_angle + stddev, color='orange', label=f'Ideal + 1 std: {ideal_angle + stddev:.2f}')
        else:
            axes_hip[i].text(0.5, 0.5, f'No detections found for {pose_class}',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes_hip[i].transAxes, fontsize=10, color='red')

        axes_hip[i].legend()
        axes_hip[i].set_title(f'Hip Tilt: {pose_class}')
        axes_hip[i].set_xlabel('Angle (degrees)')
        axes_hip[i].set_ylabel('Probability Density')
        axes_hip[i].grid()

    plt.tight_layout()
    
    
    cap.release()
    if save:
        out_video.release()
        print("[INFO] Output Video Saved as 'VideoOutput/output.mp4'")
        os.makedirs('Plots', exist_ok=True)
        shoulder_plot_path = os.path.join('Plots', 'shoulder_tilt_distribution.png')
        hip_plot_path = os.path.join('Plots', 'hip_tilt_distribution.png')
        fig_shoulder.savefig(shoulder_plot_path)
        fig_hip.savefig(hip_plot_path)
        print(f'[INFO] Shoulder Tilt Plot saved at {shoulder_plot_path}')
        print(f'[INFO] Hip Tilt Plot saved at {hip_plot_path}')
    #plt.show()
    cv2.destroyAllWindows()
    print('[INFO] Inference on Video Stream is Ended...')

print(pose_class_angles)

