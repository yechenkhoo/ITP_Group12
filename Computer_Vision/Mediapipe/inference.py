import os
from keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import argparse
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to saved .h5 model, eg: dir/model.h5")
ap.add_argument("-c", "--conf", type=float, required=True,
                help="min prediction conf to detect pose class (0<conf<1)")
ap.add_argument("-i", "--source", type=str, required=True,
                help="path to sample image or video source")
ap.add_argument("--save", action='store_true',
                help="Save video")

args = vars(ap.parse_args())
source = args["source"]
path_saved_model = args["model"]
threshold = args["conf"]
save = args['save']

##############
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
    'P1','P10', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'
]
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Face
    (9, 10),  # Shoulders
    (11, 12), (11, 13), (13, 15),  # Left Arm
    (12, 14), (14, 16),  # Right Arm
    (11, 23), (12, 24),  # Torso
    (23, 24), (23, 25), (24, 26),  # Hips
    (25, 27), (27, 29), (29, 31),  # Left Leg
    (26, 28), (28, 30), (30, 32)  # Right Leg
]
##############

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

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

# Load saved model
model = load_model(path_saved_model, compile=True)

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

def is_next_class_valid(current_class_index, previous_class_index):
    return current_class_index == previous_class_index + 1
# previous_class = 'Unknown Pose'
# def is_next_class_valid(current_class, previous_class):
#     if current_class in class_names and previous_class in class_names:
#         return class_names.index(current_class) == class_names.index(previous_class) + 1
#     return False

if source.endswith(('.jpg', '.jpeg', '.png')):
    path_to_img = source
    # Load sample Image
    img = cv2.imread(path_to_img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    if result.pose_landmarks:
        lm_list = result.pose_landmarks.landmark
        img = draw_bounding_box(img, lm_list)
        img = draw_landmarks(img, lm_list, POSE_CONNECTIONS)

        # Preprocessing and prediction code...
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

        torso_size = math.sqrt((shoulders_x - center_x)**2 + (shoulders_y - center_y)**2)
        max_distance = max(torso_size * torso_size_multiplier, max_distance)

        pre_lm = list(np.array([[(landmark.x - center_x) / max_distance, (landmark.y - center_y) / max_distance,
                                 landmark.z / max_distance, landmark.visibility] for landmark in lm_list]).flatten())
        data = pd.DataFrame([pre_lm], columns=col_names)
        predict = model.predict(data)[0]

        if max(predict) > threshold:
            pose_class = class_names[predict.argmax()]
            print('predictions: ', predict)
            print('predicted Pose Class: ', pose_class)
        else:
            pose_class = 'Unknown Pose'
            print('[INFO] Predictions is below given Confidence!!')

        # Show Result
        img = cv2.putText(
            img, f'{pose_class}',
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
    # Web-cam
    if source.isnumeric():
        source = int(source)

    cap = cv2.VideoCapture(source)
    source_width = int(cap.get(3))
    source_height = int(cap.get(4))

    # Write Video
    if save:
        out_video = cv2.VideoWriter('output.mp4', 
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            30, (source_width, source_height))

    while True:
        success, img = cap.read()
        if not success:
            print('[ERROR] Failed to Read Video feed')
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

            torso_size = math.sqrt((shoulders_x - center_x)**2 + (shoulders_y - center_y)**2)
            max_distance = max(torso_size * torso_size_multiplier, max_distance)

            pre_lm = list(np.array([[(landmark.x - center_x) / max_distance, (landmark.y - center_y) / max_distance,
                                     landmark.z / max_distance, landmark.visibility] for landmark in lm_list]).flatten())
            data = pd.DataFrame([pre_lm], columns=col_names)
            predict = model.predict(data)[0]

            if max(predict) > threshold:
                current_class_index = predict.argmax()
                print('predictions: ', predict)
                print('predicted Pose Class: ', class_names[current_class_index])
                if previous_class_index == -1 or is_next_class_valid(current_class_index, previous_class_index):
                    previous_class_index = current_class_index
                else:
                    current_class_index = previous_class_index
            else:
                if previous_class_index == class_names.index('P3'):
                    current_class_index = class_names.index('P4')
                    previous_class_index = current_class_index  # Update previous class to P4
                else:
                    current_class_index = -1  # Unknown Pose
                print('[INFO] Predictions is below given Confidence!!')
                if current_class_index != -1:
                    print(f'[INFO] Using predicted Pose Class: {class_names[current_class_index]}')

            # Show Result
            pose_class_text = class_names[current_class_index] if current_class_index != -1 else 'Unknown Pose'
            img = cv2.putText(
                img, pose_class_text,
                (40, 50), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2
            )
                # Write Video
            if save:
                out_video.write(img)

        cv2.imshow('Output Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if save:
        out_video.release()
        print("[INFO] Out video Saved as 'output.avi'")
    cv2.destroyAllWindows()
    print('[INFO] Inference on Videostream has Ended...')
