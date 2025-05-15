import os
from keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt

# Argument parser for command-line options
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to saved .h5 model, e.g., dir/model.h5")
ap.add_argument("-c", "--conf", type=float, required=True,
                help="min prediction conf to detect pose class (0<conf<1)")
ap.add_argument("-d", "--dataset", type=str, required=True,
                help="path to folder containing validation images")
ap.add_argument("--confusion_matrix", type=str, required=True,
                help="path to save the confusion matrix plot, e.g., dir/confusion_matrix.png")

args = vars(ap.parse_args())
path_saved_model = args["model"]
threshold = args["conf"]
path_dataset = args["dataset"]
path_confusion_matrix = args["confusion_matrix"]

# Output directory for validated images
validated_output_dir = "validatedOutput"
os.makedirs(validated_output_dir, exist_ok=True)

##############
torso_size_multiplier = 2.5
n_landmarks = 33
landmark_names = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
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
class_names = ['P1','P10', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']

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

# Initialize lists to hold the true and predicted labels
y_true = []
y_pred = []
y_true_display = []
y_pred_display = []

# Process each class folder
for class_name in class_names:
    class_folder = os.path.join(path_dataset, class_name)
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        
        # Load and preprocess the image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        
        if result.pose_landmarks:
            lm_list = result.pose_landmarks.landmark
            
            # Preprocessing and prediction code
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
                predicted_class = class_names[predict.argmax()]
            else:
                predicted_class = 'Unknown Pose'
            
            # Append true and predicted labels for display
            y_true_display.append(class_name)
            y_pred_display.append(predicted_class)

            # Add the true and predicted class text to the image using OpenCV
            text_true = f"True: {class_name}"
            if predicted_class == class_name:
                text_pred = f"Predicted: {predicted_class}"
                color = (0, 255, 0)  # Green for correct predictions
            else:
                text_pred = f"Predicted: {predicted_class}"
                color = (0, 0, 255)  # Red for incorrect predictions
            
            cv2.putText(img, text_true, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(img, text_pred, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Create class folder in validatedOutput if it doesn't exist
            class_output_folder = os.path.join(validated_output_dir, class_name)
            os.makedirs(class_output_folder, exist_ok=True)

            # Save the image to validatedOutput with appropriate naming
            output_img_name = f"{predicted_class}_{img_name}"
            if predicted_class != class_name:
                output_img_name = f"Wrong_{output_img_name}"
            
            output_path = os.path.join(class_output_folder, output_img_name)
            cv2.imwrite(output_path, img)

            # Append true and predicted labels for evaluation (excluding "Unknown Pose")
            if predicted_class != 'Unknown Pose':
                y_true.append(class_name)
                y_pred.append(predicted_class)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", report)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=class_names)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save confusion matrix plot
if os.path.exists(path_confusion_matrix):
    os.remove(path_confusion_matrix)
plt.savefig(path_confusion_matrix, bbox_inches='tight')
print(f"[INFO] Successfully saved confusion matrix plot at {path_confusion_matrix}")

#
