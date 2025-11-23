# NO SYNCING INVOLVED IN THIS
# Purpose: get best frames per video, for FO-Only

import os
import tensorflow as tf
import pandas as pd
import numpy as np

from MainScripts.poseLandmark_csv import extract_landmarks
from videoToFrames import process_all_videos_in_directory

def parse_frame_name(frame_name):
    base = os.path.splitext(frame_name)[0]  # remove .jpg
    parts = base.split("_")
    camera_angle = "FO"
    frame_number = parts[-1]
    person = base.replace(frame_number, "")[:-1]
    return person, camera_angle, frame_number

# process videos and extract landmarks (multiple FO videos)
input_FO_directory = 'FO_videos'
output_FO_directory = 'FO_videos/out'
process_all_videos_in_directory(input_FO_directory, output_FO_directory)
os.makedirs('FO_videos/landmarks', exist_ok=True)
extract_landmarks(output_FO_directory, 'FO_videos/landmarks/FO_landmarks.csv')


# predict positions using FO model and select best frames
model = tf.keras.models.load_model("best_model.keras")
FO_input_csv = "FO_videos/landmarks/FO_landmarks.csv"
data = pd.read_csv(FO_input_csv)
X = data.drop(columns=["Frame_Name", "Image_Path", "Pose_Class"], errors="ignore")
predictions = model.predict(X)
predicted_indices = np.argmax(predictions, axis=1)
pose_labels = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]
predicted_labels = [pose_labels[i] for i in predicted_indices]
predicted_confidences = predictions[np.arange(len(predicted_indices)), predicted_indices]
data["Position"] = predicted_labels
data["Confidence"] = predicted_confidences
output = data[["Frame_Name", "Position", "Confidence"] + [c for c in data.columns if c not in ["Frame_Name", "Position", "Confidence"]]]
output_csv = os.path.splitext(FO_input_csv)[0] + "_predicted.csv"
output.to_csv(output_csv, index=False)
print(f"Saved predictions to {output_csv}")
top_frames_list = []
body_coords = [c for c in data.columns if c not in ["Frame_Name", "Image_Path", "Pose_Class", "Position", "Confidence"]]
data[["Person", "Camera_Angle", "Frame_Number"]] = data["Frame_Name"].apply(
    lambda x: pd.Series(parse_frame_name(x))
)
for pos in pose_labels:
    pos_frames = data[data["Position"] == pos]
    for person in pos_frames["Person"].unique():
        person_frames = pos_frames[pos_frames["Person"] == person]
        person_frames = person_frames.dropna(subset=body_coords)
        if not person_frames.empty:
            person_frames_sorted = person_frames.sort_values(by="Confidence", ascending=False)
            top_1 = person_frames_sorted.head(1)
            top_frames_list.append(top_1)
if top_frames_list:
    top_frames_df = pd.concat(top_frames_list)
    cols = ["Frame_Name", "Person", "Camera_Angle", "Frame_Number", "Position"]
    top_frames_df = top_frames_df[cols]
    top_output_csv = os.path.splitext(FO_input_csv)[0] + "_best_frames.csv"
    top_frames_df.to_csv(top_output_csv, index=False)
    print(f"Saved best frames per person to {top_output_csv}")
else:
    print("No FO frames found for best frame selection.")