# NO SYNCING INVOLVED IN THIS
# Purpose: use existing DTL data to analyse body positions

import os
import tensorflow as tf
import pandas as pd
import numpy as np

from MainScripts.poseLandmark_csv import extract_landmarks
from videoToFrames import process_all_videos_in_directory

input_FO_directory = 'DTL_frames'
output_FO_directory = 'DTL_frames/out'

def parse_frame_name(frame_name):
    base = os.path.splitext(frame_name)[0]  # remove .jpg
    parts = base.split("_")
    camera_angle = "DTL"
    frame_number = parts[-1]
    person = base.replace(frame_number, "")[:-1]
    
    # frame_path, position = find_frame_path_and_position(frame_name, output_FO_directory)
    return person, camera_angle, frame_number

# def find_frame_path_and_position(frame_name, base_dir):
#     for pos in [f"P{i}" for i in range(1, 11)]:
#         folder = os.path.join(base_dir, pos)
#         frame_path = os.path.join(folder, frame_name)
#         if os.path.isfile(frame_path):
#             return frame_path, pos
#     return None, None 

os.makedirs('DTL_frames/landmarks', exist_ok=True)
extract_landmarks(output_FO_directory, 'DTL_frames/landmarks/DTL_landmarks.csv')

# # predict positions using FO model and select best frames
# model = tf.keras.models.load_model("best_model.keras")
DTL_input_csv = "DTL_frames/landmarks/DTL_landmarks.csv"
data = pd.read_csv(DTL_input_csv)
X = data.drop(columns=["Frame_Name", "Image_Path", "Pose_Class"], errors="ignore")
pose_labels = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]

top_frames_list = []
data[["Person", "Camera_Angle", "Frame_Number"]] = data["Frame_Name"].apply(
    lambda x: pd.Series(parse_frame_name(x))
)

# MERGE WITH PRIOR LABELS
map_csv = "DTL_frames/_map_to_labels.csv"
mapping = pd.read_csv(map_csv)
label_map = dict(zip(mapping["filename"], mapping["label"]))
data["Position"] = data["Frame_Name"].map(label_map)
##################### 

if "NOSE_X" in data.columns:
    before_count = len(data)
    data = data.dropna(subset=["NOSE_X"])
    after_count = len(data)
    print(f"[INFO] Dropped {before_count - after_count} rows with missing NOSE_X.")
else:
    print("[WARN] Column 'NOSE_X' not found in data — skipping drop.")

cols = ["Frame_Name", "Person", "Camera_Angle", "Frame_Number", "Position"]
data = data[cols]
data.to_csv(os.path.splitext(DTL_input_csv)[0] + "_best_frames.csv")

# for pos in pose_labels:
#     pos_frames = data[data["Position"] == pos]
#     for person in pos_frames["Person"].unique():
#         person_frames = pos_frames[pos_frames["Person"] == person]
#         person_frames = person_frames.dropna(subset=body_coords)
#         if not person_frames.empty:
#             person_frames_sorted = person_frames.sort_values(by="Confidence", ascending=False)
#             top_1 = person_frames_sorted.head(1)
#             top_frames_list.append(top_1)
# if top_frames_list:
#     top_frames_df = pd.concat(top_frames_list)
#     cols = ["Frame_Name", "Person", "Camera_Angle", "Frame_Number", "Position"]
#     top_frames_df = top_frames_df[cols]
#     top_output_csv = os.path.splitext(FO_input_csv)[0] + "_best_frames.csv"
#     top_frames_df.to_csv(top_output_csv, index=False)
#     print(f"Saved best frames per person to {top_output_csv}")
# else:
#     print("No FO frames found for best frame selection.")