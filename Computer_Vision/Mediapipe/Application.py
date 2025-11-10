import os
import tensorflow as tf
import pandas as pd
import numpy as np

from MainScripts.poseLandmark_csv import extract_landmarks
from videoToFrames import process_all_videos_in_directory

def parse_frame_name(frame_name):
    base = os.path.splitext(frame_name)[0]  # remove .jpg
    parts = base.split("_")
    if len(parts) >= 3:
        person = parts[0]
        camera_angle = parts[1]
        frame_number = "_".join(parts[2:])  # in case frame has underscores
    else:
        person, camera_angle, frame_number = base, "", ""
    return person, camera_angle, frame_number

# 1. split video into frames 
input_FO_directory = 'FO/input'     
input_DTL_directory = 'DTL/input'           
output_FO_directory = 'FO/output'     
output_DTL_directory = 'DTL/output' 

process_all_videos_in_directory(input_FO_directory, output_FO_directory)
process_all_videos_in_directory(input_DTL_directory, output_DTL_directory)

# 2. get coordinates from mediapipe
os.makedirs('FO/landmarks', exist_ok=True)
os.makedirs('DTL/landmarks', exist_ok=True)

extract_landmarks(output_FO_directory, 'FO/landmarks/FO_landmarks.csv')
extract_landmarks(output_DTL_directory, 'DTL/landmarks/DTL_landmarks.csv')

# 3. predict positions using FO model

# load FO model
model = tf.keras.models.load_model("best_model.keras")

# load landmarks csv
FO_input_csv = "FO/landmarks/FO_landmarks.csv"
data = pd.read_csv(FO_input_csv)

frame_names = data["Frame_Name"] if "Frame_Name" in data.columns else None

X = data.drop(columns=["Frame_Name", "Image_Path", "Pose_Class"], errors="ignore")

predictions = model.predict(X)
predicted_indices = np.argmax(predictions, axis=1)

pose_labels = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]
predicted_labels = [pose_labels[i] for i in predicted_indices]
predicted_confidences = predictions[np.arange(len(predicted_indices)), predicted_indices]

data["Position"] = predicted_labels
data["Confidence"] = predicted_confidences

# save predictions in csv
output = data[["Frame_Name", "Position", "Confidence"] + [c for c in data.columns if c not in ["Frame_Name", "Position", "Confidence"]]]
output_csv = os.path.splitext(FO_input_csv)[0] + "_predicted.csv"
output.to_csv(output_csv, index=False)
print(f" Saved predictions to {output_csv}")

# 4. select top 1 frame from each postition
top_frames_list = []

body_coords = [c for c in data.columns if c not in ["Frame_Name", "Image_Path", "Pose_Class", "Position", "Confidence"]]

for pos in pose_labels:
    pos_frames = data[data["Position"] == pos]
    pos_frames = pos_frames.dropna(subset=body_coords)
    
    if not pos_frames.empty:
        pos_frames_sorted = pos_frames.sort_values(by="Confidence", ascending=False)
        top_1 = pos_frames_sorted.head(1)
        top_frames_list.append(top_1)

top_frames_df = pd.concat(top_frames_list)

# save frames into csv
top_frames_df[["Person", "Camera_Angle", "Frame_Number"]] = top_frames_df["Frame_Name"].apply(
    lambda x: pd.Series(parse_frame_name(x))
)

body_coords = [c for c in top_frames_df.columns if c not in ["Frame_Name", "Position", "Confidence", "Person", "Camera_Angle", "Frame_Number"]]
cols = ["Frame_Name", "Person", "Camera_Angle", "Position", "Frame_Number"] + body_coords
top_frames_df = top_frames_df[cols]

top_output_csv = os.path.splitext(FO_input_csv)[0] + "_best_frames.csv"
top_frames_df.to_csv(top_output_csv, index=False)

print(f"Saved top 10 frames per position to {top_output_csv}")

# 5. generate DTL csv by matching FO frames
fo_best_csv = top_output_csv
dtl_best_csv = fo_best_csv.replace("FO", "DTL")

fo_best_df = pd.read_csv(fo_best_csv)

dtl_best_df = fo_best_df.copy()
dtl_best_df["Frame_Name"] = dtl_best_df["Frame_Name"].str.replace("_FO_", "_DTL_", regex=False)
dtl_best_df["Camera_Angle"] = dtl_best_df["Camera_Angle"].replace("FO", "DTL")

# retrieve dtl landmarks csv
dtl_landmarks_csv = "DTL/landmarks/DTL_landmarks.csv"
dtl_landmarks_df = pd.read_csv(dtl_landmarks_csv)

body_coords = [c for c in dtl_landmarks_df.columns if c != "Frame_Name"]

non_coord_cols = ["Frame_Name", "Person", "Camera_Angle", "Position", "Frame_Number"]
dtl_best_df = dtl_best_df[non_coord_cols]

# merge with dtl landmarks csv
dtl_best_df = pd.merge(dtl_best_df, dtl_landmarks_df, on="Frame_Name", how="left")
cols = non_coord_cols + body_coords
dtl_best_df = dtl_best_df[cols]

dtl_best_df.to_csv(dtl_best_csv, index=False)
print(f"Saved DTL synced best frames CSV to {dtl_best_csv}")