"""
To run:
python -m BodyAngleOptimisation.pipeline
"""
import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from angle_utils import (
    calculate_and_draw_shoulder_tilt,
    calculate_and_draw_hip_tilt
)

import numpy as np

CSV_PATH = "FO/landmarks/FO_landmarks_best_frames.csv"
UNNORMALISED_PATH = "FO/landmarks/FO_landmarks_unnormalised.csv"
IMAGES_DIR = "FO/output/Grant_FO"
OUT_CSV = "BodyAngleOptimisation/Grant_FO_unnormalized_with_angles.csv"
OUT_IMG_DIR = "BodyAngleOptimisation/output_visuals"

# ---------- SETUP ----------
os.makedirs(OUT_IMG_DIR, exist_ok=True)
mp_pose = mp.solutions.pose
pose_draw = mp.solutions.drawing_utils

# 1) Load meta only from best-frames CSV
meta_cols = ["Frame_Name", "Person", "Camera_Angle", "Position", "Frame_Number"]
df_meta = pd.read_csv(CSV_PATH, usecols=[c for c in meta_cols if c in pd.read_csv(CSV_PATH, nrows=0).columns])

# 2) Load UNNORMALISED (pixel) coordinates
df_unnorm = pd.read_csv(UNNORMALISED_PATH)

# 3) Merge by Frame_Name, keep unnormalized coords
df = pd.merge(df_meta, df_unnorm, on="Frame_Name", how="inner")
df["shoulder_tilt_deg"] = np.nan
df["hip_tilt_deg"] = np.nan

# ---------- PROCESS ----------
def calculate_rotation_angle(p1, p2):
    """ADDED: Calculate rotation angle (degrees) between two points (shoulder or hip) and horizontal axis."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def process_frame(row):
    frame_name = row["Frame_Name"]
    frame_path = os.path.join(IMAGES_DIR, frame_name)
    if not os.path.isfile(frame_path):
        print(f"[WARN] Missing image: {frame_path}")
        return row

    img = cv2.imread(frame_path)
    if img is None:
        print(f"[WARN] Failed to load {frame_path}")
        return row

    H, W, _ = img.shape

    # --- Reconstruct Mediapipe landmark list from CSV ---
    lm_list = []
    for lm in mp_pose.PoseLandmark:
        name = lm.name
        try:
            landmark = type("Landmark", (object,), {})()
            landmark.x = row[f"{name}_X"] / W
            landmark.y = row[f"{name}_Y"] / H
            landmark.z = row[f"{name}_Z"] / W
            landmark.visibility = row[f"{name}_V"]
            lm_list.append(landmark)
        except KeyError:
            # In case of NaNs or missing columns
            print(f"[WARN] Missing landmark {name} for {frame_name}")
            return row

    # --- Compute angles ---
    pose_class = row.get("Position", "P1")
    annotated_img = img.copy()

    shoulder_tilt_deg = calculate_and_draw_shoulder_tilt(annotated_img, lm_list, pose_class)
    hip_tilt_deg = calculate_and_draw_hip_tilt(annotated_img, lm_list, pose_class)

    row["shoulder_tilt_deg"] = shoulder_tilt_deg
    row["hip_tilt_deg"] = hip_tilt_deg

    # --- Draw pose skeleton (optional; uses only landmark positions) ---
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(lm_list) and end_idx < len(lm_list):
            p1 = (int(lm_list[start_idx].x * W), int(lm_list[start_idx].y * H))
            p2 = (int(lm_list[end_idx].x * W), int(lm_list[end_idx].y * H))
            cv2.line(annotated_img, p1, p2, (0, 255, 0), 2)

    for lm in lm_list:
        cv2.circle(
            annotated_img,
            (int(lm.x * W), int(lm.y * H)),
            2,
            (0, 0, 255),
            -1
        )

    # --- Label angles ---
    cv2.putText(annotated_img, f"Shoulder Tilt: {shoulder_tilt_deg:.1f}°",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(annotated_img, f"Hip Tilt: {hip_tilt_deg:.1f}°",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_img, f"{pose_class}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    out_path = os.path.join(OUT_IMG_DIR, f"{os.path.splitext(frame_name)[0]}_angles.jpg")
    cv2.imwrite(out_path, annotated_img)
    print(f"[OK] Saved: {out_path} | Shoulder={shoulder_tilt_deg:.2f}°, Hip={hip_tilt_deg:.2f}°")

    return row


updated_rows = [process_frame(row) for _, row in df.iterrows()]
df_out = pd.DataFrame(updated_rows)
df_out.to_csv(OUT_CSV, index=False)

print(f"Annotated images saved to: {OUT_IMG_DIR}")
print(f"CSV with angles saved to: {OUT_CSV}")