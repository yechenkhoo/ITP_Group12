import cv2
import numpy as np
import mediapipe as mp
from math import degrees, atan2

def calculate_angle(a, b, c):
    """Calculates the angle between three points.

    Args:
        a: The first point as a list or tuple [x, y].
        b: The second (vertex) point as a list or tuple [x, y].
        c: The third point as a list or tuple [x, y].

    Returns:
        The angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def calculate_midpoint(point1, point2):
    """Calculates the midpoint between two points.

    Args:
        point1: The first point as a list or tuple [x, y].
        point2: The second point as a list or tuple [x, y].

    Returns:
        The midpoint as a list [x, y].
    """
    return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]

def draw_angle(image, point1, point2, point3, angle, label):
    """Draws the angle between three points on the image with a label.

    Args:
        image: The image on which to draw.
        point1: The first point as a list or tuple [x, y].
        point2: The second (vertex) point as a list or tuple [x, y].
        point3: The third point as a list or tuple [x, y].
        angle: The angle to draw.
        label: The label describing the angle.
    """
    # Visualization parameters
    text_color = (0, 0, 255)  # Red color for the text
    line_color = (255, 0, 255)  # Magenta color for the lines
    font_size = 1
    font_thickness = 2
    
    # Draw the Line between the points and the text of the angle
    cv2.line(image, tuple(map(int, point1)), tuple(map(int, point2)), line_color, 2) 
    cv2.line(image, tuple(map(int, point2)), tuple(map(int, point3)), line_color, 2)  
    text_position = (int(point2[0] + 10), int(point2[1] - 10)) 
    cv2.putText(image, f"{label}: {round(angle, 2)}", text_position, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)


def calculate_and_draw_shoulder_tilt(img, lm_list, pose_class):
    """Calculates and draws the shoulder tilt automatically based on direction."""
    left = lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right = lm_list[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]

    left_shoulder = (left.x * img.shape[1], left.y * img.shape[0])
    right_shoulder = (right.x * img.shape[1], right.y * img.shape[0])

    dy = left_shoulder[1] - right_shoulder[1]

    # Choose which side to anchor the vertical reference line (lower shoulder)
    if dy > 0:
        # Left shoulder lower — tilt down to right
        line_coord = (left_shoulder[0], right_shoulder[1])
        angle = calculate_angle(left_shoulder, right_shoulder, line_coord)
        draw_angle(img, left_shoulder, right_shoulder, line_coord, angle, "Shoulder Tilt")
    else:
        # Right shoulder lower — tilt down to left
        line_coord = (right_shoulder[0], left_shoulder[1])
        angle = calculate_angle(right_shoulder, left_shoulder, line_coord)
        draw_angle(img, right_shoulder, left_shoulder, line_coord, angle, "Shoulder Tilt")

    return angle

def calculate_and_draw_hip_tilt(img, lm_list, pose_class):
    left = lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right = lm_list[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

    left_hip = (left.x * img.shape[1], left.y * img.shape[0])
    right_hip = (right.x * img.shape[1], right.y * img.shape[0])

    dy = left_hip[1] - right_hip[1]

    if dy > 0:
        # Left hip lower — pelvis tilted down to right
        line_coord = (left_hip[0], right_hip[1])
        angle = calculate_angle(left_hip, right_hip, line_coord)
        draw_angle(img, left_hip, right_hip, line_coord, angle, "Hip Tilt")
    else:
        # Right hip lower — pelvis tilted down to left
        line_coord = (right_hip[0], left_hip[1])
        angle = calculate_angle(right_hip, left_hip, line_coord)
        draw_angle(img, right_hip, left_hip, line_coord, angle, "Hip Tilt")

    return angle

def calculate_and_draw_shoulder_rotation(img, lm_list, pose_class):
    """Calculates and draws shoulder rotation (angle in x–z plane) using a reference line."""
    # Get shoulder landmarks
    left_shoulder = lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = lm_list[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Project to top-down (x–z) plane
    L = (left_shoulder.x * img.shape[1], left_shoulder.z * img.shape[1])
    R = (right_shoulder.x * img.shape[1], right_shoulder.z * img.shape[1])

    # Create a reference horizontal line across x-axis in x–z plane
    ref = (left_shoulder.x * img.shape[1], right_shoulder.z * img.shape[1])

    # Compute the angle between actual shoulder line and reference
    angle = calculate_angle(L, R, ref)


    # cv2.putText(img, f"Shoulder Rotation: {angle:.1f}",
    #             (20, img.shape[0] - 150), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.7, (0, 255, 165), 2)

    return angle

def calculate_and_draw_hip_rotation(img, lm_list, pose_class):
    """Calculates and draws hip rotation (angle in x–z plane) using a reference line."""
    # Get hip landmarks
    left_hip = lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm_list[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

    # Project to top-down (x–z) plane
    L = (left_hip.x * img.shape[1], left_hip.z * img.shape[1])
    R = (right_hip.x * img.shape[1], right_hip.z * img.shape[1])

    # Create reference horizontal line across x-axis in x–z plane
    ref = (left_hip.x * img.shape[1], right_hip.z * img.shape[1])

    # Compute the angle between actual hip line and reference
    angle = calculate_angle(L, R, ref)


    # cv2.putText(img, f"Hip Rotation: {angle:.1f}",
    #             (20, img.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.7, (255, 255, 255), 2)

    return angle

def calculate_and_draw_forward_tilt_dtl(img, lm_list, pose_class):
    """
    Calculates and draws forward tilt (torso lean) in the X–Y plane.
    The angle represents the deviation of the torso from vertical.
    """
    # Extract key landmarks
    left_shoulder = lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = lm_list[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm_list[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

    # Convert normalized landmarks to pixel coordinates
    ls = [left_shoulder.x * img.shape[1], left_shoulder.y * img.shape[0]]
    rs = [right_shoulder.x * img.shape[1], right_shoulder.y * img.shape[0]]
    lh = [left_hip.x * img.shape[1], left_hip.y * img.shape[0]]
    rh = [right_hip.x * img.shape[1], right_hip.y * img.shape[0]]

    # Calculate midpoints
    mid_shoulder = calculate_midpoint(ls, rs)
    mid_hip = calculate_midpoint(lh, rh)

    # Create reference line and calculate angle
    torso_length = np.linalg.norm(np.array(mid_shoulder) - np.array(mid_hip))
    vertical_ref = [mid_hip[0], mid_hip[1] - torso_length]
    angle = calculate_angle(mid_shoulder, mid_hip, vertical_ref)

    # Visualization overlay
    cv2.line(
        img,
        (int(mid_hip[0]), int(mid_hip[1])),
        (int(mid_shoulder[0]), int(mid_shoulder[1])),
        (0, 255, 255),
        2
    )
    cv2.line(
        img,
        (int(mid_hip[0]), int(mid_hip[1])),
        (int(vertical_ref[0]), int(vertical_ref[1])),
        (0, 255, 255),
        2,
        lineType=cv2.LINE_AA
    )

    cv2.putText(
        img,
        f"Forward Tilt: {angle:.1f}",
        (20, img.shape[0] - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    return angle

def calculate_and_draw_lead_arm_angle(img, lm_list, pose_class):
    """
    Calculates and draws the lead arm straightness (left arm for right-handed golfers).
    The angle is measured at the elbow joint (shoulder–elbow–wrist).
    """
    # Extract key landmarks
    left_shoulder = lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow    = lm_list[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist    = lm_list[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]

    # Convert normalized landmarks to pixel coordinates
    shoulder_coord = [left_shoulder.x * img.shape[1], left_shoulder.y * img.shape[0]]
    elbow_coord    = [left_elbow.x   * img.shape[1], left_elbow.y   * img.shape[0]]
    wrist_coord    = [left_wrist.x   * img.shape[1], left_wrist.y   * img.shape[0]]

    # Compute elbow (lead arm) angle
    angle = calculate_angle(shoulder_coord, elbow_coord, wrist_coord)

    #  Visualization overlay
    cv2.line(
        img,
        (int(shoulder_coord[0]), int(shoulder_coord[1])),
        (int(elbow_coord[0]), int(elbow_coord[1])),
        (0, 255, 255),
        2,
        lineType=cv2.LINE_AA
    )

    # Draw elbow to wrist
    cv2.line(
        img,
        (int(elbow_coord[0]), int(elbow_coord[1])),
        (int(wrist_coord[0]), int(wrist_coord[1])),
        (0, 255, 255),
        2,
        lineType=cv2.LINE_AA
    )

    # Mark joints
    cv2.circle(img, (int(shoulder_coord[0]), int(shoulder_coord[1])), 5, (0, 255, 255), -1)
    cv2.circle(img, (int(elbow_coord[0]), int(elbow_coord[1])), 5, (0, 255, 255), -1)
    cv2.circle(img, (int(wrist_coord[0]), int(wrist_coord[1])), 5, (0, 255, 255), -1)

    cv2.putText(
        img,
        f"Lead Arm Angle: {angle:.1f}",
        (20, img.shape[0] - 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    return angle

def calculate_and_draw_knee_bend(img, lm_list, pose_class):
    """
    Calculates and draws the lead knee bend angle (left leg for right-handed golfers).
    The angle is measured at the knee joint (hip–knee–ankle).
    """

    # Extract key landmarks
    left_hip   = lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    left_knee  = lm_list[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = lm_list[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value]

    # Convert normalized landmarks to pixel coordinates 
    hip_coord   = [left_hip.x * img.shape[1], left_hip.y * img.shape[0]]
    knee_coord  = [left_knee.x * img.shape[1], left_knee.y * img.shape[0]]
    ankle_coord = [left_ankle.x * img.shape[1], left_ankle.y * img.shape[0]]

    # Compute the knee bend angle (hip–knee–ankle)
    angle = calculate_angle(hip_coord, knee_coord, ankle_coord)

    # Draw hip to knee
    cv2.line(
        img,
        (int(hip_coord[0]), int(hip_coord[1])),
        (int(knee_coord[0]), int(knee_coord[1])),
        (255, 255, 0), 
        2,
        lineType=cv2.LINE_AA
    )

    # Draw knee to ankle
    cv2.line(
        img,
        (int(knee_coord[0]), int(knee_coord[1])),
        (int(ankle_coord[0]), int(ankle_coord[1])),
        (255, 255, 0),
        2,
        lineType=cv2.LINE_AA
    )

    # Mark joints for clarity
    cv2.circle(img, (int(hip_coord[0]), int(hip_coord[1])), 5, (255, 255, 0), -1)
    cv2.circle(img, (int(knee_coord[0]), int(knee_coord[1])), 5, (255, 255, 0), -1)
    cv2.circle(img, (int(ankle_coord[0]), int(ankle_coord[1])), 5, (255, 255, 0), -1)

    # Display text
    cv2.putText(
        img,
        f"Knee Bend: {angle:.1f}",
        (20, img.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    return angle


## NOT USED
def calculate_and_draw_forward_tilt_faceon(img, lm_list, pose_class):
    """
    Calculates and draws forward tilt (torso lean) in the Y–Z plane
    for the Face-On camera angle (golfer facing camera).
    The angle represents how much the torso leans toward or away from the camera.
    """
    # --- Extract landmarks
    left_shoulder = lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = lm_list[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
    right_hip = lm_list[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]

    # --- Use (y, z) coordinates for Face-On forward tilt
    # Scale by image height for y, and width for z to keep similar magnitudes
    ls = [left_shoulder.y * img.shape[0], left_shoulder.z * img.shape[1]]
    rs = [right_shoulder.y * img.shape[0], right_shoulder.z * img.shape[1]]
    lh = [left_hip.y * img.shape[0], left_hip.z * img.shape[1]]
    rh = [right_hip.y * img.shape[0], right_hip.z * img.shape[1]]

    # --- Compute midpoints and calculate angle
    mid_shoulder = calculate_midpoint(ls, rs)
    mid_hip = calculate_midpoint(lh, rh)
    torso_length = np.linalg.norm(np.array(mid_shoulder) - np.array(mid_hip))
    vertical_ref = [mid_hip[0] - torso_length, mid_hip[1]]  # goes upward (smaller y)
    angle = calculate_angle(mid_shoulder, mid_hip, vertical_ref)

    # --- Visualization projected into 2D image space
    cv2.putText(
        img,
        f"Forward Tilt (Face-On): {angle:.1f}",
        (20, img.shape[0] - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    return angle

