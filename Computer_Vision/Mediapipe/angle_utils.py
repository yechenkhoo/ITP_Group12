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
    """Calculates and draws the shoulder tilt based on the detected pose class."""
    if pose_class in ['P2', 'P3', 'P4', 'P5']:
        # Calculate left shoulder tilt
        left_shoulder_coord = (lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1], lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0])
        right_shoulder_coord = (lm_list[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1], lm_list[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0])
        line_coord = (lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1], lm_list[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0])
        angle = calculate_angle(left_shoulder_coord, right_shoulder_coord, line_coord)
        draw_angle(img, left_shoulder_coord, right_shoulder_coord, line_coord, angle, 'Left Shoulder Tilt')
    else:
        # Calculate right shoulder tilt
        left_shoulder_coord = (lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1], lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0])
        right_shoulder_coord = (lm_list[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1], lm_list[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0])
        line_coord = (lm_list[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1], lm_list[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0])
        angle = calculate_angle(right_shoulder_coord, left_shoulder_coord, line_coord)
        draw_angle(img, right_shoulder_coord, left_shoulder_coord, line_coord, angle, 'Right Shoulder Tilt')    
    return angle


def calculate_and_draw_hip_tilt(img, lm_list, pose_class):
    """Calculates and draws the hip tilt based on the detected pose class."""
    if pose_class in ['P2', 'P3', 'P4', 'P5']:
        # Calculate left hip tilt
        left_hip_coord = (
            lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * img.shape[1],
            lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * img.shape[0]
        )
        right_hip_coord = (
            lm_list[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * img.shape[1],
            lm_list[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * img.shape[0]
        )
        line_coord = (
            lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * img.shape[1],
            lm_list[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * img.shape[0]
        )
        angle = calculate_angle(left_hip_coord, right_hip_coord, line_coord)
        draw_angle(img, left_hip_coord, right_hip_coord, line_coord, angle, 'Left Hip Tilt')
    else:
        # Calculate right hip tilt
        left_hip_coord = (
            lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * img.shape[1],
            lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * img.shape[0]
        )
        right_hip_coord = (
            lm_list[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * img.shape[1],
            lm_list[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * img.shape[0]
        )
        line_coord = (
            lm_list[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * img.shape[1],
            lm_list[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * img.shape[0]
        )
        angle = calculate_angle(right_hip_coord, left_hip_coord, line_coord)
        draw_angle(img, right_hip_coord, left_hip_coord, line_coord, angle, 'Right Hip Tilt')
    
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

    # Draw a simple visual guide near bottom of frame
    p1 = (int(L[0]), img.shape[0] - 60)
    p2 = (int(R[0]), img.shape[0] - 60)
    cv2.line(img, p1, p2, (255, 165, 0), 2)
    cv2.putText(img, f"Shoulder Rotation: {angle:.1f}",
                (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 165, 0), 2)

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

    # Draw a simple visual guide near bottom of frame
    p1 = (int(L[0]), img.shape[0] - 120)
    p2 = (int(R[0]), img.shape[0] - 120)
    cv2.line(img, p1, p2, (0, 165, 255), 2)
    cv2.putText(img, f"Hip Rotation: {angle:.1f}",
                (20, img.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 165, 255), 2)

    return angle
