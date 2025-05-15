import cv2
import numpy as np
import mediapipe as mp

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






