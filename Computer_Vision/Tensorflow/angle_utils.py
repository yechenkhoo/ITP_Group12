import cv2
import numpy as np

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
    global angle_counter

    # Visualization parameters
    text_color = (0, 0, 255)  
    font_size = 1
    font_thickness = 2
    
    # Draw lines
    cv2.line(image, tuple(map(int, point1)), tuple(map(int, point2)), (255, 0, 0), 2)
    cv2.line(image, tuple(map(int, point2)), tuple(map(int, point3)), (255, 0, 0), 2)

    # Put angle text next to point2
    cv2.putText(image, f"{label}: {round(angle, 2)}", tuple(map(int, point2)), cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)