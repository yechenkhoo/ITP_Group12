from flask import Flask, Response, jsonify
import requests
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
from libcamera import Transform
import cv2
import threading
import time

app = Flask(__name__)

# Initialize Picamera2 and face detection
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)  # Set preview resolution
picam2.preview_configuration.main.format = "RGB888"  # 8 bits
picam2.preview_configuration.transform = Transform(vflip=True)
encoder = H264Encoder(10000000)  # Video encoder

# Threading event to control recording
recording_active = threading.Event()

# Global variable to store the current detection element
current_element = 'Unknown'

# Change the URL to your testing/hosted server
FLASK_URL = "https://ad80-2400-79e0-8070-cb06-4d9f-b400-fe15-ad8a.ngrok-free.app/"

def get_ml_prediction(img):
    """Send image to ML Flask app for prediction."""
    _, img_encoded = cv2.imencode('.jpg', img)
    predict = FLASK_URL + "dashboard/predict/"
    response = requests.post(predict, files={'image': img_encoded.tobytes()})
    if response.status_code == 200:
        return response.json().get('pose', 'Unknown')
    return 'Unknown'

def run_camera_preview():
    """Function to run the camera preview and show the live feed."""
    global current_element
    picam2.start()
    while True:
        im = picam2.capture_array()
        if current_element is not None:
            cv2.putText(im, f'Detected: {current_element}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
#         predicted_pose = get_ml_prediction(im)
#         print(predicted_pose)
#         if predicted_pose != 'Unknown':
#             current_element = predicted_pose
        _, buffer = cv2.imencode('.jpg', im)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/start_live_cam')
def start_live_cam():
    """Start the live view camera feed."""
    return Response(run_camera_preview(), content_type='multipart/x-mixed-replace; boundary=frame')

# Detection array for simulation
arr = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']

def detection_sequence():
    """Detection sequence to handle recording based on the detection array."""
    global current_element  # Access global current_element
    for element in arr:
        current_element = element  # Update current element to be displayed
        time.sleep(1)  # Simulate detection time delay
        print(f"Detected: {element}")
        if element == 'p1' and not recording_active.is_set():
            video_output_path = "/home/user/Videos/video1.mp4"
            video_output = FfmpegOutput(video_output_path)
            print("Starting video recording...")
            picam2.start_recording(encoder, output=video_output)
            recording_active.set()

        if element == 'p10' and recording_active.is_set():
            print("p10 detected, stopping recording in 5 seconds...")
            time.sleep(5)
            picam2.stop_recording()
            picam2.start()
            print("Recording finished.")
            recording_active.clear()
            
            # Clear the putText display after P10
            current_element = None
            current_element = 'Recording finished, uploading file now'
            
            # Send the video file to the main computer Flask server
            try:
                with open(video_output_path, 'rb') as f:
                    upload_video_url = FLASK_URL + "home/upload_from_pi/"
                    response = requests.post(upload_video_url, files={'file': f})
                    if response.status_code == 200:
                        print("Video file uploaded successfully.")
                        current_element = None
                        current_element = 'Video file uploaded successfully'
                        time.sleep(2)
                        
                    else:
                        print("Failed to upload video file.")
            except Exception as e:
                print(f"Error uploading video: {e}")
            
            # Clear the putText display after P10
            current_element = None
            current_element = 'Unknown'
            break

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start recording using a separate thread."""
    if not recording_active.is_set():
        recording_thread = threading.Thread(target=detection_sequence)
        recording_thread.start()
        return jsonify({"message": "Recording started"}), 200
    else:
        return jsonify({"message": "Recording already in progress"}), 400

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop recording if currently active."""
    if recording_active.is_set():
        picam2.stop_recording()
        recording_active.clear()
        return jsonify({"message": "Recording stopped"}), 200
    else:
        return jsonify({"message": "No recording to stop"}), 400

# Main thread for Flask app
if __name__ == '__main__':
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000))
    flask_thread.start()


