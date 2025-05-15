from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
from ml import Classifier, Movenet, Posenet, MoveNetMultiPose
import utils
import io
from PIL import Image
import logging

# Initialize Flask app
app = Flask(__name__)

# Load TFLite models based on parameters
MODEL_DICT = {
    'movenet_lightning': 'models/movenet_lightning.tflite',
    'movenet_thunder': 'models/movenet_thunder.tflite',
    'posenet': 'models/posenet.tflite',
    'movenet_multipose': 'models/movenet_multipose.tflite'
}

# Utility function to load the correct pose detection model
def load_pose_detector(model_name, tracker_type=None):
    if model_name in ['movenet_lightning', 'movenet_thunder']:
        return Movenet(MODEL_DICT[model_name])
    elif model_name == 'posenet':
        return Posenet(MODEL_DICT[model_name])
    elif model_name == 'movenet_multipose':
        return MoveNetMultiPose(MODEL_DICT[model_name], tracker_type)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

@app.route('/')
def index():
    return "Welcome to the Pose Estimation Cloud Function!", 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON request
        data = request.get_json()
        image_base64 = data.get("image")
        model_type = data.get("model", "movenet_lightning")
        tracker_type = data.get("tracker", "bounding_box")
        classification_model = data.get("classifier", None)
        label_file = data.get("label_file", "labels.txt")

        # Load the pose detection model
        pose_detector = load_pose_detector(model_type, tracker_type)

        # Load the pose classification model (if provided)
        classifier = None
        if classification_model:
            classifier = Classifier(classification_model, label_file)

        # Convert base64 image to OpenCV image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Run pose estimation on the input image
        if model_type == 'movenet_multipose':
            list_persons = pose_detector.detect(image)
        else:
            list_persons = [pose_detector.detect(image)]

        image_with_keypoints = utils.visualize(image, list_persons)

        classification_result = None
        if classifier:
            person = list_persons[0]
            classification_result = classifier.classify_pose(person)

        _, buffer = cv2.imencode('.jpg', image_with_keypoints)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image_with_keypoints': image_base64,
            'classification_result': [result.to_dict() for result in classification_result] if classification_result else None
        })

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

# Correct Cloud Function entry point
def predict_function(request):
    """This function serves as the Cloud Function entry point."""
    with app.app_context():
        # Use Flask's test_client to handle the request
        response = app.test_client().post('/predict', json=request.get_json())
        return response.get_data(), response.status_code, response.headers

# Local server configuration - not required for Cloud Functions
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
