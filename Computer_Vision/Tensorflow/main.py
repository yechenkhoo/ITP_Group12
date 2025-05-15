from flask import Flask, request, jsonify
import os
import cv2
from google.cloud import storage
from ml import Classifier, Movenet, MoveNetMultiPose, Posenet
import utils
from data import Person, KeyPoint, BodyPart
import angle_utils
import csv
from tqdm import tqdm

app = Flask(__name__)

# Function to download a blob from Google Cloud Storage
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from {bucket_name} to {destination_file_name}")

# Function to upload a file to Google Cloud Storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")

# Combined route to handle video download from GCS and pose estimation
@app.route('/run_pose_estimation', methods=['POST'])
def run_pose_estimation():
    try:
        # Get the request data
        data = request.json
        print(f"Received request: {data}")

        # Extract information from the request
        estimation_model = data['estimation_model']
        classification_model = data['classification_model']
        label_file = data['label_file']
        video_path = data['video_path']  # Path in Google Cloud Storage
        output_dir = '/tmp/testData'  # Temporary directory for output
        width = int(data.get('width', 640))
        height = int(data.get('height', 480))

        # Bucket name where the models and video are stored
        bucket_name = 'itp-se-team13'

        # Log parameters
        print(f"Estimation model: {estimation_model}")
        print(f"Classification model: {classification_model}")
        print(f"Label file: {label_file}")
        print(f"Video path: {video_path}")

        # Download the required model, label, and video from Google Cloud Storage to /tmp/
        download_blob(bucket_name, estimation_model, '/tmp/' + os.path.basename(estimation_model))
        download_blob(bucket_name, classification_model, '/tmp/' + os.path.basename(classification_model))
        download_blob(bucket_name, label_file, '/tmp/' + os.path.basename(label_file))
        download_blob(bucket_name, video_path, '/tmp/' + os.path.basename(video_path))

        # Update the paths to use the downloaded files in /tmp/
        estimation_model = '/tmp/' + os.path.basename(estimation_model)
        classification_model = '/tmp/' + os.path.basename(classification_model)
        label_file = '/tmp/' + os.path.basename(label_file)
        video_local_path = '/tmp/' + os.path.basename(video_path)

        # Initialize pose detector (based on your existing logic)
        if estimation_model.endswith('movenet_lightning.tflite') or estimation_model.endswith('movenet_thunder.tflite'):
            pose_detector = Movenet(estimation_model)
        elif estimation_model.endswith('posenet.tflite'):
            pose_detector = Posenet(estimation_model)
        elif estimation_model.endswith('movenet_multipose.tflite'):
            pose_detector = MoveNetMultiPose(estimation_model, 'bounding_box')
        else:
            print(f"Model {estimation_model} is not supported.")
            return jsonify({'error': 'Model not supported'}), 400

        print("Pose detector initialized successfully.")

        # Initialize video capture from the downloaded video
        cap = cv2.VideoCapture(video_local_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        csv_path = os.path.join(output_dir, 'keypoints.csv')
        with open(csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Generate CSV header
            list_name = [[bodypart.name + '_x', bodypart.name + '_y', bodypart.name + '_score'] for bodypart in BodyPart]
            header_name = ['file_name', 'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']
            for columns_name in list_name:
                header_name += columns_name
            header_name.append('class_name')
            csv_writer.writerow(header_name)

            # Process frames for pose estimation and classification
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_count in tqdm(range(total_frames), desc="Processing frames"):
                success, image = cap.read()
                if not success:
                    break

                if estimation_model.endswith('movenet_multipose.tflite'):
                    list_persons = pose_detector.detect(image)
                else:
                    list_persons = [pose_detector.detect(image)]

                # Visualize keypoints
                image = utils.visualize(image, list_persons)

                class_name = None
                if classification_model:
                    classifier = Classifier(classification_model, label_file)
                    person = list_persons[0]
                    min_score = min([keypoint.score for keypoint in person.keypoints])
                    if min_score >= 0.1:
                        prob_list = classifier.classify_pose(person)
                        class_name = prob_list[0].label

                # Log keypoints to CSV
                frame_filename = f"frame_{frame_count:04d}.jpg"
                csv_writer.writerow([frame_filename])

        cap.release()

        # Upload the CSV file to Google Cloud Storage
        csv_gcs_path = f'output/keypoints_{os.path.basename(video_path)}.csv'
        upload_blob(bucket_name, csv_path, csv_gcs_path)

        return jsonify({'status': 'Processing complete', 'csv_file': f'gs://{bucket_name}/{csv_gcs_path}'})

    except Exception as e:
        print(f"Error during pose estimation: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "API is up and runninging"}), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
