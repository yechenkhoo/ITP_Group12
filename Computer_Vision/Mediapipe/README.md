# CustomPose-Classification-Mediapipe

Creating a Custom pose classification using Mediapipe with help of OpenCV

<p align="center">
  <img src='https://miro.medium.com/max/434/1*iy_qNrpaHWkfJTZ3TrAuKA.png'/>
</p>

**Sample Video Output:**<br>

<p align="center">
  <img src='https://user-images.githubusercontent.com/88816150/189837009-a7344d98-d795-4bc4-b1fd-640e772221f7.gif' alt="animated" />
</p>

**Sample Image Output:**<br>

<div class="row">
  <div class="column">
    <img src="https://github.com/naseemap47/CustomPose-Classification-Mediapipe/blob/master/ImageOutput/chair.jpg">
    <img src="https://github.com/naseemap47/CustomPose-Classification-Mediapipe/blob/master/ImageOutput/cobra.jpg">
    <img src="https://github.com/naseemap47/CustomPose-Classification-Mediapipe/blob/master/ImageOutput/dog.jpg">
  </div>
  <div class="column">
  <img src="https://github.com/naseemap47/CustomPose-Classification-Mediapipe/blob/master/ImageOutput/tree.jpg">
  <img src="https://github.com/naseemap47/CustomPose-Classification-Mediapipe/blob/master/ImageOutput/warrior.jpg">
  </div>
</div>

# (Demo) Let's Get Started...

Using this Custom Pose Classification, I am going to Create a Yoga Pose Classification. Using **Yoga Poses Dataset**.

### Clone this Repository

```
git clone https://github.com/naseemap47/CustomPose-Classification-Mediapipe.git
cd CustomPose-Classification-Mediapipe
```

### Install Dependency

```
pip3 install -r requirements.txt
```

### 1.Download Dataset:

**Yoga Poses Dataset:**

```
wget -O yoga_poses.zip http://download.tensorflow.org/data/pose_classification/yoga_poses.zip
```

About Dataset:

- 5 Classes: **Chair, Cobra, Dog, Tree and Warrior**
- Contain Train and Test data
- Combain both Train and Test data

**Dataset Structure:**

```
├── Dataset
│   ├── Chair
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   ├── Cobra
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
.   .
.   .
```

### 2.Create Landmark Dataset for each Classes

```
python3 poseLandmark_csv.py -i <path_to_data_dir> -o <path_to_save_csv>
```

Example:

```
python3 poseLandmark_csv.py -i data/ -o data.csv
```
#### Creating Existing Dataset Path (pose.csv):
```
python poseLandmark_csv.py -i Dataset -o output/pose.csv
```
CSV file will be saved in **<path_to_save_csv>**

### 3.Create DeepLearinng Model to predict Human Pose

```
python3 poseModel.py -i <path_to_save_csv> -o <path_to_save_model>
```

Example:

```
python3 poseModel.py -i data.csv -o model.h5
```
#### Using Existing Dataset Path (pose.csv):
```
python poseModel.py -i output/pose.csv -o posecsv.h5
```
Model will saved in **<path_to_save_model>** and Model Metrics saved in **metrics.png**

### 4.Inference

Show Predicted Pose Class on Test Image or Video or Web-cam <br>
**To Save:**

- `--save`: It will save Images (on **ImageOutput** Dir) or Videos ("**output.avi**")

```
python3 inference.py --model <path_to_model> \
                     --conf <model_prediction_confidence> \
                     --source <image or video or web-cam>

# to save
python3 inference.py --model <path_to_model> \
                     --conf <model_prediction_confidence> \
                     --source <image or video or web-cam> \
                     --save
```

Example:

```
python3 inference.py --model model.h5 --conf 0.75 --source data/test/image.jpg
python3 inference.py --model model.h5 --conf 0.75 --source data/test/video.mp4
python3 inference.py --model model.h5 --conf 0.75 --source 0  # web-cam

# to save
python3 inference.py --model model.h5 --conf 0.75 --source data/test/image.jpg --save
python3 inference.py --model model.h5 --conf 0.75 --source data/test/video.mp4 --save
python3 inference.py --model model.h5 --conf 0.75 --source 0 --save # web-cam
```

**To Exit Window - Press Q-key**

# Custom Pose Classification

### Clone this Repository

```
git clone https://github.com/naseemap47/CustomPose-Classification-Mediapipe.git
cd CustomPose-Classification-Mediapipe
git checkout custom
```

### 1.Take your Custom Pose Dataset

**Dataset Structure:**

```
├── Dataset
│   ├── Pose_1
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   ├── Pose_2
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
.   .
.   .
```

### 2.Create Landmark Dataset for each Classes

CSV file will be saved in **<path_to_save_csv>**

```
python3 poseLandmark_csv.py -i <path_to_data_dir> -o <path_to_save_csv>
```

### 3.Create DeepLearinng Model to predict Human Pose

Model will saved in **<path_to_save_model>** and Model Metrics saved in **metrics.png**

```
python3 poseModel.py -i <path_to_save_csv> -o <path_to_save_model>
```

### 4.Inference

Open **inference.py**

change **Line-43**:
According to your Class Names, Write Class Order <br>
**To Save:**

- `--save`: It will save Images (on **ImageOutput** Dir) or Videos ("**output.avi**")

```
python3 inference.py --model <path_to_model> \
                     --conf <model_prediction_confidence> \
                     --source <image or video or web-cam> \

# to save
python3 inference.py --model <path_to_model> \
                     --conf <model_prediction_confidence> \
                     --source <image or video or web-cam> \
                     --save

```

## Inference on Video

python inferenceAngle.py --model models/frontalV2.keras --conf 0.9 --source videos/rorymcilroy.mp4 --save

## Inference on Image

python inferenceAngle.py --model models/frontal.keras --conf 0.9 --source test/P1/P1.png --save

python inferenceAngle.py --model models/frontal.keras --conf 0.9 --source test/P3/P3.png --save

Show Predicted Pose Class on Test Image or Video or Web-cam

**To Exit Window - Press Q-key**

---

## Testing

### Local Testing for `/process-video` Endpoint

The `/process-video` endpoint (deployed as a Cloud Run service) can be tested locally without redeployment. The integration test mocks cloud dependencies and saves actual output files for manual inspection.

#### Quick Start

```bash
# Navigate to Mediapipe directory
cd Computer_Vision/Mediapipe

# Run the integration test
python tests/test_with_local_output.py
```

#### What This Does

- Processes the complete video with real MediaPipe and TensorFlow
- Saves all output files to `tests/test_output/test_[timestamp]/`
- Displays detailed statistics (pose distribution, angle status, sample predictions)
- Creates timestamped folders for each test run
- Allows manual inspection of:
  - Processed video with annotations
  - CSV files with predictions and angles
  - Pose class thumbnail images (P1-P10)

#### Test Documentation

- **`tests/README.md`** - Complete testing guide with setup instructions
- **`tests/test_with_local_output.py`** - Integration test with local outputs

#### What's Tested

- Complete video processing workflow
- MediaPipe pose detection integration
- TensorFlow model predictions
- Angle calculations (shoulder/hip tilt)
- Output file generation (video, CSVs, images)
- Response structure verification

#### Requirements

- Test video: `FO_videos/grant.mp4` (included)
- Test model: `best_model.keras` (included)
- Python dependencies: `flask tensorflow mediapipe opencv-python`

For detailed testing instructions and mocking explanation, see **[tests/README.md](tests/README.md)** and **[tests/HOW_TESTING_WORKS.md](tests/HOW_TESTING_WORKS.md)**
