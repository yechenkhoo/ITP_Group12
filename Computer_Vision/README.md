# Computer Vision

## 1. Prerequisites

- Python version must be <= 3.11
- Ensure `pip` is installed

## 2. Installation

- Navigate to the project directory (cd Mediapipe or cd tensorflow).
- Create and activate a virtual environment
  - `python -m venv .venv`
- Activate the virtual environment:
  - Mac/Linux: `source .venv/bin/activate`
  - Windows: `.venv\Scripts\activate`
- Install the required packages by running
  - `pip install -r requirements.txt`

## 3. Running the Application

### Mediapipe

`python inferenceAngle.py --model 'PathOfModel' --conf 'confidence_score' --source 'PathOfVideo' --save"`

### Validation

`python validation.py --model 'PathOfModel' --dataset 'PathOfDataset' --conf 'confidence_score' --confusion_matrix 'PathofConfusionMatrix'`

#### Models available

Current list of model:

1. model.keras

### Tensorflow

`python pose_estimation.py --classifier 'PathOfClassifer' --label_file 'PathOfLabel'`

`python video_estimation.py --classifier 'PathOfClassifer' --label_file 'PathOfLabel' --videoPath 'PathOFVideo' --outputDir 'PathOfOutput'`

#### Models available

Current list of models and label can be found in the models folder:

1. P1andP2model.tflite and P1andP2Label.txt
2. P1toP10model.tflite and P1toP10label.txt

test
