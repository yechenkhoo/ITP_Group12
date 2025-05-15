# Steps

## Installation

1. pip install -r requirements.txt

## For Training

1. run setup.sh (dont need if the tflite models are installed)
2. python dataprocessing.py (run script)
3. python train.py (run script)

## For Inference

### For Webcam use 1, for video use 2

1. python pose_estimation.py --classifier models/P1toP10model.tflite --label_file models/P1toP10label.txt
2. python video_estimation.py --classifier models/P1toP10model.tflite --label_file models/P1toP10label.txt --videoPath videos\Golf.mp4 --outputDir testData

#### Models available

Current List of models can be found in the models folder.

1. P1andP2model.tflite and P1andP2Label.txt
2. P1toP10model.tflite and P1toP10label.txt
