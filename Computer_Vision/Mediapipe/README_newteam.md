- pose.csv was generated using current Dataset folder (provided dataset)
- data.csv is legacy csv file (Unknown data source)

### 1. Create Landmark Dataset for each Classes

```
python3 poseLandmark_csv.py -i <path_to_data_dir> -o <path_to_save_csv>
```
#### Example: Creating Existing Dataset Path (pose.csv):
```
python poseLandmark_csv.py -i Dataset -o output/pose.csv
```
CSV file will be saved in **<path_to_save_csv>**, used for the next step.

### 2. Create DeepLearning Model to predict Human Pose (Simplified Code)

```
python3 poseModel.py -i <path_to_save_csv> -o <path_to_save_model>
```

#### USING SIMPLIFIED SCRIPT:
```
python poseModelSimplified.py -i output/pose.csv -o output/pose.h5
```

#### USING OLD SCRIPT:
```
python poseModel.py -i output/pose.csv -o output/posecsv.h5
```