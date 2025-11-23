# Tests for Video Processing Service

This directory contains an integration test for the `/process-video` endpoint in `main.py`.

## Overview

The video processing service is deployed as a Cloud Run service on Google Cloud. Testing it requires mocking cloud dependencies (Google Cloud Storage, MongoDB) to enable local testing without redeployment.

## Test File

### `test_with_local_output.py`
Integration test that saves actual output files locally for manual inspection.

**Features:**
- Full integration testing of video processing pipeline
- Mocks GCS (Google Cloud Storage) operations
- Mocks MongoDB database operations
- Saves processed video, CSVs, and pose images to `tests/test_output/`
- Allows manual inspection of all outputs
- Detailed statistics and sample output display

**Output files:**
- `processed_video.mp4` - Annotated video with predictions
- `predictions.csv` - Frame-by-frame predictions
- `angles.csv` - Angle calculations per pose
- `pose_images/` - Thumbnail images for each pose class (P1.jpg, P2.jpg, etc.)

Each test run creates a new `test_[timestamp]/` folder containing all outputs.

## Running the Test

### Prerequisites

1. **Required files:**
   - Test video: `Computer_Vision/Mediapipe/FO_videos/grant.mp4`
   - Test model: `Computer_Vision/Mediapipe/best_model.keras`

2. **Python dependencies:**
   ```bash
   pip install flask tensorflow mediapipe opencv-python
   ```

### Run the Test
```bash
# From the Mediapipe directory
cd Computer_Vision/Mediapipe

# Run the test
python tests/test_with_local_output.py
```

**What this does:**
- Processes the entire video with real MediaPipe
- Saves all output files to `tests/test_output/test_[timestamp]/`

**Output files location:**
```
tests/test_output/
├── test_20251108_143056/        ← First test run
│   ├── processed_video.mp4
│   ├── predictions.csv
│   ├── angles.csv
│   └── pose_images/
│       ├── P1.jpg
│       ├── P2.jpg
│       └── ... (up to P10.jpg)
├── test_20251108_154521/        ← Second test run
│   ├── processed_video.mp4
│   ├── predictions.csv
│   ├── angles.csv
│   └── pose_images/
│       └── ...
└── test_20251108_165632/        ← Third test run
    └── ...
```

## Endpoint Analysis

### Input Specification

**Endpoint:** `POST /process-video`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "video_id": "string (MongoDB ObjectId)",
  "video_path": "string (GCS blob path)",
  "classification_model": "string (GCS blob path to .keras model)",
  "output_video_path": "string (optional, GCS destination)",
  "output_csv_path": "string (optional, GCS destination)",
  "output_angle_csv_path": "string (optional, GCS destination)",
  "bucket_name": "string (default: 'golf-swing-models')"
}
```

**Required Fields:**
- `video_id`: Identifies the video in MongoDB
- `video_path`: GCS path to input video
- `classification_model`: GCS path to TensorFlow model

**Optional Fields:**
- `output_video_path`: Custom output video path
- `output_csv_path`: Custom predictions CSV path
- `output_angle_csv_path`: Custom angles CSV path
- `bucket_name`: GCS bucket name (defaults to 'golf-swing-models')

### Output Specification

**Success Response (200):**
```json
{
  "status": "Processing complete",
  "predictions": [
    {
      "frame": 0,
      "predicted_class": "P1",
      "confidence": 0.95,
      "video_time": 0.0,
      "shoulder_tilt_angle": 15.5,
      "hip_tilt_angle": 8.2,
      "shoulder_tilt_status": "Good",
      "hip_tilt_status": "Good",
      "overall_status": "Good"
    }
  ],
  "output_video": "https://storage.googleapis.com/.../video.mp4",
  "output_csv": "https://storage.googleapis.com/.../predictions.csv",
  "output_angle_csv": "https://storage.googleapis.com/.../angles.csv",
  "output_pose_images": {
    "P1": "https://storage.googleapis.com/.../P1.png",
    "P2": "https://storage.googleapis.com/.../P2.png",
    ...
    "P10": "https://storage.googleapis.com/.../P10.png"
  }
}
```

**Error Response (400/500):**
```json
{
  "error": "Error message describing what went wrong"
}
```

## How the Test Works

### 1. Mocking Strategy

**Google Cloud Storage (GCS):**
- Mocks `storage.Client` to avoid real GCS calls
- `download_blob()` → copies local test files instead of downloading from cloud
- `upload_blob()` → saves files locally instead of uploading to cloud
- Returns fake GCS URLs to satisfy the endpoint

**MongoDB:**
- Mocks database collection operations
- `find_one()` → returns fake document
- `update_one()` → returns success

**FFmpeg:**
- Mocks video conversion
- Just copies the file instead of running actual H264 encoding

### 2. Test Flow

```
1. Setup mocks for GCS, MongoDB, and FFmpeg
2. Configure blob operations to use local files
3. Send POST request with test payload
4. Endpoint processes video using local files (REAL processing)
5. Mocks intercept upload operations and save files locally
6. Display response structure and statistics
7. Save all outputs to timestamped folder
```

### 3. What's Mocked vs What's Real

| Component | Status | Reason |
|-----------|--------|--------|
| Google Cloud Storage | **MOCKED** | Avoid cloud dependencies and costs |
| MongoDB | **MOCKED** | Avoid database setup |
| FFmpeg conversion | **MOCKED** | Speed up testing |
| **Video processing** | **REAL** | Core functionality |
| **MediaPipe pose detection** | **REAL** | Core functionality |
| **ML model inference** | **REAL** | Core functionality |
| **Angle calculations** | **REAL** | Core functionality |

