"""
Test with Local Output Files

This test runs the video processing and saves actual output files locally
for manual inspection. Perfect for debugging and verifying results visually.

Output files are saved to: tests/test_output/
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import Mock, patch, MagicMock
import json
import shutil
import tempfile
from datetime import datetime

# Mock only cloud/production dependencies BEFORE importing main
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.storage'] = MagicMock()
sys.modules['pymongo'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['bson'] = MagicMock()
sys.modules['ffmpeg'] = MagicMock()

from main import app


def print_header(text):
    """Print a formatted header"""
    width = 80
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")


def print_section(text):
    """Print a formatted section header"""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}")


def run_test_with_local_output():
    """
    Run video processing test and save actual output files locally
    """
    
    print_header("TEST WITH LOCAL OUTPUT FILES")
    
    # Setup paths
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mediapipe_dir = os.path.dirname(test_dir)
    test_video = os.path.join(mediapipe_dir, "tests/test_videos", "Grant_FO.mp4")
    # test_video = os.path.join(mediapipe_dir, "tests/test_videos", "Grant_DTL.mp4")
    test_model = os.path.join(mediapipe_dir, "best_model.keras")
    
    # Create output directory
    output_base_dir = os.path.join(test_dir, "test_output")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create timestamped folder for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_dir = os.path.join(output_base_dir, f"test_{timestamp}")
    os.makedirs(test_run_dir, exist_ok=True)
    
    print_section("Step 1: Verify Required Files")
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False
    print(f"✓ Test video: {os.path.basename(test_video)}")
    
    if not os.path.exists(test_model):
        print(f"❌ Test model not found: {test_model}")
        return False
    print(f"✓ Test model: {os.path.basename(test_model)}")
    
    print(f"✓ Output directory: {test_run_dir}")
    
    # Setup Flask test client
    print_section("Step 2: Setting Up Test Environment")
    
    app.config['TESTING'] = True
    client = app.test_client()
    print("✓ Flask test client initialized")
    
    # Prepare output file paths (all in the same timestamped folder)
    local_video_output = os.path.join(test_run_dir, "processed_video.mp4")
    local_csv_output = os.path.join(test_run_dir, "predictions.csv")
    local_angle_csv_output = os.path.join(test_run_dir, "angles.csv")
    local_pose_dir = os.path.join(test_run_dir, "pose_images")
    os.makedirs(local_pose_dir, exist_ok=True)
    
    print_section("Step 3: Mocking Cloud Operations")
    
    with patch('main.storage.Client') as mock_storage_client, \
         patch('main.download_blob') as mock_download, \
         patch('main.upload_blob') as mock_upload, \
         patch('main.upload_video_blob') as mock_upload_video, \
         patch('main.Videos_Collection') as mock_collection, \
         patch('main.convert_to_h264') as mock_convert_h264:
        
        print("✓ GCS Client mocked")
        
        # Configure mocks
        mock_client = MagicMock()
        mock_storage_client.return_value = mock_client
        
        # Track what files were "uploaded"
        uploaded_files = {}
        
        def mock_download_func(bucket, blob, dest):
            """Copy local files instead of downloading from GCS"""
            if 'best_model.keras' in blob or 'model' in blob:
                shutil.copy(test_model, dest)
                print(f"   📥 Downloaded model: {blob}")
            elif '.mp4' in blob or 'video' in blob:
                shutil.copy(test_video, dest)
                print(f"   📥 Downloaded video: {blob}")
        
        def mock_convert_h264_func(input_path, output_path):
            """Mock ffmpeg conversion - just copy the file"""
            if os.path.exists(input_path):
                shutil.copy(input_path, output_path)
                print(f"   🎬 [Mocked] Converted to H264: {os.path.basename(output_path)}")
        
        def mock_upload_func(bucket, src, dest):
            """Save file locally AND return fake URL"""
            # Determine output location
            if 'video' in dest or dest.endswith('.mp4'):
                local_path = local_video_output
            elif 'predictions' in dest and dest.endswith('.csv'):
                local_path = local_csv_output
            elif 'angles' in dest or 'angle' in dest:
                local_path = local_angle_csv_output
            elif any(f'P{i}' in dest for i in range(1, 11)):
                # Pose image
                filename = os.path.basename(dest)
                local_path = os.path.join(local_pose_dir, filename)
            else:
                # Other file
                local_path = os.path.join(test_run_dir, os.path.basename(dest))
            
            # Copy the file locally
            if os.path.exists(src):
                shutil.copy(src, local_path)
                uploaded_files[dest] = local_path
                print(f"   💾 Saved locally: {os.path.basename(local_path)}")
            
            # Return fake GCS URL
            url = f"https://storage.googleapis.com/{bucket}/{dest}"
            return url
        
        mock_download.side_effect = mock_download_func
        mock_upload.side_effect = mock_upload_func
        mock_upload_video.side_effect = mock_upload_func
        mock_convert_h264.side_effect = mock_convert_h264_func
        
        # Mock MongoDB collection operations
        mock_collection.find_one.return_value = {"_id": "test_id"}
        mock_collection.update_one.return_value = Mock(modified_count=1)
        
        print("✓ Download/Upload functions mocked")
        print("✓ MongoDB mocked")
        
        # Prepare request
        print_section("Step 4: Processing Video")
        
        payload = {
            "video_id": f"test_{timestamp}",
            "video_path": test_video,
            "classification_model": "models/best_model.keras",
            "output_video_path": "output/processed_video.mp4",
            "output_csv_path": "output/predictions.csv",
            "output_angle_csv_path": "output/angles.csv",
            "bucket_name": "golf-swing-models"
        }
        
        print("Request payload:")
        print(json.dumps(payload, indent=2))
        print("\n⏳ Processing video (this may take a minute)...\n")
        
        # Make request
        response = client.post(
            '/process-video',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Display results
        print_section("Step 5: Results")
        
        print(f"HTTP Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print("\n❌ Test Failed")
            print(f"Error Response: {response.data.decode()}")
            return False
        
        data = json.loads(response.data)
        
        print(f"✓ Status: {data['status']}")
        print(f"✓ Predictions Generated: {len(data['predictions'])}")
        
        # Display statistics
        print_section("Statistics")
        
        # Pose class distribution
        pose_counts = {}
        for pred in data['predictions']:
            pose = pred['predicted_class']
            pose_counts[pose] = pose_counts.get(pose, 0) + 1
        
        print("Pose Class Distribution:")
        for pose in sorted(pose_counts.keys()):
            count = pose_counts[pose]
            percentage = (count / len(data['predictions'])) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {pose}: {bar} {count} frames ({percentage:.1f}%)")
        
        # Angle status distribution
        print("\nAngle Status Distribution:")
        status_counts = {'Good': 0, 'Bad': 0, 'Very Bad': 0}
        for pred in data['predictions']:
            status = pred.get('overall_status', 'Unknown')
            if status in status_counts:
                status_counts[status] += 1
        
        total_with_status = sum(status_counts.values())
        if total_with_status > 0:
            for status, count in status_counts.items():
                percentage = (count / total_with_status) * 100
                print(f"  {status}: {count} frames ({percentage:.1f}%)")
        
        # Sample predictions
        print_section("Sample Predictions (First 5 Frames)")
        
        print(f"{'Frame':<8} {'Pose':<6} {'Conf':<6} {'Time':<8} {'Shoulder':<10} {'Hip':<8} {'Status':<12}")
        print("─" * 80)
        
        for pred in data['predictions'][:5]:
            print(f"{pred['frame']:<8} "
                  f"{pred['predicted_class']:<6} "
                  f"{pred['confidence']:<6.2f} "
                  f"{pred['video_time']:<8.2f} "
                  f"{pred['shoulder_tilt_angle']:<10.1f} "
                  f"{pred['hip_tilt_angle']:<8.1f} "
                  f"{pred['overall_status']:<12}")
        
        # Output Files
        print_section("📁 OUTPUT FILES")
        
        print(f"\n🎯 All output files saved to: {test_run_dir}\n")
        
        # Check which files were actually created
        actual_files = []
        
        if os.path.exists(local_video_output):
            size_mb = os.path.getsize(local_video_output) / (1024 * 1024)
            print(f"✓ Processed Video: {os.path.basename(local_video_output)} ({size_mb:.2f} MB)")
            actual_files.append(local_video_output)
        else:
            print(f"⚠️  Processed Video: Not created")
        
        if os.path.exists(local_csv_output):
            with open(local_csv_output, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"✓ Predictions CSV: {os.path.basename(local_csv_output)} ({line_count} lines)")
            actual_files.append(local_csv_output)
        else:
            print(f"⚠️  Predictions CSV: Not created")
        
        if os.path.exists(local_angle_csv_output):
            with open(local_angle_csv_output, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"✓ Angles CSV: {os.path.basename(local_angle_csv_output)} ({line_count} lines)")
            actual_files.append(local_angle_csv_output)
        else:
            print(f"⚠️  Angles CSV: Not created")
        
        # Check pose images
        if os.path.exists(local_pose_dir):
            pose_images = [f for f in os.listdir(local_pose_dir) if f.endswith(('.jpg', '.png'))]
            if pose_images:
                print(f"✓ Pose Images: {len(pose_images)} images in {os.path.basename(local_pose_dir)}/")
                for img in sorted(pose_images)[:5]:  # Show first 5
                    print(f"  - {img}")
                if len(pose_images) > 5:
                    print(f"  ... and {len(pose_images) - 5} more")
            else:
                print(f"⚠️  Pose Images: No images found")
        
        # Final summary
        print_header("✓ TEST COMPLETE")
        
        print("Summary:")
        print(f"  • Video processed successfully")
        print(f"  • {len(data['predictions'])} frames analyzed")
        print(f"  • {len(pose_counts)} unique poses detected")
        print(f"  • {len(actual_files)} output files saved")
        print(f"\n📂 Open this folder to view results:")
        print(f"  {os.path.abspath(test_run_dir)}")
        print()
        
        return True


def main():
    """Main entry point"""
    try:
        if run_test_with_local_output():
            return 0
        else:
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
