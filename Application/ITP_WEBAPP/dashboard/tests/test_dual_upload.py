from django.test import SimpleTestCase, Client
from django.urls import reverse, resolve
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch, MagicMock
import importlib


class DualUploadFlowTests(SimpleTestCase):
    def setUp(self):
        self.client = Client()
        # Use a valid ObjectId-like string for URL args
        self.student_id = '507f1f77bcf86cd799439011'
        # Session setup to simulate logged-in student
        session = self.client.session
        session['Id'] = self.student_id
        session['Role'] = 'student'
        session.save()

    def test_dual_upload_post_calls_upload_dual_videos(self):
        """
        Integration test: POSTing two files with videoType='both' should call
        Video.upload_dual_videos with the provided files and redirect.
        """
        # Resolve the view module from the URL so we patch the correct module path
        url = reverse('dashboard_dataSpace', args=[self.student_id])
        resolved = resolve(url)
        views_mod = __import__(resolved.func.__module__, fromlist=['*'])
        # Patch User and Video on the resolved views module
        with patch.object(views_mod, 'User') as mock_user, patch.object(views_mod, 'Video') as mock_video_class:
            # Include Role so view code can read user['Role']
            mock_user.find_user_by_id.return_value = {'_id': self.student_id, 'Name': 'Test Student', 'Role': 'student'}

            # Prepare fake uploaded files
            face_file = SimpleUploadedFile('face.mp4', b'fake-video-data', content_type='video/mp4')
            dtl_file = SimpleUploadedFile('dtl.mp4', b'fake-video-data', content_type='video/mp4')

            response = self.client.post(url, {
                'videoType': 'both',
                'face_on_file': face_file,
                'dtl_file': dtl_file
            })

            # Ensure upload_dual_videos was called
            self.assertTrue(mock_video_class.upload_dual_videos.called, "upload_dual_videos was not called")
            # Response should be a redirect back to the dataSpace page
            self.assertIn(response.status_code, (302, 301))

    def test_dashboard_results_detects_dual_upload_and_combines(self):
        """
        Unit/integration test: Given two videos sharing a session_id (face-on and down-the-line),
        the `dashboard_results` view should detect `is_dual_upload` and provide combined columns/data.
        """
        # Setup a pair of videos with same session_id
        session_id = 'session123'
        face_id = 'face123'
        dtl_id = 'dtl123'

        face_video = {
            'id': face_id,
            'session_id': session_id,
            'Type': 'face-on',
            'Title': 'Face On Sample',
            'rawVideoLink': ''
        }
        dtl_video = {
            'id': dtl_id,
            'session_id': session_id,
            'Type': 'down-the-line',
            'Title': 'Down The Line Sample',
            'rawVideoLink': ''
        }

        url = reverse('results', args=[self.student_id, face_id])
        resolved = resolve(url)
        views_mod = __import__(resolved.func.__module__, fromlist=['*'])
        with patch.object(views_mod, 'Video') as mock_video_class, patch.object(views_mod, 'User') as mock_user, patch.object(views_mod, 'process_csv_data') as mock_process_csv:
            mock_user.find_user_by_id.return_value = {'_id': self.student_id, 'Name': 'Test Student', 'Role': 'student'}

            mock_video_class.get_all_videos.return_value = [face_video, dtl_video]
            mock_video_class.get_video_url.return_value = ''
            mock_video_class.get_output_csv_url.return_value = 'http://example.com/dtl.csv'
            mock_video_class.get_csv_url.return_value = 'http://example.com/fo.csv'
            mock_video_class.get_video_status.return_value = {}
            mock_video_class.get_all_video_comments.return_value = []

            # Create very small FO and DTL CSV-parsed lists
            fo_data = [
                {'Pose Class': 'P1', 'Time Frame': 1, 'Shoulder Tilt': 10, 'Shoulder Tilt Status': 'Good'}
            ]
            fo_cols = ['Pose Class', 'Time Frame', 'Shoulder Tilt', 'Shoulder Tilt Status']

            dtl_data = [
                {'Reference Pose (FO)': 'P1', 'Time Frame': 1, 'Shoulder Tilt': 11, 'Shoulder Tilt Status': 'Average'}
            ]
            dtl_cols = ['Reference Pose (FO)', 'Time Frame', 'Shoulder Tilt', 'Shoulder Tilt Status']

            # process_csv_data is called twice (video1, video2) -> use side_effect
            mock_process_csv.side_effect = [(fo_data, fo_cols), (dtl_data, dtl_cols)]

            # Ensure session indicates logged-in user
            session = self.client.session
            session['Id'] = self.student_id
            session['Role'] = 'student'
            session.save()

            # Call the results view
            response = self.client.get(url)

            self.assertEqual(response.status_code, 200)
            # The template context should flag a dual upload
            self.assertTrue(response.context['is_dual_upload'])
            # Combined columns should be present (string includes 'Shoulder Tilt')
            self.assertIn('Shoulder Tilt', response.context['columns'])
