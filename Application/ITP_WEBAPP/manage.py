#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import json
from dotenv import load_dotenv
from google.oauth2 import service_account
from google.cloud import storage
from dashboard.google_cloud import get_google_cloud_storage_client


def main():
    """Run administrative tasks."""
    # Initialize Google Cloud Storage before running any command
    get_google_cloud_storage_client()

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ITP_WEBAPP.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()