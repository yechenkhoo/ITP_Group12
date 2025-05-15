# Raspberry Pi

## Prerequisites
- Raspberry Pi with Raspbian (or Raspberry Pi OS) installed.
- Internet connection for updating packages.

## Steps
### 1. Update and Upgrade System
Ensure your Raspberry Pi is up to date by running the following commands:
```bash
sudo apt update && sudo apt upgrade -y
```
### 2. Install Python Package Manager (pip)
```bash
sudo apt install python3-pip -y
```
### 3. Install Dependencies from requirements.txt
```bash
pip install -r requirements.txt
```
### 4. Verify Installation
Check if Flask was installed correctly by running:
```bash
python3 -c "import flask; print(flask.__version__)"
```
If Flask is installed correctly, you should see the version number without errors.

## Making Changes to IP Addresses
### 1. Update the Flask Server IP
Configure your Flask app to run on a specific server, modify the app's URL. In your raspberry_pi.py file, update the following
```python
FLASK_URL = "http://<YOUR_SERVER_IP>:<PORT>"
```
Replace <YOUR_SERVER_IP> with the IP address of the website/server where the Flask app will run.

### 2. Update the Raspberry Pi IP in the Client-side Dashboard
Navigate to the following location:
**Application\ITP_WEBAPP\dashboard\views.py**
and find the RASPBERRY_PI_URL variable.

Update it with your Raspberry Pi’s IP address:
```python
RASPBERRY_PI_URL = "http://<RPI_IP>:<PORT>"
```
Replace <RPI_IP> with the Raspberry Pi’s IP address.

## Running the Flask App on Raspberry Pi
### 1. Open Thonny IDE
Launch Thonny, the Python IDE available on your Raspberry Pi.

### 2. Navigate to raspberry_pi.py
In Thonny, open the raspberry_pi.py file where your Flask app is defined.

### 3. Run the Flask App
Click the Run button in Thonny to start the Flask application. Once running, the app should be accessible on the specified IP address and port.
