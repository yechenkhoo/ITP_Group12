# Prerequisites
Node.js: Ensure that Node.js is installed before proceeding.
- Then add this to ITP WEBAPP SETTINGS
- NPM_BIN_PATH = r"C:\Program Files\nodejs\npm.cmd"

Python: Python should also be installed on your system.

Env: .env file has to be populated with the necessary data and placed at root path    ITP_SE_TEAM13

# Installation
- Open CMD by running as Administrator
- Change directory to the project folder "...Application/ITP_WEBAPP"

Run the following command in order.
```
py -m venv .venv
```
```
.venv\Scripts\activate
```
```
python -m pip install -r ITP_WEBAPP/requirements.txt
```

# Design Contraints
Mongo-session is out of date and you are required to change the syntax within the library, we have saved the updated library files in "...application/LibraryFix".
- Copy those files to venv, find mongo-sessions and paste those files into it after installing all libraries

# Running the project
Start the Backend Server: 
- Make sure the virtual environment (.venv) is activated in ../Application/ITP_WEBAPP. 
- Then, start the Django server by running:

```
python manage.py runserver
```
Run Tailwind for Frontend: To compile any frontend changes you’ve made, ensure the Tailwind listener is active:
```
python manage.py tailwind start
```


