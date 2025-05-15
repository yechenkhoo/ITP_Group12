from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(dotenv_path="../../env")

MongoDB_Username = os.getenv("MONGO_USERNAME")
MongoDB_Password = os.getenv("MONGO_PASSWORD")

# MongoDB connection string
url = f"mongodb+srv://{MongoDB_Username}:{MongoDB_Password}@itpteam13.wtajb.mongodb.net/"

# Establish MongoDB connection
connection = MongoClient(url)
MONGO_CLIENT = connection.ITP
Videos_Collection = MONGO_CLIENT['Videos']
