from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path="../../env")

MongoDB_Username = os.getenv("MONGO_USERNAME")
MongoDB_Password = os.getenv("MONGO_PASSWORD")

url = f"mongodb+srv://{MongoDB_Username}:{MongoDB_Password}@itpteam13.wtajb.mongodb.net/?retryWrites=true&w=majority&appName=ITPTEAM13"

connection = MongoClient(url)
MONGO_CLIENT = connection.ITP

MONGO_SESSIONS_COLLECTION = MONGO_CLIENT['Sessions']  # Default collection name
MONGO_SESSIONS_TTL = 3600



