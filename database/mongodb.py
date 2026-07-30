from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from config import MONGODB_URL
load_dotenv()

client=MongoClient(
    MONGODB_URL,
    server_api=ServerApi('1'),
    serverSelectionTimeoutMS=5000
)


db=client["sample_mflix"]
def get_collection():
    return db["documents"]

