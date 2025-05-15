from django.db import models
from db_connection import MONGO_CLIENT
from bson import ObjectId

Users_Collection = MONGO_CLIENT['Users']

class User:
    @staticmethod
    def find_user_by_email(email):
        """Fetch a user document from MongoDB by email."""
        return Users_Collection.find_one({'Email': email})

    @staticmethod
    def verify_password(user, password):
        """Verify if the provided password matches the stored password."""
        return user['Password'] == password
    
    @staticmethod
    def find_user_by_id(id):
        """Fetch a user document from MongoDB by id."""
        return Users_Collection.find_one({'_id': ObjectId(id)})
    
    
        
        
    
    