import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "default-dev-secret-key")
    DEBUG = True