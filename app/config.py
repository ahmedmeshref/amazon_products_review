import os


SECRET_KEY = os.urandom(32)
WTF_CSRF_ENABLED = True
SQLALCHEMY_TRACK_MODIFICATIONS = False
