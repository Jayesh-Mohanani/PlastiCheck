import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
SQLALCHEMY_DATABASE_URI = os.getenv(
    'DATABASE_URL',
    f'sqlite:///{os.path.join(BASE_DIR, "app.db")}'
)
SQLALCHEMY_TRACK_MODIFICATIONS = False