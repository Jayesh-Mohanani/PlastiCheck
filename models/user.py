from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    email = db.Column(db.String(128), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ConsumptionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    selected_categories = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    regime = db.Column(db.String(32))
    percentages = db.Column(db.Text)
    box1_path = db.Column(db.String(256))
    box2_path = db.Column(db.String(256))
    pie_path = db.Column(db.String(256))