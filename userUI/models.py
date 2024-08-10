# models.py
from extensions import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    filter_keyword = db.Column(db.String(120), nullable=True)  # Make sure this column exists

    def __repr__(self):
        return f'<User {self.username}>'
