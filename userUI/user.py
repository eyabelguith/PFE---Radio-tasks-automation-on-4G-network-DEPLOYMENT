from extensions import db

class User(db.Model):
    username = db.Column(db.String(80), unique=True, nullable=False, primary_key=True)
    password = db.Column(db.String(120), nullable=False)
    filter_keyword = db.Column(db.String(100))  # Change the length as needed