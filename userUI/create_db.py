from app import app, db

# Use the app context to create the database tables
with app.app_context():
    db.create_all()
    print("Database tables created.")
