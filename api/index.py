"""
Vercel serverless function entry point for Flask app
"""
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# This is the entry point for Vercel
# Vercel expects the app to be available as the default export

# Export the Flask app for Vercel
if __name__ == "__main__":
    app.run(debug=True)