# AI Study Notes Generator using Prompt Engineering

A Flask web app that generates structured study notes with Google Gemini based on a topic.

## Features
- Short definition, key points, examples, MCQs, common mistakes, quick revision tips
- Fixed prompt in the backend for consistent formatting
- Simple centered UI for quick use

## Setup
1. Create a virtual environment (optional).
2. Install dependencies:
   `pip install -r requirements.txt`
3. Add your key to .env:
   `GEMINI_API_KEY=YOUR_API_KEY_HERE`
4. Run the app:
   `python app.py`
5. Open the local address shown in the terminal.

## Notes
- Keep the .env file private and do not commit real API keys.
