# Gemini Triplet Extractor using Google Cloud Vertex AI

This project uses Gemini API on Google Cloud to extract knowledge triplets from Wikipedia articles.

## Features
- Authentication via service account (`.json` key file)
- Fetches content from Wikipedia using `wikipedia` module
- Sends prompt to Gemini endpoint and extracts triplets
- Saves results as structured JSON

## Technologies
- Python 3
- Google Cloud Vertex AI
- Wikipedia API
- JSON output format

## Files
- `gemini_api_calls.py`: main logic
- `pelagic-campus-*.json`: service account credentials (not uploaded to Git)
- `requirements.txt`: required dependencies

## How to Run
```bash
python gemini_api_calls.py
