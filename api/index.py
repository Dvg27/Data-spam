"""
Vercel Serverless Function Handler for SpamGuard AI SMS Classifier.
Supports GET (health/info), POST (SMS prediction), and OPTIONS (CORS preflight).
"""
from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import nltk

# ---------------------------------------------------------------------------
# Path setup — ensure project root and bundled nltk_data are on the right paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Point NLTK at the bundled data directory (included via vercel.json includeFiles)
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
if os.path.isdir(NLTK_DATA_DIR) and NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

from src.prediction import predict_sms


class handler(BaseHTTPRequestHandler):
    """
    Vercel Serverless Function Handler for SpamGuard AI SMS Classifier.
    Supports GET (health/info), POST (SMS prediction), and OPTIONS (CORS preflight).
    """

    def _send_json(self, status_code: int, data: dict):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self._send_json(200, {"status": "ok"})

    def do_GET(self):
        """Health check and API metadata endpoint."""
        self._send_json(200, {
            "status": "healthy",
            "service": "SpamGuard AI — SMS Classifier API",
            "version": "1.0.0",
            "endpoints": {
                "POST /api/predict": {
                    "description": "Classify an SMS message as SPAM or HAM",
                    "payload": {"message": "string"}
                }
            }
        })

    def do_POST(self):
        """Perform SMS spam classification."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json(400, {
                    "error": "Request body cannot be empty. Please send a JSON object with a 'message' field."
                })
                return

            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body)

            # Accept either 'message' or 'text'
            message = payload.get("message", payload.get("text", ""))

            if not isinstance(message, str) or not message.strip():
                self._send_json(400, {
                    "error": "Please provide a valid non-empty SMS 'message' string."
                })
                return

            # Perform prediction using trained model artifacts
            result = predict_sms(message)
            self._send_json(200, result)

        except json.JSONDecodeError:
            self._send_json(400, {
                "error": "Malformed JSON payload. Please send valid JSON."
            })
        except Exception as exc:
            self._send_json(500, {
                "error": f"Internal inference error: {str(exc)}"
            })
