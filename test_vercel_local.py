"""
Local Testing Server simulating Vercel's routing environment.
Runs static frontend from public/ and routes /api/predict to api.index.handler.
Usage: python test_vercel_local.py
"""

import os
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
import mimetypes

# Set project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.index import handler as ApiHandler


class VercelLocalDevHandler(ApiHandler, SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        SimpleHTTPRequestHandler.__init__(self, *args, directory=PUBLIC_DIR, **kwargs)

    def do_OPTIONS(self):
        if self.path.startswith("/api"):
            return ApiHandler.do_OPTIONS(self)
        return SimpleHTTPRequestHandler.do_OPTIONS(self)

    def do_GET(self):
        if self.path.startswith("/api"):
            return ApiHandler.do_GET(self)
        
        # Route root to index.html
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
            
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path.startswith("/api"):
            return ApiHandler.do_POST(self)
        self.send_error(404, "Endpoint not found")


def run(port=3000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, VercelLocalDevHandler)
    print(f"\n=======================================================")
    print(f" SpamGuard AI — Vercel Local Emulation Server")
    print(f" Local Web App:  http://localhost:{port}")
    print(f" API Endpoint:   http://localhost:{port}/api/predict")
    print(f" Press Ctrl+C to stop the server.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    run(port)
