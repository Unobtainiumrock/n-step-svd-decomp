import json
import os
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            'message': 'Tone Calibration API',
            'method': 'GET',
            'path': self.path,
            'timestamp': '2024-01-01T00:00:00Z'
        }
        
        self.wfile.write(json.dumps(response).encode())
        return
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            'message': 'Tone Calibration API',
            'method': 'POST',
            'path': self.path,
            'timestamp': '2024-01-01T00:00:00Z'
        }
        
        self.wfile.write(json.dumps(response).encode())
        return
