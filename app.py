from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from groq import Groq
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://medibites11.vercel.app/"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize APIs
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

SYSTEM_PROMPT = """You are a professional doctor providing medical advice. Analyze the symptoms described and:
1. Provide a concise diagnosis (1-2 sentences)
2. Suggest immediate remedies (1-2 sentences)
3. Recommend when to see a doctor
4. Keep the tone professional but compassionate
5. Never use medical jargon without explanation
6. Always start directly with your assessment, no preamble"""

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        text_input = data.get('text', '')
        transcribed_text = data.get('transcribed_text', '')
        
        if not text_input and not transcribed_text:
            return jsonify({"error": "No input provided"}), 400

        full_prompt = f"{SYSTEM_PROMPT}\n\nPatient description: {text_input}\n\nAdditional notes: {transcribed_text}"

        # Get AI response
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            model="llama3-70b-8192",
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content

        return jsonify({
            "text_response": ai_response,
            "transcribed_text": transcribed_text
        })

    except Exception as e:
        logging.error(f"Error in /api/analyze: {str(e)}")
        return jsonify({"error": "Service temporarily unavailable"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
