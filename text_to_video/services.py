from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
from generate_video import text_to_video;
app = Flask(__name__)
CORS(app)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'text is required'}), 400

    text = data['text']

    try:
        path = text_to_video(text)
        return send_file(path, as_attachment=True)
       
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)