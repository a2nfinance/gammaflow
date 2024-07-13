from flask import Flask, send_file, request, jsonify
from zip_utils import zip_folder

app = Flask(__name__)

@app.route('/download', methods=['POST'])
def download_file():
    data = request.get_json()
    if 'path' not in data:
        return jsonify({'error': 'path is required'}), 400

    path = data['path']
    need_zip = data["need_zip"]
    output_file = data["output_file"]

    try:
        if (need_zip):
            zip_folder(path, output_file)
            return send_file(output_file, as_attachment=True)
        else:
            return send_file(path, as_attachment=True)
       
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000, debug=False)