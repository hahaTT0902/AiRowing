from flask import Blueprint, request, jsonify
import os
from app.config import Config

upload_blueprint = Blueprint('upload', __name__)

@upload_blueprint.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({'message': 'File uploaded', 'path': file_path}), 200
