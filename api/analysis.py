from flask import Blueprint, request, jsonify

analysis_blueprint = Blueprint('analysis', __name__)

@analysis_blueprint.route('/', methods=['POST'])
def analyze_video():
    data = request.get_json()
    video_path = data.get('path')
    # Placeholder response
    result = {
        'overall_score': 0.0,
        'back_angle': 0.0,
        'leg_drive_angle': 0.0,
        'arm_angle': 0.0
    }
    return jsonify(result)
