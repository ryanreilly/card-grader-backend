from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from grader import grade_card

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB in bytes

@app.route('/grade', methods=['POST'])
def grade_image():
    try:
        # Create the uploads directory if it doesn't exist
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

        # Check if an image file is included in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        # Check image size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        if file_size > MAX_IMAGE_SIZE:
            return jsonify({'error': 'Image file too large. Maximum size is 5MB.'}), 400
        file.seek(0)  # Reset file pointer to the beginning

        # Save the image
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Process the image
        grades = grade_card(image_path)

        # Clean up
        os.remove(image_path)

        return jsonify(grades)

    except Exception as e:
        return jsonify({'error': str(e)}), 500