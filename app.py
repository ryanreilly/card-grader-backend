from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from grader import grade_card

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from your Netlify frontend

# Create a directory to store uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/grade', methods=['POST'])
def grade_image():
    try:
        # Check if an image file is included in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        # Save the image temporarily
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Process the image using your grade_card function
        grades = grade_card(image_path)

        # Delete the temporary image file
        os.remove(image_path)

        # Return the grades as JSON
        return jsonify(grades)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)