from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('skin_disease_model.h5')

# Image size parameters (should match your model input)
img_height, img_width = 224, 224

# Define the class names (updated to include 'leprosy')
class_names = ['dermatomyositis', 'leprosy', 'morphea', 'normal', 'pityrasis_alba', 'psoriasis', 'vitiligo']

# Upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/app')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Prepare the image for prediction
        img = image.load_img(file_path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        # Make the prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)
        predicted_class = class_names[predicted_class_index[0]]

        # Return the result as JSON
        return jsonify({'prediction': predicted_class})

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
