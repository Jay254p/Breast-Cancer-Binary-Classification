import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import os

img_height, img_width = 150, 150  

model = load_model('breast_cancer_detection_1.h5')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Breast Cancer Detection</title>
    </head>
    <body>
        <h1>Breast Cancer Detection</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload and Predict">
        </form>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        img = image.load_img(filepath, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        result = 'Benign' if confidence < 0.5 else 'Malignant'
        confidence = 1 - confidence if result == 'Benign' else confidence
        
        return jsonify({'prediction': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
