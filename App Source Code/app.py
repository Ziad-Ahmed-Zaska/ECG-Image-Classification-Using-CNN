import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template
from PIL import Image

#Creating a Flask application and loading the best saved model weights from the specified file.
app = Flask(__name__)
model = tf.keras.models.load_model('Output Model/ECG Diagnosis Model_Best/CNN_model.h5')

#Creating a list of class names for the classification problem.
class_names = ['Fusion of ventricular and normal beat', 'Myocardial Infarction', 'Normal Beat', 'Premature Ventricular Contraction',
               'Supraventricular Premature Beat', 'Unclassifiable Beat']

@app.route('/')
def home():
    return render_template("index1.html")

@app.route('/predict', methods=['POST'])

#Defining a predict function to perform ECG image classification using the loaded model and class names list.
def predict():

    # Obtaining the ECG image file from the HTTP request and preprocessing it for model input.
    img = request.files['image']
    img = Image.open(img)
    img = img.resize((224, 224))
    img = img.convert('L')
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Using the loaded model to make predictions on the preprocessed ECG image and obtaining the predicted class and confidence.
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    diagnosis = class_names[predicted_class]
    confidence = predictions[0][predicted_class]

    # Formatting the prediction results into a string and returning it as the HTTP response.
    result = f"The ECG image is classified as {diagnosis} with {confidence*100:.2f}% confidence."
    return result

if __name__ == '__main__':
    app.run()