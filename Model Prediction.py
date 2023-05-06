import tensorflow as tf
import numpy as np
from PIL import Image

# Preprocess the ECG image
img_path = 'archive/Random Internet Samples/Supraventricular Premature Beat.png'
img = Image.open(img_path)
img = img.resize((224, 224))  # model was trained on 224x224 images
img = img.convert('L')  # convert to grayscale
img = np.array(img) / 255.0  # normalize pixel values

# Load the pre-trained CNN model
model_path = 'Output Model/ECG Diagnosis Model_Best/CNN_model.h5'
model = tf.keras.models.load_model(model_path)

# Make predictions on the ECG image
img = np.expand_dims(img, axis=0)  # add batch dimension
predictions = model.predict(img)
predicted_class = np.argmax(predictions)

# Interpret the predictions
class_names = ['Fusion of ventricular and normal beat', 'Myocardial Infarction', 'Normal Beat', 'Premature Ventricular Contraction', 'Supraventricular Premature Beat', 'Unclassifiable Beat']  # replace with actual class labels
diagnosis = class_names[predicted_class]
confidence = predictions[0][predicted_class]
print(f"The ECG image is classified as {diagnosis} with {confidence*100:.2f}% confidence.")
