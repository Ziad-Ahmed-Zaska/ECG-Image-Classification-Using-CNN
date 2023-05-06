import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os


class ECGDiagnosisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Diagnosis")
        self.root.geometry("600x500")

        # Create a frame for the image and diagnosis
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(pady=10)

        # Create a label to display the selected image
        self.image_label = tk.Label(self.top_frame)
        self.image_label.pack(side=tk.LEFT, padx=10)

        # Create a frame for the diagnosis and confidence
        self.diagnosis_frame = tk.Frame(self.top_frame)
        self.diagnosis_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        # Create a label to display the diagnosis
        self.diagnosis_label = tk.Label(self.diagnosis_frame, font=("Arial", 20))
        self.diagnosis_label.pack(pady=10)

        # Create a label to display the confidence
        self.confidence_label = tk.Label(self.diagnosis_frame, font=("Arial", 14))
        self.confidence_label.pack(pady=10)

        # Create a button to select an image
        self.browse_button = tk.Button(self.root, text="Select an ECG image", command=self.select_image)
        self.browse_button.pack(pady=10)

        # Create a button to save the diagnosis
        self.save_button = tk.Button(self.root, text="Save Diagnosis", command=self.save_diagnosis, state=tk.DISABLED)
        self.save_button.pack(pady=10)

        # Load the pre-trained CNN model
        model_path = 'Output Model/ECG Diagnosis Model_Best/CNN_model.h5'
        self.model = tf.keras.models.load_model(model_path)

    def select_image(self):
        # Open a file dialog to select an image
        file_path = filedialog.askopenfilename(initialdir='archive', title="Select an ECG image",
                                               filetypes=[("JPEG files", ".jpg"), ("PNG files", ".png")])
        if not file_path:
            return

        # Preprocess the image and display it
        img = Image.open(file_path)
        img = img.resize((224, 224))  # model was trained on 224x224 images
        img = img.convert('L')  # convert to grayscale
        img = np.array(img) / 255.0  # normalize pixel values
        img = np.expand_dims(img, axis=0)  # add batch dimension

        # Make predictions on the image
        predictions = self.model.predict(img)
        predicted_class = np.argmax(predictions)

        # Interpret the predictions
        class_names = ['Fusion of ventricular and normal beat', 'Myocardial Infarction', 'Normal Beat',
                       'Premature Ventricular Contraction', 'Supraventricular Premature Beat',
                       'Unclassifiable Beat']  # replace with actual class labels
        diagnosis = class_names[predicted_class]
        confidence = predictions[0][predicted_class]

        # Display the image and diagnosis
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img)
        self.image_label.image = img
        self.diagnosis_label.configure(text=diagnosis)
        self.confidence_label.configure(text=f"Confidence: {confidence * 100:.2f}%")
        self.save_button.configure(state=tk.NORMAL)
        self.selected_image_path = file_path

    def save_diagnosis(self):
        # Create a directory for the saved diagnoses if it doesn't exist
        if not os.path.exists('Saved Diagnoses'):
            os.mkdir('Saved Diagnoses')

        # Save the diagnosis to a text file
        diagnosis_text = f"The ECG image '{os.path.basename(self.selected_image_path)}' is classified as {self.diagnosis_label.cget('text')} with {self.confidence_label.cget('text')} confidence."
        with open(f"Saved Diagnoses/{os.path.basename(self.selected_image_path)}.txt", 'w') as f:
            f.write(diagnosis_text)

        # Display a message box to confirm the diagnosis was saved
        messagebox.showinfo("Diagnosis Saved", "The diagnosis has been saved to the 'Saved Diagnoses' folder.")

    def run(self):
        self.root.mainloop()


root = tk.Tk()
ECGDiagnosisGUI(root).run()
