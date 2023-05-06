import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os


class ECGClassificationPage:
    def __init__(self, root):
        self.root = root

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
        messagebox.showinfo("Diagnosis Saved", "The diagnosis has been saved to 'Saved Diagnoses' folder.")


class ECGClassificationInfoPage:
    def __init__(self, root):
        self.root = root

    # Create a label to display the ECG [classification information](poe://www.poe.com/_api/key_phrase?phrase=classification%20information&prompt=Tell%20me%20more%20about%20classification%20information.)
        self.info_label = tk.Label(self.root, text="ECG classification is a technique used to identify different types of [heart rhythms](poe://www.poe.com/_api/key_phrase?phrase=heart%20rhythms&prompt=Tell%20me%20more%20about%20heart%20rhythms.) from an [ECG signal](poe://www.poe.com/_api/key_phrase?phrase=ECG%20signal&prompt=Tell%20me%20more%20about%20ECG%20signal.). [ECG classification](poe://www.poe.com/_api/key_phrase?phrase=ECG%20classification&prompt=Tell%20me%20more%20about%20ECG%20classification.) can help in the prevention of heart attacks by identifying [abnormal heart rhythms](poe://www.poe.com/_api/key_phrase?phrase=abnormal%20heart%20rhythms&prompt=Tell%20me%20more%20about%20abnormal%20heart%20rhythms.) before they cause serious damage.\n\nThere are several types of [ECG classifications](poe://www.poe.com/_api/key_phrase?phrase=ECG%20classifications&prompt=Tell%20me%20more%20about%20ECG%20classifications.), including:\n\n1. Fusion of ventricular and normal beat\n2. [Myocardial Infarction](poe://www.poe.com/_api/key_phrase?phrase=Myocardial%20Infarction&prompt=Tell%20me%20more%20about%20Myocardial%20Infarction.)\n3. Normal Beat\n4. Premature Ventricular Contraction\n5. Supraventricular Premature Beat\n6. Unclassifiable Beat\n\nBy accurately predicting the ECG classification, doctors can take appropriate measures to prevent heart attacks and other [heart diseases](poe://www.poe.com/_api/key_phrase?phrase=heart%20diseases&prompt=Tell%20me%20more%20about%20heart%20diseases.).")

        self.info_label.pack(pady=10)


class ECGProgramInfoPage:
    def __init__(self, root):
        self.root = root

    # Create a label to display the [program information](poe://www.poe.com/_api/key_phrase?phrase=program%20information&prompt=Tell%20me%20more%20about%20program%20information.)
        self.info_label = tk.Label(self.root, text="The [ECG Diagnosis](poe://www.poe.com/_api/key_phrase?phrase=ECG%20Diagnosis&prompt=Tell%20me%20more%20about%20ECG%20Diagnosis.) program is designed to help doctors and medical professionals accurately identify different types of heart rhythms from an ECG signal. The program uses [machine learning algorithms](poe://www.poe.com/_api/key_phrase?phrase=machine%20learning%20algorithms&prompt=Tell%20me%20more%20about%20machine%20learning%20algorithms.) to analyze the ECG signal and predict the type of [heart rhythm](poe://www.poe.com/_api/key_phrase?phrase=heart%20rhythm&prompt=Tell%20me%20more%20about%20heart%20rhythm.).\n\nThe program is easy to use and can save doctors and medical professionals a significant amount of time by automating the ECG classification process. The program can also help in the prevention of heart attacks by identifying abnormal heart rhythms before they cause serious damage.\n\nThe program is designed to be user-friendly and can be used by doctors and medical professionals with minimal training.")

        self.info_label.pack(pady=10)


class ECGFilePage:
    def __init__(self, root):
        self.root = root

        # Create a label to display the instructions
        self.instructions_label = tk.Label(self.root, text="Please select the ECG file to diagnose:")
        self.instructions_label.pack(pady=10)

        # Create a button to open the file dialog
        self.file_button = tk.Button(self.root, text="Select File", command=self.open_file_dialog)
        self.file_button.pack(pady=10)

        # Create a label to display the selected file path
        self.file_path_label = tk.Label(self.root, text="")
        self.file_path_label.pack(pady=10)

    def open_file_dialog(self):
        # Open the file dialog
        file_path = tk.filedialog.askopenfilename()

        # Update the file path label with the selected file path
        self.file_path_label.config(text=file_path)


class SavedDiagnosisPage:
    def __init__(self, root):
        self.root = root

        # Create a label to display the saved diagnoses
        self.saved_diagnoses_label = tk.Label(self.root, text="List of saved diagnoses:")
        self.saved_diagnoses_label.pack(pady=10)

        # Create a listbox to display the saved diagnoses
        self.saved_diagnoses_listbox = tk.Listbox(self.root)
        self.saved_diagnoses_listbox.pack()

        # Create a button to delete selected diagnosis
        self.remove_button = tk.Button(self.root, text="Remove Selected Diagnosis", command=self.remove_selected_diagnosis)
        self.remove_button.pack(pady=10)

    def remove_selected_diagnosis(self):
        # Get the selected diagnosis from the listbox
        selected_diagnosis = self.saved_diagnoses_listbox.get(tk.ACTIVE)

        # Delete the selected diagnosis from the listbox
        self.saved_diagnoses_listbox.delete(tk.ACTIVE)

        # Delete the selected diagnosis file
        os.remove(selected_diagnosis)

#Create the main application window


root = tk.Tk()

#Create the tabs
tab_control = ttk.Notebook(root)

ecg_file_tab = ttk.Frame(tab_control)
tab_control.add(ecg_file_tab, text="ECG File")

ecg_classification_tab = ttk.Frame(tab_control)
tab_control.add(ecg_classification_tab, text="ECG Classification")

program_info_tab = ttk.Frame(tab_control)
tab_control.add(program_info_tab, text="Program Information")

#Create the pages for each tab
ecg_file_page = ECGFilePage(ecg_file_tab)
ecg_classification_page = ECGClassificationPage(ecg_classification_tab)
saved_diagnoses_page = SavedDiagnosisPage(ecg_classification_tab)
ecg_classification_info_page = ECGClassificationInfoPage(ecg_classification_tab)
ecg_program_info_page = ECGProgramInfoPage(program_info_tab)

#Pack the tabs
tab_control.pack(expand=1, fill="both")

root.geometry("600x500")
root.title("ECG Diagnosis Program")

#Start the main event loop
root.mainloop()
