from flask import Flask, render_template, url_for, redirect, request,send_from_directory,flash
import json

from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField

import urllib.request
import os
from wtforms.validators import InputRequired

#from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
# from flask_bootstrap import Bootstrap
UPLOAD_FOLDER = 'static/brainimages/'
app = Flask(_name_, template_folder='templates')
app = Flask(_name_)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
@app.route('/index1')
def homePage():
    return render_template("index1.html", title="heart disease", custom_css="index")

@app.route('/prediction', methods=['POST', 'GET'])
def classification():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            # Use this werkzeug method to secure filename.
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # getPrediction(filename)
            label = getPrediction(filename)
            print(label)
            arr_list = label.tolist()

            # convert the list to a JSON string
            json_str = json.dumps(arr_list)
            flash(json_str)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            flash(full_filename)
            return redirect('/classification')

    # This code will execute for GET requests to /classification
    return render_template("prediction.html", title="prediction", custom_css="index",custom_js="index")
if _name_ == '_main_':
    app.run(debug=True,use_reloader=False, port=8000)