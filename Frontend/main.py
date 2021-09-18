import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.models import load_model
model = load_model('Vggnet.hdf5')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
class_names=['Cattle', 'Horse', 'Hyena', 'Leopard', 'Lion', 'Tiger','Wolf']

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		sunflower_path = 'static/uploads/'+filename

		img = tf.keras.preprocessing.image.load_img(
    		sunflower_path, target_size=(180, 180)
		)
		img_array = tf.keras.preprocessing.image.img_to_array(img)
		img_array = tf.expand_dims(img_array, 0) # Create a batch

		predictions = model.predict(img_array)
		score = tf.nn.softmax(predictions[0])

		a=(
    		"This image most likely belongs to {} ."
    		.format(class_names[np.argmax(score)], 100 * np.max(score))
		)
		return render_template(
            "qwe.html",
            data=a,
            filename=filename 
        )

	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()