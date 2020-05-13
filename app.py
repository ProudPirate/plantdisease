import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, Markup
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

#app.config['SERVER_NAME'] = '0.0.0.0:5000'

# app.config.from_pyfile(config.cfg)
# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

MODEL_DICT = {
    "Apple": "model.h5",
    "Blueberry": "model.h5",
    "Cherry": "model.h5",
    "Corn": "model.h5",
    "Grape": "model.h5",
    "Orange": "model.h5",
    "Peach": "model.h5",
    "Pepper": "model.h5",
    "Potato": "model.h5",
    "Raspberry": "model.h5",
    "Soybean": "model.h5",
    "Squash": "model.h5",
    "Strawberry": "model.h5",
    "Tomato": "model.h5"
}


def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    dropdown_options = [k for k, v in MODEL_DICT.items()]
    return render_template('index.html', options=dropdown_options)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        pname = request.form["plant_name"]
        model = tf.keras.models.load_model(
            MODEL_DICT[pname], compile=False)

        print('Model loaded!!')

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        # x = x.reshape([64, 64]);
        disease_class = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy',
                         'Cherry_(including_sour)___Powdery_mildew',
                         'Cherry_(including_sour)___healthy',
                         'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                         'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                         'Corn_(maize)___healthy', 'Grape___Black_rot',
                         'Grape___Esca_(Black_Measles)',
                         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                         'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                         'Peach___healthy', 'Pepper_bell___Bacterial_spot',
                         'Pepper_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                         'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                         'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                         'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                         'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                         'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                         'Tomato___healthy']
        a = preds[0]
        ind = np.argmax(a)
        print('Prediction:', disease_class[ind])
        result = disease_class[ind]
        result = tuple(result.split('___'))
        print(result)
        return {
            "plant": pname,
            "status": result[1],
        }
    return None


if __name__ == '__main__':
    # app.run(port=5000, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()
