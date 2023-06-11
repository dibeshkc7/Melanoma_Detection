from flask import Flask, render_template, request
import keras
import tensorflow as tf
from keras.models import load_model
import numpy as np

app = Flask(__name__)

dic = {0: 'Benign', 1: 'Malignant'}

model = load_model('E:/Cancer-detection-classification-CNN-deep-learning-main/Cancer-detection-classification-CNN-deep-learning-main/model_3.h5')

model.make_predict_function()

def predict_label(img_path):
    i = keras.utils.load_img(img_path, target_size=(224,224))
    i = tf.keras.utils.img_to_array(i)/255.0
    i = i.reshape(1,224,224,3)
    p = np.argmax(model.predict(i), axis = -1)
    return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        image = request.files['my_image']

        save_directory = "E:/Cancer-detection-classification-CNN-deep-learning-main/Cancer-detection-classification-CNN-deep-learning-main/static/images"
        app.config['UPLOAD'] = save_directory

        file_name = image.filename

        image.save(f"{save_directory}/{image.filename}")

        p = predict_label(f"{save_directory}/{image.filename}")

    return render_template("index.html", prediction = p, file_name = file_name)

if __name__ == '__main__':
    app.run(debug=True)
