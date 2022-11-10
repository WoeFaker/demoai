from email import message
from flask import Flask, render_template, request, url_for

import os

from keras.utils.image_utils import img_to_array
from keras.utils.image_utils import load_img
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16

from keras.models import load_model
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from keras.applications.resnet_v2 import ResNet50V2

IMAGES_FOLDER = os.path.join('images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

model = ResNet50V2()
model2 = load_model('../.keras/models/medical_trial_model.h5')
@app.route('/', methods=['GET'])
def hello_world():
    # train_samples = []
    # train_labels = []

    # for i in range(50):
    #         #5% of young experienced side effects
    #     random_young = randint(13, 64)
    #     train_samples.append(random_young)
    #     train_labels.append(1)

    #         #5% of old experienced no side effects
    #     random_old = randint(65, 100)
    #     train_samples.append(random_old)
    #     train_labels.append(0)

    # for i in range(1000):
    #         #95% of young experienced no side effects
    #     random_young = randint(13, 64)
    #     train_samples.append(random_young)
    #     train_labels.append(0)

    #         #95% of old experienced side effects
    #     random_old = randint(65,100)
    #     train_samples.append(random_old)
    #     train_labels.append(1)

    # # change the format to be lisible for the AI
    # train_labels = np.array(train_labels)
    # train_samples = np.array(train_samples)
    # train_samples, train_labels = shuffle(train_samples, train_labels)

    # #To help the AI being faster, we transforme data in number between 0 and 1
    # scaler = MinMaxScaler(feature_range=(0,1))
    # scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

    # #my thing
    # train_samples1 = [70,15,70]
    # train_samples1 = np.array(train_samples1)
    # scaled_train_samples1 = scaler.fit_transform(train_samples1.reshape(-1,1))
    # predictions1 = model2.predict(x=scaled_train_samples1, batch_size=10, verbose=0)
    # rounded_prediction1 = np.argmax(predictions1, axis=-1)

    # predictions = model2.predict(x=scaled_train_samples, batch_size=10, verbose=0)
    # rounded_prediction = np.argmax(predictions, axis=-1)

    # return render_template('index.html', resultat = rounded_prediction)
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files["imagefile"]
    image_path = "./static/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2],))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    
    return render_template('index.html', prediction = classification, yeps = image_path)

# def predict2():
#     scaler = MinMaxScaler(feature_range=(0,1))
#     train_samples1 = [100,26]
#     age= request.values["age"]
#     train_samples1.append(age)
#     train_samples1 = np.array(train_samples1)
#     scaled_train_samples1 = scaler.fit_transform(train_samples1.reshape(-1,1))
#     predictions1 = model2.predict(x=scaled_train_samples1, batch_size=10, verbose=0)
#     rounded_prediction1 = np.argmax(predictions1, axis=-1)
#     if rounded_prediction1[2] == 0:
#         message = "No side effects"
#     elif rounded_prediction1[2] == 1:
#         message = "You will have side effects"

#     return render_template('index.html', resultat1 = message)


if __name__ == '__main__':
    app.run(port=3000, debug=True)