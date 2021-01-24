from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from IPython.display import FileLink
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)
import pickle
import random
import shutil
import cv2
import os
import keras
from keras.applications import VGG16
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator

from flask_cors import CORS,cross_origin


app = Flask(__name__)
CORS(app, support_credentials=True)


# load the models
RFC_Model = pickle.load(open('./model/RF_model.pkl', 'rb'))
LR_Model = pickle.load(open('./model/LR_model.pkl', 'rb'))
SVM_Model = pickle.load(open('./model/SVM_model.pkl', 'rb
reconstructed_model = keras.models.load_model('./model/keras_model')



#### define functions 

RF_list = []
def predict_Covid_RF(img_file):
    'function to take image and return prediction'
    test_image = cv2.imread(img_file)
    test_image = cv2.cvtColor(test_image, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (224, 224))
    test_img = test_image.flatten().reshape(1, -1)
    

    RFC_pred_prob = RFC_Model.predict_proba(test_img)
    RFC_pred = RFC_Model.predict(test_img)

    RF_list = ['RF_Covid', RFC_pred_prob[0,0], RFC_pred[0]]

    return (RF_list)
    
LR_list = []
def predict_Covid_LR(img_file):
    'function to take image and return prediction'
    test_image = cv2.imread(img_file)
    test_image = cv2.cvtColor(test_image, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (224, 224))
    test_img = test_image.flatten().reshape(1, -1)
    

    LR_pred_prob = LR_Model.predict_proba(test_img)
    LR_pred = LR_Model.predict(test_img)

    LR_list = ['LR_Covid', LR_pred_prob[0,0], LR_pred[0]]

    return (LR_list)

def iload(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (150, 150, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def pneumonia_CNN(img_file): 
    iload(img_file)
    image = iload(img_file)
    pred_pnemonia = (1-reconstructed_model.predict(image))
    if reconstructed_model.predict(image) <0.5: 
        out_pneu= 'normal'
    if reconstructed_model.predict(image) >0.5: 
        out_pneu = 'pneumonia'
    arrayList = ['Pneumonia_CNN', pred_pnemonia[0,0], out_pneu]
    return(arrayList)
    
SVM_list = []
def predict_Covid_SVM(img_file):
    'function to take image and return prediction'
    test_image = cv2.imread(img_file)
    test_image = cv2.cvtColor(test_image, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (224, 224))
    test_img = test_image.flatten().reshape(1, -1)
    

    SVM_pred_prob = 'Na'
    SVM_pred = SVM_Model.predict(test_img)

    SVM_list = ['SVM_Covid', SVM_pred_prob, SVM_pred[0]]

    return (SVM_list)

def predict_single(img_file):
    heading = ['model', 'probability', 'prediction']
    covid_RF = predict_Covid_RF(img_file)
    covid_LR = predict_Covid_LR(img_file)
    covid_SVM = predict_Covid_RF(img_file)
    pneu = pneumonia_CNN(img_file)
    out = np.vstack((heading, covid_RF, covid_LR, covid_SVM, pneu))
    return (out)


# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))

def api_response():
    from flask import jsonify
    if request.method == 'POST':
        return jsonify(**request.json)

if __name__ == '__main__':
    app.debug = True
    app.run()
