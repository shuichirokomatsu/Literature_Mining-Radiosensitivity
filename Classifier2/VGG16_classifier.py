import os, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2

top_model_weights_path = 'bottleneck_fc_model.h5'

def predict(file):
    # load the class_indices
    class_dictionary = np.load('class_indices.npy').item()
    num_classes = len(class_dictionary)
    image_path = file
    orig = cv2.imread(image_path)
    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    model = applications.VGG16(include_top=False, weights='imagenet')
    bottleneck_prediction = model.predict(image)
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.load_weights(top_model_weights_path)
    class_predicted = model.predict_classes(bottleneck_prediction)
    probabilities = model.predict_proba(bottleneck_prediction)
    inID = class_predicted[0]
    inv_map = {v: k for k, v in class_dictionary.items()}
    label = inv_map[inID]

    # get the prediction label
    print("File_name: {}, Image ID: {}, Label: {}".format(os.path.basename(file), inID, label))

for file in glob.glob(os.path.join("/pictures/", "*.jpeg")):
    predict(file)