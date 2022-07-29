import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
def preprocessing(file):
    file_names = os.listdir(file)
    label = []
    for i in (file_names):
        if "dog" in i:
            label.append("dog")
        else:
            label.append("cat")
    Data = []

    for i in (range(25000)):
        full_path = os.path.join("train", file_names[(i)])
        Data.append(full_path)
    final_data = pd.DataFrame(list(zip(Data, label)), columns = ["image_path", "labels"])
    train_data, test_data = tts(final_data, test_size=0.2)
    return(train_data,test_data)

def predictirl(imagename):
    Model = tf.keras.models.load_model('CatorDog1.h5')

    my_image = load_img(imagename, target_size=(150, 150))

    #preprocess the image
    my_image = img_to_array(my_image)
    my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
    my_image = preprocess_input(my_image)

    #make the prediction
    prediction = Model.predict(my_image)
    return(np.round(prediction))
