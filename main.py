import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
import preprocessing as x

(train_data,test_data)=x.preprocessing('train')

train_datagen=ImageDataGenerator(
rotation_range=15,
rescale=1./255,
shear_range=0.1,
zoom_range=0.2,
horizontal_flip=True,
width_shift_range=0.1,
height_shift_range=0.1)

train_datagenerator=train_datagen.flow_from_dataframe(dataframe=train_data,
                                                     x_col="image_path",
                                                     y_col="labels",
                                                     target_size=(150, 150),
                                                     class_mode="binary",
                                                     batch_size=64)
test_datagen = ImageDataGenerator(rotation_range=15,
rescale=1./255,
shear_range=0.1,
zoom_range=0.2,
horizontal_flip=True,
width_shift_range=0.1,
height_shift_range=0.1)
test_datagenerator=test_datagen.flow_from_dataframe(dataframe=test_data,
                                                   x_col="image_path",
                                                   y_col="labels",
                                                   target_size=(150, 150),
                                                   class_mode="binary",
                                                   batch_size=64)
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(train_datagenerator, epochs=50)
predictions = model.predict(x=test_datagenerator)


v = np.round(predictions)
print(classification_report(v,test_datagenerator.classes))
model.save("CatorDog.h5")
