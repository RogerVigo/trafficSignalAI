import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image
import pickle as pkl
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential


class TrainingModel:
    def __init__(self):
        self.batch_size = 100
        self.epochs = 10

        self.train = pkl.load(open("data/train.pickle", "rb"))
        self.test = pkl.load(open("data/test.pickle", "rb"))
        self.valid = pkl.load(open("data/valid.pickle", "rb"))

        self.label_names = pd.read_csv("data/label_names.csv")
        self.labels = pkl.load(open("data/labels.pickle", "rb"))

        #print(self.train.keys())
        #print(np.size(self.train['features'][0]))

        #cv.imshow("test", self.train['features'][3000])
        #cv.waitKey(0)
        #cv.destroyAllWindows()

    def run(self):
        x_train = self.train['features']
        y_train = self.train['labels']

        x_test = self.test['features']
        y_test = self.test['labels']

        x_valid = self.valid['features']
        y_valid = self.valid['labels']

        model = Sequential()

        model.add(Input(shape=np.shape(x_train)))
        model.add(Dense(256, activation="relu"))
        output_size = np.size(self.labels)
        model.add(Dense(output_size))

        print(model.input_shape)
        print(model.output_shape)

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        result = model.fit(x_train, y_train, epochs=self.epochs,
                           validation_data=(x_test, y_test),
                           batch_size=self.batch_size)