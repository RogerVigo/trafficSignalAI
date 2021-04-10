import pandas as pd
import numpy as np
from datetime import datetime as dt
import os
import cv2 as cv
from PIL import Image
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, AveragePooling2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential


class TrainingModel:
    def __init__(self, save=False):
        self.save = save
        self.model_save_path = "./saved_models/"
        try:
            os.mkdir(self.model_save_path)
        except:
            print("Saved model folder already exists")

        self.batch_size = 500
        self.epochs = 10

        self.train = pkl.load(open("data/train.pickle", "rb"))
        self.test = pkl.load(open("data/test.pickle", "rb"))
        self.valid = pkl.load(open("data/valid.pickle", "rb"))

        self.label_names = pd.read_csv("data/label_names.csv")
        self.labels = pkl.load(open("data/labels.pickle", "rb"))

        # print(self.train.keys())
        # print(np.size(self.train['features'][0]))

        # cv.imshow("test", self.train['features'][3000])
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    def createModel(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))

        output_size = np.size(self.labels)
        model.add(Dense(output_size, activation="softmax"))

        return model

    def loadModel(self):
        files = os.listdir(self.model_save_path)
        if len(files) == 0:
            return False

        print("Loading model!")
        return tf.keras.models.load_model(self.model_save_path + files[len(files) - 1])

    def evaluate(self, model, img_valid, lab_valid):
        loss, acc = model.evaluate(img_valid, lab_valid, verbose=2)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

        print(model.predict(img_valid).shape)

    def run(self):
        x_train = self.train['features']
        y_train = self.train['labels']

        x_test = self.test['features']
        y_test = self.test['labels']

        x_valid = self.valid['features']
        y_valid = self.valid['labels']

        model = self.loadModel()
        if not bool(model):
            model = self.createModel()
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            result = model.fit(x_train, y_train, epochs=self.epochs,
                               validation_data=(x_test, y_test),
                               batch_size=self.batch_size)

            path = self.model_save_path + str(dt.now())
            if self.save:
                model.save(path)

        self.evaluate(model, x_valid, y_valid)


class ProductionModel(TrainingModel):
    def __init__(self):
        super(ProductionModel, self).__init__()
        self.model = self.loadModel()
        if not self.model:
            raise Exception("ModelNotLoaded")


    def run(self):
        image = cv.imread("/home/roger/Imágenes/signals/carretera.png", cv.IMREAD_COLOR)
        reshaped = np.reshape(image,(1502,744,3))
        #image = image / 255
        image_shape = np.shape(reshaped)
        print(image_shape)

        print(image[0][0])
        for i in image_shape:
            #print(i)
            pass