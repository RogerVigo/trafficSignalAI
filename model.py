import pandas as pd
import numpy as np
from datetime import datetime as dt
import os
import cv2 as cv
from PIL import Image
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv2D, AveragePooling2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential


class TrainingModel:
    def __init__(self, save=False):
        self.save = save
        self.model_save_path = "./saved_models/"
        try:
            os.mkdir(self.model_save_path)
        except:
            print("Saved model folder already exists")

        self.batch_size = 100
        self.epochs = 10

        self.images_path = "./data/myData/"

        self.images, self.labels = self.load_images(self.images_path)

        self.label_names = pd.read_csv("./data/labels.csv")
        # print(len(self.train_images[]))

        # cv.imshow("test", self.train['features'][3000])
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    def load_images(self, path):
        images_p = "./data/images.p"
        labels_p = "./data/labels.p"

        if os.path.isfile(images_p) and os.path.isfile(labels_p):
            images = pkl.load(open(images_p, "rb"))
            labels = pkl.load(open(labels_p, "rb"))

            return images, labels

        images = {"train": [], "test": [], "valid": []}
        labels = {"train": [], "test": [], "valid": []}

        # r=root, d=directories, f = files
        target = np.sort(np.array(os.listdir(path), dtype="int16"))

        for i in target:
            images_path = path + "/" + str(i) + "/"
            dir_len = len(os.listdir(images_path))
            print("STARTING FOLDER " + str(i))

            for x, img in enumerate(os.listdir(images_path)):
                # print(str(x) + " OF " + str(dir_len))
                loaded = cv.imread(images_path + img)

                if x <= dir_len * 0.7:
                    images["train"].append(loaded)
                    labels["train"].append(i)

                elif x > dir_len * 0.7 and x <= dir_len * 0.9:
                    images["test"].append(loaded)
                    labels["test"].append(i)
                else:
                    images["valid"].append(loaded)
                    labels["valid"].append(i)

        images["train"] = np.array(images["train"])
        images["test"] = np.array(images["test"])
        images["valid"] = np.array(images["valid"])

        labels["train"] = np.array(labels["train"])
        labels["test"] = np.array(labels["test"])
        labels["valid"] = np.array(labels["valid"])

        pkl.dump(images, open(images_p, "wb"))
        pkl.dump(labels, open(labels_p, "wb"))

        return images, labels

    def createModel(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32,3)))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))

        output_size = 43
        model.add(Dense(output_size, activation="softmax"))

        return model

    def loadModel(self, compile=True):
        files = os.listdir(self.model_save_path)
        if len(files) == 0:
            return False
        return False
        print("Loading model!")
        return tf.keras.models.load_model(self.model_save_path + files[len(files) - 1], compile=compile)

    def evaluate(self, model, img_valid, lab_valid):
        loss, acc = model.evaluate(img_valid, lab_valid, verbose=2)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

        print(np.amax(model.predict(img_valid) * 100))

    def test(self):
        original_image = cv.imread("/home/roger/Imágenes/signals/speed.png")
        image = cv.resize(original_image, (32, 32)) / 255
        print(np.shape(image))
        self.model.predict(image)

    def run(self):
        x_train = self.images["train"] / 255
        y_train = self.labels["train"]

        x_test = self.images["test"] / 255
        y_test = self.labels["test"]

        x_valid = self.images["valid"] / 255
        y_valid = self.labels["valid"]

        model = self.loadModel()

        if not bool(model):
            model = self.createModel()
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            result = model.fit(x_train, y_train, epochs=self.epochs,
                               validation_data=(x_test, y_test),
                               batch_size=self.batch_size)

            path = self.model_save_path + str(dt.now())
            if self.save:
                model.save(path + "/myModel.h5")

        self.evaluate(model, x_valid, y_valid)
        self.model = model


class ProductionModel():
    def __init__(self):
        self.model_save_path = "./saved_models/"
        self.model = self.loadModel(compile=False)
        self.threshold = 0.85
        self.draw = np.empty((0, 2), dtype='uint32')
        self.label_names = pd.read_csv("./data/labels.csv")

        if not self.model:
            raise Exception("ModelNotLoaded")

    def loadModel(self, compile=True):
        files = os.listdir(self.model_save_path)
        if len(files) == 0:
            return False

        print("Loading model!")
        return tf.keras.models.load_model(self.model_save_path + files[len(files) - 1] + "/myModel.h5", compile=compile)

    def show_image(self, image):
        cv.imshow("image", image)
        cv.waitKey(0)
        # cv.destroyAllWindows()

    def draw_matches(self, image, point, kernel):
        image = cv.rectangle(image, point, (point[0] + kernel, point[1] + kernel), (255, 0, 0), 3)

        return image

    def runImage(self):
        original_image = cv.imread("/home/roger/Imágenes/signals/speed.png")
        image = cv.resize(original_image, (32,32))
        print(np.shape(image))
        result = self.model.predict(np.reshape(image, (1,32,32,3)))
        print(np.amax(result))
        print(result * 100)

        cv.destroyAllWindows()

    def predict(self, image):
        shape = np.shape(image)
        print(shape)
        kernel = 32
        x = 0
        y = 0

        while y + kernel <= shape[0]:
            while x + kernel <= shape[1]:
                final = image[y:y + kernel, x:x + kernel, :]
                if final.shape != (32,32,3): continue
                result = self.model.predict(np.reshape(final, (1, 32, 32, 3)))
                #self.show_image(final)
                if np.amax(result) >= self.threshold:
                    image = self.draw_matches(image, (y,x), kernel)
                    #self.show_image(image)
                    label = np.where(result == np.amax(result))
                    #print(self.label_names["Name"][label[1][0]])
                y += kernel // 2

            x += kernel // 2
            y = 0

        self.show_image(image)

    def runCamera(self):
        video = cv.VideoCapture(0)

        while True:
            ret, image = video.read()
            self.predict(image)

        cv.destroyAllWindows()
