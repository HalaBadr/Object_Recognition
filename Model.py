import sys;

sys.path.insert(1, '../image')

# TODO: resolve the dependeny on reader (when the model is done)
from reader import ObjectsNames
import cv2 as cv
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import os
import pickle
import tensorflow as tf
import keras
from random import randint
from keras.layers import Input, Conv3D, MaxPooling2D, Flatten, Dense, BatchNormalization, concatenate, Dropout, \
    MaxPooling3D, Conv2D, regularizers
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import MaxPooling2D


class ObjectsModel(object):

    def __init__(self, classes=100, img_dim=32):
        # Load the train and test data
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        print('x_train shape:', x_train.shape)
        print('train: ', x_train.shape[0])
        print('test : ', x_test.shape[0] )
        # Convert integer to binary
        self.y_train = keras.utils.to_categorical(y_train, classes)
        self.y_test = keras.utils.to_categorical(y_test, classes)
        # Make pixel_value as float
        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        # Divide on 255 to make normalize to pixel_values
        self.x_train /= 255.
        self.x_test /= 255.

        self.model_path = 'E:\Hala\MyComputer\BOOKS\programming\Python\Object_Recognition\saved-models\Objects_model.f5'
        self.weight_path = 'E:\Hala\MyComputer\BOOKS\programming\Python\Object_Recognition\saved-weights\Objects_weight.h5'

        self.batch_size = 128
        self.epochs = 1
        self.weight_decay = 1e-6
        self.image_shape = [img_dim, img_dim, 3]

        if  self.is_saved():
            self.model = tf.keras.models.load_model(self.model_path)
            self.trained = True
        else:
            self.model = self.__create_model__(classes)
            self.trained = False

    def add_layer(self,model, num):
        model.add(Conv2D(num, (3 , 3), padding='same'))#, kernel_regularizer=regularizers.l2(self.weight_decay)
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        return model

    def fit(self):
        train_fit=self.model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=1,
                  validation_data=(self.x_test, self.y_test),
                  shuffle=False)
        self.trained = True
        self.model.save(self.model_path)
        self.model.save_weights(self.weight_path)

        return train_fit

    def test(self):
        if not self.trained: raise Exception("Model not trained yet")
        # Load label names
        label_list_path = 'datasets/cifar-100-python/meta'
        keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
        datadir_base = os.path.expanduser(keras_dir)
        if not os.access(datadir_base, os.W_OK):
            datadir_base = os.path.join('/tmp', '.keras')
        label_list_path = os.path.join(datadir_base, label_list_path)
        with open(label_list_path, mode='rb') as f:
            labels = pickle.load(f)

        '''
        evaluation = self.model.predict(self.x_test)

        # How many predictions are wrong
        is_not_correct = (np.argmax(evaluation, 1) != np.argmax(self.y_test, 1))
        print(is_not_correct)

        loss = sum(is_not_correct) / len(is_not_correct)
        print("Loss in prediction: ", loss)'''
        # Evaluate model
        evaluation = self.model.evaluate(self.x_test, self.y_test)
        print('Model Accuracy = %.2f' % (evaluation[1]))
        predict_gen = self.model.predict(self.x_test)
        print(self.y_test.size)
        i = 0
        for predict_index, predicted_y in enumerate(predict_gen):
            actual_label = labels['fine_label_names'][np.argmax(self.y_test[predict_index])]
            predicted_label = labels['fine_label_names'][np.argmax(predicted_y)]

            if(np.argmax(self.y_test[predict_index])==np.argmax(predicted_y)):
              i+=1
              print('Actual Label = %s vs. Predicted Label = %s' % (actual_label,
                                                                  predicted_label))
            if predict_index == 9999:
                print(i)
                break

        return evaluation

    def predict(self, objects):
        if not self.trained: raise Exception("Model not trained yet")
        # Load label names to use in prediction results
        label_list_path = 'datasets/cifar-100-python/meta'

        keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
        datadir_base = os.path.expanduser(keras_dir)
        if not os.access(datadir_base, os.W_OK):
            datadir_base = os.path.join('/tmp', '.keras')
        label_list_path = os.path.join(datadir_base, label_list_path)

        with open(label_list_path, mode='rb') as f:
            labels = pickle.load(f)
        print(labels)
        allObjects=[]

        # Look at the first 9 images from the dataset
        plt.figure(figsize=(10, 10))
        images = range(0, 5)
        for i in images:
            pre = self.model.predict(np.reshape(self.x_train[i],(1,32,32,3)))
            plt.subplot(550 + 1 + i)
            plt.imshow(labels['fine_label_names'][np.argmax(self.y_test[i])],self.x_test[i])

         #Show the plot
        plt.show()
        for i in range (len(objects)) :
             object = objects[i][0]
             object=np.resize(object,(32,32,3))
             allObjects.append(object)
        allObjects=np.array(allObjects)
        allObjects = allObjects.astype('float32')
        allObjects /= 255.

        #pre = self.model.predict(allObjects)
        pre = pre.tolist()
        print(pre)
        for i, predicted_y in enumerate(pre):
            pre[i] = labels['fine_label_names'][np.argmax(predicted_y)]
            print(pre[i])

        return pre

    def is_saved(self):
        model_path = Path(self.model_path)
        return model_path.is_file()

    def __create_model__(self):

        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.image_shape))#, kernel_regularizer=regularizers.l2(self.weight_decay)
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model = self.add_layer(model, 64 )

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model = self.add_layer(model, 128)

        model = self.add_layer(model, 128)

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model = self.add_layer(model, 256)

        model = self.add_layer(model, 256)

        model = self.add_layer(model, 256)

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model = self.add_layer(model, 512)

        model = self.add_layer(model, 512)

        model = self.add_layer(model, 512)

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model = self.add_layer(model, 512)

        model = self.add_layer(model, 512)

        model = self.add_layer(model, 512)

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(100))
        model.add(Activation('softmax'))

        model.load_weights('cifar1001.h5')

        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        return model