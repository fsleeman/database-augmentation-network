import os
from os.path import isfile, join, exists

import keras
import numpy as np
from keras import layers
from keras.callbacks import EarlyStopping

data_modes = ['a', 'b', 'ca', 'cb', 'ca_a', 'cb_b', 'a_blend', 'b_blend']


# FIXME - is this even used
class Classifier:
    def __init__(self, num_features, num_classes):
        self.classifier_model = None
        self.num_features = num_features
        self.num_classes = num_classes
        self.create_model()

    def get_classifier(self):
        return self.classifier_model

    def create_model(self):
        input_layer = keras.Input(shape=self.num_features)
        x = layers.Dense(self.num_features, activation="relu")(input_layer)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.num_features / 2, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.num_features / 4, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        output_layer = layers.Dense(self.num_classes, activation="softmax")(x)
        self.classifier_model = keras.Model(input_layer, output_layer)


class Autoencoder:
    def __init__(self, x_train_1, y_train_1, x_val_1, y_val_1, x_train_2=None, y_train_2=None,
                 x_val_2=None, y_val_2=None, data_mode_1='', data_mode_2=None, dataset=None, descriptor=None,
                 num_classes=None, num_features=None, fold=None, percent_string=''):
        self.dataset = dataset
        self.descriptor = descriptor
        self.num_classes = num_classes
        self.num_features = num_features
        self.fold = fold
        self.percent_string = percent_string

        self.classifier_history = None
        self.autoencoder_history = None

        if self.dataset is None:
            assert 'Dataset must be set'
        elif self.num_classes is None:
            assert 'num_classes must be set'
        elif self.num_features is None:
            assert 'num_features must be set'

        self.x_train_1 = x_train_1
        self.x_val_1 = x_val_1
        self.y_train_1 = y_train_1  # TODO remove these, not used
        self.y_val_1 = y_val_1

        self.data_mode_1 = data_mode_1
        if data_mode_2 is None:
            self.data_mode_2 = data_mode_1
            self.x_train_2 = self.x_train_1
            self.x_val_2 = self.x_val_1
            self.y_train_2 = self.y_train_1
            self.y_val_2 = self.y_val_1
        else:
            self.data_mode_2 = data_mode_2
            self.x_train_2 = x_train_2
            self.x_val_2 = x_val_2
            self.y_train_2 = y_train_2
            self.y_val_2 = y_val_2

        self.num_features_1 = self.x_train_1.shape[1]
        self.num_features_2 = self.x_train_2.shape[1]

        if self.data_mode_1 not in data_modes:
            assert f'ERROR: Data mode 1 {self.data_mode_1} is not valid'

        if self.fold is None:
            folder = dataset + self.percent_string
        else:
            folder = join(dataset + self.percent_string, str(self.fold))
        if not exists(folder):
            os.makedirs(folder, exist_ok=True)

        if 'blend' in self.data_mode_1:
            self.autoencoder_model_path = join(folder, self.data_mode_1 + '_autoencoder.h5')
            self.classifier_model_path = join(folder, self.data_mode_1 + '_classifier.h5')
        else:
            self.autoencoder_model_path = join(folder, self.data_mode_1 + '_' + self.data_mode_2 + '_autoencoder.h5')
            self.classifier_model_path = join(folder, self.data_mode_1 + '_' + self.data_mode_2 + '_classifier.h5')

        self.input_img = None
        self.autoencoder_model = None
        self.classifier_model = None
        self.encoder_layer = None
        self.decoder_layer = None
        self.input_layer = None
        self.create_autoencoder_model()
        self.create_classifier_model()

    def get_autoencoder_model(self):
        if isfile(self.autoencoder_model_path):
            self.autoencoder_model.load_weights(self.autoencoder_model_path)
        else:
            self.autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')
            self.autoencoder_model.fit(self.x_train_1, self.x_train_2,
                                       epochs=1000,
                                       batch_size=64,
                                       shuffle=True,
                                       validation_data=(self.x_val_1, self.x_val_2),
                                       callbacks=[EarlyStopping(monitor='val_loss', patience=50)])
            self.autoencoder_model.save_weights(self.autoencoder_model_path)
        return self.autoencoder_model

    def get_encoder_layer(self):
        return self.encoder_layer

    def encode_data(self, data, flat=True):
        encoder_network = keras.Model(self.input_layer, outputs=self.encoder_layer)
        encoded_data = encoder_network.predict(data)
        if flat:
            return encoded_data.reshape(encoded_data.shape[0], np.product(encoded_data.shape[1:]))
        else:
            return encoded_data

    def create_classifier_model(self):
        input_shape = int(np.product(self.get_encoder_layer().shape[1:]))
        input_layer = keras.Input(shape=input_shape)

        x = layers.Dense(input_shape * 4, activation="relu")(input_layer)  # (input_layer)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(input_shape * 2, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(input_shape, activation="relu")(x)
        output_layer = layers.Dense(self.num_classes, activation="softmax")(x)
        self.classifier_model = keras.Model(input_layer, output_layer)

    def create_autoencoder_model(self):
        self.input_layer = keras.Input(shape=(self.num_features_1,))
        encoded_h2 = layers.Dense(64, activation='relu')(self.input_layer)
        encoded_h3 = layers.Dense(32, activation='relu')(encoded_h2)
        self.encoder_layer = layers.Dense(32, activation='relu')(encoded_h3)
        decoder_h3 = layers.Dense(32, activation='relu')(self.encoder_layer)
        decoder_h2 = layers.Dense(64, activation='relu')(decoder_h3)
        self.decoder_layer = layers.Dense(self.num_features_2, activation='relu')(decoder_h2)
        self.autoencoder_model = keras.Model(self.input_layer, self.decoder_layer)
