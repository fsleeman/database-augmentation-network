from os import makedirs
from os.path import isfile, join, exists
import keras
from keras import layers
from keras.callbacks import EarlyStopping


class Classifier:
    def __init__(self, X_train, y_train, X_val=None, y_val=None, input_shape=(28, 28, 1), num_classes=None,
                 dataset=None, common_cols=None, rotation_angle=None, fold=None, name=None, epochs=250):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.fold = fold
        self.dataset = dataset
        self.common_cols = common_cols
        if rotation_angle is not None:
            self.rotation_angle = str(rotation_angle) + 'deg_'
        else:
            self.rotation_angle = ''
        self.epochs = epochs
        self.monitor_loss = None
        if self.X_val is None or self.y_val is None:
            self.validation_data = None
            self.monitor_loss = 'loss'
        else:
            self.validation_data = (self.X_val, self.y_val)
            self.monitor_loss = 'val_loss'
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.name = name
        self.history = None
        self.model = self.get_model_architecture()
        self.get_classifier()

    def get_classifier(self):

        if self.fold is None:
            folder = self.dataset + '_' + self.rotation_angle + str(self.common_cols)
        else:
            folder = join(self.dataset + '_' + self.rotation_angle + str(self.common_cols), str(self.fold))

        if not exists(folder):
            makedirs(folder, exist_ok=True)
        full_path = join(folder, self.name + '_cls.h5')

        if isfile(full_path):
            self.model.load_weights(full_path)
        else:
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            self.history = self.model.fit(self.X_train, self.y_train,
                                          epochs=self.epochs,
                                          batch_size=128,
                                          shuffle=True,
                                          validation_data=self.validation_data,
                                          callbacks=[EarlyStopping(monitor=self.monitor_loss, patience=50)])
            self.model.save_weights(full_path)

    def get_model_architecture(self):
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, 3, activation="relu")(inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64)(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return keras.Model(inputs, outputs, name="classifier")
