from os import makedirs
from os.path import join, isfile
import keras
import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping
from tensorflow import keras

data_modes = ['a', 'b', 'ca', 'cb', 'ca_a', 'cb_b', 'a_blend', 'b_blend']


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, x_train_1, x_train_2=None, data_mode_1=None, data_mode_2=None, dataset=None, fold=None,
                 rotation_angle=None, common_cols=None, shape=(28, 28, 1), epochs=250, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.x_dim = shape[0]
        self.y_dim = shape[1]
        self.channels = shape[2]
        self.common_cols = common_cols
        self.latent_dim = 32
        if rotation_angle is not None:
            self.rotation_angle = str(rotation_angle) + 'deg_'
        else:
            self.rotation_angle = ''
        self.epochs = epochs

        if dataset is None:
            assert 'Dataset must be set'
        else:
            self.dataset = dataset

        if data_mode_1 not in data_modes:
            assert f'ERROR: Data mode 1 {self.data_mode_1} is not valid'
        self.data_mode_1 = data_mode_1
        if data_mode_2 is None:
            self.data_mode_2 = self.data_mode_1
        else:
            self.data_mode_2 = data_mode_2

        self.x_train_1 = x_train_1
        if x_train_2 is None:
            self.x_train_2 = self.x_train_1
        else:
            self.x_train_2 = x_train_2

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        if fold is None:
            path = self.dataset + '_' + self.rotation_angle + str(self.common_cols)
        else:
            path = join(self.dataset + '_' + self.rotation_angle + str(self.common_cols), str(fold))

        if 'blend' in self.data_mode_1:
            encoder_path = join(path, self.data_mode_1 + '_vae_encoder.h5')
            decoder_path = join(path, self.data_mode_1 + '_vae_decoder.h5')
        else:
            encoder_path = join(path, self.data_mode_1 + '_' + self.data_mode_2 + '_vae_encoder.h5')
            decoder_path = join(path, self.data_mode_1 + '_' + self.data_mode_2 + '_vae_decoder.h5')

        if isfile(encoder_path) and isfile(decoder_path):
            self.encoder.load_weights(encoder_path)
            self.decoder.load_weights(decoder_path)
        else:
            self.compile(optimizer=keras.optimizers.Adam())
            self.fit(self.x_train_1, self.x_train_2, epochs=self.epochs, batch_size=128, shuffle=True,
                     callbacks=[EarlyStopping(monitor='reconstruction_loss', patience=50)])
            makedirs(path, exist_ok=True)
            self.encoder.save_weights(encoder_path)
            self.decoder.save_weights(decoder_path)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def get_encoder(self):
        encoder_inputs = keras.Input(shape=(self.x_dim, self.y_dim, self.channels))
        x = layers.Conv2D(8, 3, activation="relu", strides=1, padding="same")(encoder_inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, 3, activation="relu", strides=1, padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def get_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(64, activation="relu")(latent_inputs)
        x = layers.Dense(4 * 4 * 32, activation="relu")(x)
        x = layers.Reshape((4, 4, 32))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        if self.channels == 1:
            x = layers.Conv2D(8, (3, 3), activation='relu', padding='valid')(x)
        elif self.channels == 3:
            x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        else:
            assert f'Invalid channel count {self.channels}'

        x = layers.UpSampling2D((2, 2))(x)
        decoder_outputs = layers.Conv2D(self.channels, (3, 3), activation='sigmoid', padding='same', name='decoder')(x)
        return keras.Model(latent_inputs, decoder_outputs, name="decoder")

    def train_step(self, data):
        encode_data = data[0]
        decode_data = data[1]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(encode_data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(decode_data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
