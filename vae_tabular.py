import tensorflow as tf
import keras
from keras import layers
from tensorflow import keras

data_modes = ['a', 'b', 'ca', 'cb', 'ca_a', 'cb_b', 'a_blend', 'b_blend']


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs): # **kwargs?
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, num_features_1, num_features_2, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.num_features_1 = num_features_1
        self.num_features_2 = num_features_2
        self.latent_dim = 32
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def get_encoder(self):
        input_layer = keras.Input(shape=self.num_features_1)
        x = layers.Dense(64, activation='relu')(input_layer)
        x = layers.Dense(32, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(input_layer, [z_mean, z_log_var, z], name="encoder")

    def get_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(32, activation="relu")(latent_inputs)
        x = layers.Dense(64, activation="relu")(x)
        print(self.num_features_2)
        output_layer = layers.Dense(self.num_features_2, activation="relu")(x)
        return keras.Model(latent_inputs, output_layer, name="decoder")

    def train_step(self, data):
        encode_data = data[0]
        decode_data = data[1]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(encode_data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(decode_data, reconstruction), axis=0
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
