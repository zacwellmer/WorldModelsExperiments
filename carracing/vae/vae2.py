# ConvVAE model

import numpy as np
import tensorflow as tf
import os

class CVAE(tf.keras.Model):
    '''
    convolutional variational auto encoder
    '''
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001,
                kl_tolerance=0.5):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.kl_tolerance = kl_tolerance
        super(CVAE, self).__init__()

        self.inference_net_base = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(
              filters=32, kernel_size=4, strides=(2, 2), activation='relu',
              name="enc_conv1"),
            tf.keras.layers.Conv2D(
              filters=64, kernel_size=4, strides=(2, 2), activation='relu',
              name="enc_conv2"),
            tf.keras.layers.Conv2D(
              filters=128, kernel_size=4, strides=(2, 2), activation='relu',
              name="enc_conv3"),
            tf.keras.layers.Conv2D(
              filters=256, kernel_size=4, strides=(2, 2), activation='relu',
              name="enc_conv4"),
            tf.keras.layers.Flatten()])
        
        self.mu_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1024)),
            tf.keras.layers.Dense(self.z_size, name="enc_fc_mu")])
        self.logvar_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1024)),
            tf.keras.layers.Dense(self.z_size, name="enc_fc_log_var")])

        self.generative_net = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.z_size,)),
            tf.keras.layers.Dense(units=4*256, activation=tf.nn.relu, name="dec_dense1"),
            tf.keras.layers.Reshape(target_shape=(1, 1, 4*256)),
            tf.keras.layers.Conv2DTranspose(
              filters=128,
              kernel_size=5,
              strides=(2, 2),
              padding="valid",
              activation='relu',
              name="dec_deconv1"),
            tf.keras.layers.Conv2DTranspose(
              filters=64,
              kernel_size=5,
              strides=(2, 2),
              padding="valid",
              activation='relu',
              name="dec_deconv2"),
            tf.keras.layers.Conv2DTranspose(
              filters=32,
              kernel_size=6,
              strides=(2, 2),
              padding="valid",
              activation='relu',
              name="dec_deconv3"),
            tf.keras.layers.Conv2DTranspose(
              filters=3,
              kernel_size=6,
              strides=(2, 2),
              padding="valid",
              activation="sigmoid",
              name="dec_deconv4")])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.z_size))
        return self.decode(eps)

    def encode(self, x):
        x = self.inference_net_base(x)
        mean = self.mu_net(x)
        logvar = self.logvar_net(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        probs = self.generative_net(z)
        return probs

    def get_loss(self):
        z_size = self.z_size
        kl_tolerance = self.kl_tolerance
        
        def reconstruction_loss_func(y_true, y_pred):
            # reconstruction loss
            reconstruction_loss = tf.reduce_sum(
              input_tensor=tf.square(y_true - y_pred),
              axis = [1,2,3]
            )
            reconstruction_loss = tf.reduce_mean(input_tensor=reconstruction_loss)
            return reconstruction_loss

        def kl_loss_func(_, y_pred): # _ is where y_true goes, but we don't need it for kl loss
            mean, logvar = y_pred[:, :z_size], y_pred[:, z_size:]

            eps = 1e-6 # avoid taking log of zero
            # augmented kl loss per dim
            kl_loss = - 0.5 * tf.reduce_sum(
              input_tensor=(1 + logvar - tf.square(mean) - tf.exp(logvar)),
              axis = 1
            )
            kl_loss = tf.maximum(kl_loss, kl_tolerance * z_size)
            kl_loss = tf.reduce_mean(input_tensor=kl_loss)

            return kl_loss
        return [reconstruction_loss_func, kl_loss_func]

    def call(self, inputs, training=True):
        return self.__call__(inputs, training)


    def __call__(self, inputs, training=True):
        # not sure why keras forces us to use the trainng flag
        mean, logvar = self.encode(inputs)
        z = self.reparameterize(mean, logvar)
        y = self.decode(z)
        mean_and_logvar = tf.concat([mean, logvar], axis=-1)
        return [y, mean_and_logvar]
