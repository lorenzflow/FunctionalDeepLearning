# Code for FFTVAE on GP prior samples

# Build the encoder with KLAddLossLayer
import GP_sample
import get_VAE
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

# generate data sets for training and testing
n_points = 100
x_upper = 4
X,Y_train = sample_batch(10**4, n_points, GP, 0, x_upper, x=1)
X,Y_val = sample_batch(10**3, n_points, GP, 0, x_upper, x=1)
# apply fourier transform to data
Y_train = np.transpose(np.squeeze(Y_train))
fourier_train = np.fft.fft(Y_train)

# above changes to data
fourier_train = fourier_train[:,:(n_points//2)]
real_train = fourier_train.real
imag_train = fourier_train.imag

full_train = np.concatenate([real_train,imag_train], axis=1)

fourier_val = fourier_train[:,:(n_points//2)]
real_val = fourier_val.real
imag_val = fourier_val.imag

full_val = np.concatenate([real_val,imag_val], axis=1)

# define configuration
latent_size=10
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_size), scale=1), reinterpreted_batch_ndims=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.MeanSquaredError()
input_shape=(100,)

# get compiled model
FVAE_concat = get_VAE(input_shape, latent_size, prior=prior, loss=loss, optimizer=optimizer)

# callback
early_stopping = EarlyStopping(patience=10)

# train
history = FVAE_concat.fit(full_train, full_train, validation_data=(full_val, full_val), epochs=400, batch_size=100, callbacks=[early_stopping])
