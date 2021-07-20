# Code for FFTVAE on GP prior samples

# Build the encoder with KLAddLossLayer
import GP_sample.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers




#from tensorflow.keras.layers.experimental import RandomFourierFeatures
# assuming beta_i has 1x20 dims
latent_size=10
input_shape=(100,)

# Define the prior
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_size), scale=1), reinterpreted_batch_ndims=1)

encoder = Sequential([
    Dense(80, activation='relu', input_shape=input_shape), # adjust shape
    Dense(60, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(40, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(tfpl.IndependentNormal.params_size(latent_size)),
    tfpl.IndependentNormal(latent_size),
    tfpl.KLDivergenceAddLoss(prior, use_exact_kl=False, weight=1, test_points_fn= lambda q: q.sample(10), 
                             test_points_reduce_axis=None) # could possibly use exact KL divergence since Normal
          ])

decoder = Sequential([
    Dense(40, activation='relu', input_shape=(latent_size,)),
    Dense(60, activation='relu'),
    Dense(80, activation='relu'),
    Dense(tfpl.IndependentNormal.params_size(input_shape)),
    tfpl.IndependentNormal(input_shape)
          ])


# combine encoder and decoder in model
FVAE = Model(encoder.inputs, decoder(encoder.outputs))

# generate data sets
X,Y_train = sample_batch(10**4, 100, GP, 0, 4, x=1)
X,Y_val = sample_batch(10**3, 100, GP, 0, 4, x=1)
# apply fourier transform to data
Y_train = np.transpose(np.squeeze(Y_train))
fourier_train = np.fft.fft(Y_train)

# apply fourier transform to data
Y_val = np.transpose(np.squeeze(Y_val))
fourier_val = np.fft.fft(Y_val)