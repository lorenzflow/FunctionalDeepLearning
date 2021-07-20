# get_VAE.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

def get_VAE(input_shape, latent_size, encoder_dims=[80,60,40], decoder_dims=[40,60,80], prior, loss, optimizer):

  encoder = Sequential([
      Dense(80, activation='relu', input_shape=input_shape), # adjust shape #
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
      
  FVAE_concat = Model(encoder.inputs, decoder(encoder.outputs))
  FVAE_concat.compile(optimizer, loss=loss)
  # compile model
  return FVAE_concat