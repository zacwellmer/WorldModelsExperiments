'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
from vae.vae import ConvVAE, reset_graph

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#config = ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 1.0
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)


DATA_DIR = "record"
SERIES_DIR = "series"
model_path_name = "tf_vae"

if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)
Z_SIZE = 32
def ds_gen():
    dirname = 'record'
    filenames = os.listdir(dirname)[:10000] # only use first 10k episodes
    n = len(filenames)
    for j, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(dirname, fname)
        with np.load(file_path) as data:
            N = data['obs'].shape[0]
            for i, img in enumerate(data['obs']):
                action = data['action'][i]
                img_i = img / 255.0
                zerod_outputs = np.zeros([2*Z_SIZE])
                yield img_i, action 

def create_tf_dataset():
    dataset = tf.data.Dataset.from_generator(ds_gen, output_types=(tf.float32, tf.float32), output_shapes=((64, 64, 3), (3)))
    return dataset

def encode_batch(batch_img):
  simple_obs = np.copy(batch_img).astype(np.float)/255.0
  simple_obs = simple_obs.reshape(batch_size, 64, 64, 3)
  mu, logvar = vae.encode_mu_logvar(simple_obs)
  z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))
  return mu, logvar, z

def decode_batch(batch_z):
  # decode the latent vector
  batch_img = vae.decode(z.reshape(batch_size, z_size)) * 255.
  batch_img = np.round(batch_img).astype(np.uint8)
  batch_img = batch_img.reshape(batch_size, 64, 64, 3)
  return batch_img


# Hyperparameters for ConvVAE
z_size=32
batch_size=1000 # treat every episode as a batch of 1000!
learning_rate=0.0001
kl_tolerance=0.5

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]
#dataset, action_dataset = load_raw_data_list(filelist)
dataset = create_tf_dataset()
dataset = dataset.batch(batch_size)

reset_graph()
vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=False,
              reuse=False,
              gpu_mode=True) # use GPU on batchsize of 1000 -> much faster

vae.load_json(os.path.join(model_path_name, 'vae.json'))
mu_dataset = []
logvar_dataset = []
action_dataset = []
i=0
for obs_batch, action_batch in dataset:
  i += 1
  mu, logvar, _ = encode_batch(obs_batch)

  mu_dataset.append(mu.astype(np.float16))
  logvar_dataset.append(logvar.astype(np.float16))
  action_dataset.append(action_batch.numpy())
  if ((i+1) % 100 == 0):
    print(i+1)

action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset)
