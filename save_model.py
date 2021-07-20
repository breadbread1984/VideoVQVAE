#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
from absl import flags, app;
import tensorflow as tf;
from models import VideoVQVAE_Trainer,VideoVQVAE

FLAGS = flags.FLAGS;
flags.DEFINE_boolean('use_2d', default = False, help = 'whether to use 2d to replace 3d conv');

def main(unused_argv):
  if not exists('models'): mkdir('models');
  trainer = VideoVQVAE_Trainer(use_2d = FLAGS.use_2d);
  trainer.load_weights('./checkpoints/ckpt');
  trainer.save(join('models', 'trainer.h5'));
  trainer.layers[1].save(join('models', 'encoder.h5'));
  trainer.layers[2].save_weights(join('models', 'pre_vq_conv.h5'));
  trainer.layers[3].save(join('models', 'codebook.h5'));
  trainer.layers[4].save_weights(join('models', 'post_vq_conv.h5'));
  trainer.layers[5].save(join('models', 'decoder.h5'));
  video_vqvae = VideoVQVAE(use_2d = FLAGS.use_2d);
  video_vqvae.encoder = tf.keras.models.load_model(join('models', 'encoder.h5'));
  video_vqvae.decoder = tf.keras.models.load_model(join('models', 'decoder.h5'));
  video_vqvae.pre_vq_conv.load_weights(join('models', 'pre_vq_conv.h5'));
  video_vqvae.post_vq_conv.load_weights(join('models', 'post_vq_conv.h5'));
  video_vqvae.codebook = tf.keras.models.load_model(join('models', 'codebook.h5'));
  video_vqvae.save_weights(join('models', 'video_vqvae_weights.h5'));

if __name__ == "__main__":

  app.run(main);  
