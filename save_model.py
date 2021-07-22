#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
from absl import flags, app;
import numpy as np;
import tensorflow as tf;
from models import VideoVQVAE_Trainer

FLAGS = flags.FLAGS;
flags.DEFINE_boolean('use_2d', default = False, help = 'whether to use 2d to replace 3d conv');

def main(unused_argv):
  if not exists('models'): mkdir('models');
  trainer = VideoVQVAE_Trainer(use_2d = FLAGS.use_2d);
  trainer.load_weights('./checkpoints/ckpt');
  trainer.layers[3].set_trainable(False);
  encoder = tf.keras.Model(inputs = trainer.layers[0].input, outputs = trainer.layers[3].output);
  encoder.save(join('models', 'encoder_use2d.h5' if FLAGS.use_2d else 'encoder.h5'));
  quantized = tf.keras.Input(encoder.output[0].shape[1:]);
  results = trainer.layers[4](quantized);
  recon = trainer.layers[5](results);
  decoder = tf.keras.Model(inputs = quantized, outputs = recon);
  decoder.save(join('models', 'decoder_use2d.h5' if FLAGS.use_2d else 'decoder.h5'));

if __name__ == "__main__":

  app.run(main);  
