#!/usr/bin/python3

from os.path import join;
from absl import app, flags;
import cv2;
import tensorflow as tf;
from models import CodeBook;
from create_dataset import load_ucf101;

FLAGS = flags.FLAGS;
flags.DEFINE_string('testset', 'testset.tfrecord', help = "testset");
flags.DEFINE_integer('length', 16, help = "video length");
flags.DEFINE_boolean('use_2d', False, 'whether to use 2d conv to replace 3d conv')

def main(unused_argv):
  testset = load_ucf101(FLAGS.testset, FLAGS.length).repeat(-1).batch(1);
  encoder = tf.keras.models.load_model(join('models', 'encoder_use2d.h5' if FLAGS.use_2d else 'encoder.h5'), custom_objects = {'CodeBook': CodeBook});
  decoder = tf.keras.models.load_model(join('models', 'decoder_use2d.h5' if FLAGS.use_2d else 'decoder.h5'));
  for clip, label_dict in testset:
    quantized, tokens, quant_loss = encoder(clip);
    recon = decoder(quantized);
    recon = tf.cast((recon + 0.5) * 255., dtype = tf.uint8)[0]; # recon.shape = (length, h, w, c)
    original = tf.cast((clip + 0.5) * 255., dtype = tf.uint8)[0]; # original.shape = (length, h, w, c)
    stacked = tf.concat([original, recon], axis = 2).numpy(); # stacked.shape = (length, h, 2 * w, c)
    for frame in stacked:
      cv2.imshow('clip', frame);
      cv2.waitKey(25);

if __name__ == "__main__":

  app.run(main);
