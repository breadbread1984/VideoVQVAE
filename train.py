#!/usr/bin/python3

from os.path import exists, join;
from math import ceil;
from absl import app, flags;
import tensorflow as tf;
from models import VideoVQVAE_Trainer, CodeBook;
from create_dataset import load_ucf101;

FLAGS = flags.FLAGS;
flags.DEFINE_boolean('use_2d', default = False, help = 'whether to use 2d to replace 3d conv');
flags.DEFINE_integer('batch_size', default = 32, help = 'batch size');
flags.DEFINE_integer('length', default = 16, help = 'video length');

TRAINSET_SIZE = 9537;
TESTSET_SIZE = 3783;

def recon_loss(labels, outputs):
  return tf.keras.losses.MeanSquaredError()(labels, outputs);

def quant_loss(_, outputs):
  return outputs;

def main(unused_argv):
  
  if exists('./checkpoints/ckpt'):
    trainer = tf.keras.models.load_model('./checkpoints/ckpt',
                                         custom_objects = {'tf': tf,
                                                           'CodeBook': CodeBook,
                                                           'recon_loss': recon_loss,
                                                           'quant_loss': quant_loss},
                                         compile = True);
    optimizer = trainer.optimizer;
  else:
    trainer = VideoVQVAE_Trainer(use_2d = FLAGS.use_2d);
    optimizer = tf.keras.optimizers.Adam(3e-4);
    trainer.compile(optimizer = optimizer,
                    loss = {'model_88': recon_loss,
                            'code_book': quant_loss},
                    loss_weights = {'model_88': 16.67, 'code_book': 1});

  # load ucf101 dataset
  trainset = load_ucf101('trainset.tfrecord', FLAGS.length).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = load_ucf101('testset.tfrecord', FLAGS.length).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = './checkpoints'),
    tf.keras.callbacks.ModelCheckpoint(filepath = './checkpoints/ckpt', save_freq = 10000)
  ];
  trainer.fit(trainset, steps_per_epoch = ceil(TRAINSET_SIZE / FLAGS.batch_size), epochs = 560, validation_data = testset, validation_steps = ceil(TESETSET_SIZE / FLAGS.batch_size), callbacks = callbacks);
  trainer.save('trainer.h5');

if __name__ == "__main__":

  app.run(main);
