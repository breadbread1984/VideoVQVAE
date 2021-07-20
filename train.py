#!/usr/bin/python3

from os.path import exists, join;
from absl import app, flags;
import tensorflow as tf;
from models import VideoVQVAE_Trainer;
from create_dataset import load_ucf101;

FLAGS = flags.FLAGS;
flags.DEFINE_boolean('use_2d', default = False, help = 'whether to use 2d to replace 3d conv');
flags.DEFINE_integer('batch_size', default = 32, help = 'batch size');
flags.DEFINE_integer('length', default = 16, help = 'video length');

def main(unused_argv):
  
  trainer = VideoVQVAE_Trainer(use_2d = FLAGS.use_2d);
  if exists('./checkpoints/ckpt'): trainer.load_weights('./checkpoints/ckpt');
  optimizer = tf.keras.optimizers.Adam(3e-4);
  trainer.compile(optimizer = optimizer,
                  loss = {'model_88': lambda labels, outputs: tf.keras.losses.MeanSquaredError()(labels, outputs),
                          'code_book': lambda dummy, outputs: outputs},
                  loss_weights = {'model_88': 16.67, 'code_book': 1});
  class SummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, eval_freq = 100):
      self.eval_freq = eval_freq;
      testset = load_ucf101('testset.tfrecord', FLAGS.length).repeat(-1).batch(1);
      self.iter = iter(testset);
      self.recon_loss = tf.keras.metrics.Mean(name = 'recon_loss', dtype = tf.float32);
      self.quant_loss = tf.keras.metrics.Mean(name = 'quant_loss', dtype = tf.float32);
      self.log = tf.summary.create_file_writer('./checkpoints');
    def on_batch_begin(self, batch, logs = None):
      pass;
    def on_batch_end(self, batch, logs = None):
      image, label_dict = next(self.iter);
      recon, quant_loss = trainer(image); # recon.shape = (1, 16, 64, 64, 3)
      recon_loss = tf.keras.losses.MeanSquaredError()(image, recon);
      self.recon_loss.update_state(recon_loss);
      self.quant_loss.update_state(quant_loss);
      if batch % self.eval_freq == 0:
        recon = tf.cast((recon + 0.5) * 255., dtype = tf.uint8);
        rows = list();
        for i in range(4):
          row = tf.concat([recon[:,i * 4 + c,...] for c in range(4)], axis = 2); # row.shape = (1, 64, 64 * 4, 3)
          rows.append(row);
        total = tf.concat(rows, axis = 1); # total.shape = (1, 64 * 4, 64 * 4, 3)
        with self.log.as_default():
          tf.summary.scalar('reconstruction loss', self.recon_loss.result(), step = optimizer.iterations);
          tf.summary.scalar('quantize loss', self.quant_loss.result(), step = optimizer.iterations);
          tf.summary.image('reconstructed video', total, step = optimizer.iterations);
        self.recon_loss.reset_states();
        self.quant_loss.reset_states();
    def on_epoch_begin(self, epoch, logs = None):
      pass;
    def on_epoch_end(self, epoch, logs = None):
      pass;

  # load ucf101 dataset
  trainset = load_ucf101('trainset.tfrecord', FLAGS.length).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = load_ucf101('testset.tfrecord', FLAGS.length).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = './checkpoints'),
    tf.keras.callbacks.ModelCheckpoint(filepath = './checkpoints/ckpt', save_freq = 10000),
    SummaryCallback()
  ];
  trainer.fit(trainset, epochs = 560, validation_data = testset, callbacks = callbacks);
  trainer.save('trainer.h5');

if __name__ == "__main__":

  app.run(main);
