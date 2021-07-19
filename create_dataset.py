#!/usr/bin/python3

from absl import flags, app;
from os.path import exists, join;
import cv2;
import numpy as np;
import tensorflow as tf;

FLAGS = flags.FLAGS;

flags.DEFINE_string('ucf_root', default = None, help = 'root directory of ucf101 dataset');
flags.DEFINE_string('train_list', default = None, help = 'video list for trainset');
flags.DEFINE_string('test_list', default = None, help = 'video list for testset');

def generate_dataset(video_root, video_list, size = (64,64), output_file):
  if not exists(video_list):
    raise Exception('invalid video list!');
  videos = list();
  with open(video_list, 'r') as f:
    for line in f:
      path, classid = line.strip().split(' ');
      classid = int(classid);
      self.videos.append(join(video_root, path));
  writer = tf.io.TFRecordWriter(output_file);
  for i in range(len(videos)):
    print('%d/%d' % (i+1, len(videos)));
    video = videos[i];
    cap = cv2.VideoCapture(video);
    if False == cap.isOpened():
      print('can\'t open video %s' % video);
      continue;
    frames = np.zeros((0, size[0], size[1], 3), dtype = np.uint8);
    while True:
      ret, frame = cap.read();
      if False == ret: break;
      frame = np.expand_dims(cv2.resize(frame, size), axis = 0); # frame.shape = (1, h, w, c)
      frames = np.concat([frames, frame], axis = 0); # frames.shape = (length, h, w, c)
    trainsample = tf.train.Example(features = tf.train.Features(
      feature = {
        'video': tf.train.Feature(bytes_list = tf.train.BytesList(value = tf.io.serialize_tensor(frames))),
      }
    ));
    writer.write(trainsample.SerializeToString());
  writer.close();

def main(unused_argv):
  generate_dataset(FLAGS.ucf_root, FLAGS.train_list, output_file = 'trainset.tfrecord');
  generate_dataset(FLAGS.ucf_root, FLAGS.test_list, output_file = 'testset.tfrecord');

def parse_function(serialized_example):
  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'video': tf.io.FixedLenFeature((,), dtype = tf.string)
    });
  video = tf.io.parse_tensor(feature['video']); # video.shape = (length, h, w, c)
  return video;

def load_ucf101(filename):
  return tf.data.TFRecordDataset(filename).map(parse_function).repeat(-1);

if __name__ == "__main__":
  app.run(main);
