#!/usr/bin/python3

from os.path import exists, join;
import cv2;
import numpy as np;
import tensorflow as tf;

def generate_dataset(video_root, video_list, size = (64,64), output_file):
  if not exists(video_list):
    raise Exception('invalid video list!');
  videos = list();
  with open(video_list, 'r') as f:
    for line in f:
      path, classid = line.strip().split(' ');
      self.videos.append(join(video_root, path));
  writer = tf.io.TFRecordWriter(output_file);
  for video in videos:
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
        'video': tf.train.Feature(bytes_list = )
      }
    ));

    
