# Created by albert aparicio on 11/05/17
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os
from itertools import chain, repeat

import numpy as np
from scipy.misc import imread, imsave
from tensorflow.contrib.keras.python.keras.utils.np_utils import to_categorical

from tf_unet import util


def save_prediction_color_code(ground_truth, prediction, save_path, filename):
  color_code_dict = [
    [0, 0, 0],  # black
    # [0, 0, 0],  # black
    [1, 0, 0],  # red
    [1, 0.4392156863, 0],  # orange
    [1, 1, 1],  # white
    [1, 0, 1],  # magenta
    [0, 0, 1],  # blue
    [0, 1, 0],  # green
    [0, 1, 1]  # cyan
    ]

  # Crop ground truth image
  crop_gt = util.crop_to_shape(ground_truth, prediction.shape)

  # Argmax to remove one-hot encoding
  gt_categorical = np.argmax(crop_gt, axis=3).squeeze()
  pr_categorical = np.argmax(prediction, axis=3).squeeze()

  gt_mat = np.zeros(gt_categorical.shape + (3,))
  pr_mat = np.zeros(pr_categorical.shape + (3,))

  for num in range(len(color_code_dict)):
    gt_mat[gt_categorical == num, :] = color_code_dict[num]
    pr_mat[pr_categorical == num, :] = color_code_dict[num]

  if not os.path.exists(save_path):
    os.makedirs(save_path)

  imsave(os.path.join(save_path, filename + '_gt.png'), gt_mat)
  imsave(os.path.join(save_path, filename + '_pr.png'), pr_mat)


class UNetGeneratorClass(object):
  def __init__(self, files_list, num_classes, batch_size, data_path='data',
               img_path='img', labels_path='labels'):

    filenames = open(os.path.join(data_path, files_list)).readlines()

    self.files_list = []
    [self.files_list.append(os.path.splitext(name.split('\n')[0])) for name in
     sorted(filenames)]
    self.training_iters = len(self.files_list) - 4

    self.num_classes = num_classes
    self.data_path = data_path
    self.img_path = img_path
    self.labels_path = os.path.join(self.data_path, labels_path)
    self.image_path = os.path.join(self.data_path, self.img_path)
    self.generator = self.provide_images(batch_size)

  def __call__(self, *args, **kwargs):
    return next(self.generator)

  def provide_images(self, batch_size):
    images = []
    labels = []

    # Iterate indefinitely over files list
    for image_name in chain.from_iterable(repeat(sorted(self.files_list))):
      # Read image and its labels
      images.append(
          imread(os.path.join(self.image_path, image_name[0]) + image_name[1]))

      label = imread(os.path.join(self.labels_path, image_name[0] + '.png'))
      labels.append(to_categorical(label, self.num_classes).reshape(
          label.shape + (self.num_classes,)))

      assert len(images) == len(labels)

      if len(images) == batch_size:
        # Return data and re-initialize lists

        np_images = np.array(images)
        np_labels = np.array(labels)

        images = []
        labels = []

        yield (np_images, np_labels)
