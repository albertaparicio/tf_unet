# Created by albert aparicio on 11/05/17
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os
from itertools import chain, repeat
from random import shuffle

import numpy as np
from scipy.misc import imread, imsave
from skimage.transform import rotate
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


def data_augmentation(data):
  """Perform data augmentation on the provided image and labels.
  The data augmentation techniques used in this function are rotations,
  horizontal and vertical mirrorings and rotated mirrorings"""

  # Pre-compute horizontal and vertical mirrorings
  h_i_mirror = np.fliplr(data)
  v_i_mirror = np.flipud(data)

  # Return list with all data augmentations
  # The 270º rotations are done in two steps because if they are done on one
  # step, the result image is not resized well
  return [
    # Original data
    data,

    # Rotations
    rotate(data, angle=180, resize=False),
    rotate(data, angle=90, resize=True),
    rotate(rotate(data, angle=90, resize=True), angle=180, resize=False),

    # Mirrorings
    h_i_mirror,
    v_i_mirror,

    # Rotated mirrorings
    rotate(h_i_mirror, angle=180, resize=False),
    rotate(h_i_mirror, angle=90, resize=True),
    rotate(rotate(h_i_mirror, angle=90, resize=True), angle=180, resize=False),

    rotate(v_i_mirror, angle=180, resize=False),
    rotate(v_i_mirror, angle=90, resize=True),
    rotate(rotate(v_i_mirror, angle=90, resize=True), angle=180, resize=False)
    ]


class UNetGeneratorClass(object):
  def __init__(self, files_list, num_classes, data_path='data',
               img_path='img', labels_path='labels'):

    filenames = open(os.path.join(data_path, files_list)).readlines()

    self.files_list = []
    [self.files_list.append(os.path.splitext(name.split('\n')[0])) for name in
     sorted(filenames)]
    self.training_iters = (6 * len(self.files_list)) - 1

    self.num_classes = num_classes
    self.data_path = data_path
    self.img_path = img_path
    self.labels_path = os.path.join(self.data_path, labels_path)
    self.image_path = os.path.join(self.data_path, self.img_path)
    self.generator = self.provide_images(batch_size=1)

  def __call__(self, *args, **kwargs):
    return next(self.generator)

  def provide_images(self, batch_size=1):
    images = []
    labels = []
    count = 0

    # Iterate indefinitely over files list
    for image_name in chain.from_iterable(repeat(sorted(self.files_list))):
      # Read image and its labels
      image = imread(
          os.path.join(self.image_path, image_name[0]) + image_name[1])
      label = imread(os.path.join(self.labels_path, image_name[0] + '.png'))

      images.extend(data_augmentation(image))
      labels.extend(data_augmentation(
          to_categorical(label, self.num_classes).reshape(
              label.shape + (self.num_classes,))))

      assert len(images) == len(labels)
      count += 1

      if count == 5:
        # Shuffle images and labels list
        shuf_order = list(range(len(images)))
        shuffle(shuf_order)

        # np_images = np.array(images)[shuf_order]
        # np_labels = np.array(labels)[shuf_order]

        # Yield batches of images
        for i in shuf_order:
          yield (np.array(images[i]).reshape(((1,)+images[i].shape)),
                 np.array(labels[i]).reshape(((1,)+labels[i].shape)))

        images = []
        labels = []
        count = 0
