# Copyright 2017 Albert Aparicio Isarn
#
# This file is part of tf_unet.
#
# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


"""Created on 20 Jul 2017

author: albertaparicio"""

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os
from itertools import chain, repeat
from queue import Queue
from random import shuffle

import numpy as np
from scipy.misc import imread, imsave
from skimage.transform import rotate
from sklearn.feature_extraction.image import extract_patches
from tensorflow.contrib.keras.python.keras.utils.np_utils import to_categorical

from tf_unet import util


class UNetGeneratorClass(object):
  def __init__(self, files_list, num_classes, batch_size=1, data_path='data',
               img_path='img', labels_path='labels', patch_width=500,
               patch_overlap=150):

    filenames = open(os.path.join(data_path, files_list)).readlines()

    self.files_list = []
    [self.files_list.append(os.path.splitext(name.split('\n')[0])) for name in
     sorted(filenames)]
    shuffle(self.files_list)

    self.training_iters = (6 * len(self.files_list)) - 1
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.data_path = data_path
    self.img_path = img_path
    self.labels_path = os.path.join(self.data_path, labels_path)
    self.image_path = os.path.join(self.data_path, self.img_path)
    self.patch_width = patch_width
    self.patch_overlap = patch_overlap
    self.patch_stride = self.patch_width - self.patch_overlap
    # self.patch_size = (self.patch_width, self.patch_width, 3)

    self.generator = self.provide_images(batch_size=self.batch_size)

  def __call__(self, *args, **kwargs):
    return next(self.generator)

  @staticmethod
  def save_prediction_color_code(ground_truth, prediction, save_path,
                                 filename):
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

  @staticmethod
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
      rotate(rotate(h_i_mirror, angle=90, resize=True), angle=180,
             resize=False),

      rotate(v_i_mirror, angle=180, resize=False),
      rotate(v_i_mirror, angle=90, resize=True),
      rotate(rotate(v_i_mirror, angle=90, resize=True), angle=180, resize=False)
      ]

  def get_pad_size_num_patches(self, dim_size):
    # Compute number of needed patches
    num_patches = 1 + np.ceil((dim_size - self.patch_width) / self.patch_stride)

    # Compute size of padded image
    padded_size = int(num_patches * self.patch_stride + self.patch_overlap)

    return padded_size - dim_size

  def get_patches(self, image, label):
    assert image.ndim == 3 and label.ndim == 3
    assert image.shape[:2] == label.shape[:2]

    v_pad = self.get_pad_size_num_patches(image.shape[0])
    h_pad = self.get_pad_size_num_patches(image.shape[1])

    # Pad image and label
    pad_image = np.pad(image, ((0, v_pad), (0, h_pad), (0, 0)), mode='reflect')
    pad_label = np.pad(label, ((0, v_pad), (0, h_pad), (0, 0)), mode='reflect')

    # Extract patches
    # self.patch_size = (self.patch_width, self.patch_width, 3)
    image_patches = extract_patches(
        pad_image, (self.patch_width, self.patch_width, 3), self.patch_stride
        ).reshape((-1,) + (self.patch_width, self.patch_width, 3))
    label_patches = extract_patches(
        pad_label, (self.patch_width, self.patch_width, self.num_classes), self.patch_stride
        ).reshape((-1,) + (self.patch_width, self.patch_width, self.num_classes))

    return image_patches, label_patches

  def provide_images(self, batch_size=1):
    image_patch_queue = Queue()
    label_patch_queue = Queue()

    # Iterate indefinitely over files list
    for image_name in chain.from_iterable(repeat(sorted(self.files_list))):
      # Read image and its labels
      image = imread(
          os.path.join(self.image_path, image_name[0]) + image_name[1])
      label = imread(os.path.join(self.labels_path, image_name[0] + '.png'))

      augmented_image = self.data_augmentation(image)
      augmented_label = self.data_augmentation(
          to_categorical(label, self.num_classes).reshape(
              label.shape + (self.num_classes,)))

      assert len(augmented_image) == len(augmented_label)

      for aug_image, aug_label in zip(augmented_image, augmented_label):
        # Extract image patches
        img_patches, lab_patches = self.get_patches(aug_image, aug_label)

        # Put patches into queue
        [image_patch_queue.put(item) for item in img_patches[:]]
        [label_patch_queue.put(item) for item in lab_patches[:]]

        assert image_patch_queue.qsize() == label_patch_queue.qsize()

        while image_patch_queue.qsize() >= batch_size:
          # Pop a batch from the queue
          img_batch = []
          [img_batch.append(image_patch_queue.get()) for _ in range(batch_size)]

          lab_batch = []
          [lab_batch.append(label_patch_queue.get()) for _ in range(batch_size)]

          yield np.array(img_batch), np.array(lab_batch)
