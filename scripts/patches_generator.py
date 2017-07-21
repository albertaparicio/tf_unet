# Created by albert aparicio on 11/05/17
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os
from itertools import chain, repeat

import numpy as np
from keras.utils import to_categorical
from scipy.misc import imread


class PatchesGeneratorClass(object):
  def __init__(self, files_list, window_size, num_classes,
               data_path='data', img_path='img', labels_path='labels',
               display=False):

    filenames = open(os.path.join(data_path, files_list)).readlines()

    self.files_list = []
    [self.files_list.append(os.path.splitext(name.split('\n')[0])) for name in
     sorted(filenames)]

    if display:
      assert len(self.files_list) == 1

    self.w = window_size
    self.d = int(self.w // 2)
    self.d_plus = self.w % 2
    self.num_classes = num_classes
    self.data_path = data_path
    self.img_path = img_path
    self.labels_path = labels_path
    self.image_path = os.path.join(self.data_path, self.img_path)

  def count_patches(self):
    num_patches = 0

    for image in sorted(self.files_list):
      # Select pixels that are not from background to be used for patches
      mask = imread(os.path.join(
          self.image_path, image[0][:3] + 'N' + image[0][3:]) + '.png')[:, :, 1]

      # mask = regions[:, :, 1]

      # Normalize and pad the mask with False (masked) values
      mask_norm = (mask / np.max(mask)).astype(np.bool)
      mask_padded = np.pad(mask_norm, self.d, mode='constant',
                           constant_values=False)

      # Get and accumulate number of True pixels i.e. patches
      num_patches += np.sum(mask_padded)

    return num_patches

  def generate_patches(self, batch_size, validation=False, full_patch=False):
    # Iterate indefinitely over files list
    for image in chain.from_iterable(repeat(sorted(self.files_list))):

      # Read image, regions and labels
      im = imread(os.path.join(self.image_path, image[0]) + image[1])
      # Channel 0 contains cell nucleus
      # Channel 1 contains cell membrane + nucleus
      # Channel 2 contains non nucleus
      regions = imread(
          os.path.join(self.image_path,
                       image[0][:3] + 'N' + image[0][3:]) + '.png')
      labels_c1 = imread(os.path.join(self.data_path, self.labels_path,
                                      image[0] + '.1' + '.png'))
      labels_c2 = imread(os.path.join(self.data_path, self.labels_path,
                                      image[0] + '.2' + '.png'))

      # Select pixels that are not from background to be used for patches
      mask = regions[:, :, 1]

      # Apply mirroring on the edges of image
      im_padded = np.pad(im, ((self.d, self.d), (self.d, self.d), (0, 0)),
                         mode='reflect')

      # Normalize and pad the mask with False (masked) values
      mask_norm = (mask / np.max(mask)).astype(np.bool)

      # Normalize and Pad the labels
      labels_c1_n = (labels_c1 / 255).astype(np.uint8)
      labels_c2_n = (labels_c2 / 255).astype(np.uint8)

      labels_c1_pad = 1 * np.pad(labels_c1_n,
                                 ((self.d, self.d), (self.d, self.d)),
                                 mode='constant', constant_values=0)
      labels_c2_pad = 2 * np.pad(labels_c2_n,
                                 ((self.d, self.d), (self.d, self.d)),
                                 mode='constant', constant_values=0)

      # Iterate over the pixels of the padded mask
      patches = []
      labels_list = []
      num_it = 0
      pixels_list = []

      # Iterate over unpadded mask, saving data from padded matrices
      it = np.nditer(mask_norm, flags=['multi_index'])
      while not it.finished:
        if it[0]:
          # Pixel belongs to cytoplasm or cell nucleus
          (n_orig, m_orig) = it.multi_index
          n = n_orig + self.d
          m = m_orig + self.d

          pixels_list.append((n_orig, m_orig))
          # Take patch centered at current pixel (becomes w-width window)
          patches.append(im_padded[
                         n - self.d:n + self.d + self.d_plus,
                         m - self.d:m + self.d + self.d_plus,
                         :])

          # The modulus operation ensures that labels are only 0, 1 or 2 i.e.
          # when a pixel belongs to two classes at once, it becomes background
          if full_patch:
            labels_list.append(to_categorical(
                (labels_c1_pad[
                 n - self.d:n + self.d + self.d_plus,
                 m - self.d:m + self.d + self.d_plus
                 ] + labels_c2_pad[
                     n - self.d:n + self.d + self.d_plus,
                     m - self.d:m + self.d + self.d_plus]
                 ) % self.num_classes, self.num_classes).reshape(
                (self.w, self.w, self.num_classes)))
          else:
            labels_list.append(to_categorical(
                (labels_c1_pad[n, m] + labels_c2_pad[n, m]
                 ) % self.num_classes, self.num_classes).reshape(
                (1, 1, self.num_classes)))

          # Each 'batch_size' iterations, yield the collected patches
          num_it += 1

          if num_it == batch_size:
            yield (np.array(patches),
                   np.array(labels_list),
                   pixels_list)

            num_it = 0
            patches = []
            labels_list = []
            pixels_list = []

        it.iternext()
