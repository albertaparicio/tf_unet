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


'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse

from scripts.unet_generator import UNetGeneratorClass
from tf_unet import unet, util

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description="Classify image patches with a U-Net")
  parser.add_argument('--data_path', type=str, default="data")
  parser.add_argument('--img_path', type=str, default="img")
  parser.add_argument('--labels_path', type=str, default="labels")
  parser.add_argument('--train_list', type=str, default="training.list")
  parser.add_argument('--test_list', type=str, default="test.list")
  parser.add_argument('--batch_size', type=int, default=5)
  parser.add_argument('--num_classes', type=int, default=8)
  parser.add_argument('--epochs', type=int, default=1)
  parser.add_argument('--start_epoch', type=int, default=0)
  parser.add_argument('--summary', dest='show_summary', action='store_true',
                      help='Show summary of model')
  parser.add_argument('--dev', dest='dev', action='store_true',
                      help='Development mode')

  parser.set_defaults(dev=False, show_summary=False, display=False)
  args = parser.parse_args()

  # nx = 572
  # ny = 572

  # training_iters = 20
  # epochs = 100
  dropout = 0.75  # Dropout, probability to keep units
  display_step = 2
  restore = False

  # generator = image_gen.RgbDataProvider(nx, ny, cnt=20, rectangles=False)
  train_generator = UNetGeneratorClass(args.train_list, args.num_classes,
                                       args.batch_size, args.data_path,
                                       args.img_path, args.labels_path)
  test_generator = UNetGeneratorClass(args.test_list, args.num_classes,
                                      args.batch_size, args.data_path,
                                      args.img_path, args.labels_path)

  net = unet.Unet(channels=3,
                  n_class=args.num_classes,
                  layers=1,
                  features_root=16,
                  cost="dice_coefficient")

  trainer = unet.Trainer(net, optimizer="momentum",
                         opt_kwargs=dict(momentum=0.2))
  path = trainer.train(train_generator, "./unet_trained",
                       training_iters=train_generator.training_iters,
                       epochs=args.epochs,
                       dropout=dropout,
                       display_step=display_step,
                       restore=restore)

  x_test, y_test = test_generator(args.batch_size)
  prediction = net.predict(path, x_test)

  print("Testing error rate: {:.2f}%".format(
      unet.error_rate(prediction,
                      util.crop_to_shape(y_test, prediction.shape))))
