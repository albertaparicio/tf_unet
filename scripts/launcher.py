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


"""
Created on Jul 28, 2016

author: jakeret
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os

from unet_generator import UNetGeneratorClass
from tf_unet import unet, util

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description="Classify image patches with a U-Net")
  parser.add_argument('--data_path', type=str, default="data")
  parser.add_argument('--img_path', type=str, default="img")
  parser.add_argument('--labels_path', type=str, default="labels")
  parser.add_argument('--train_list', type=str, default="training.list")
  parser.add_argument('--test_list', type=str, default="test.list")
  parser.add_argument('--display_step', type=int, default=5)
  parser.add_argument('--num_classes', type=int, default=8)
  parser.add_argument('--batch_size', type=int, default=5)
  parser.add_argument('--patch_size', type=int, default=500)
  parser.add_argument('--patch_overlap', type=int, default=150)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--start_epoch', type=int, default=0)
  parser.add_argument('--summary', dest='show_summary', action='store_true',
                      help='Show summary of model')
  parser.add_argument('--dev', dest='dev', action='store_true',
                      help='Development mode')
  parser.add_argument('--no-train', dest='do_train',
                      action='store_false', help='Flag to train or not.')
  parser.add_argument('--no-test', dest='do_test',
                      action='store_false', help='Flag to test or not.')

  parser.set_defaults(dev=False, show_summary=False, display=False,
                      do_train=True, do_test=True)
  args = parser.parse_args()

  dropout = 0.75  # Dropout, probability to keep units
  restore = False

  if args.do_train:
    epochs = args.epochs
  else:
    epochs = 0

  # generator = image_gen.RgbDataProvider(nx, ny, cnt=20, rectangles=False)
  train_generator = UNetGeneratorClass(args.train_list, args.num_classes,
                                       args.batch_size, args.data_path,
                                       args.img_path, args.labels_path,
                                       args.patch_size, args.patch_overlap)
  test_generator = UNetGeneratorClass(args.test_list, args.num_classes, 1,
                                      args.data_path, args.img_path,
                                      args.labels_path, args.patch_size,
                                      args.patch_overlap)
  net = unet.Unet(channels=3, n_class=args.num_classes, layers=3,
                  features_root=16, cost="cross_entropy")

  trainer = unet.Trainer(net, batch_size=args.batch_size, optimizer="adam")  # ,
  # opt_kwargs=dict(momentum=0.2))

  path = trainer.train(train_generator, "./unet_trained",
                       training_iters=train_generator.training_iters,
                       epochs=epochs, dropout=dropout,
                       display_step=args.display_step, restore=restore)

  if args.do_test:
    x_test, y_test = test_generator(1)
    prediction = net.predict(path, x_test)

    print("Testing error rate: {:.2f}%".format(
        unet.error_rate(prediction,
                        util.crop_to_shape(y_test, prediction.shape))))

    UNetGeneratorClass.save_prediction_color_code(
        y_test, prediction,
        os.path.join(args.data_path,'res'),test_generator.files_list[0][0])
