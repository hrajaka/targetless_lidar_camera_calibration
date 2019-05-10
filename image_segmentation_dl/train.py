#! /usr/bin/env python

import argparse
import json
import os

from src.frontend import Model

argparser = argparse.ArgumentParser(
    description='Train and validate Kitti Road Segmentation Model')

argparser.add_argument(
    '-c',
    '--conf', default="config.json",
    help='path to configuration file')


def _main_(args):
    """
    :param args: command line argument
    """
    # check for data existence
    if not os.path.exists('data'):
        print('No data folder found, please download data at:')
        print('https://drive.google.com/file/d/15jBrlQlMJ51A1BcOSasFwXohaipZL7Kz/view?usp=sharing')
        return 

    config_path = args.conf

    # load the json files
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    # read the json files and build the architecture
    backend = config["model"]["backend"]
    input_size = (config["model"]["im_width"], config["model"]["im_height"])
    classes = config["model"]["classes"]
    data_dir = config["train"]["data_directory"]

    # define the model and train
    model = Model(backend, input_size, classes)
    model.train(config["train"])


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
