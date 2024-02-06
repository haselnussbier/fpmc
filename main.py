import optparse
import sys

import yaml
import re
from os import listdir
from os.path import isfile, join
from model import run
from methods import random_factor, base_score

from plot import plot, save_config, init_result
import pickle

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

usage_str = "%prog [options]"
description_str = "ML-script"
epilog_str = "Examples"

parser = optparse.OptionParser(usage=usage_str,
                               description=description_str,
                               epilog=epilog_str,
                               add_help_option=False,
                               version="%prog version 0.1")

parser.add_option('-h',
                  action='store_true',
                  dest='help',
                  default=False,
                  help="Show help notes")

parser.add_option('-d',
                  action='store_true',
                  dest='directory',
                  default=False,
                  help="Show list of available training sets")

parser.add_option("-l",
                  dest="layer",
                  type="int",
                  help="Set the amount of layers")

parser.add_option("-n",
                  dest="neuron",
                  type="int",
                  help="Set the amount of neurons per layer")

parser.add_option("-z",
                  dest="hidden_size",
                  type="int",
                  help="Set the size of the hidden information array")

parser.add_option("-p",
                  dest="steps_to_stop",
                  type="int",
                  help="Set the amount steps of training with no improvement")

parser.add_option("-r",
                  dest="learning_rate",
                  type="float",
                  help="Set the learning rate")

parser.add_option("-b",
                  dest="batch_size",
                  type="int",
                  help="Set the batch size")

parser.add_option("-f",
                  dest="file",
                  type="str",
                  help=".pkl file of the training set")


(options, args) = parser.parse_args()

if options.help:
    parser.print_help()
    sys.exit()

if options.directory:
    regex = re.compile("[\S]*\.pkl")
    mypath = "graphs/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and re.match(regex, f)]
    for f in onlyfiles:
        print(f)
    sys.exit()

if options.layer and options.neuron and options.hidden_size and options.steps_to_stop and options.learning_rate and options.file and options.batch_size:
    config['model']['layers'] = options.layer
    config['model']['neurons'] = options.neuron
    config['model']['hidden_size'] = options.hidden_size
    config['model']['steps_to_stop'] = options.steps_to_stop
    config['model']['learning_rate'] = options.learning_rate
    config['file'] = "graphs/" + options.file
    config['model']['batch_size'] = options.batch_size
else:
    print("Incomplete model parameters. Please use -h to get a list of necessary input.")
    #sys.exit()

run(config)
random_factor(config)
base_score(config)

