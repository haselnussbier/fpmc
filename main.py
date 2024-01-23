import optparse
import sys

import yaml
import re
from os import listdir
from os.path import isfile, join
from model import run, random_factor

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
    mypath = "permanent/"
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
    config['file'] = "permanent/" + options.file
    config['model']['batch_size'] = options.batch_size
else:
    print("Incomplete model parameters. Please use -h to get a list of necessary input.")
    #sys.exit()


"""
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

parser.add_option("-c",
                  dest="core",
                  type="int",
                  help="Set the amount of cores")

parser.add_option("-t",
                  dest="task",
                  type="int",
                  help="Set the amount of tasks")

parser.add_option("-e",
                  dest="edge",
                  type="int",
                  help="Set the edge probability")

parser.add_option("-s",
                  dest="training_set",
                  type="int",
                  help="Set the amount Graphs generated")

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

parser.add_option("-j",
                  dest="jobs",
                  type="int",
                  help="Set Amount of Threads for Graph generation")

parser.add_option("-u",
                  action="store_true",
                  dest="single",
                  help="Perform Model on a single graph")

(options, args) = parser.parse_args()

if options.help:
    parser.print_help()
    sys.exit()

if not (options.layer is None):
    config['model']['layers'] = options.layer

if not (options.neuron is None):
    config['model']['neurons'] = options.neuron

if not (options.hidden_size is None):
    config['model']['hidden_size'] = options.hidden_size

if not (options.core is None):
    config['graphs']['cores'] = options.core

if not (options.task is None):
    config['graphs']['tasks'] = options.task

if not (options.edge is None):
    config['graphs']['edge'] = options.edge

if not (options.training_set is None):
    config['training_set'] = options.training_set

if not (options.steps_to_stop is None):
    config['model']['steps_to_stop'] = options.steps_to_stop

if not (options.learning_rate is None):
    config['model']['learning_rate'] = options.learning_rate

if not (options.batch_size is None):
    config['model']['batch_size'] = options.batch_size

if not (options.jobs is None):
    config['jobs'] = options.jobs
"""

util_model, p_model = run(config)
util_frac, p_frac = random_factor(config['file'], config['model']['batch_size'])

"""
if options.single:
# if 1:
    config['training_set'] = 5
    config['model']['batch_size'] = 1

    init_result()

    train_set, validate_set = generate_sets(nTasks=config['graphs']['tasks'],
                                            nDags=config['graphs']['dags'],
                                            nCores=config['graphs']['cores'],
                                            pEdge=config['graphs']['edge'],
                                            set_size=config['training_set'],
                                            nJobs=config['jobs'],
                                            split=1)

    single = [train_set.pop()]

    model_config = ModelConfig(
        num_hidden_size=config['model']['hidden_size'],
        num_hidden_neurons=config['model']['neurons'],
        num_hidden_layers=config['model']['layers']
    )

    single = batch(single, config['model']['batch_size'])

    net, params = init_net(model_config=model_config, sample=single[0])

    trained_params = train_model(net=net,
                                 params=params,
                                 train_set=single,
                                 validate_set=single[0],
                                 model_config=config['model'])

    plot()

    loss, utilization, p_task_overrun, wcets = predict_model(net, trained_params, single[0], config['model'])

    print("*****************************************")
    print("Test-Batch finished with a loss of  ", loss)
    print("An average utilization of ", utilization, " per Graph.")
    print("And a probability of task overrun of ", p_task_overrun, " per Graph.")
    print("Starting wcets: ", single[0].node_features)
    print("The best wcets are: ", wcets)
    print("*****************************************")

    save_config(config)

else:

    init_result()

    train_set, validate_set = generate_sets(nTasks=config['graphs']['tasks'],
                                            nDags=config['graphs']['dags'],
                                            nCores=config['graphs']['cores'],
                                            pEdge=config['graphs']['edge'],
                                            set_size=config['training_set'],
                                            nJobs=config['jobs'])

    model_config = ModelConfig(
        num_hidden_size=config['model']['hidden_size'],
        num_hidden_neurons=config['model']['neurons'],
        num_hidden_layers=config['model']['layers']
    )
    batched_train = batch(train_set, config['model']['batch_size'])
    batched_val = batch(validate_set, config['model']['batch_size'])
    test = batched_train.pop()
    sample = batched_train[0]

    net, params = init_net(model_config=model_config, sample=sample)

    trained_params = train_model(net=net,
                                 params=params,
                                 train_set=batched_train,
                                 validate_set=batched_val[0],
                                 model_config=config['model'])

    plot()

    loss, utilization, p_task_overrun, wcets = predict_model(net, trained_params, test, config['model'])

    print("*****************************************")
    print("Test-Batch finished with a loss of  ", loss)
    print("An average utilization of ", utilization, " per Graph.")
    print("And a probability of task overrun of ", p_task_overrun, " per Graph.")
    print("*****************************************")

    save_config(config)


config['model']['batch_size'] = 1

init_result()

with open("permanent/1graphs-c2-t30.pkl", "rb") as f:
    graph = pickle.load(f)


train_set = graph[:80]
validate_set = graph[80:]
test_set = validate_set[10:]
validate_set = validate_set[:10]



model_config = ModelConfig(
    num_hidden_size=config['model']['hidden_size'],
    num_hidden_neurons=config['model']['neurons'],
    num_hidden_layers=config['model']['layers']
)

train_set = batch(graph, config['model']['batch_size'])
validate_set = batch(graph, config['model']['batch_size'])
test_set = batch(graph, config['model']['batch_size'])

net, params = init_net(model_config=model_config, sample=train_set[0])

trained_params = train_model(net=net,
                             params=params,
                             train_set=train_set,
                             validate_set=validate_set[0],
                             model_config=config['model'])

plot()

loss, utilization, p_task_overrun, wcets = predict_model(net, trained_params, test_set[0], config['model'])

print("*****************************************")
print("Test-Batch finished with a loss of  ", loss)
print("An average utilization of ", utilization, " per Graph.")
print("And a probability of task overrun of ", p_task_overrun, " per Graph.")
print("Starting wcets: ", test_set[0].node_features)
print("The best wcets are: ", wcets)
print("*****************************************")

save_config(config)
"""

