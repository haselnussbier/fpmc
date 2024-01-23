import pickle
import os
import xmltodict
import sys
from typing import *
import numpy as np
import optparse
import jax
import jax.numpy as jnp
import random
from model import Graph, Step
import yaml
import time
from generator import build_levels, batch

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

usage_str = "%prog [options]"
description_str = "Training-Set generator"
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

parser.add_option("-c",
                  dest="cores",
                  type="int",
                  help="Set the amount of cores")

parser.add_option("-l",
                  dest="levels",
                  type="int",
                  help="Set the amount of graph levels")

parser.add_option("-t",
                  dest="tasks",
                  type="int",
                  help="Set the amount of tasks")

parser.add_option("-s",
                  dest="size",
                  type="int",
                  help="Set the amount graphs in the training set")

parser.add_option("-e",
                  dest="edge",
                  type="int",
                  help="Set the chance of edge creation in percent")


(options, args) = parser.parse_args()

if options.help:
    parser.print_help()
    sys.exit()

if options.cores and options.levels and options.tasks and options.size and options.edge:
    config['graphs']['cores'] = options.cores
    config['graphs']['edge'] = options.edge
    config['graphs']['tasks'] = options.tasks
    config['graphs']['levels'] = options.levels
    config['amount'] = options.size
else:
    print("Incomplete information input for thye generator. Use -h to get a list of necessary flags.")
    sys.exit()

def build_levels(ports: Dict):
    dst = list()
    src = list()
    i = 0
    for port in ports:
        if port['@dstActor'] not in dst:
            dst.append(port['@dstActor'])
        if port['@srcActor'] not in src:
            src.append(port['@srcActor'])
        i = i + 1

    starter = list()
    for source in src:
        if source not in dst:
            starter.append(source)

    order = list()
    edges = ports
    while starter:

        node = starter.pop()
        order.append(node)
        new_edges = list()
        for edge in edges:
            if edge['@srcActor'] == node:
                new_node = edge['@dstActor']
                i = 0
                for edge2 in edges:
                    if edge2['@dstActor'] == new_node:
                        i = i + 1
                if i == 1:
                    starter.append(new_node)
            else:
                new_edges.append(edge)
        edges = new_edges

    return order


def check_wholeness(steps):
    starter = list([steps[0].sender])
    nodes = list()

    while starter:
        node = starter.pop()
        nodes.append(node)
        for step in steps:
            if node == step.sender:
                if (not step.receiver in starter) and (not step.receiver in nodes):
                    starter.append(step.receiver)
            if node == step.receiver:
                if (not step.sender in starter) and (not step.sender in nodes):
                    starter.append(step.sender)

    if len(nodes) == config['graphs']['tasks']:
        return True
    else:
        return False

set_size = int(config['amount'])
nJobs = 1

g = 0
graphs = list()
print(config['graphs']['cores'], config['graphs']['levels'], config['graphs']['tasks'], set_size, config['graphs']['edge'])
while g < config['amount']:
    time.sleep(2)
    cmd = "java -jar mcdag/generator.jar " \
          "-mu " + str(config['graphs']['cores'] - 0.6) + \
          " -nd 1" \
          " -l " + str(config['graphs']['levels']) + \
          " -nt " + str(config['graphs']['tasks']) + \
          " -nf " + str(set_size) + \
          " -e " + str(config['graphs']['edge']) + \
          " -o permanent/graph.xml" \
          " -p 1" \
          " -j 1" # + str(nJobs)

    ret = os.system(cmd)
    if ret != 0:
        print("MC-DAG script > ERROR unexpected behavior for the generation. Exiting...")
        continue

    for i in range(set_size):
        cmd = "java -jar mcdag/scheduler.jar -i permanent/graph-"+str(i)+".xml -j " + str(nJobs) + " -os"
        ret = os.system(cmd)
        if ret != 0:
            print("Graph not schedulable, skipping")
            continue
    #sys.exit()
    for i in range(set_size):
        dict_graph = xmltodict.parse(open(
            "permanent/graph-" + str(i) + ".xml").read())
        dict_sched = xmltodict.parse(open(
            "permanent/graph-" + str(i) + "-sched.xml").read())

        check = False
        for slot in dict_sched['sched']['Mode-0']['core'][0]['slot']:
            if slot['#text'] != '-':
                check = True
                break

        if not check:
            print("Scheduler Error, skipping")
            continue
        time.sleep(2)
        order = build_levels(dict_graph['mcsystem']['mcdag']['ports']['port'])

        if len(dict_graph['mcsystem']['mcdag']['actor']) != config['graphs']['tasks']:
            print("wtf too many tasks")
            continue

        if len(order) < len(dict_graph['mcsystem']['mcdag']['actor']):
            print("Graph contains unreachable nodes")
            continue

        node_features = np.zeros(shape=(len(dict_graph['mcsystem']['mcdag']['actor']) + 1, 5))

        for actor in dict_graph['mcsystem']['mcdag']['actor']:

            # Create NodeFeatures:
            # Criticality
            # wcet_hi
            # acet
            # Standard deviation
            if actor['wcet'][1]['#text'] == '0':
                node_features[order.index(actor['@name'])] = jnp.asarray([[0, int(actor['wcet'][0]['#text']), 0, 0, 0]],
                                                                         dtype=jnp.float32)


            else:
                node_features[order.index(actor['@name'])] = jnp.asarray([[1,
                                                                           int(actor['wcet'][0]['#text']),
                                                                           int(actor['wcet'][1]['#text']),
                                                                           float(actor['wcet'][1][
                                                                                     '#text']) * random.uniform(0.2,
                                                                                                                1 / 3),
                                                                           float(actor['wcet'][1][
                                                                                     '#text']) * random.uniform(0.05,
                                                                                                                0.1)]],
                                                                         dtype=jnp.float32)
        # create dummy node for padding
        node_features[-1] = jnp.asarray([[0, 0, 0, 0, 0]], dtype=jnp.float32)

        steps = list()
        for node in order:
            for edge in dict_graph['mcsystem']['mcdag']['ports']['port']:
                if edge['@srcActor'] == node:
                    steps.append(Step(sender=jnp.asarray([order.index(node)], dtype=jnp.int32),
                                      receiver=jnp.asarray([order.index(edge['@dstActor'])], dtype=jnp.int32)))

        if not check_wholeness(steps):
            print("contains islands")
            continue

        graph = Graph(node_features=jnp.asarray(node_features),
                      node_values=None,
                      steps=steps,
                      deadline=[int(dict_graph['mcsystem']['mcdag']['@deadline']) * config['graphs']['cores']])

        graphs += [graph]
        g = g + 1
        if len(graphs) == config['amount']:
            break

with open('permanent/' + str(config['amount']) + 'graphs-c' + str(config['graphs']['cores']) + '-t' + str(config['graphs']['tasks']) + '.pkl', 'wb') as f:
    pickle.dump(graphs, f)






