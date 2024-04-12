import pickle
import os
from os import listdir
from os.path import isfile, join
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
import re
import math
import time
import itertools
import dict2xml
import dicttoxml
from alive_progress import alive_bar

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
                  help="Set the amount of criticality levels")

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
    print("Incomplete information input for the generator. Use -h to get a list of necessary flags.")
    sys.exit()

def build_levels(ports: Dict):
    dst_set = set()
    src_set = set()
    dst = list()
    src = list()
    for port in ports:
        dst_set = dst_set | {port['@dstActor']}
        dst.append(port['@dstActor'])
        src_set = src_set | {port['@srcActor']}
        src.append(port['@srcActor'])

    roots = list(src_set-dst_set)

    roots_n = [i for i in roots]
    # root out islands

    for root in roots_n:
        queue = [root]
        connected = list()
        while queue:
            node = queue.pop()
            connected.append(node)
            for port in ports:
                if node == port['@srcActor']:
                    if (not port['@dstActor'] in queue) and (not port['@dstActor'] in connected):
                        queue.append(port['@dstActor'])
                if node == port['@dstActor']:
                    if (not port['@srcActor'] in queue) and (not port['@srcActor'] in connected):
                        queue.append(port['@srcActor'])

        if len(connected) < config['graphs']['tasks']:
            roots.remove(root)

    order = list()
    edges = [i for i in ports]
    while roots:
        node = roots.pop()
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
                    roots.append(new_node)
            else:
                new_edges.append(edge)
        edges = new_edges

    if len(order) < config['graphs']['tasks']:
        return False

    while len(order) > config['graphs']['tasks']:
        for node in order:
            if src.count(node) == 0 and dst.count(node) == 1:
                del src[dst.index(node)]
                del dst[dst.index(node)]
                order.remove(node)
                break
            return False

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


def blockprint():
    sys.stdout = open(os.devnull, 'w')


def enableprint():
    sys.stdout = sys.__stdout__


set_size = int(math.ceil(config['amount']/10))
nJobs = set_size

g = 0
graphs = list()
warning_buffer = jnp.zeros(5)
with alive_bar(config['amount'], title='Graph-Generator', spinner='classic', bar='solid') as bar:
    while g < config['amount']:
        cmd = "java -jar mcdag/generator.jar " \
              "-mu " + str(config['graphs']['cores']-(config['graphs']['cores']/10)) + \
              " -nd 1" \
              " -l " + str(config['graphs']['levels']) + \
              " -nt " + str(config['graphs']['tasks']+5) + \
              " -nf " + str(set_size) + \
              " -e " + str(config['graphs']['edge']) + \
              " -o graphs/graph.xml" \
              " -p 1" \
              " -j " + str(nJobs)

        ret = os.system(cmd)
        if ret != 0:
            print("MC-DAG script > ERROR unexpected behavior for the generation. Exiting...")
            continue

        for i in range(set_size):
            dict_graph = xmltodict.parse(open("graphs/graph-" + str(i) + ".xml").read())

            # calculate node order and remove nodes if necessary

            # TODO: add 'check_wholeness'
            order = build_levels(dict_graph['mcsystem']['mcdag']['ports']['port'])

            if not order:
                os.remove("graphs/graph-" + str(i) + ".xml")
                continue

            # remove nodes

            actors = dict_graph['mcsystem']['mcdag']['actor']
            actors_new = list()
            for actor in actors:
                if actor['@name'] in order:
                    actors_new.append(actor)

            dict_graph['mcsystem']['mcdag']['actor'] = actors_new

            ports = dict_graph['mcsystem']['mcdag']['ports']['port']
            ports_new = list()
            for port in ports:
                if port['@dstActor'] in order and port['@srcActor'] in order:
                    ports_new.append(port)

            dict_graph['mcsystem']['mcdag']['ports']['port'] = ports_new

            with open("graphs/graph-" + str(i) + ".xml", "w") as f:
                f.write(xmltodict.unparse(dict_graph))
            dict_graph = xmltodict.parse(open("graphs/graph-" + str(i) + ".xml").read())

            # schedule graph with removed nodes and check schedulability

            cmd = "java -jar mcdag/scheduler.jar -i graphs/graph-" + str(i) + ".xml -j 1 -os"
            ret = os.system(cmd)
            if ret != 0:
                continue

            try:
                dict_sched = xmltodict.parse(open("graphs/graph-" + str(i) + "-sched.xml").read())
            except:
                os.remove("graphs/graph-" + str(i) + ".xml")
                continue
            check = False
            for slot in dict_sched['sched']['Mode-0']['core'][0]['slot']:
                if slot['#text'] != '-':
                    check = True
                    break

            if not check:
                os.remove("graphs/graph-" + str(i) + ".xml")
                os.remove("graphs/graph-" + str(i) + "-sched.xml")
                continue

            # create graph

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
                continue

            graph = Graph(node_features=jnp.asarray(node_features),
                          node_values=None,
                          steps=steps,
                          deadline=[int(dict_graph['mcsystem']['mcdag']['@deadline']) * config['graphs']['cores']])

            bar()
            graphs += [graph]
            g = g + 1
            os.remove("graphs/graph-"+str(i)+".xml")
            os.remove("graphs/graph-" + str(i) + "-sched.xml")
            if len(graphs) == config['amount']:
                regex = re.compile("[\S]*\.xml")
                mypath = "graphs/"
                onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and re.match(regex, f)]
                for f in onlyfiles:
                    os.remove(join(mypath, f))
                break

with open('graphs/' + str(config['amount']) + 'graphs-c' + str(config['graphs']['cores']) + '-t' + str(config['graphs']['tasks']) + '.pkl', 'wb') as f:
    pickle.dump(graphs, f)






