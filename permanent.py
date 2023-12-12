import pickle
import os
import xmltodict
import sys
from typing import *
import numpy as np
import jax
import jax.numpy as jnp
import random
from model import Graph, Step
from generator import build_levels, batch

nCores = 2
nDags = 1
nLevels = 2
nTasks = 20
set_size = 1
pEdge = 11
nJobs = 1
amount = 100


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

    if len(nodes) == nTasks:
        return True
    else:
        return False

g = 0
graphs = list()
while g < amount:
    cmd = "java -jar mcdag/generator.jar " \
          "-mu " + str(nCores - 0.2) + \
          " -nd " + str(nDags) + \
          " -l " + str(nLevels) + \
          " -nt " + str(nTasks) + \
          " -nf " + str(set_size) + \
          " -e " + str(pEdge) + \
          " -o permanent/graph.xml" \
          " -p 5" \
          " -j " + str(nJobs)

    ret = os.system(cmd)
    if ret != 0:
        print("MC-DAG script > ERROR unexpected behavior for the generation. Exiting...")
        continue

    cmd = "java -jar mcdag/scheduler.jar -i permanent/graph-0.xml -j 1 -os"
    ret = os.system(cmd)
    if ret != 0:
        print("Graph not schedulable, skipping")
        continue

    dict_graph = xmltodict.parse(open(
        "permanent/graph-0.xml").read())
    dict_sched = xmltodict.parse(open(
        "permanent/graph-0-sched.xml").read())

    check = False
    for slot in dict_sched['sched']['Mode-0']['core'][0]['slot']:
        if slot['#text'] != '-':
            check = True
            break

    if not check:
        print("Scheduler Error, skipping")
        continue

    order = build_levels(dict_graph['mcsystem']['mcdag']['ports']['port'])

    if len(dict_graph['mcsystem']['mcdag']['actor']) != nTasks:
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
                                                                                 '#text']) * random.uniform(0.1,
                                                                                                            0.2)]],
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
                  deadline=[int(dict_graph['mcsystem']['mcdag']['@deadline']) * nCores])
    print(graph)
    graphs += [graph]
    g = g + 1


with open('permanent/' + str(amount) + 'graphs-c' + str(nCores) + '-t' + str(nTasks) + '.pkl', 'wb') as f:
    pickle.dump(graphs, f)






