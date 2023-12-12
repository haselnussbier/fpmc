import os
import random
from typing import *

import jax
import jax.numpy as jnp
import numpy as np
import xmltodict

from model import Graph, Step


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


def calc_leftover(schedules):
    leftover = 0
    for core_sched in schedules:
        for i in range(len(core_sched)):
            core_sched[i] = core_sched[i]['#text']
        for i in range(len(core_sched)):
            if core_sched[i] == '-':
                leftover += 1
    return leftover


def generate_sets(nTasks: int, nDags: int, nCores: int, pEdge: float, set_size: int, nLevels=2, nJobs=1, split=0.8):
    print("uwu", os.path.exists("mcdag/generator.jar"))
    cmd = "java -jar mcdag/generator.jar " \
          "-mu " + str(nCores - 0.9) + \
          " -nd " + str(nDags) + \
          " -l " + str(nLevels) + \
          " -nt " + str(nTasks) + \
          " -nf " + str(set_size) + \
          " -e " + str(pEdge) + \
          " -o graphs/graph-c" + str(nCores) + "-t" + str(nTasks) + "-e" + str(pEdge) + ".xml " \
                                                                                        "-p 1 " \
                                                                                        "-j " + str(nJobs)
    ret = os.system(cmd)
    if ret != 0:
        print("MC-DAG script > ERROR unexpected behavior for the generation. Exiting...")
        return -1

    graphs = list()

    for i in range(set_size):
        """
        cmd = "java -jar mcdag/scheduler.jar -i graphs/graph-c" + str(nCores) + "-t" + str(nTasks) + "-e" + str(
            pEdge) + "-" + str(i) + ".xml -j 1 -os"
        ret = os.system(cmd)
        if ret != 0:
            print("Graph not schedulable, skipping")
            continue
        """
        dict_graph = xmltodict.parse(open(
            "graphs/graph-c" + str(nCores) + "-t" + str(nTasks) + "-e" + str(pEdge) + "-" + str(i) + ".xml").read())
        """
        dict_sched = xmltodict.parse(open(
            "graphs/graph-c" + str(nCores) + "-t" + str(nTasks) + "-e" + str(pEdge) + "-" + str(
                i) + "-sched.xml").read())
        """
        order = build_levels(dict_graph['mcsystem']['mcdag']['ports']['port'])
        if len(order) != nTasks:
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
        """
        core_schedules = list()
        for i in range(nCores):
            core_schedules.append(dict_sched['sched']['Mode-0']['core'][i]['slot'])
        """

        graph = Graph(node_features=jnp.asarray(node_features),
                      node_values=None,
                      steps=steps,
                      deadline=[int(dict_graph['mcsystem']['mcdag']['@deadline']) * nCores])

        graphs += [graph]

    train_set = graphs[:int(len(graphs) * split)]
    validate_set = graphs[int(len(graphs) * split):]

    return train_set, validate_set


def pad_steps(graphs: list):
    max_steps = 0
    for graph in graphs:
        if len(graph.steps) > max_steps:
            max_steps = len(graph.steps)
    for graph in graphs:
        while len(graph.steps) < max_steps:
            graph.steps.append(Step(jnp.asarray([len(graph.node_features) - 1], dtype=jnp.int32),
                                    jnp.asarray([len(graph.node_features) - 1], dtype=jnp.int32)))
    return graphs, max_steps


def batch(graphs: list, batch_size: int):
    batched_graphs = list()
    for batch in range(0, len(graphs), batch_size):
        next_batch = graphs[batch: batch + batch_size]
        next_batch, max_steps = pad_steps(next_batch)
        if len(next_batch) < batch_size:
            continue
        node_features = list()
        col_steps = list()
        deadlines = list()
        tasks = 0
        for graph in next_batch:
            tasks = len(graph.node_features)
            node_features.append(graph.node_features)
            col_steps.append(graph.steps)
            deadlines.append(graph.deadline)
        conc_nf = np.concatenate(node_features)
        conc_steps = list()
        for i in range(max_steps):
            senders = list()
            receivers = list()
            for j in range(len(col_steps)):
                offset = j * tasks
                senders.append(col_steps[j][i].sender + offset)
                receivers.append(col_steps[j][i].receiver + offset)
            conc_senders = np.concatenate(senders)
            conc_receivers = np.concatenate(receivers)
            step = Step(sender=conc_senders, receiver=conc_receivers)
            conc_steps.append(step)

        conc_steps = jax.tree_map(lambda x: np.expand_dims(x, 1), conc_steps)
        conc_steps = jax.tree_multimap(lambda *args: np.concatenate(args, 1).transpose(), *conc_steps)

        graph = Graph(node_features=conc_nf,
                      node_values=None,
                      steps=conc_steps,
                      deadline=deadlines)
        batched_graphs.append(graph)
    return batched_graphs
