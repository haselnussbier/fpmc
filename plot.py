import csv
import os
import pickle
from datetime import datetime

import matplotlib
import pandas as pd
import yaml
from matplotlib import pyplot as plt

matplotlib.use("Agg")
DATA = list()
SCORES = list()
TIMESTAMP: str


def init_result():
    now = datetime.now()
    timestamp = '{:%y.%m.%d-%H:%M:%S}'.format(now)
    global TIMESTAMP
    TIMESTAMP = timestamp
    os.mkdir("results/" + TIMESTAMP)


def append_data(data):
    global DATA
    new = DATA
    new.append(data)
    DATA = new


def append_score(name, u, p, s):
    global SCORES
    new = SCORES
    new.append([name, u, p, s])
    SCORES = new


def write_csv():
    global TIMESTAMP
    with open("results/" + TIMESTAMP + "/training.csv", "w", newline='') as f:
        writer = csv.writer(f)
        global DATA
        writer.writerow(['step', 'loss', 'util', 'prob'])
        for row in DATA:
            writer.writerow(row)


def plot():
    global TIMESTAMP
    # TIMESTAMP = "16.11-12:47"
    train = pd.read_csv("results/" + TIMESTAMP + "/training.csv")

    fig, ax = plt.subplots()
    ax.plot('step', 'loss', data=train, color='red', label='loss')
    ax.plot('step', 'util', data=train, color='blue', label='util')
    ax.plot('step', 'prob', data=train, color='green', label='prob')
    plt.legend(loc='best', title='legend', frameon=False)
    fig.savefig("results/" + TIMESTAMP + "/plot.png")
    # plot_loss = p9.ggplot(train) + p9.aes('step', ['loss', 'util', 'p']) + p9.geom_line()
    # plot_loss.save(filename="plot-"+TIMESTAMP+".pdf", path="model/results/")


def save_config(config: dict):
    global TIMESTAMP
    with open("results/" + TIMESTAMP + "/config.yaml", "w") as f:
        yaml.dump(config['model'], f)

def write_score():
    global TIMESTAMP
    with open("results/" + TIMESTAMP + "/scores.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'utilization', 'probability', 'score'])
        global SCORES
        for row in SCORES:
            writer.writerow(row)


def save_graph(file):
    cmd = "cp " + file + " results/" + TIMESTAMP
    os.system(cmd)

    
def save_model(state):
    # trained_params = jax.experimental.optimizers.unpack_optimizer_state(state)
    # pickle.dump(trained_params, open("results/" + TIMESTAMP + "/params.pkl", "wb"))
    return

