import yaml

from generator import *
from model import *
from plot import plot


def pad_steps(graphs: list):
    max_steps = 0
    for graph in graphs:
        if len(graph.steps) > max_steps:
            max_steps = len(graph.steps)
    for graph in graphs:
        while len(graph.steps) < max_steps:
            graph.steps.append(Step(jnp.asarray([], dtype=jnp.int8), jnp.asarray([], dtype=jnp.int8)))
    return graphs

def batch(graphs: list, batch_size: int):
    graphs = pad_steps(graphs)
    batched_graphs = list()
    return batched_graphs


with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

train_set, validate_set = generate_sets(nLevels=config['graphs']['levels'],
                                        nTasks=config['graphs']['tasks'],
                                        nDags=config['graphs']['dags'],
                                        nCores=config['graphs']['cores'],
                                        pEdge=config['graphs']['edge'],
                                        set_size=config['set'])

model_config = ModelConfig(
    num_hidden_size=config['model']['hidden_size'],
    num_hidden_neurons=config['model']['neurons'],
    num_hidden_layers=config['model']['layers']
)
sample = train_set[0]

net, params = init_net(model_config=model_config, sample=sample)

batch_size=10

# train_set = batch(train_set, batch_size)

trained_params = train_model(net=net,
                             params=params,
                             sample=sample,
                             num_steps=config['steps_to_stop'],
                             learning_rate=float(config['learning_rate']))

plot()

optimal_wcets, utilization, p_task_overrun = predict_model(net, trained_params, sample)

print("*****************************************")
print("Optimal wcet's for the graph: ", optimal_wcets)
print("Have utilization of: ", utilization)
print("And a probability of task overrun of: ", p_task_overrun)
print("*****************************************")
