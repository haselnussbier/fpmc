from model import *
from generator import *
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

train_set, validate_set = generate_sets(nLevels=config['graphs']['levels'],
                                        nTasks=config['graphs']['tasks'],
                                        nDags=config['graphs']['dags'],
                                        nCores=config['graphs']['cores'],
                                        pEdge=config['graphs']['edge'],
                                        set_size=config['set'])

model_config = ModelConfig(
    propagation_steps=2,
    num_hidden_size=config['model']['hidden_size'],
    num_hidden_neurons=config['model']['neurons'],
    num_hidden_layers=config['model']['layers']
)
sample = train_set[0]

net, params = init_net(model_config=model_config, sample=sample)

trained_params = train_model(net=net,
                             params=params,
                             sample=sample,
                             num_steps=100)

optimal_wcets, utilization, p_task_overrun=predict_model(net, trained_params, sample)

print("*****************************************")
print("Optimal wcet's for the graph: ",optimal_wcets)
print("Have utilization of: ", utilization)
print("And a probability of task overrun of: ", p_task_overrun)
print("*****************************************")

