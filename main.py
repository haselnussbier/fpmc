from model import *

#       0(HC)---
#               \
#                --- 2(LC)
#               /
#       1(HC)---

Node0 = Node(wcet=jnp.asarray([0]),
             features=jnp.asarray([
                 1,
                 2,
                 3,
             ])
             )

Node1 = Node(wcet=jnp.asarray([0]),
             features=jnp.asarray([
                 1,
                 4,
                 5,
             ])
             )

Node2 = Node(wcet=jnp.asarray([0]),
             features=jnp.asarray([
                 0,
                 6,
                 7,
             ])
             )

Step1 = Step(sender=0, receiver=2)
Step2 = Step(sender=1, receiver=2)

exampleGraph = Graph(nodes=[Node0, Node1, Node2], steps=[Step1, Step2])

model_config = ModelConfig(
    propagation_steps=2,
    num_hidden_size=8,
    num_hidden_neurons=2,
    num_hidden_layers=2
)
sample = exampleGraph

net, params = init_net(model_config=model_config, sample=sample)

trained_params = train_model(net=net,
                       params=params,
                       sample=exampleGraph,
                       num_steps=1000)

#optimal_wcets=predict_model()
