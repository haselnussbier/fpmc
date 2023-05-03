from model import *


#       0(HC)--->
#                 \
#                 2(LC)
#                 /
#       1(HC)--->

exampleGraph = Graph(node_features=jnp.asarray([jnp.asarray([1, 2, 3], dtype=jnp.float32),
                                                jnp.asarray([1, 4, 5], dtype=jnp.float32),
                                                jnp.asarray([0, 6, 7], dtype=jnp.float32)], dtype=jnp.float32),
                     node_values=jnp.asarray([jnp.asarray([1], dtype=jnp.float32),
                                              jnp.asarray([1], dtype=jnp.float32),
                                              jnp.asarray([0], dtype=jnp.float32)], dtype=jnp.float32),
                     steps=[Step(sender=jnp.asarray([0], dtype=jnp.int8), receiver=jnp.asarray([2], dtype=jnp.int8)),
                            Step(sender=jnp.asarray([1], dtype=jnp.int8), receiver=jnp.asarray([2], dtype=jnp.int8))])

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
                       num_steps=10)

# optimal_wcets=predict_model()

