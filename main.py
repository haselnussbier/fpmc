from model import *
import xmltodict

def build_levels(ports: Dict):
    dst = list()
    src = list()
    i=0
    for port in ports:
        if port['@dstActor'] not in dst:
            dst.append(port['@dstActor'])
        if port['@srcActor'] not in src:
            src.append(port['@srcActor'])
        i=i+1

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
                i=0
                for edge2 in edges:
                    if edge2['@dstActor'] == new_node:
                        i = i + 1
                if i == 1:
                    starter.append(new_node)
            else:
                new_edges.append(edge)
        edges = new_edges

    return order

dict = xmltodict.parse(open("graphs/Benchmarks/2&4Cores/c4/100/test-2.0-0.xml").read())

order = build_levels(dict['mcsystem']['mcdag']['ports']['port'])
node_features = np.zeros(shape=(len(dict['mcsystem']['mcdag']['actor']), 2))

for actor in dict['mcsystem']['mcdag']['actor']:
    if actor['wcet'][0]['#text'] == actor['wcet'][1]['#text']:
        node_features[order.index(actor['@name'])] = jnp.asarray([[0, float(actor['wcet'][1]['#text'])]])
        #node_features = jnp.append(node_features, jnp.asarray([[1, float(actor['maxpow'][0]['@number']), float(actor['maxpow'][1]['@number'])]], dtype=jnp.float32), axis=0)
    else:
        node_features[order.index(actor['@name'])] = jnp.asarray([[1, float(actor['wcet'][1]['#text'])]])
        #node_features = jnp.append(node_features, jnp.asarray([[0, float(actor['maxpow'][0]['@number']), float(actor['maxpow'][1]['@number'])]], dtype=jnp.float32), axis=0)

    #node_values[order.index(actor['@name'])] = jnp.asarray([[float(actor['wcet'][0]['#text']), float(actor['wcet'][1]['#text'])]])
    #node_values = jnp.append(node_values, jnp.asarray([[float(actor['wcet'][0]['#text']), float(actor['wcet'][1]['#text'])]], dtype=jnp.float32), axis=0)

steps = list()
for node in order:
    for edge in dict['mcsystem']['mcdag']['ports']['port']:
        if edge['@srcActor'] == node:
            steps.append(Step(sender=jnp.asarray([order.index(node)], dtype=jnp.int8), receiver=jnp.asarray([order.index(edge['@dstActor'])], dtype=jnp.int8)))

example_graph = Graph(node_features=jnp.asarray(node_features, dtype=jnp.float32),
                      node_values=None,
                      steps=steps)

example_graph2 = Graph(node_features=jnp.asarray([[1, 100], [0, 40], [0, 50]], dtype=jnp.float32),
                       node_values=None,
                       steps=[Step(sender=jnp.asarray([0], dtype=jnp.int8), receiver=jnp.asarray([1], dtype=jnp.int8)),
                              Step(sender=jnp.asarray([0], dtype=jnp.int8), receiver=jnp.asarray([2], dtype=jnp.int8))])




"""
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
"""
model_config = ModelConfig(
    propagation_steps=2,
    num_hidden_size=16,
    num_hidden_neurons=16,
    num_hidden_layers=8
)
sample = example_graph

net, params = init_net(model_config=model_config, sample=sample)

trained_params = train_model(net=net,
                       params=params,
                       sample=example_graph,
                       num_steps=10000)

# optimal_wcets=predict_model()


