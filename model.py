from typing import *

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
import optax
import random

NodeValue = jnp.ndarray
NodeFeatures = jnp.ndarray
AnyNested = Union[Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]

IFn = Callable[[NodeFeatures], NodeValue]
CFn = Callable[[NodeValue, NodeValue], NodeValue]
AFn = Callable[[NodeValue, NodeFeatures], NodeValue]
RFn = Callable[[NodeValue], NodeValue]


class Graph(NamedTuple):
    node_features: NodeFeatures
    node_values: NodeValue
    steps: AnyNested


class Step(NamedTuple):
    sender: jnp.ndarray
    receiver: jnp.ndarray


def Init(i_fn: IFn):
    def _Init(graph: Graph):
        # initialise a nodes wcet by the node features
        node_features = graph.node_features
        """
        for node_features in node_features:
            new_wcet = jax.tree_map(lambda x: i_fn(x), node_features)
            new_wcets.append(new_wcet)
        """
        new_wcets = jax.tree_map(lambda x: i_fn(x), node_features)
        return graph._replace(node_values=jnp.asarray(new_wcets))

    return _Init

def Collect(c_fn: CFn):
    def _Collect(graph: Graph, step: Step):

        sender = step.sender
        receiver = step.receiver
        wcets = graph.node_values

        # collect wcet from incoming edges and combine with own
        inc_wcet = jax.tree_map(lambda x: x[sender], wcets)
        own_wcet = jax.tree_map(lambda x: x[receiver], wcets)
        new_wcet = jax.tree_multimap(lambda x, y: c_fn(x, y), inc_wcet, own_wcet)

        values_new = jax.tree_multimap(
            lambda x, y: jax.ops.index_update(x, step.receiver, y),
            graph.node_values,
            new_wcet,
        )

        return graph._replace(node_values=values_new)

    return _Collect

def Apply(a_fn: AFn):
    def _Apply(graph: Graph, step: Step):

        receiver = step.receiver
        wcets = graph.node_values
        nf = graph.node_features

        # calculate wcet based on collected wcet value and own task information
        wcet = jax.tree_map(lambda x: x[receiver], wcets)
        node_features = jax.tree_map(lambda x: x[receiver], nf)
        new_wcet = jax.tree_map(lambda x, y: a_fn(x, y), node_features, wcet)

        values_new = jax.tree_multimap(
            lambda x, y: jax.ops.index_update(x, step.receiver, y),
            graph.node_values,
            new_wcet,
        )

        return graph._replace(node_values=values_new)

    return _Apply

def Output(r_fn: RFn):
    def _Output(graph: Graph):
        # reduce wcets to singular value

        wcets = graph.node_values
        """
        for node_features in node_features:
            new_wcet = jax.tree_map(lambda x: r_fn(x), node_features)
            new_wcets.append(np.abs(new_wcet))
        """
        new_wcets = jax.tree_map(lambda x: r_fn(x), wcets)
        return graph._replace(node_values=new_wcets)


    return _Output


class ModelConfig(NamedTuple):
    propagation_steps: int
    num_hidden_layers: int
    num_hidden_neurons: int
    num_hidden_size: int


class ModelBase:
    def make_xfn(self, num_hidden_layers, num_hidden_neurons, num_hidden_size, name):
        layers = []
        for i in range(num_hidden_layers):
            layers += hk.Linear(num_hidden_neurons, name=name + '_linear_' + str(i)), jax.nn.tanh,
        layers += hk.Linear(num_hidden_size, name=name + '_linear_out'), jax.nn.tanh,

        return hk.Sequential(layers)

    def make_rfn(self, num_input_size, num_output_size, name):
        layers = []
        i = 0
        while num_input_size > num_output_size:
            layers += hk.Linear(num_input_size, name=name + '_linear_' + str(i)), jax.nn.tanh,
            num_input_size = int(num_input_size / 2)
            i += 1
        layers += hk.Linear(num_output_size, name=name + '_linear_out'), jax.nn.tanh,

        return hk.Sequential(layers)


class Model(ModelBase):
    def __init__(self, model_config):
        self.propagation_steps = model_config.propagation_steps
        self.num_hidden_layers = model_config.num_hidden_layers
        self.num_hidden_neurons = model_config.num_hidden_neurons
        self.num_hidden_size = model_config.num_hidden_size

        self.i = None
        self.c = None
        self.a = None
        self.r = None

    def i_fn(self, nf: NodeFeatures):
        if not self.i:
            self.i = self.make_xfn(
                self.num_hidden_layers,
                self.num_hidden_neurons,
                self.num_hidden_size,
                "i_fn"
            )

        return jax.tree_map(lambda x: self.i(x), nf)

    def c_fn(self, iv: NodeValue, cv: NodeValue):
        iv_and_cv = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=1), iv, cv
        )

        if not self.c:
            self.c = self.make_xfn(
                self.num_hidden_layers,
                self.num_hidden_neurons,
                self.num_hidden_size,
                "c_fn")

        return jax.tree_map(lambda x: self.c(x), iv_and_cv)

    def a_fn(self, nf: NodeFeatures, cv: NodeValue):
        if not self.a:
            self.a = self.make_xfn(
                self.num_hidden_layers,
                self.num_hidden_neurons,
                self.num_hidden_size,
                "a_fn")

        nf_and_cv = jnp.concatenate([nf, cv], axis=1)

        return jax.tree_map(lambda x: self.a(x), nf_and_cv)

    def r_fn(self, cv: NodeValue):
        if not self.r:
            self.r = self.make_rfn(
                self.num_hidden_size,
                1,
                "r_fn"
            )

        return jax.tree_map(lambda x: self.r(x), cv)

    def get_net_definition(self):
        def _get_net_definition(graph: Graph, debug_mode=True):
            init = Init(i_fn=self.i_fn)
            graph = init(graph)

            def f_scan(graph, step):
                collect = Collect(c_fn=self.c_fn)
                graph = collect(graph, step)

                apply = Apply(a_fn=self.a_fn)
                graph = apply(graph, step)

                return graph, step

            if debug_mode:
                n_steps = len(graph.steps)
                for i in range(n_steps):
                    step = graph.steps[i] # jax.tree_map(lambda x: x[i], graph.steps)
                    graph, _ = f_scan(graph, step)
            else:
                graph, extra = hk.scan(f_scan, graph, graph.steps)

            output = Output(r_fn=self.r_fn)
            graph = output(graph)
            out = graph.node_values

            return out

        return _get_net_definition


def init_net(model_config, sample):
    net = Model(model_config)
    net_def = net.get_net_definition()
    net = hk.without_apply_rng(hk.transform(net_def))
    params = net.init(jax.random.PRNGKey(42), sample)
    return net, params


def train_model(net, params, sample, num_steps):

    #@jax.jit
    def utilization(wcets_lo):
        #do magic

        utilization = jnp.sum(wcets_lo)/4000

        return utilization

    #@jax.jit
    def mode_switch_p(wcets_lo, wcets_hi):
        # do magic
        p_task = jnp.zeros_like(wcets_lo)
        i = 0

        for wcet_lo, wcet_hi in zip(wcets_lo, wcets_hi):
            acet = wcet_hi*random.uniform(0.2, 1/3)
            d = wcet_hi*random.uniform(0.1,0.2)
            n=(wcet_lo-acet)/d
            n=n.astype(int)
            p_task = p_task.at[i].set(1/(1+jnp.power(n, 2)))
            i = i + 1

        p_sys = 1 - jnp.prod(1-jnp.asarray(p_task))
        return p_sys
    #@jax.jit
    def prediction_loss(params, sample):
        # implement proper loss calculation
        wcets_d = net.apply(params, sample)
        wcets_hi = jnp.expand_dims(sample.node_features[:, 1], axis=1)
        wcets_d = 1 - wcets_d
        wcets_lo = jnp.multiply(wcets_d, wcets_hi)

        util = utilization(wcets_lo)
        p = mode_switch_p(wcets_lo, wcets_hi)
        loss = (1 - util*(1-p))

        return loss

    opt_init, opt_update = optax.adam(2e-4)
    opt_state = opt_init(params)

    #@jax.jit
    def update(params, opt_state, sample):
        g = jax.grad(prediction_loss)(params, sample)
        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    for step in range(num_steps):
        params, opt_state = update(params, opt_state, sample)
        loss = prediction_loss(params, sample)
        print("step: %d, loss: %f" % (step, loss))

    return params



