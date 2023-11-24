from typing import *

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from plot import append_data, write_csv

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

    # additional information not included in calculations
    deadline: list
    leftover_time: list


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
        return graph._replace(node_values=jnp.asarray(new_wcets, dtype=jnp.float64))

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
                    step = graph.steps[i]  # jax.tree_map(lambda x: x[i], graph.steps)
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
    params = 0
    test = False
    if test:
        for i in range(100):
            print(i)
            params = net.init(jax.random.PRNGKey(i), sample)
            output = net.apply(params, sample)
            wcets_p = jnp.subtract(1, output)
            wcets_hi = jnp.expand_dims(sample.node_features[:, 1], axis=1)
            wcets_lo = jnp.asarray(jnp.multiply(wcets_p, wcets_hi), dtype=jnp.int32)
            print(wcets_lo)
    else:
        params = net.init(jax.random.PRNGKey(69), sample)
    return net, params


def train_model(net, params, train_set, validate_set, num_steps, learning_rate, batch_size):

    # @jax.jit
    def get_metrics(params, sample):
        # model returns values (ret) between -1 and 1
        # these values represent how the wcet_hi should change, wcet_lo = wcet_hi * wcet_p
        # return value < 1 -> wcet_p > 1 -> wcet_low increases by percentage abs(ret)
        # return value > 1 -> wcet_p < 1 -> wcet_low decreases by percentage ret
        output = net.apply(params, sample)
        wcets_p = jnp.subtract(1, output)

        # get given high-wcets of each task from graph
        wcets_hi = jnp.expand_dims(sample.node_features[:, 1], axis=1)
        # calculate new low-wcets for each task based on model returns
        wcets_lo = jnp.multiply(wcets_p, wcets_hi)
        # get given acet of each task from graph
        acets = jnp.expand_dims(sample.node_features[:, 2], axis=1)
        # get given standard deviation of each task from graph
        st_ds = jnp.expand_dims(sample.node_features[:, 3], axis=1)

        # split into respective graphs (unbatch)
        wcets_lo = jnp.asarray(jnp.split(wcets_lo, batch_size))
        acets = jnp.asarray(jnp.split(acets, batch_size))
        st_ds = jnp.asarray(jnp.split(st_ds, batch_size))

        # ----------------------------
        # Calculate Utilization:

        wcets_sum = jnp.sum(wcets_lo, axis=1)
        used_timeslots_old = jnp.subtract(jnp.multiply(jnp.asarray(sample.deadline), 2),
                                          jnp.asarray(sample.leftover_time))
        s = jnp.subtract(used_timeslots_old, wcets_sum)

        # get criticality of nodes
        crit = jnp.expand_dims(sample.node_features[:, 0], axis=1)
        crit = jnp.asarray(jnp.split(crit, batch_size))

        # get low criticality wcets
        wcets_lc = jnp.where(crit == 0, wcets_lo, 0)
        wcets_lc = jnp.sum(wcets_lc, axis=1)

        ovr = jnp.asarray(jnp.add(jnp.asarray(s), jnp.asarray(wcets_lc)))
        util = jnp.divide(ovr, jnp.asarray(sample.deadline))

        # util = jnp.divide(jnp.subtract(jnp.asarray(sample.leftover_time), leftover), jnp.asarray(sample.leftover_time))

        # ----------------------------
        # Calculate p_task_overrun:

        n = jnp.asarray(jnp.divide(jnp.subtract(wcets_lo, acets), st_ds), dtype=jnp.int32)
        p_task = jnp.divide(1, jnp.add(1, jnp.power(n, 2)))
        p = jnp.subtract(1, jnp.product(jnp.subtract(1, p_task), axis=1))

        return wcets_lo, util, p
    # @jax.jit
    def prediction_loss(params, sample):
        # model returns values (ret) between -1 and 1
        # these values represent how the wcet_hi should change, wcet_lo = wcet_hi * wcet_p
        # return value < 1 -> wcet_p > 1 -> wcet_low increases by percentage abs(ret)
        # return value > 1 -> wcet_p < 1 -> wcet_low decreases by percentage ret
        output = net.apply(params, sample)
        wcets_p = jnp.subtract(1, output)

        # get given high-wcets of each task from graph
        wcets_hi = jnp.expand_dims(sample.node_features[:, 1], axis=1)
        # calculate new low-wcets for each task based on model returns
        wcets_lo = jnp.multiply(wcets_p, wcets_hi)
        # get given acet of each task from graph
        acets = jnp.expand_dims(sample.node_features[:, 2], axis=1)
        # get given standard deviation of each task from graph
        st_ds = jnp.expand_dims(sample.node_features[:, 3], axis=1)

        # split into respective graphs (unbatch)
        wcets_lo = jnp.asarray(jnp.split(wcets_lo, batch_size))
        acets = jnp.asarray(jnp.split(acets, batch_size))
        st_ds = jnp.asarray(jnp.split(st_ds, batch_size))


        # ----------------------------
        # Calculate Utilization:

        wcets_sum = jnp.sum(wcets_lo, axis=1)
        used_timeslots_old = jnp.subtract(jnp.multiply(jnp.asarray(sample.deadline), 2), jnp.asarray(sample.leftover_time))
        s = jnp.subtract(used_timeslots_old, wcets_sum)

        # get criticality of nodes
        crit = jnp.expand_dims(sample.node_features[:, 0], axis=1)
        crit = jnp.asarray(jnp.split(crit, batch_size))

        # get low criticality wcets
        wcets_lc = jnp.where(crit==0, wcets_lo, 0)
        wcets_lc = jnp.sum(wcets_lc, axis=1)

        ovr = jnp.asarray(jnp.add(jnp.asarray(s), jnp.asarray(wcets_lc)))
        util = jnp.divide(ovr, jnp.asarray(sample.deadline))


        #util = jnp.divide(jnp.subtract(jnp.asarray(sample.leftover_time), leftover), jnp.asarray(sample.leftover_time))

        # ----------------------------
        # Calculate p_task_overrun:

        n = jnp.asarray(jnp.divide(jnp.subtract(wcets_lo, acets), st_ds), dtype=jnp.int32)
        p_task = jnp.divide(1, jnp.add(1, jnp.power(n, 2)))
        p = jnp.subtract(1, jnp.product(jnp.subtract(1, p_task), axis=1))
        # p = 1 - jnp.prod(1 - jnp.asarray(p_task))
        losses = jnp.subtract(1, jnp.multiply(util, jnp.subtract(1, p)))

        loss = jnp.divide(jnp.sum(losses),batch_size)
        return loss

    opt_init, opt_update = optax.adam(learning_rate)
    opt_state = opt_init(params)

    # @jax.jit
    def update(params, opt_state, sample):
        g = jax.grad(prediction_loss)(params, sample)
        updates, opt_state = opt_update(g, opt_state, params)
        a = optax.apply_updates(params, updates)
        return a, opt_state

    step = 0
    min_loss = 1
    params_best = 0
    steps_to_stop = num_steps
    while steps_to_stop > 0:
        loss_acc = 0
        for graph in train_set:
            params, opt_state = update(params, opt_state, graph)
            loss_acc += prediction_loss(params, graph)
        loss = loss_acc/len(train_set)
        print("step: %d, loss: %f, mode: train" % (step, loss))
        loss = prediction_loss(params, validate_set)
        print("step: %d, loss: %f, mode: validate" % (step, loss))
        _, util, p = get_metrics(params, validate_set)
        append_data([step, loss, util, p])
        step += 1
        if loss < min_loss:
            steps_to_stop = num_steps
            params_best = params
            min_loss = loss
            print("improved")
        else:
            steps_to_stop -= 1
            print("not improved, steps left: ", steps_to_stop)

    write_csv()
    return params_best


def predict_model(net, params, sample, batch_size):
    output = net.apply(params, sample)
    wcets_p = jnp.subtract(1, output)
    wcets_hi = jnp.expand_dims(sample.node_features[:, 1], axis=1)
    wcets_lo = jnp.multiply(wcets_p, wcets_hi)
    acets = jnp.expand_dims(sample.node_features[:, 2], axis=1)
    st_ds = jnp.expand_dims(sample.node_features[:, 3], axis=1)
    wcets_lo = jnp.asarray(jnp.split(wcets_lo, batch_size))
    acets = jnp.asarray(jnp.split(acets, batch_size))
    st_ds = jnp.asarray(jnp.split(st_ds, batch_size))
    wcets_sum = jnp.sum(wcets_lo, axis=1)
    used_timeslots_old = jnp.subtract(jnp.multiply(jnp.asarray(sample.deadline), 2), jnp.asarray(sample.leftover_time))
    s = jnp.subtract(used_timeslots_old, wcets_sum)
    crit = jnp.expand_dims(sample.node_features[:, 0], axis=1)
    crit = jnp.asarray(jnp.split(crit, batch_size))
    wcets_lc = jnp.where(crit == 0, wcets_lo, 0)
    wcets_lc = jnp.sum(wcets_lc, axis=1)
    ovr = jnp.asarray(jnp.add(jnp.asarray(s), jnp.asarray(wcets_lc)))
    util = jnp.divide(ovr, jnp.asarray(sample.deadline))
    n = jnp.asarray(jnp.divide(jnp.subtract(wcets_lo, acets), st_ds), dtype=jnp.int32)
    p_task = jnp.divide(1, jnp.add(1, jnp.power(n, 2)))
    p = jnp.subtract(1, jnp.product(jnp.subtract(1, p_task), axis=1))
    losses = jnp.subtract(1, jnp.multiply(util, jnp.subtract(1, p)))

    util = jnp.divide(jnp.sum(util), batch_size)
    p = jnp.divide(jnp.sum(p), batch_size)
    loss = jnp.divide(jnp.sum(losses), batch_size)

    return loss, util, p
