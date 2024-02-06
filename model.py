import random
from typing import *
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from plot import *

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
    deadline: AnyNested


class Step(NamedTuple):
    sender: jnp.ndarray
    receiver: jnp.ndarray


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
        return graph._replace(node_values=jnp.asarray(new_wcets, dtype=jnp.float32))

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
        def _get_net_definition(graph: Graph, debug_mode=False):
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
                    step = jax.tree_map(lambda x: x[i], graph.steps)
                    graph, _ = f_scan(graph, step)
            else:
                steps = graph.steps
                graph, extra = hk.scan(f_scan, graph, steps)

            output = Output(r_fn=self.r_fn)
            graph = output(graph)
            out = graph.node_values

            return out

        return _get_net_definition


def init_net(model_config, sample):
    net = Model(model_config)
    net_def = net.get_net_definition()
    net = hk.without_apply_rng(hk.transform(net_def))
    """
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
        
    """
    params = net.init(jax.random.PRNGKey(69), sample)
    return net, params


def train_model(net, params, train_set, validate_set, model_config):
    @jax.jit
    def get_metrics(params, sample):
        # model returns values (ret) between -1 and 1
        # these values represent how the wcet_hi should change, wcet_lo = wcet_hi * wcet_p
        # return value < 1 -> wcet_p > 1 -> wcet_low increases by percentage abs(ret)
        # return value > 1 -> wcet_p < 1 -> wcet_low decreases by percentage ret
        output = net.apply(params, sample)
        wcets_p = jnp.subtract(1, output)

        # get node features
        crit = jnp.expand_dims(sample.node_features[:, 0], axis=1)
        wcets_lo = jnp.expand_dims(sample.node_features[:, 1], axis=1)
        wcets_hi = jnp.expand_dims(sample.node_features[:, 2], axis=1)
        acets = jnp.expand_dims(sample.node_features[:, 3], axis=1)
        st_ds = jnp.expand_dims(sample.node_features[:, 4], axis=1)

        # calculate new wcets_lo
        wcets_lo_new = jnp.multiply(wcets_p, wcets_lo)
        # experimental:
        wcets_lo_new = jnp.where(crit == 0, wcets_lo, wcets_lo_new)

        # split into respective graphs (unbatch)
        crit = jnp.asarray(jnp.split(crit, model_config['batch_size']))
        crit = jnp.delete(crit, -1, axis=1)
        wcets_lo = jnp.asarray(jnp.split(wcets_lo, model_config['batch_size']))
        wcets_lo = jnp.delete(wcets_lo, -1, axis=1)
        wcets_hi = jnp.asarray(jnp.split(wcets_hi, model_config['batch_size']))
        wcets_hi = jnp.delete(wcets_hi, -1, axis=1)
        acets = jnp.asarray(jnp.split(acets, model_config['batch_size']))
        acets = jnp.delete(acets, -1, axis=1)
        st_ds = jnp.asarray(jnp.split(st_ds, model_config['batch_size']))
        st_ds = jnp.delete(st_ds, -1, axis=1)

        wcets_lo_new = jnp.asarray(jnp.split(wcets_lo_new, model_config['batch_size']))
        wcets_lo_new = jnp.delete(wcets_lo_new, -1, axis=1)

        # Calculate Utilization:

        # calculate difference between old and new wcets_lo_hc
        wcets_lo_hc_old = jnp.where(crit == 1, wcets_lo, 0)
        wcets_lo_hc_new = jnp.where(crit == 1, wcets_lo_new, 0)
        s = jnp.subtract(jnp.sum(wcets_lo_hc_old, axis=1), jnp.sum(wcets_lo_hc_new, axis=1))

        # calculate overall utilization

        ovr = jnp.subtract(jnp.asarray(sample.deadline), jnp.sum(wcets_lo_new, axis=1))

        # utilization
        util = jnp.divide(jnp.add(s, ovr), jnp.asarray(sample.deadline))

        # ----------------------------
        # Calculate p_task_overrun:

        n = jnp.asarray(jnp.divide(jnp.subtract(wcets_lo_new, acets), st_ds), dtype=jnp.int32)
        p_task = jnp.divide(1, jnp.add(1, jnp.power(n, 2)))
        # replace probability of task overrun for lc tasks with 0, so PI(1-p_taskoverrun) only multiplies hc tasks probability
        p_task = jnp.where(crit == 1, p_task, 0)
        p_full = jnp.subtract(1, jnp.product(jnp.subtract(1, p_task), axis=1))

        return jnp.divide(jnp.sum(util), model_config['batch_size']), jnp.divide(jnp.sum(p_full),
                                                                                 model_config['batch_size'])

    @jax.jit
    def prediction_loss(params, sample):
        # model returns values (ret) between -1 and 1
        # these values represent how the wcet_hi should change, wcet_lo = wcet_hi * wcet_p
        # return value < 1 -> wcet_p > 1 -> wcet_low increases by percentage abs(ret)
        # return value > 1 -> wcet_p < 1 -> wcet_low decreases by percentage ret
        output = net.apply(params, sample)
        wcets_p = jnp.subtract(1, output)

        # get node features
        crit = jnp.expand_dims(sample.node_features[:, 0], axis=1)
        wcets_lo = jnp.expand_dims(sample.node_features[:, 1], axis=1)
        wcets_hi = jnp.expand_dims(sample.node_features[:, 2], axis=1)
        acets = jnp.expand_dims(sample.node_features[:, 3], axis=1)
        st_ds = jnp.expand_dims(sample.node_features[:, 4], axis=1)

        # calculate new wcets_lo
        wcets_lo_new = jnp.multiply(wcets_p, wcets_lo)
        # experimental:
        wcets_lo_new = jnp.where(crit == 0, wcets_lo, wcets_lo_new)

        # split into respective graphs (unbatch)
        crit = jnp.asarray(jnp.split(crit, model_config['batch_size']))
        crit = jnp.delete(crit, -1, axis=1)
        wcets_lo = jnp.asarray(jnp.split(wcets_lo, model_config['batch_size']))
        wcets_lo = jnp.delete(wcets_lo, -1, axis=1)
        wcets_hi = jnp.asarray(jnp.split(wcets_hi, model_config['batch_size']))
        wcets_hi = jnp.delete(wcets_hi, -1, axis=1)
        acets = jnp.asarray(jnp.split(acets, model_config['batch_size']))
        acets = jnp.delete(acets, -1, axis=1)
        st_ds = jnp.asarray(jnp.split(st_ds, model_config['batch_size']))
        st_ds = jnp.delete(st_ds, -1, axis=1)

        wcets_lo_new = jnp.asarray(jnp.split(wcets_lo_new, model_config['batch_size']))
        wcets_lo_new = jnp.delete(wcets_lo_new, -1, axis=1)

        # Calculate Utilization:

        # calculate difference between old and new wcets_lo_hc
        wcets_lo_hc_old = jnp.where(crit == 1, wcets_lo, 0)
        wcets_lo_hc_new = jnp.where(crit == 1, wcets_lo_new, 0)
        s = jnp.subtract(jnp.sum(wcets_lo_hc_old, axis=1), jnp.sum(wcets_lo_hc_new, axis=1))

        # calculate overall utilization

        ovr = jnp.subtract(jnp.asarray(sample.deadline), jnp.sum(wcets_lo_new, axis=1))

        # utilization
        util = jnp.divide(jnp.add(s, ovr), jnp.asarray(sample.deadline))

        # ----------------------------
        # Calculate p_task_overrun:

        n = jnp.asarray(jnp.divide(jnp.subtract(wcets_lo_new, acets), st_ds), dtype=jnp.int32)
        p_task = jnp.divide(1, jnp.add(1, jnp.power(n, 2)))
        # replace probability of task overrun for lc tasks with 0, so PI(1-p_taskoverrun) only multiplies hc tasks probability
        p_task = jnp.where(crit == 1, p_task, 0)
        p_full = jnp.subtract(1, jnp.product(jnp.subtract(1, p_task), axis=1))
        losses = jnp.subtract(1, jnp.multiply(util, jnp.subtract(1, p_full)))

        mean_loss = jnp.divide(jnp.sum(losses), model_config['batch_size'])
        return mean_loss

    opt_init, opt_update = optax.adam(float(model_config['learning_rate']))
    opt_state = opt_init(params)

    @jax.jit
    def update(params, state, sample):
        g = jax.grad(prediction_loss)(params, sample)
        updates, state = opt_update(g, state, params)
        return optax.apply_updates(params, updates), state

    step = 0
    min_loss = 2
    params_best = params
    state_best = 0
    steps_to_stop = model_config['steps_to_stop']
    while steps_to_stop > 0:
        loss_acc = 0
        for graph in train_set:
            params, opt_state = update(params, opt_state, graph)
            loss_acc += prediction_loss(params, graph)
        loss = loss_acc / len(train_set)
        print("step: %d, loss: %f, mode: train" % (step, loss))
        loss = prediction_loss(params, validate_set)
        print("step: %d, loss: %f, mode: validate" % (step, loss))
        util, p = get_metrics(params, validate_set)
        append_data([step, loss, util, p])
        step += 1
        if loss < min_loss:
            steps_to_stop = model_config['steps_to_stop']
            params_best = params
            state_best = opt_state
            min_loss = loss
            print("improved")
        else:
            steps_to_stop -= 1
            print("not improved, steps left: ", steps_to_stop)

    write_csv()
    # save_model(state_best)

    return params_best


def predict_model(net, params, sample, model_config):
    # model returns values (ret) between -1 and 1
    # these values represent how the wcet_hi should change, wcet_lo = wcet_hi * wcet_p
    # return value < 1 -> wcet_p > 1 -> wcet_low increases by percentage abs(ret)
    # return value > 1 -> wcet_p < 1 -> wcet_low decreases by percentage ret
    output = net.apply(params, sample)
    wcets_p = jnp.subtract(1, output)

    # get node features
    crit = jnp.expand_dims(sample.node_features[:, 0], axis=1)
    wcets_lo = jnp.expand_dims(sample.node_features[:, 1], axis=1)
    wcets_hi = jnp.expand_dims(sample.node_features[:, 2], axis=1)
    acets = jnp.expand_dims(sample.node_features[:, 3], axis=1)
    st_ds = jnp.expand_dims(sample.node_features[:, 4], axis=1)

    # calculate new wcets_lo
    wcets_lo_new = jnp.multiply(wcets_p, wcets_lo)
    # experimental:
    wcets_lo_new = jnp.where(crit == 0, wcets_lo, wcets_lo_new)

    # split into respective graphs (unbatch)
    crit = jnp.asarray(jnp.split(crit, model_config['batch_size']))
    crit = jnp.delete(crit, -1, axis=1)
    wcets_lo = jnp.asarray(jnp.split(wcets_lo, model_config['batch_size']))
    wcets_lo = jnp.delete(wcets_lo, -1, axis=1)
    wcets_hi = jnp.asarray(jnp.split(wcets_hi, model_config['batch_size']))
    wcets_hi = jnp.delete(wcets_hi, -1, axis=1)
    acets = jnp.asarray(jnp.split(acets, model_config['batch_size']))
    acets = jnp.delete(acets, -1, axis=1)
    st_ds = jnp.asarray(jnp.split(st_ds, model_config['batch_size']))
    st_ds = jnp.delete(st_ds, -1, axis=1)

    wcets_lo_new = jnp.asarray(jnp.split(wcets_lo_new, model_config['batch_size']))
    wcets_lo_new = jnp.delete(wcets_lo_new, -1, axis=1)

    # Calculate Utilization:

    # calculate difference between old and new wcets_lo_hc
    wcets_lo_hc_old = jnp.where(crit == 1, wcets_lo, 0)
    wcets_lo_hc_new = jnp.where(crit == 1, wcets_lo_new, 0)
    s = jnp.subtract(jnp.sum(wcets_lo_hc_old, axis=1), jnp.sum(wcets_lo_hc_new, axis=1))

    # calculate overall utilization

    ovr = jnp.subtract(jnp.asarray(sample.deadline), jnp.sum(wcets_lo_new, axis=1))

    # utilization
    util = jnp.divide(jnp.add(s, ovr), jnp.asarray(sample.deadline))

    # ----------------------------
    # Calculate p_task_overrun:

    n = jnp.asarray(jnp.divide(jnp.subtract(wcets_lo_new, acets), st_ds), dtype=jnp.int32)
    p_task = jnp.divide(1, jnp.add(1, jnp.power(n, 2)))
    # replace probability of task overrun for lc tasks with 0, so PI(1-p_taskoverrun) only multiplies hc tasks probability
    p_task = jnp.where(crit == 1, p_task, 0)
    p_full = jnp.subtract(1, jnp.product(jnp.subtract(1, p_task), axis=1))
    losses = jnp.subtract(1, jnp.multiply(util, jnp.subtract(1, p_full)))

    loss = jnp.divide(jnp.sum(losses), model_config['batch_size'])

    return loss, jnp.divide(jnp.sum(util), model_config['batch_size']), jnp.divide(jnp.sum(p_full),
                                                                                   model_config['batch_size']), jnp.asarray(wcets_lo_new, dtype=jnp.int32)


def run(config):
    init_result()

    with open(config['file'], "rb") as f:
        graphs = pickle.load(f)

    if config['file'][7] == '1':
        train_set = batch(graphs, 1)
        validate_set = batch(graphs, 1)

    else:
        train_set = graphs[:0.8*len(graphs)]
        validate_set = graphs[0.8*len(graphs):]
        train_set = batch(train_set, config['model']['batch_size'])
        validate_set = batch(validate_set, config['model']['batch_size'])

    model_config = ModelConfig(
        num_hidden_size=config['model']['hidden_size'],
        num_hidden_neurons=config['model']['neurons'],
        num_hidden_layers=config['model']['layers']
    )

    net, params = init_net(model_config=model_config, sample=train_set[0])

    trained_params = train_model(net=net,
                                 params=params,
                                 train_set=train_set,
                                 validate_set=validate_set[0],
                                 model_config=config['model'])

    plot()

    loss, utilization, p_task_overrun, wcets = predict_model(net, trained_params, validate_set[0], config['model'])
    wcets_high_old = jnp.expand_dims(validate_set[0].node_features[:, 2], axis=1)
    wcets_high_old = jnp.delete(wcets_high_old, -1, axis=0)
    wcets = wcets[0]
    wcets = jnp.concatenate([wcets, wcets_high_old], axis=1)
    print("*************************************************************")
    print("Model results.")
    print("Utilization: " + str(round(utilization * 100, 2)) + "%.")
    print("Probability of task overrun: " + str(round(p_task_overrun * 100, 2)) + "%.")
    print("Combined score (u*(1-p)): ", round(utilization * (1 - p_task_overrun), 5))
    print("Starting worstcase execution times (low, high)")
    print(jnp.asarray(jnp.delete(validate_set[0].node_features[:, (1, 2)], -1, axis=0), dtype=jnp.int32))
    print("The new calculated worstcase execution times are:")
    print(jnp.asarray(wcets, dtype=jnp.int32))
    print("*************************************************************")

    save_config(config)

