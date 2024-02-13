import pickle
import jax.numpy as jnp
from numpy import random

def random_factor(config):

    with open(config['file'], "rb") as f:
        graphs = pickle.load(f)

    for graph in graphs:
        wcets_hi = jnp.expand_dims(graph.node_features[:, 2], axis=1)
        random_fac = random.uniform(low=2/3, high=1, size=len(wcets_hi))
        while 1 in random_fac:
            random_fac = random.uniform(2 / 3, 1)



        # get node features
        crit = jnp.expand_dims(graph.node_features[:, 0], axis=1)
        wcets_lo = jnp.expand_dims(graph.node_features[:, 1], axis=1)
        wcets_hi = jnp.expand_dims(graph.node_features[:, 2], axis=1)
        acets = jnp.expand_dims(graph.node_features[:, 3], axis=1)
        st_ds = jnp.expand_dims(graph.node_features[:, 4], axis=1)

        # calculate new wcets_lo
        random_fac = jnp.expand_dims(jnp.asarray(random_fac), axis=1)
        wcets_lo_new = jnp.multiply(wcets_hi, random_fac)
        wcets_lo_new = jnp.where(wcets_lo_new == 0, wcets_lo, wcets_lo_new)
        crit = jnp.asarray(jnp.split(crit, config['model']['batch_size']))
        crit = jnp.delete(crit, -1, axis=1)
        wcets_lo = jnp.asarray(jnp.split(wcets_lo, config['model']['batch_size']))
        wcets_lo = jnp.delete(wcets_lo, -1, axis=1)
        wcets_hi = jnp.asarray(jnp.split(wcets_hi, config['model']['batch_size']))
        wcets_hi = jnp.delete(wcets_hi, -1, axis=1)
        acets = jnp.asarray(jnp.split(acets, config['model']['batch_size']))
        acets = jnp.delete(acets, -1, axis=1)
        st_ds = jnp.asarray(jnp.split(st_ds, config['model']['batch_size']))
        st_ds = jnp.delete(st_ds, -1, axis=1)

        wcets_lo_new = jnp.asarray(jnp.split(wcets_lo_new, config['model']['batch_size']))
        wcets_lo_new = jnp.delete(wcets_lo_new, -1, axis=1)

        # Calculate Utilization:

        # calculate difference between old and new wcets_lo_hc
        wcets_lo_hc_old = jnp.where(crit == 1, wcets_lo, 0)
        wcets_lo_hc_new = jnp.where(crit == 1, wcets_lo_new, 0)
        s = jnp.subtract(jnp.sum(wcets_lo_hc_old, axis=1), jnp.sum(wcets_lo_hc_new, axis=1))

        # calculate overall utilization

        ovr = jnp.subtract(jnp.asarray(graph.deadline), jnp.sum(wcets_lo_new, axis=1))

        # utilization
        util = jnp.divide(jnp.add(s, ovr), jnp.asarray(graph.deadline))

        # ----------------------------
        # Calculate p_task_overrun:

        n = jnp.asarray(jnp.divide(jnp.subtract(wcets_lo_new, acets), st_ds), dtype=jnp.int32)
        p_task = jnp.divide(1, jnp.add(1, jnp.power(n, 2)))
        # replace probability of task overrun for lc tasks with 0, so PI(1-p_taskoverrun) only multiplies hc tasks probability
        p_task = jnp.where(crit == 1, p_task, 0)
        p_full = jnp.subtract(1, jnp.product(jnp.subtract(1, p_task), axis=1))

        print("*************************************************************")
        print("Fraction based method results.")
        print("Utilization: " + str(round(util[0][0] * 100, 2)) + "%.")
        print("Probability of task overrun: " + str(round(p_full[0][0] * 100, 2)) + "%.")
        print("Combined score (u*(1-p)): ", round(jnp.multiply(util, jnp.subtract(1, p_full))[0][0], 5))
        print("*************************************************************")


def base_score(config):

    with open(config['file'], "rb") as f:
        graphs = pickle.load(f)

    for graph in graphs:

        # get node features
        crit = jnp.expand_dims(graph.node_features[:, 0], axis=1)
        crit = jnp.asarray(jnp.split(crit, config['model']['batch_size']))
        crit = jnp.delete(crit, -1, axis=1)
        wcets_lo = jnp.expand_dims(graph.node_features[:, 1], axis=1)
        wcets_lo = jnp.asarray(jnp.split(wcets_lo, config['model']['batch_size']))
        wcets_lo = jnp.delete(wcets_lo, -1, axis=1)
        wcets_hi = jnp.expand_dims(graph.node_features[:, 2], axis=1)
        wcets_hi = jnp.asarray(jnp.split(wcets_hi, config['model']['batch_size']))
        wcets_hi = jnp.delete(wcets_hi, -1, axis=1)
        acets = jnp.expand_dims(graph.node_features[:, 3], axis=1)
        acets = jnp.asarray(jnp.split(acets, config['model']['batch_size']))
        acets = jnp.delete(acets, -1, axis=1)
        st_ds = jnp.expand_dims(graph.node_features[:, 4], axis=1)
        st_ds = jnp.asarray(jnp.split(st_ds, config['model']['batch_size']))
        st_ds = jnp.delete(st_ds, -1, axis=1)

        # Calculate Utilization:

        # calculate difference between old and new wcets_lo_hc
        wcets_lo_hc_old = jnp.where(crit == 1, wcets_lo, 0)
        wcets_lo_hc_new = jnp.where(crit == 1, wcets_lo, 0)
        s = jnp.subtract(jnp.sum(wcets_lo_hc_old, axis=1), jnp.sum(wcets_lo_hc_new, axis=1))

        # calculate overall utilization

        ovr = jnp.subtract(jnp.asarray(graph.deadline), jnp.sum(wcets_lo, axis=1))

        # utilization
        util = jnp.divide(jnp.add(s, ovr), jnp.asarray(graph.deadline))

        # ----------------------------
        # Calculate p_task_overrun:

        n = jnp.asarray(jnp.divide(jnp.subtract(wcets_lo, acets), st_ds), dtype=jnp.int32)
        p_task = jnp.divide(1, jnp.add(1, jnp.power(n, 2)))
        # replace probability of task overrun for lc tasks with 0, so PI(1-p_taskoverrun) only multiplies hc tasks probability
        p_task = jnp.where(crit == 1, p_task, 0)
        p_full = jnp.subtract(1, jnp.product(jnp.subtract(1, p_task), axis=1))

        print("*************************************************************")
        print("Unaltered results.")
        print("Utilization: " + str(round(util[0][0] * 100, 2)) + "%.")
        print("Probability of task overrun: " + str(round(p_full[0][0] * 100, 2)) + "%.")
        print("Combined score (u*(1-p)): ", round(jnp.multiply(util, jnp.subtract(1, p_full))[0][0], 5))
        print("*************************************************************")

