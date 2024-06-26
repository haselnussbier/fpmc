import pickle
import jax.numpy as jnp
from numpy import random
from plot import append_score
import time

def random_factor(validate_set, config):
    start = time.time()
    for set in validate_set:

        random_fac = random.uniform(low=2/3, high=1, size=jnp.shape(set.node_features[:, 2]))
        while 1 in random_fac:
            random_fac = random.uniform(low=2/3, high=1, size=jnp.shape(set.node_features[:, 2]))

        # get node features
        crit = jnp.expand_dims(set.node_features[:, 0], axis=1)
        wcets_lo = jnp.expand_dims(set.node_features[:, 1], axis=1)
        wcets_hi = jnp.expand_dims(set.node_features[:, 2], axis=1)
        acets = jnp.expand_dims(set.node_features[:, 3], axis=1)
        st_ds = jnp.expand_dims(set.node_features[:, 4], axis=1)

        random_fac = jnp.expand_dims(jnp.asarray(random_fac), axis=1)

        # calculate new wcets_lo
        wcets_lo_new = jnp.multiply(wcets_hi, random_fac)
        wcets_lo_new = jnp.where(crit == 0, wcets_lo, wcets_lo_new)
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

        ovr = jnp.subtract(jnp.asarray(set.deadline), jnp.sum(wcets_lo_new, axis=1))

        # utilization
        util = jnp.divide(jnp.add(s, ovr), jnp.asarray(set.deadline))

        # ----------------------------
        # Calculate p_task_overrun:

        n = jnp.asarray(jnp.divide(jnp.subtract(wcets_lo_new, acets), st_ds), dtype=jnp.int32)
        p_task = jnp.divide(1, jnp.add(1, jnp.power(n, 2)))
        # replace probability of task overrun for lc tasks with 0, so PI(1-p_taskoverrun) only multiplies hc tasks probability
        p_task = jnp.where(crit == 1, p_task, 0)
        p_full = jnp.subtract(1, jnp.product(jnp.subtract(1, p_task), axis=1))

        prob_clean = list()
        util_clean = list()
        for ut, prob in zip(util, p_full):
            if prob != 1 and ut > 0:
                util_clean.append(ut)
                prob_clean.append(prob)

        util = jnp.asarray(util_clean)
        p_full = jnp.asarray(prob_clean)

        #get average
        util = jnp.sum(util)/len(util)
        p_taskoverrun = jnp.sum(p_full) / len(p_full)
        end = time.time()
        u = round(util * 100, 2)
        p = round(p_taskoverrun * 100, 2)
        s = round(jnp.multiply(util, jnp.subtract(1, p_taskoverrun)), 5)

        print("*************************************************************")
        print("Fraction based method results (average over validation set batches).")
        print("Utilization: " + str(u) + "%.")
        print("Probability of task overrun: " + str(p) + "%.")
        print("Combined score (u*(1-p)): ", s)
        print("Time: ", end-start)
        print("*************************************************************")
        append_score("random_factor", u, p, s)


def base_score(validate_set, config):


    for set in validate_set:
        start = time.time()
        # get node features
        crit = jnp.expand_dims(set.node_features[:, 0], axis=1)
        crit = jnp.asarray(jnp.split(crit, config['model']['batch_size']))
        crit = jnp.delete(crit, -1, axis=1)
        wcets_lo = jnp.expand_dims(set.node_features[:, 1], axis=1)
        wcets_lo = jnp.asarray(jnp.split(wcets_lo, config['model']['batch_size']))
        wcets_lo = jnp.delete(wcets_lo, -1, axis=1)
        wcets_hi = jnp.expand_dims(set.node_features[:, 2], axis=1)
        wcets_hi = jnp.asarray(jnp.split(wcets_hi, config['model']['batch_size']))
        wcets_hi = jnp.delete(wcets_hi, -1, axis=1)
        acets = jnp.expand_dims(set.node_features[:, 3], axis=1)
        acets = jnp.asarray(jnp.split(acets, config['model']['batch_size']))
        acets = jnp.delete(acets, -1, axis=1)
        st_ds = jnp.expand_dims(set.node_features[:, 4], axis=1)
        st_ds = jnp.asarray(jnp.split(st_ds, config['model']['batch_size']))
        st_ds = jnp.delete(st_ds, -1, axis=1)

        # Calculate Utilization:

        # calculate difference between old and new wcets_lo_hc
        wcets_lo_hc_old = jnp.where(crit == 1, wcets_lo, 0)
        wcets_lo_hc_new = jnp.where(crit == 1, wcets_lo, 0)
        s = jnp.subtract(jnp.sum(wcets_lo_hc_old, axis=1), jnp.sum(wcets_lo_hc_new, axis=1))

        # calculate overall utilization

        ovr = jnp.subtract(jnp.asarray(set.deadline), jnp.sum(wcets_lo, axis=1))

        # utilization
        util = jnp.divide(jnp.add(s, ovr), jnp.asarray(set.deadline))

        # ----------------------------
        # Calculate p_task_overrun:

        n = jnp.asarray(jnp.divide(jnp.subtract(wcets_lo, acets), st_ds), dtype=jnp.int32)
        p_task = jnp.divide(1, jnp.add(1, jnp.power(n, 2)))
        # replace probability of task overrun for lc tasks with 0, so PI(1-p_taskoverrun) only multiplies hc tasks probability
        p_task = jnp.where(crit == 1, p_task, 0)
        p_full = jnp.subtract(1, jnp.product(jnp.subtract(1, p_task), axis=1))

        prob_clean = list()
        util_clean = list()
        for ut, prob in zip(util, p_full):
            if prob != 1 and ut > 0:
                util_clean.append(ut)
                prob_clean.append(prob)

        util = jnp.asarray(util_clean)
        p_full = jnp.asarray(prob_clean)

        # get average
        util = jnp.sum(util) / len(util)
        p_taskoverrun = jnp.sum(p_full) / len(p_full)
        end = time.time()
        u = round(util * 100, 2)
        p = round(p_taskoverrun * 100, 2)
        s = round(jnp.multiply(util, jnp.subtract(1, p_taskoverrun)), 5)

        print("*************************************************************")
        print("Unaltered results.")
        print("Utilization: " + str(u) + "%.")
        print("Probability of task overrun: " + str(p) + "%.")
        print("Combined score (u*(1-p)): ", s)
        print("Time: ", end-start)
        print("*************************************************************")
        append_score("base", u, p, s)


def same_n(validate_set, config):
    for set in validate_set:
        start = time.time()
        # get node features
        crit = jnp.expand_dims(set.node_features[:, 0], axis=1)
        crit = jnp.asarray(jnp.split(crit, config['model']['batch_size']))
        crit = jnp.delete(crit, -1, axis=1)
        wcets_lo = jnp.expand_dims(set.node_features[:, 1], axis=1)
        wcets_lo = jnp.asarray(jnp.split(wcets_lo, config['model']['batch_size']))
        wcets_lo = jnp.delete(wcets_lo, -1, axis=1)
        wcets_hi = jnp.expand_dims(set.node_features[:, 2], axis=1)
        wcets_hi = jnp.asarray(jnp.split(wcets_hi, config['model']['batch_size']))
        wcets_hi = jnp.delete(wcets_hi, -1, axis=1)
        acets = jnp.expand_dims(set.node_features[:, 3], axis=1)
        acets = jnp.asarray(jnp.split(acets, config['model']['batch_size']))
        acets = jnp.delete(acets, -1, axis=1)
        st_ds = jnp.expand_dims(set.node_features[:, 4], axis=1)
        st_ds = jnp.asarray(jnp.split(st_ds, config['model']['batch_size']))
        st_ds = jnp.delete(st_ds, -1, axis=1)

        util_all = list()
        p_all = list()
        score_all = list()

        for critg,wcets_log, wcets_hig, acetsg, st_dsg, deadlineg in zip(crit, wcets_lo, wcets_hi, acets, st_ds, set.deadline):
            best_score = 0
            best_util = 0
            best_p = 0
            for current_n in range(10):
                # calculate p with current n
                n = jnp.full_like(critg, current_n)
                p_task = jnp.divide(1, jnp.add(1, jnp.power(n, 2)))
                # replace probability of task overrun for lc tasks with 0, so PI(1-p_taskoverrun) only multiplies hc tasks probability
                p_task = jnp.where(critg == 1, p_task, 0)
                p_full = jnp.subtract(1, jnp.product(jnp.subtract(1, p_task), axis=0))

                #calculate new wcets based on n
                wcets_lo_new = jnp.add(acetsg, jnp.multiply(st_dsg, n))
                wcets_lo_new = jnp.where(critg==1, wcets_lo_new, wcets_log)


                # Calculate Utilization:

                # calculate difference between old and new wcets_lo_hc
                wcets_lo_hc_old = jnp.where(critg == 1, wcets_log, 0)
                wcets_lo_hc_new = jnp.where(critg == 1, wcets_lo_new, 0)
                s = jnp.subtract(jnp.sum(wcets_lo_hc_old, axis=0), jnp.sum(wcets_lo_hc_new, axis=0))

                # calculate overall utilization

                ovr = jnp.subtract(jnp.asarray(deadlineg), jnp.sum(wcets_lo_new, axis=0))

                # utilization
                util = jnp.divide(jnp.add(s, ovr), jnp.asarray(deadlineg))

                score = jnp.multiply(util, jnp.subtract(1, p_full))

                if score > best_score:
                    best_score = score
                    best_util = util
                    best_p = p_full
                    best_n = current_n

            if best_score != 0 and best_p != 1 and best_util > 0:
                score_all.append(best_score)
                p_all.append(best_p)
                util_all.append(best_util)
        util_all = jnp.asarray(util_all)
        p_all = jnp.asarray(p_all)
        score_all = jnp.asarray(score_all)
        util = jnp.sum(util_all) / len(util_all)
        p_taskoverrun = jnp.sum(p_all) / len(p_all)
        score = jnp.sum(score_all) / len(score_all)

        end = time.time()

        u = round(util * 100, 2)
        p = round(p_taskoverrun * 100, 2)
        s = round(score, 5)

        print("*************************************************************")
        print("DATE2021.")
        print("Utilization: " + str(u) + "%.")
        print("Probability of task overrun: " + str(p) + "%.")
        print("Combined score (u*(1-p)): ", s)
        print("Time: ", end-start)
        print("*************************************************************")
        append_score("date", u, p, s)
