import jax.numpy as jnp
import random
from model import Step, Graph
import pickle

node_features = jnp.asarray([[1, 30, 30, 30 * random.uniform(0.2, 1 / 3), 30 * random.uniform(0.05, 0.1)],
                             [1, 60, 80, 80 * random.uniform(0.2, 1 / 3), 80 * random.uniform(0.05, 0.1)],
                             [1, 20, 50, 50 * random.uniform(0.2, 1 / 3), 50 * random.uniform(0.05, 0.1)],
                             [0, 20, 0, 0, 0],
                             [0, 30, 0, 0, 0],
                             [0, 60, 0, 0, 0],
                             [0, 20, 0, 0, 0],
                             [0, 20, 0, 0, 0],
                             [0, 0, 0, 0, 0]], dtype=jnp.float32)

steps = [Step(sender=jnp.asarray([0], dtype=jnp.int32),
              receiver=jnp.asarray([1], dtype=jnp.int32)),
         Step(sender=jnp.asarray([0], dtype=jnp.int32),
              receiver=jnp.asarray([3], dtype=jnp.int32)),
         Step(sender=jnp.asarray([1], dtype=jnp.int32),
              receiver=jnp.asarray([2], dtype=jnp.int32)),
         Step(sender=jnp.asarray([1], dtype=jnp.int32),
              receiver=jnp.asarray([3], dtype=jnp.int32)),
         Step(sender=jnp.asarray([2], dtype=jnp.int32),
              receiver=jnp.asarray([3], dtype=jnp.int32)),
         Step(sender=jnp.asarray([3], dtype=jnp.int32),
              receiver=jnp.asarray([4], dtype=jnp.int32)),
         Step(sender=jnp.asarray([6], dtype=jnp.int32),
              receiver=jnp.asarray([7], dtype=jnp.int32))]

graph = Graph(node_features=jnp.asarray(node_features),
                          node_values=None,
                          steps=steps,
                          deadline=[320])

with open('graphs/uav.pkl', 'wb') as f:
    pickle.dump([graph], f)

