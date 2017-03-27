
import tensorflow as tf

# https://github.com/tensorflow/tensorflow/blob/b07791f6e9b306937eb58f7bb6c3300cd26583af/tensorflow/python/training/optimizer.py
# https://github.com/tensorflow/tensorflow/blob/b07791f6e9b306937eb58f7bb6c3300cd26583af/tensorflow/python/training/adam.py

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops

class HyperSGD(optimizer.Optimizer):
    """
    Hyper SGD. Does gradient descent on learning rate as well.
    """
    def __init__(self, learning_rate, beta=None, use_locking=False,
                 name="HyperSGD"):
        super(HyperSGD, self).__init__(use_locking, name)
        self.lr = learning_rate
        self.beta = beta if beta is not None else learning_rate

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "g", self._name)
            self._get_or_make_slot(var, self.lr, "a", self._name)

    def _apply_dense(self, g_t, v):
        g = self.get_slot(v, "g")
        a = self.get_slot(v, "a")

        h_t = math_ops.reduce_sum((g_t * -g))  # lazy matmul..
        a_t = a - self.beta*h_t
        tf.summary.scalar('a_'+v.name, a)

        # v_t = v - a_t*g
        v_update = state_ops.assign_add(v, -a_t*g_t)
        g_update = state_ops.assign(g, g_t)
        a_update = state_ops.assign(a, a_t)

        return control_flow_ops.group(v_update, g_update, a_update)


class HyperAdam(optimizer.Optimizer):
    """Optimizer that implements the hyper Adam algorithm.

    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980) and
    [Baydin et al. 2017](https://arxiv.org/abs/1703.04782)).
    """
    def __init__(self, learning_rate=0.0001, beta=1e-7, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, use_locking=False, name="HyperAdam"):
        super(HyperAdam, self).__init__(use_locking, name)
        self.lr = learning_rate
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

    def _create_slots(self, var_list):
        # if (self.beta1_t is None or
        #     self.beta1_t.graph is not var_list[0].graph):
        with ops.colocate_with(var_list[0]):
            self.beta1_t = tf.Variable(self.beta1,
                                              name="beta1_t",
                                              trainable=False)
            self.beta2_t = tf.Variable(self.beta2,
                                              name="beta2_t",
                                              trainable=False)
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._get_or_make_slot(v, self.lr, "a", self._name)

    def _apply_dense(self, g_t, var):
        # 1st mom. estimate. m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, self.beta1 * m + (1 - self.beta1) * g_t,
                               use_locking=self._use_locking)

        # 2nd mom. estimate. v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_t = state_ops.assign(v, self.beta2 * v + (1 - self.beta2) * (g_t * g_t),
                               use_locking=self._use_locking)

        # # bias correction.
        # m_hat_t = m_t / (1 - beta1)
        # v_hat_t = v_t / (1 - beta2)
        # can rearrage if we assume that eps ~= 0. See Kingma et al. section 2.
        # (although, the impememntations are not equivalent...
        # this one seems to work.)
        moving_avg = math_ops.sqrt(1 - self.beta2_t) / (1 - self.beta1_t)

        # hypergradient
        # dot prod w/o worrying about shapes = sum(mul(x, y))
        h_t = moving_avg * tf.reduce_sum(g_t * (-m_t / (math_ops.sqrt(v_t) + self.eps)))

        # lr update. a_t = a - beta.h
        a = self.get_slot(var, 'a')
        a_t = state_ops.assign_sub(a, self.beta * h_t)
        tf.summary.scalar('a_'+var.name, a)

        # var update. -a * m_t / (sqrt(v_t) + eps)
        v_update = state_ops.assign_sub(var, a_t * moving_avg *
        m_t / (math_ops.sqrt(v_t) + self.eps),
                use_locking=self._use_locking)
        return control_flow_ops.group(*[v_update, m_t, v_t, a_t])

    def _apply_sparse(self, g_t, var):
        # TODO
        pass


    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
          with ops.colocate_with(self.beta1_t):
            update_beta1 = self.beta1_t.assign(self.beta1_t * self.beta1,
                use_locking=self._use_locking)
            update_beta2 = self.beta2_t.assign(self.beta2_t * self.beta2,
                use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                      name=name_scope)
