import numpy as np
import tensorflow as tf

float_type_tf = tf.float32


def add_f(limit=None, reuse=None):
    if reuse:
        with tf.variable_scope("mLSTM", reuse=True):
            # tf.get_variable_scope().reuse_variables()
            W_e = tf.get_variable(
                name='W_mi'
                )
    else:
        with tf.variable_scope("mLSTM", reuse=None):
            # tf.get_variable_scope().reuse_variables()
            W_e = tf.get_variable(
                name='W_mi',
                initializer=tf.random_uniform_initializer(
                    -limit, limit,
                    dtype=float_type_tf
                    ),
                shape=(),
                trainable=True,
                dtype=float_type_tf
                )
    W_e += 10
    return W_e

a_1 = add_f(1)
a_2 = add_f(10, reuse=True)
a_3 = add_f(100, reuse=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result_1 = sess.run(a_1)
    result_2 = sess.run(a_2)
    result_3 = sess.run(a_3)
    a_1_name = sess.run(a_1.name)

print(result_1)
print(result_2)
print(result_3)
print(a_1)
print(a_2)
