import numpy as np
import tensorflow as tf

float_type_tf = tf.float64
rnn_dim = 2

source_enc = tf.constant([
    [[111, 112],
     [121, 122],
     [131, 132]],
    [[211, 212],
     [221, 222],
     [231, 232]]
], dtype=float_type_tf)

target_enc = tf.constant([
    [[111, 112],
     [121, 122],
     [131, 132]],
    [[211, 212],
     [221, 222],
     [231, 232]]
], dtype=float_type_tf)


W_e = tf.get_variable(
    name='W_e',
    initializer=tf.random_uniform_initializer(
        -0.1, 0.1,
        dtype=float_type_tf
        ),
    shape=[rnn_dim, 1],
    trainable=True,
    dtype=float_type_tf
    )
W_s = tf.get_variable(
    name='W_s',
    initializer=tf.random_uniform_initializer(
        -0.1, 0.1,
        dtype=float_type_tf
        ),
    shape=[rnn_dim, rnn_dim],
    trainable=True,
    dtype=float_type_tf
    )
W_t = tf.get_variable(
    name='W_t',
    initializer=tf.random_uniform_initializer(
        -0.1, 0.1,
        dtype=float_type_tf
        ),
    shape=[rnn_dim, rnn_dim],
    trainable=True,
    dtype=float_type_tf
    )
W_a = tf.get_variable(
    name='W_a',
    initializer=tf.random_uniform_initializer(
        -0.1, 0.1,
        dtype=float_type_tf
        ),
    shape=[rnn_dim, rnn_dim],
    trainable=True,
    dtype=float_type_tf
    )
V_a = tf.get_variable(
    name='V_a',
    initializer=tf.random_uniform_initializer(
        -0.1, 0.1,
        dtype=float_type_tf
        ),
    shape=[rnn_dim, rnn_dim],
    trainable=True,
    dtype=float_type_tf
    )

# unstack along time axis
source_list = tf.unstack(source_enc, axis=1)
target_list = tf.unstack(target_enc, axis=1)
h_a_prev = tf.zeros_like(source_list[0])
a = []
for h_k in target_list:
    e_k = []
    for h_j in source_list:
        e_kj = tf.matmul(
            tf.tanh(
                tf.matmul(h_j, W_s) +
                tf.matmul(h_k, W_t) +
                tf.matmul(h_a_prev, W_a)
            ),
            W_e
        )
        e_k.append(tf.squeeze(e_kj))

    e_k = tf.stack(e_k, axis=1)
    maxs = tf.reduce_max(e_k, axis=1, keep_dims=True)
    e_k_exped = tf.exp(e_k - maxs)
    alpha_k = e_k_exped / tf.reduce_sum(e_k_exped, axis=1, keep_dims=True)

    # # computing a_k
    a_k = 0
    for i, h_j in enumerate(source_list):
        a_k += tf.multiply(tf.reshape(alpha_k[:, i], [-1, 1]), h_j)
    h_a_prev = a_k + tf.tanh(tf.matmul(h_a_prev, V_a))
    a.append(a_k)
# a = tf.stack(a, axis=1)
a_last = a[-1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(a_last)
print(result)
