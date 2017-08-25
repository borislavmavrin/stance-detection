import numpy as np
import tensorflow as tf


float_type_tf = tf.float64
rnn_dim = 2


'''Change bias init to constant.'''

def mLSTM(a_k, h_k, h_m_prev, c_m_prev, resue=None):
    with tf.variable_scope("mLSTM"):
        """Init variables."""
        if reuse:
            tf.get_variable_scope().reuse_variables()
            W_mi = tf.get_variable(
                name='W_mi',
                dtype=float_type_tf
                )
            W_mf = tf.get_variable(
                name='W_mf',
                dtype=float_type_tf
                )
            W_mo = tf.get_variable(
                name='W_mo',
                dtype=float_type_tf
                )
            W_mc = tf.get_variable(
                name='W_mc',
                dtype=float_type_tf
                )
            V_mi = tf.get_variable(
                name='V_mi',
                dtype=float_type_tf
                )
            V_mf = tf.get_variable(
                name='V_mf',
                dtype=float_type_tf
                )
            V_mo = tf.get_variable(
                name='V_mo',
                dtype=float_type_tf
                )
            V_mc = tf.get_variable(
                name='V_mc',
                dtype=float_type_tf
                )
            b_mi = tf.get_variable(
                name='b_mi',
                dtype=float_type_tf
                )
            b_mf = tf.get_variable(
                name='b_mf',
                dtype=float_type_tf
                )
            b_mo = tf.get_variable(
                name='b_mo',
                dtype=float_type_tf
                )
            b_mc = tf.get_variable(
                name='b_mc',
                dtype=float_type_tf
                )
        else:
            W_mi = tf.get_variable(
                name='W_mi',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[2 * rnn_dim, rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
            W_mf = tf.get_variable(
                name='W_mf',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[2 * rnn_dim, rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
            W_mo = tf.get_variable(
                name='W_mo',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[2 * rnn_dim, rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
            W_mc = tf.get_variable(
                name='W_mc',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[2 * rnn_dim, rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
            V_mi = tf.get_variable(
                name='V_mi',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[rnn_dim, rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
            V_mf = tf.get_variable(
                name='V_mf',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[rnn_dim, rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
            V_mo = tf.get_variable(
                name='V_mo',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[rnn_dim, rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
            V_mc = tf.get_variable(
                name='V_mc',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[rnn_dim, rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
            b_mi = tf.get_variable(
                name='b_mi',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
            b_mf = tf.get_variable(
                name='b_mf',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
            b_mo = tf.get_variable(
                name='b_mo',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
            b_mc = tf.get_variable(
                name='b_mc',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=float_type_tf
                    ),
                shape=[rnn_dim],
                trainable=True,
                dtype=float_type_tf
                )
        m_k = tf.concat([a_k, h_k], axis=1)
        i_mk = tf.sigmoid(
            tf.matmul(m_k, W_mi) +
            tf.matmul(h_m_prev, V_mi) +
            b_mi
        )
        f_mk = tf.sigmoid(
            tf.matmul(m_k, W_mf) +
            tf.matmul(h_m_prev, V_mf) +
            b_mf
        )
        o_mk = tf.sigmoid(
            tf.matmul(m_k, W_mo) +
            tf.matmul(h_m_prev, V_mo) +
            b_mo
        )
        c_m_next = (
            tf.multiply(f_mk, c_m_prev) +
            tf.multiply(
                i_mk,
                tf.tanh(
                    tf.matmul(m_k, W_mc) +
                    tf.matmul(h_m_prev, V_mc) +
                    b_mc
                )
            )
        )
        h_m_next = tf.multiply(
            o_mk,
            tf.tanh(c_m_next)
        )



    return h_m_next, c_m_next




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
h_m_prev = tf.zeros_like(source_list[0])
c_m_prev = tf.zeros_like(source_list[0])
a = []
for k, h_k in enumerate(target_list):
    e_k = []
    for h_j in source_list:
        e_kj = tf.matmul(
            tf.tanh(
                tf.matmul(h_j, W_s) +
                tf.matmul(h_k, W_t) +
                tf.matmul(h_m_prev, W_a)
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
    if k > 0:
        reuse = True
    else:
        reuse = None
    h_m_prev, c_m_prev = mLSTM(a_k, h_k, h_m_prev, c_m_prev, reuse)
    a.append(a_k)
# a = tf.stack(a, axis=1)
a_last = a[-1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(a_last)
print(result)
