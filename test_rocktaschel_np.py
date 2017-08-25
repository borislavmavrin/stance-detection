import numpy as np
import tensorflow as tf

float_type_np = np.float64
rnn_dim = 2
time_steps = 3

source_enc = np.array([
    [[111.1, 112.1],
     [121.1, 122.1],
     [131.1, 132.1]],
    [[211.1, 212.1],
     [221.1, 222.1],
     [231.1, 232.1]]
], dtype=float_type_np)

target_enc = np.array([
    [[111, 112],
     [121, 122],
     [131, 132]],
    [[211, 212],
     [221, 222],
     [231, 232]]
], dtype=float_type_np)


W_e = np.random.uniform(low=-0.1, high=0.1, size=(rnn_dim, 1))
W_s = np.random.uniform(low=-0.1, high=0.1, size=(rnn_dim, rnn_dim))
W_t = np.random.uniform(low=-0.1, high=0.1, size=(rnn_dim, rnn_dim))
W_a = np.random.uniform(low=-0.1, high=0.1, size=(rnn_dim, rnn_dim))
V_a = np.random.uniform(low=-0.1, high=0.1, size=(rnn_dim, rnn_dim))

# unstack along time axis
source_list = np.split(source_enc, 3, axis=1)
target_list = np.split(target_enc, 3, axis=1)
h_a_prev = np.zeros_like(np.squeeze(source_list[0]))
a = []
for h_k in target_list:
    h_k = np.squeeze(h_k)
    e_k = []
    for h_j in source_list:
        h_j = np.squeeze(h_j)
        e_kj = np.matmul(
            np.tanh(
                np.matmul(h_j, W_s) +
                np.matmul(h_k, W_t) +
                np.matmul(h_a_prev, W_a)
            ),
            W_e
        )
        e_k.append(np.squeeze(e_kj))


    e_k = np.stack(e_k, axis=1)
    maxs = np.amax(e_k, axis=1, keepdims=True)
    e_k_exped = np.exp(e_k - maxs)
    alpha_k = e_k_exped / np.sum(e_k_exped, axis=1, keepdims=True)
    # print(alpha_k.shape)

    # print(e_k.shape)

    # # computing a_k
    a_k = 0
    for j, h_j in enumerate(source_list):
        h_j = np.squeeze(h_j)
        alpha_k_j = alpha_k[:, j].reshape((-1, 1))
        a_k += alpha_k_j.dot(np.ones((1, rnn_dim))) * h_j
    h_a_prev = a_k + np.tanh(np.matmul(h_a_prev, V_a))
    a.append(a_k)
# a = tf.stack(a, axis=1)
a_last = a[-1]

print(a_last)
