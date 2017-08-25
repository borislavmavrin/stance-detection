import tensorflow as tf
from utils import *
import time
import subprocess
from gensim.models import word2vec


class Config(object):
    """Holds model hyperparams and data information."""
    float_type_tf = tf.float64
    int_type_tf = tf.int32
    float_type_np = np.float64
    int_type_np = np.int32
    source_len = 19
    target_len = 19
    num_classes = 3
    rnn_dim = 100
    feature_dim = 200
    learning_rate = 0.0001
    batch_size = 70
    epochs = 8  # 8 in sheffieldnlp (github), add reference
    keep_prob = 0.9
    debug = False
    train_data_file = "trump_autolabelled_train.txt"
    dev_data_file = "trump_autolabelled_dev.txt"
    test_data_file = "SemEval2016-Task6-subtaskB-testdata-gold.txt"
    defualt_data_file_enc = 'utf-8'
    word2vec_file = "skip_nostop_single_100features_5minwords_5context_big"
    verbose = True
    use_pretrained_word2vec = True
    tf_saver_folder = "./tf_saver_files/mLSTM/"
    tf_log_folder = "./logs/"
    model_name = "mLSTM_with_attention"


class mLSTM_model():

    def load_data(self, debug=False):
        """Loads starter word-vectors and train/dev/test data."""
        if self.config.use_pretrained_word2vec:
            print("Loading word2vec data ...")
            w2vmodel = word2vec.Word2Vec.load(self.config.word2vec_file)
            self.vocab = w2vmodel
            self.vocab_size = len(w2vmodel.vocab)
            self.pretrained_emb = w2vmodel.syn0.astype(
                dtype=self.config.float_type_np, copy=False)

        print("Loading train data ...")
        tweets, targets, labels, ids = readTweetsOfficial(
            self.config.train_data_file,
            encoding=self.config.defualt_data_file_enc)
        tweet_tokens = tokenise_tweets(tweets)
        target_tokens = tokenise_tweets(targets)
        transformed_tweets = [transform_tweet(
            w2vmodel, senttoks) for senttoks in tweet_tokens]
        transformed_targets = [transform_tweet(
            w2vmodel, senttoks) for senttoks in target_tokens]
        transformed_labels = transform_labels(labels)
        self.encoded_targets_train = np.array(
            transformed_targets, dtype=self.config.int_type_np)
        self.encoded_tweets_train = np.array(
            transformed_tweets, dtype=self.config.int_type_np)
        self.encoded_labels_train = np.array(
            transformed_labels, dtype=self.config.float_type_np)
        self.train_data_size = self.encoded_labels_train.shape[0]
        if debug:
            self.num_batches = 50
        else:
            self.num_batches = self.train_data_size // self.config.batch_size

        print("Loading dev data ...")
        tweets, targets, labels, ids = readTweetsOfficial(
            self.config.dev_data_file,
            encoding=self.config.defualt_data_file_enc)
        tweet_tokens = tokenise_tweets(tweets)
        target_tokens = tokenise_tweets(targets)
        transformed_tweets = [transform_tweet(
            w2vmodel, senttoks) for senttoks in tweet_tokens]
        transformed_targets = [transform_tweet(
            w2vmodel, senttoks) for senttoks in target_tokens]
        transformed_labels = transform_labels(labels)
        self.encoded_targets_dev = np.array(
            transformed_targets, dtype=self.config.int_type_np)
        self.encoded_tweets_dev = np.array(
            transformed_tweets, dtype=self.config.int_type_np)
        self.encoded_labels_dev = np.array(
            transformed_labels, dtype=self.config.float_type_np)
        self.dev_data_size = self.encoded_labels_dev.shape[0]

        print("Loading test data ...")
        tweets, targets, labels, ids = readTweetsOfficial(
            self.config.test_data_file,
            encoding=self.config.defualt_data_file_enc)
        self.tweets_test = tweets
        self.targets_test = targets
        self.labels_test = labels
        self.ids_test = ids
        tweet_tokens = tokenise_tweets(tweets)
        target_tokens = tokenise_tweets(targets)
        transformed_tweets = [transform_tweet(
            w2vmodel, senttoks) for senttoks in tweet_tokens]
        transformed_targets = [transform_tweet(
            w2vmodel, senttoks) for senttoks in target_tokens]
        transformed_labels = transform_labels(labels)
        self.encoded_targets_test = np.array(
            transformed_targets, dtype=self.config.int_type_np)
        self.encoded_tweets_test = np.array(
            transformed_tweets, dtype=self.config.int_type_np)
        self.encoded_labels_test = np.array(
            transformed_labels, dtype=self.config.float_type_np)
        self.test_data_size = self.encoded_labels_test.shape[0]

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors."""
        self.keep_prob_placeholder = tf.placeholder(
            dtype=self.config.float_type_tf, shape=(),
            name="keep_prob")
        with tf.variable_scope("Source"):
            self.source_placeholder = tf.placeholder(
                dtype=self.config.int_type_tf,
                shape=[None, self.config.source_len],
                name="Text")
            self.source_len_placeholder = tf.placeholder(
                dtype=self.config.int_type_tf,
                shape=[None],
                name="Length")
        with tf.variable_scope("Target"):
            self.target_placeholder = tf.placeholder(
                dtype=self.config.int_type_tf,
                shape=[None, self.config.target_len],
                name="Text")
            self.target_len_placeholder = tf.placeholder(
                dtype=self.config.int_type_tf,
                shape=[None],
                name="Length")
        self.label_placeholder = tf.placeholder(
            dtype=self.config.float_type_tf,
            shape=[None, self.config.num_classes],
            name="Label")

    def add_embedding(self, inputs, reuse=None):
        """word2vec embedding pretrained or not."""
        with tf.variable_scope("Embedding"):
            if self.config.use_pretrained_word2vec:
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                    embedding_matrix = tf.get_variable(
                        name='word2vec_emb',
                        dtype=self.config.float_type_tf
                        )
                else:
                    embedding_matrix = tf.get_variable(
                        name='word2vec_emb',
                        initializer=tf.constant(self.pretrained_emb),
                        trainable=True,
                        dtype=self.config.float_type_tf
                        )
                embedded_inputs = tf.nn.embedding_lookup(
                    embedding_matrix,
                    inputs
                    )
            else:
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                    embedding_matrix = tf.get_variable(
                        name='random_init',
                        # initializer=tf.random_uniform_initializer(
                        #     -0.1, 0.1,
                        #     dtype=self.config.float_type_tf
                        #     ),
                        # shape=[self.config.vocab_size, self.config.emb_dim],
                        # trainable=True,
                        dtype=self.config.float_type_tf
                        )
                else:
                    embedding_matrix = tf.get_variable(
                        name='random_init',
                        initializer=tf.random_uniform_initializer(
                            -0.1, 0.1,
                            dtype=self.config.float_type_tf
                            ),
                        shape=[self.config.vocab_size, self.config.emb_dim],
                        trainable=True,
                        dtype=self.config.float_type_tf
                        )
                embedded_inputs = tf.nn.embedding_lookup(
                    embedding_matrix,
                    inputs
                    )
        return embedded_inputs

    def add_source_encoder(self, source_emb, source_len, initial_state=None, reuse=None):
        """LSTM source encoder."""
        with tf.variable_scope("Source_encoder"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                LSTMCell = tf.contrib.rnn.BasicLSTMCell(
                    self.config.rnn_dim, state_is_tuple=True
                    )
                LSTMCell = tf.contrib.rnn.DropoutWrapper(
                    cell=LSTMCell, output_keep_prob=self.keep_prob_placeholder)
            else:
                LSTMCell = tf.contrib.rnn.BasicLSTMCell(
                    self.config.rnn_dim, state_is_tuple=True
                    )
                LSTMCell = tf.contrib.rnn.DropoutWrapper(
                    cell=LSTMCell, output_keep_prob=self.keep_prob_placeholder)
            # outputs shape: [batch_size, max_time, cell.output_size]
            # last_states shape: [batch_size, cell.state_size]
            all_states, last_state = tf.nn.dynamic_rnn(
                cell=LSTMCell,
                inputs=source_emb,
                initial_state=initial_state,
                sequence_length=source_len,
                dtype=self.config.float_type_tf
                )
            # all_states are the h's
        return all_states, last_state

    def add_target_encoder(self, target_emb, target_len, initial_state=None, reuse=None):
        """LSTM target encoder."""
        with tf.variable_scope("Target_encoder"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                LSTMCell = tf.contrib.rnn.BasicLSTMCell(
                    self.config.rnn_dim, state_is_tuple=True
                    )
                LSTMCell = tf.contrib.rnn.DropoutWrapper(
                    cell=LSTMCell, output_keep_prob=self.keep_prob_placeholder)
            else:
                LSTMCell = tf.contrib.rnn.BasicLSTMCell(
                    self.config.rnn_dim, state_is_tuple=True
                    )
                LSTMCell = tf.contrib.rnn.DropoutWrapper(
                    cell=LSTMCell, output_keep_prob=self.keep_prob_placeholder)
            # outputs shape: [batch_size, max_time, cell.output_size]
            # last_states shape: [batch_size, cell.state_size]
            all_states, last_state = tf.nn.dynamic_rnn(
                cell=LSTMCell,
                inputs=target_emb,
                initial_state=initial_state,
                sequence_length=target_len,
                dtype=self.config.float_type_tf
                )
            # all_states are the h's
        return all_states, last_state

    def add_attention_vector(self, source_enc, target_enc):
        """Attention mechanism.

        Sticking to the notation in Learning Natural Language Inference with LSTM.
        """
        with tf.variable_scope("Attention_vector"):
            def mLSTM(a_k, h_k, h_m_prev, c_m_prev, resue=None):
                with tf.variable_scope("mLSTM"):
                    """Init variables."""
                    if reuse:
                        tf.get_variable_scope().reuse_variables()
                        W_mi = tf.get_variable(
                            name='W_mi',
                            dtype=self.config.float_type_tf
                            )
                        W_mf = tf.get_variable(
                            name='W_mf',
                            dtype=self.config.float_type_tf
                            )
                        W_mo = tf.get_variable(
                            name='W_mo',
                            dtype=self.config.float_type_tf
                            )
                        W_mc = tf.get_variable(
                            name='W_mc',
                            dtype=self.config.float_type_tf
                            )
                        V_mi = tf.get_variable(
                            name='V_mi',
                            dtype=self.config.float_type_tf
                            )
                        V_mf = tf.get_variable(
                            name='V_mf',
                            dtype=self.config.float_type_tf
                            )
                        V_mo = tf.get_variable(
                            name='V_mo',
                            dtype=self.config.float_type_tf
                            )
                        V_mc = tf.get_variable(
                            name='V_mc',
                            dtype=self.config.float_type_tf
                            )
                        b_mi = tf.get_variable(
                            name='b_mi',
                            dtype=self.config.float_type_tf
                            )
                        b_mf = tf.get_variable(
                            name='b_mf',
                            dtype=self.config.float_type_tf
                            )
                        b_mo = tf.get_variable(
                            name='b_mo',
                            dtype=self.config.float_type_tf
                            )
                        b_mc = tf.get_variable(
                            name='b_mc',
                            dtype=self.config.float_type_tf
                            )
                    else:
                        W_mi = tf.get_variable(
                            name='W_mi',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[2 * self.config.rnn_dim, self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
                            )
                        W_mf = tf.get_variable(
                            name='W_mf',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[2 * self.config.rnn_dim, self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
                            )
                        W_mo = tf.get_variable(
                            name='W_mo',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[2 * self.config.rnn_dim, self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
                            )
                        W_mc = tf.get_variable(
                            name='W_mc',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[2 * self.config.rnn_dim, self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
                            )
                        V_mi = tf.get_variable(
                            name='V_mi',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[self.config.rnn_dim, self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
                            )
                        V_mf = tf.get_variable(
                            name='V_mf',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[self.config.rnn_dim, self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
                            )
                        V_mo = tf.get_variable(
                            name='V_mo',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[self.config.rnn_dim, self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
                            )
                        V_mc = tf.get_variable(
                            name='V_mc',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[self.config.rnn_dim, self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
                            )
                        b_mi = tf.get_variable(
                            name='b_mi',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
                            )
                        b_mf = tf.get_variable(
                            name='b_mf',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
                            )
                        b_mo = tf.get_variable(
                            name='b_mo',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
                            )
                        b_mc = tf.get_variable(
                            name='b_mc',
                            initializer=tf.random_uniform_initializer(
                                -0.1, 0.1,
                                dtype=self.config.float_type_tf
                                ),
                            shape=[self.config.rnn_dim],
                            trainable=True,
                            dtype=self.config.float_type_tf
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

            W_e = tf.get_variable(
                name='W_e',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=self.config.float_type_tf
                    ),
                shape=[self.config.rnn_dim, 1],
                trainable=True,
                dtype=self.config.float_type_tf
                )
            W_s = tf.get_variable(
                name='W_s',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=self.config.float_type_tf
                    ),
                shape=[self.config.rnn_dim, self.config.rnn_dim],
                trainable=True,
                dtype=self.config.float_type_tf
                )
            W_t = tf.get_variable(
                name='W_t',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=self.config.float_type_tf
                    ),
                shape=[self.config.rnn_dim, self.config.rnn_dim],
                trainable=True,
                dtype=self.config.float_type_tf
                )
            W_a = tf.get_variable(
                name='W_a',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=self.config.float_type_tf
                    ),
                shape=[self.config.rnn_dim, self.config.rnn_dim],
                trainable=True,
                dtype=self.config.float_type_tf
                )
            V_a = tf.get_variable(
                name='V_a',
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1,
                    dtype=self.config.float_type_tf
                    ),
                shape=[self.config.rnn_dim, self.config.rnn_dim],
                trainable=True,
                dtype=self.config.float_type_tf
                )

            # unstack along time axis
            source_list = tf.unstack(source_enc, axis=1)
            source_list = [tf.nn.dropout(i, self.keep_prob_placeholder)
                           for i in source_list]
            target_list = tf.unstack(target_enc, axis=1)
            target_list = [tf.nn.dropout(i, self.keep_prob_placeholder)
                           for i in target_list]
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
                a_k = tf.nn.dropout(a_k, self.keep_prob_placeholder)
                h_m_prev, c_m_prev = mLSTM(a_k, h_k, h_m_prev, c_m_prev, reuse)
                a.append(a_k)
            # a = tf.stack(a, axis=1)
            return a[-1]

    def add_features(self, att_last, target_enc_last):
        features = tf.concat([att_last, target_enc_last], axis=1)
        return features

    def add_projection(self, features):
        """Adds a projection layer."""
        # YOUR CODE HERE
        with tf.variable_scope("Projection"):
            # Change to Xavier init. Check the performance difference.
            W = tf.get_variable(
                'W',
                shape=[self.config.feature_dim, self.config.num_classes],
                dtype=self.config.float_type_tf,
                initializer=tf.random_normal_initializer()
            )

            b = tf.get_variable(
                'b',
                shape=[self.config.num_classes],
                dtype=self.config.float_type_tf,
                initializer=tf.constant_initializer(0.0)
            )
            logits = tf.tanh(tf.matmul(features, W) + b)
        return logits

    def add_loss_op(self, logits):
        """Adds loss ops to the computational graph."""
        with tf.variable_scope("Loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.label_placeholder,
                logits=logits)
            # original paper used loss = tf.reduce_sum(losses)
            loss = tf.reduce_mean(losses)
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops."""
        with tf.variable_scope("Projection/Optmizer"):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            train_op = optimizer.minimize(loss)
        return train_op

    def add_probs(self, logits):
        probs = tf.nn.softmax(logits, name="Projection/Probability_distro")
        return probs

    def add_predicted_labels(self, probs):
        predicted_labels = tf.argmax(
            probs, axis=1, name="Projection/Predicted_label")
        return predicted_labels

    def __init__(self, config):
        # empty logs
        self.config = config
        with tf.variable_scope(self.config.model_name):
            self.load_data(self.config.debug)
            self.add_placeholders()
            self.source_emb = self.add_embedding(self.source_placeholder, reuse=None)
            self.target_emb = self.add_embedding(self.target_placeholder, reuse=True)
            self.source_enc, self.source_last_state = self.add_source_encoder(
                self.source_emb,
                self.source_len_placeholder
                )
            self.target_enc, self.target_last_state = self.add_target_encoder(
                self.target_emb,
                self.target_len_placeholder,
                initial_state=None
                )
            self.att_last = self.add_attention_vector(self.source_enc, self.target_enc)
            self.target_enc_last = tf.unstack(self.target_enc, axis=1)[-1]
            self.features = self.add_features(self.att_last, self.target_enc_last)
            self.logits = self.add_projection(self.features)
            self.loss = self.add_loss_op(self.logits)
            self.train_step = self.add_training_op(self.loss)

            self.probs = self.add_probs(self.logits)
            self.predicted_labels = self.add_predicted_labels(self.probs)
            self.saver = tf.train.Saver()

    def train(self, session_tf):
        total_t = 0
        best_loss = float("inf")
        for epoch in xrange(self.config.epochs):
            print(' ' + '=' * 23 + "[Epoch " + str(epoch + 1) + ']' + '=' * 24)
            for batch in xrange(self.num_batches):
                start_t = time.time()
                idx = np.random.choice(
                    self.train_data_size, self.config.batch_size)
                targets_batch = self.encoded_targets_train[idx]
                targets_lens = np.array(
                    self.config.source_len *
                    np.ones(self.config.batch_size),
                    dtype=self.config.int_type_np)
                tweets_batch = self.encoded_tweets_train[idx]
                tweets_lens = np.array(
                    self.config.target_len *
                    np.ones(self.config.batch_size),
                    dtype=self.config.int_type_np)
                labels_batch = self.encoded_labels_train[idx]

                _, train_loss_np = session_tf.run(
                    [self.train_step,
                     self.loss],
                    feed_dict={
                        self.source_placeholder: targets_batch,
                        self.source_len_placeholder: targets_lens,
                        self.target_placeholder: tweets_batch,
                        self.target_len_placeholder: tweets_lens,
                        self.label_placeholder: labels_batch,
                        self.keep_prob_placeholder: self.config.keep_prob
                    }
                )

                total_t += time.time() - start_t
                if self.config.verbose and batch % 50 is 0:
                    print("[Epoch: " + str(epoch + 1) +
                          " Batch: " + '{:3}'.format(batch) + ']' +
                          " [Train loss (dropout): " + '{:4.2f}'.format(train_loss_np) + ']' +
                          " [examples/sec: " +
                          '{:6}'.format(
                              int(50 * self.config.batch_size / total_t)) + ']'
                          )
                    total_t = 0



            targets_lens = np.array(
                self.config.source_len *
                np.ones(self.train_data_size ),
                dtype=self.config.int_type_np)
            tweets_lens = np.array(
                self.config.target_len *
                np.ones(self.train_data_size),
                dtype=self.config.int_type_np)
            train_loss_wodrpt_np = session_tf.run(
                 self.loss,
                feed_dict={
                    self.source_placeholder: self.encoded_targets_train,
                    self.source_len_placeholder: targets_lens,
                    self.target_placeholder: self.encoded_tweets_train,
                    self.target_len_placeholder: tweets_lens,
                    self.label_placeholder: self.encoded_labels_train,
                    self.keep_prob_placeholder: 1.0
                }
            )
            targets_lens = np.array(
                self.config.source_len *
                np.ones(self.dev_data_size ),
                dtype=self.config.int_type_np)
            tweets_lens = np.array(
                self.config.target_len *
                np.ones(self.dev_data_size),
                dtype=self.config.int_type_np)
            dev_loss_np = session_tf.run(
                 self.loss,
                feed_dict={
                    self.source_placeholder: self.encoded_targets_dev,
                    self.source_len_placeholder: targets_lens,
                    self.target_placeholder: self.encoded_tweets_dev,
                    self.target_len_placeholder: tweets_lens,
                    self.label_placeholder: self.encoded_labels_dev,
                    self.keep_prob_placeholder: 1.0
                }
            )

            print("[Epoch: " + str(epoch + 1) +
                #   " [Train loss (dropout): " + '{:4.2f}'.format(train_loss_np) + ']' +
                  " [Tarin loss: " + '{:4.2f}'.format(train_loss_wodrpt_np) + ']' +
                  " [Dev loss: " + '{:4.2f}'.format(dev_loss_np) + ']'
                  )



            if dev_loss_np < best_loss:
                best_loss = train_loss_np
                best_loss_epoch = epoch
                self.saver.save(session_tf, self.config.tf_saver_folder +
                                self.config.model_name)
                # self.saver.save(session_tf, './logs/biRNN')

    def predict(self, session_tf, targets, tweets):
        self.saver.restore(session_tf, self.config.tf_saver_folder +
                           self.config.model_name)
        keep_prob = 1.0
        targets = np.array(targets, dtype=self.config.int_type_np)
        targets_lens = np.array(
            self.config.source_len * np.ones(targets.shape[0]),
            dtype=self.config.int_type_np
        )
        tweets = np.array(tweets, dtype=self.config.int_type_np)
        tweets_lens = np.array(
            self.config.target_len * np.ones(tweets.shape[0]),
            dtype=self.config.int_type_np
        )

        predicted_labels = session_tf.run(
            self.predicted_labels,
            feed_dict={
                self.source_placeholder: targets,
                self.source_len_placeholder: targets_lens,
                self.target_placeholder: tweets,
                self.target_len_placeholder: tweets_lens,
                self.keep_prob_placeholder: keep_prob
            }
        )
        return predicted_labels

    def test_data_evaluation(self, session_tf):
        self.saver.restore(session_tf, self.config.tf_saver_folder +
                           self.config.model_name)
        keep_prob = 1.0
        targets_lens = np.array(
            self.config.source_len * np.ones(self.test_data_size),
            dtype=self.config.int_type_np
        )
        tweets_lens = np.array(
            self.config.target_len * np.ones(self.test_data_size),
            dtype=self.config.int_type_np
        )
        predicted_probs = session_tf.run(
            self.probs,
            feed_dict={
                self.source_placeholder: self.encoded_targets_test,
                self.source_len_placeholder: targets_lens,
                self.target_placeholder: self.encoded_tweets_test,
                self.target_len_placeholder: tweets_lens,
                self.keep_prob_placeholder: keep_prob
            }
        )
        '''Postprocessing of predicted_labels in order to refine 'NONE'.
        If a tweet has one of the keywords for Trump, its predicted label is
        forced to be AGAINST/FAVOR acoording to the predicted distribution.'''
        predicted_labels = np.argmax(predicted_probs, axis=1)
        target_keywords = ['donald trump', 'trump', 'donald']
        for idx, tweet in enumerate(self.tweets_test):
            for key in target_keywords:
                if key.lower() in tweet.lower():
                    if predicted_probs[idx][1] > predicted_probs[idx][2]:
                        predicted_labels[idx] = 1
                    else:
                        predicted_labels[idx] = 2
                    break

        predictions_lines = []
        label_names = np.array(["NONE", "AGAINST", "FAVOR"])
        header = "ID\tTarget\tTweet\tStance\n"
        labels = label_names[predicted_labels]
        predictions_lines.append(header)
        for i in xrange(len(self.ids_test)):
            line = '\t'.join((
                str(self.ids_test[i][0].astype('int')),
                self.targets_test[i],
                self.tweets_test[i],
                labels[i],
                '\n'))
            predictions_lines.append(line)
        predictions_file = "predictions.txt"
        with open(predictions_file, 'w') as f:
            f.writelines(predictions_lines)
        bashCommand = "perl eval.pl SemEval2016-Task6-subtaskB-testdata-gold.txt predictions.txt"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        return output


if __name__ == "__main__":

    config = Config()
    model = mLSTM_model(config)

    init = tf.global_variables_initializer()
    with tf.Session() as session_tf:
        # graph for tensorboard
        log_writer = tf.summary.FileWriter(model.config.tf_log_folder,
                                           session_tf.graph)
        session_tf.run(init)
        model.train(session_tf)
        output = model.test_data_evaluation(session_tf)

    print(output)
