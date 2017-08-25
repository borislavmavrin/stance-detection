import tensorflow as tf
from utils import *
from bicond_tf10 import create_bicond_embeddings_reader
import time
import subprocess
from gensim.models import word2vec


class Config(object):
    """Holds model hyperparams and data information."""
    float_type_tf = tf.float64
    int_type_tf = tf.int32
    float_type_np = np.float64
    int_type_np = np.int32
    first_seq_len = 19
    second_seq_len = 19
    num_classes = 3
    rnn_dim = 100
    features_dim = 200
    learning_rate = 0.0001
    batch_size = 70
    epochs = 8  # 8 in sheffieldnlp (github), add reference
    keep_prob = 0.9
    debug = False
    train_data_file = "trump_autolabelled.txt"
    test_data_file = "SemEval2016-Task6-subtaskB-testdata-gold.txt"
    defualt_data_file_enc = 'utf-8'
    word2vec_file = "skip_nostop_single_100features_5minwords_5context_big"
    verbose = True
    use_pretrained_word2vec = True
    tf_saver_file = "./tf_saver_files/biRNN"


class biRNN_model():

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
        with tf.variable_scope("Target"):
            self.first_seq_placeholder = tf.placeholder(
                dtype=self.config.int_type_tf,
                shape=[None, self.config.first_seq_len],
                name="Text")
            self.first_seq_lens_placeholder = tf.placeholder(
                dtype=self.config.int_type_tf,
                shape=[None],
                name="Length")
        with tf.variable_scope("Tweet"):
            self.second_seq_placeholder = tf.placeholder(
                dtype=self.config.int_type_tf,
                shape=[None, self.config.second_seq_len],
                name="Text")
            self.second_seq_lens_placeholder = tf.placeholder(
                dtype=self.config.int_type_tf,
                shape=[None],
                name="Length")
        self.labels_placeholder = tf.placeholder(
            dtype=self.config.float_type_tf,
            shape=[None, self.config.num_classes],
            name="Stance")

    def add_embedding(self, inputs, reuse=True):
        """word2vec embedding pretrained or not."""
        with tf.variable_scope("Embedding"):
            if self.config.use_pretrained_word2vec:
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                    embedding_matrix = tf.get_variable('word2vec_emb', initializer=tf.constant(
                        self.pretrained_emb), trainable=True, dtype=self.config.float_type_tf,)
                else:
                    embedding_matrix = tf.get_variable('word2vec_emb', initializer=tf.constant(
                        self.pretrained_emb), trainable=True, dtype=self.config.float_type_tf,)
                embedded_inputs = tf.nn.embedding_lookup(
                    embedding_matrix, inputs)
            else:
                embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.config.rnn_dim], -0.1, 0.1, dtype=self.config.float_type_tf),
                                               name=emb_name, trainable=True, dtype=self.config.float_type_tf)
                embedded_inputs = tf.nn.embedding_lookup(
                    embedding_matrix, inputs)
        return embedded_inputs

    def add_first_seq_encoder(self, first_seq_emb):
        """Bidirectional RNN ecnoder for the first sequence."""
        with tf.variable_scope("Target_encoder"):
            with tf.variable_scope("Forward") as scope:
                cell_fw = tf.contrib.rnn.LSTMCell(
                    self.config.rnn_dim, state_is_tuple=True)
                cell_fw = tf.contrib.rnn.DropoutWrapper(
                    cell=cell_fw, output_keep_prob=self.keep_prob_placeholder)
                # outputs shape: [batch_size, max_time, cell.output_size]
                # last_states shape: [batch_size, cell.state_size]
                outputs_fw, last_state_fw = tf.nn.dynamic_rnn(
                    cell=cell_fw,
                    dtype=self.config.float_type_tf,
                    sequence_length=self.first_seq_lens_placeholder,
                    inputs=first_seq_emb)

            with tf.variable_scope("Backward") as scope:
                first_seq_emb_rev = tf.reverse(first_seq_emb, [1])  # reverse the sequence
                cell_bw = tf.contrib.rnn.LSTMCell(
                    self.config.rnn_dim, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.DropoutWrapper(
                    cell=cell_bw, output_keep_prob=self.keep_prob_placeholder)
                # outputs shape: [batch_size, max_time, cell.output_size]
                # last_states shape: [batch_size, cell.state_size]
                outputs_bw, last_state_bw = tf.nn.dynamic_rnn(
                    cell=cell_bw,
                    dtype=self.config.float_type_tf,
                    sequence_length=self.first_seq_lens_placeholder,
                    inputs=first_seq_emb_rev)
        # return outputs of LSTMs, to be fed into
        # create_bi_sequence_embedding_initialise()
        return last_state_fw, last_state_bw

    def add_second_seq_encoder(self, last_state_fw, last_state_bw, second_seq_emb):
        """Conditional (first seq) bidirectional RNN ecnoder for the second sequence."""
        with tf.variable_scope("Tweet_encoder"):
            with tf.variable_scope("Forward") as scope:
                cell_fw_cond = tf.contrib.rnn.LSTMCell(
                    self.config.rnn_dim, state_is_tuple=True)
                cell_fw_cond = tf.contrib.rnn.DropoutWrapper(
                    cell=cell_fw_cond, output_keep_prob=self.keep_prob_placeholder)

                # returning [batch_size, max_time, cell.output_size]
                outputs_fw_cond, last_state_fw_cond = tf.nn.dynamic_rnn(
                    cell=cell_fw_cond,
                    dtype=self.config.float_type_tf,
                    sequence_length=self.first_seq_lens_placeholder,
                    inputs=second_seq_emb,
                    initial_state=last_state_fw
                )

            with tf.variable_scope("Backward") as scope:
                second_seq_emb_rev = tf.reverse(second_seq_emb, [1])  # reverse the sequence
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.rnn_dim, state_is_tuple=True)
                cell_fw = tf.contrib.rnn.DropoutWrapper(
                    cell=cell_fw, output_keep_prob=self.keep_prob_placeholder)

                # outputs shape: [batch_size, max_time, cell.output_size]
                # last_states shape: [batch_size, cell.state_size]
                outputs_bw_cond, last_state_bw_cond = tf.nn.dynamic_rnn(
                    cell=cell_fw,
                    dtype=self.config.float_type_tf,
                    sequence_length=self.first_seq_lens_placeholder,
                    inputs=second_seq_emb_rev,
                    initial_state=last_state_bw
                )

            with tf.variable_scope("Witchcraft") as scope:
                # version 1 for getting last output
                #last_output_fw = tfutil.get_by_index(outputs_fw_cond, seq_lengths_cond)
                #last_output_bw = tfutil.get_by_index(outputs_bw_cond, seq_lengths_cond)

                # version 2 for getting last output, without slicing, see http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
                # input, seq_lengths, seq_dim, batch_dim=None, name=None
                # might be more efficient or not, but at least memory warning
                # disappears
                # slices of input are reversed on seq_dim, but only up to seq_lengths
                outputs_fw = tf.reverse_sequence(
                    outputs_fw_cond, self.second_seq_lens_placeholder, seq_dim=1, batch_dim=0)
                # [batch_size, max_time, cell.output_size]
                dim1fw, dim2fw, dim3fw = tf.unstack(tf.shape(outputs_fw))
                last_output_fw = tf.reshape(tf.slice(outputs_fw, [0, 0, 0], [
                                            dim1fw, 1, dim3fw]), [dim1fw, dim3fw])

                # slices of input are reversed on seq_dim, but only up to seq_lengths
                outputs_bw = tf.reverse_sequence(
                    outputs_bw_cond, self.second_seq_lens_placeholder, seq_dim=1, batch_dim=0)
                # [batch_size, max_time, cell.output_size]
                dim1bw, dim2bw, dim3bw = tf.unstack(tf.shape(outputs_bw))
                last_output_bw = tf.reshape(tf.slice(outputs_bw, [0, 0, 0], [
                                            dim1bw, 1, dim3bw]), [dim1bw, dim3bw])

                features = tf.concat([last_output_fw, last_output_bw], axis=1)
        return features

    def add_model(self):
        """The model."""
        """Depends on the sheffieldnlp (github) implementaion. DEPRICATED"""
        with tf.variable_scope("bidirectional_RNN_encoder"):
            features = create_bicond_embeddings_reader(
                self.first_seq_placeholder,
                self.first_seq_lens_placeholder,
                self.second_seq_placeholder,
                self.second_seq_lens_placeholder,
                self.config.rnn_dim,
                self.vocab_size,
                emb_matrix_init=self.pretrained_emb,
                keep_prob=self.keep_prob_placeholder
            )
        return features

    def add_projection(self, features):
        """Adds a projection layer."""
        # YOUR CODE HERE
        with tf.variable_scope("Projection"):
            # Change to Xavier init. Check the performance difference.
            W = tf.get_variable(
                'W',
                shape=[self.config.features_dim, self.config.num_classes],
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
                labels=self.labels_placeholder,
                logits=logits)
            # Switch to tf.reduce_mean(losses)
            loss = tf.reduce_sum(losses)
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
        self.config = config
        self.load_data(self.config.debug)
        self.add_placeholders()
        """Model."""
        first_seq_emb = self.add_embedding(self.first_seq_placeholder, reuse=None)
        second_seq_emb = self.add_embedding(self.second_seq_placeholder, reuse=True)
        last_state_fw, last_state_bw = self.add_first_seq_encoder(first_seq_emb)
        self.features = self.add_second_seq_encoder(last_state_fw, last_state_bw, second_seq_emb)

        self.logits = self.add_projection(self.features)
        self.loss = self.add_loss_op(self.logits)
        self.train_step = self.add_training_op(self.loss)
        self.probs = self.add_probs(self.logits)
        self.predicted_labels = self.add_predicted_labels(self.probs)

    def train(self, session_tf):
        total_t = 0
        best_loss = float("inf")
        self.saver = tf.train.Saver()
        for epoch in xrange(self.config.epochs):
            print(' ' + '=' * 23 + "[Epoch " + str(epoch + 1) + ']' + '=' * 24)
            for batch in xrange(self.num_batches):
                start_t = time.time()
                idx = np.random.choice(
                    self.train_data_size, self.config.batch_size)
                targets_batch = self.encoded_targets_train[idx]
                targets_lens = np.array(
                    self.config.first_seq_len *
                    np.ones(self.config.batch_size),
                    dtype=self.config.int_type_np)
                tweets_batch = self.encoded_tweets_train[idx]
                tweets_lens = np.array(
                    self.config.second_seq_len *
                    np.ones(self.config.batch_size),
                    dtype=self.config.int_type_np)
                labels_batch = self.encoded_labels_train[idx]

                _, loss_np = session_tf.run(
                    [self.train_step,
                     self.loss],
                    feed_dict={
                        self.first_seq_placeholder: targets_batch,
                        self.first_seq_lens_placeholder: targets_lens,
                        self.second_seq_placeholder: tweets_batch,
                        self.second_seq_lens_placeholder: tweets_lens,
                        self.labels_placeholder: labels_batch,
                        self.keep_prob_placeholder: self.config.keep_prob
                    }
                )

                total_t += time.time() - start_t
                if self.config.verbose and batch % 50 is 0:
                    print("[Epoch: " + str(epoch + 1) +
                          " Batch: " + '{:3}'.format(batch) + ']' +
                          " [Loss: " + '{:4.2f}'.format(loss_np) + ']' +
                          " [examples/sec: " +
                          '{:6}'.format(
                              int(50 * self.config.batch_size / total_t)) + ']'
                          )
                    total_t = 0

            if loss_np < best_loss:
                best_loss = loss_np
                best_loss_epoch = epoch
                self.saver.save(session_tf, self.config.tf_saver_file)
                self.saver.save(session_tf, './logs/biRNN')

    def predict(self, session_tf, targets, tweets):
        self.saver.restore(session_tf, self.config.tf_saver_file)
        keep_prob = 1.0
        targets = np.array(targets, dtype=self.config.int_type_np)
        targets_lens = np.array(
            self.config.first_seq_len * np.ones(targets.shape[0]),
            dtype=self.config.int_type_np
        )
        tweets = np.array(tweets, dtype=self.config.int_type_np)
        tweets_lens = np.array(
            self.config.second_seq_len * np.ones(tweets.shape[0]),
            dtype=self.config.int_type_np
        )

        predicted_labels = session_tf.run(
            self.predicted_labels,
            feed_dict={
                self.first_seq_placeholder: targets,
                self.first_seq_lens_placeholder: targets_lens,
                self.second_seq_placeholder: tweets,
                self.second_seq_lens_placeholder: tweets_lens,
                self.keep_prob_placeholder: keep_prob
            }
        )
        return predicted_labels

    def test_data_evaluation(self, session_tf):
        self.saver.restore(session_tf, self.config.tf_saver_file)
        keep_prob = 1.0
        targets_lens = np.array(
            self.config.first_seq_len * np.ones(self.test_data_size),
            dtype=self.config.int_type_np
        )
        tweets_lens = np.array(
            self.config.second_seq_len * np.ones(self.test_data_size),
            dtype=self.config.int_type_np
        )
        predicted_probs = session_tf.run(
            self.probs,
            feed_dict={
                self.first_seq_placeholder: self.encoded_targets_test,
                self.first_seq_lens_placeholder: targets_lens,
                self.second_seq_placeholder: self.encoded_tweets_test,
                self.second_seq_lens_placeholder: tweets_lens,
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
    # empty logs
    bashCommand = "rm -R logs"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    config = Config()
    model = biRNN_model(config)
    init = tf.global_variables_initializer()
    with tf.Session() as session_tf:
        log_writer = tf.summary.FileWriter('./logs', session_tf.graph)
        session_tf.run(init)
        model.train(session_tf)
        output = model.test_data_evaluation(session_tf)

    print(output)
