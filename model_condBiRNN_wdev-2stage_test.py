import tensorflow as tf
from utils import *
from bicond_tf10 import create_bicond_embeddings_reader
import time
import subprocess
from gensim.models import word2vec
import model_condBiRNN_wdev_stage_1
import model_condBiRNN_wdev_stage_2



if __name__ == "__main__":
    # empty logs
    bashCommand = "rm -R logs"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    config_stage_1 = model_condBiRNN_wdev_stage_1.Config()
    config_stage_1.debug = True
    config_stage_1.epochs = 1
    model_stage_1 = model_condBiRNN_wdev_stage_1.biRNN_model(config_stage_1)

    config_stage_2 = model_condBiRNN_wdev_stage_2.Config()
    config_stage_2.debug = True
    config_stage_2.epochs = 1
    model_stage_2 = model_condBiRNN_wdev_stage_2.biRNN_model(config_stage_2)


    init = tf.global_variables_initializer()
    with tf.Session() as session_tf:
        log_writer = tf.summary.FileWriter('./logs', session_tf.graph)
        session_tf.run(init)
        model_stage_1.train(session_tf)
        targets_stage_1 = model_condBiRNN_wdev_stage_1.encoded_targets_test
        tweets_stage_1 = model_condBiRNN_wdev_stage_1.encoded_tweets_test

        labels_stage_1 = model_condBiRNN_wdev_stage_1.predict(
            session_tf,
            targets_stage_1,
            tweets_stage_1)
        print(labels_stage_1)
