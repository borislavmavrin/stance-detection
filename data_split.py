from utils import *

train_data_file = "trump_autolabelled.txt"
test_data_file = "SemEval2016-Task6-subtaskB-testdata-gold.txt"
defualt_data_file_enc = 'utf-8'



print("Loading train data ...")
tweets_train, targets_train, labels_train, ids_train = readTweetsOfficial(
    train_data_file,
    encoding=defualt_data_file_enc)
print(len(tweets_train))
print(len(tweets_train) * 0.1)
# tweet_tokens = tokenise_tweets(tweets)
# target_tokens = tokenise_tweets(targets)
print("Loading test data ...")
tweets_test, targets_test, labels_test, ids_test = readTweetsOfficial(
    test_data_file,
    encoding=defualt_data_file_enc)
print(len(tweets_test))


with open(train_data_file) as file_object:
    lines = file_object.readlines()

data_size = len(lines)
print("Data set size: " + str(data_size))
dev_idx = np.random.choice(data_size, 1876, replace=False)
test_idx = list(set(range(data_size)) - set(dev_idx))
print("Intersection of ttrain and dev data: " + str(len(set(dev_idx).intersection(set(test_idx)))))

lines_test = [lines[idx] for idx in test_idx]
lines_dev = [lines[idx] for idx in dev_idx]


with open("trump_autolabelled_train.txt", 'w') as file_object:
    file_object.writelines(lines_test)

with open("trump_autolabelled_dev.txt", 'w') as file_object:
    file_object.writelines(lines_dev)
