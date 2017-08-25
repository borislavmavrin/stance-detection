from utils import *

train_data_file = "trump_autolabelled_train.txt"
dev_data_file = "trump_autolabelled_dev.txt"


with open(train_data_file) as file_object:
    lines = file_object.readlines()
lines_no_nl = map(lambda x: x.replace('\n', ''), lines)
lines_split = map(lambda x: x.split('\t'), lines_no_nl)
lines_stage2 = filter(lambda x: x[3] != 'NONE', lines_split)
lines_stage2_join = map(lambda x: '\t'.join(x), lines_stage2)
lines_stage2_out = map(lambda x: x + '\n', lines_stage2_join)

with open("trump_autolabelled_train_stage2.txt", 'w') as file_object:
    file_object.writelines(lines_stage2_out)


with open(dev_data_file) as file_object:
    lines = file_object.readlines()
lines_no_nl = map(lambda x: x.replace('\n', ''), lines)
lines_split = map(lambda x: x.split('\t'), lines_no_nl)
lines_stage2 = filter(lambda x: x[3] != 'NONE', lines_split)
lines_stage2_join = map(lambda x: '\t'.join(x), lines_stage2)
lines_stage2_out = map(lambda x: x + '\n', lines_stage2_join)

with open("trump_autolabelled_dev_stage2.txt", 'w') as file_object:
    file_object.writelines(lines_stage2_out)
