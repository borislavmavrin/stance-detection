from utils import *

test_data_file = "SemEval2016-Task6-subtaskB-testdata-gold.txt"


with open(test_data_file) as file_object:
    lines = file_object.readlines()
lines = lines[1:len(lines)]
lines_no_nl = map(lambda x: x.replace('\n', ''), lines)
lines_split = map(lambda x: x.split('\t'), lines_no_nl)
lines_stage2 = filter(lambda x: x[3] != 'NONE', lines_split)
lines_stage2_join = map(lambda x: '\t'.join(x), lines_stage2)
lines_stage2_out = map(lambda x: x + '\n', lines_stage2_join)

with open("SemEval2016-Task6-subtaskB-testdata-gold_stage_2.txt", 'w') as file_object:
    file_object.writelines(lines_stage2_out)
