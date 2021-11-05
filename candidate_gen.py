import argparse
import numpy as np
import pickle

def str2bool(v):
    return v.lower() in ("true", "t", "1")


class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)

def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

parser = argparse.ArgumentParser(__doc__)
data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_name", str, "ml-1m", "Path to training data.")
data_g.add_arg("test_set_dir", str, "./bert_train/data/ml-1m-test.txt", "Path to test data.")
data_g.add_arg("vocab_path", str, "./bert_train/data/ml-1m2.0.2.vocab", "Vocabulary path.")
data_g.add_arg("save_dir", str, "./bert_train/data/", "Path to test data.")
args = parser.parse_args()

print("Generate candidates")
user_count = 0
input_ids = []
labels = []
f = open(args.test_set_dir, "r")
line = f.readline()
while line:
    parsed_line = line
    split_samples = parsed_line.split(";")
    tmp_ids = split_samples[1].split(',')
    input_ids.append([int(x) for x in tmp_ids])
    tmp_label = split_samples[5].split(',')
    labels = labels + [[int(x)] for x in tmp_label]
    user_count += 1
    line = f.readline()

input_ids = np.array(input_ids)
labels = np.array(labels)
print(user_count)
print(input_ids)
print(labels)

print('load vocab from :' + args.vocab_path)
with open(args.vocab_path, 'rb') as input_file:
    vocab = pickle.load(input_file)

keys = vocab.counter.keys()
values = vocab.counter.values()
ids = vocab.convert_tokens_to_ids(keys)
sum_value = np.sum([x for x in values])
probability = [value / sum_value for value in values]

candidates = []
for idx in range(len(input_ids)):
    rated = set(input_ids[idx])
    rated.add(0)
    rated.add(labels[idx][0])
    item_idx = [labels[idx][0]]
    if vocab is not None:
        while len(item_idx) < 101:
            sampled_ids = np.random.choice(ids, 101, replace=False, p=probability)
            sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
            item_idx.extend(sampled_ids[:])
        item_idx = item_idx[:101]
    candidates.append(item_idx)
# note that we always put the true item in the first position---[target, 100 * negative]
print(candidates)
print(len(candidates))
candidates_file_name = args.save_dir + args.data_name + '.candidate'
print('candidate file: ' + candidates_file_name)
with open(candidates_file_name, 'wb') as output_file:
    pickle.dump(candidates, output_file, protocol=2)
