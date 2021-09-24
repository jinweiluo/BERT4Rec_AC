# -*- coding: UTF-8 -*-
import os
import codecs
import collections
import random
import sys
import paddle
import six
from util import *
from vocab import *
import pickle
import multiprocessing
import time
import argparse
from args import ArgumentGroup, print_arguments

random_seed = 12345
short_seq_prob = 0  # Probability of creating sequences which are shorter than the maximum length。

parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")


model_g.add_arg(
    "pool_size", int, 10,
    "multiprocesses pool size.")

model_g.add_arg(
    "max_seq_length", int, 200,
    "max sequence length.")

model_g.add_arg(
    "max_predictions_per_seq", int, 20,
    "max_predictions_per_seq.")

model_g.add_arg(
    "masked_lm_prob", float, 0.2,  # 0.15 for ml-1m in the original github of Bert4rec
    "Masked LM probability.")

model_g.add_arg(
    "mask_prob", float, 1.0,
    "mask probabaility")

model_g.add_arg(
    "dupe_factor", int, 10,
    "Number of times to duplicate the input data (with different masks).")

model_g.add_arg("prop_sliding_window", float, 0.1, "sliding window step size.")
    
model_g.add_arg(
    "data_dir", str, './data/',
    "data dir.")

model_g.add_arg(
    "dataset_name", str, 'ml-1m',
    "dataset name.")

args = parser.parse_args()

def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, info, tokens, masked_lm_positions, masked_lm_labels):
        self.info = info  # info = [user]
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "info: %s\n" % (" ".join([printable_text(x) for x in self.info]))
        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()




def create_training_instances(output_filenamem,
                              all_documents_raw,
                              max_seq_length,
                              dupe_factor,
                              short_seq_prob,
                              masked_lm_prob,
                              max_predictions_per_seq,
                              rng,
                              vocab,
                              mask_prob,
                              prop_sliding_window,
                              pool_size,
                              force_last=False):
    """Create `TrainingInstance`s from raw text."""
    all_documents = {}
    # force_last when test
    if force_last:
        max_num_tokens = max_seq_length
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue
            all_documents[user] = [item_seq[-max_num_tokens:]]
    else:
        max_num_tokens = max_seq_length  # we need two sentence

        sliding_step = (int)(
            prop_sliding_window *
            max_num_tokens) if prop_sliding_window != -1.0 else max_num_tokens
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue

            #todo: add slide
            if len(item_seq) <= max_num_tokens:
                all_documents[user] = [item_seq]
            else:
                beg_idx = list(range(len(item_seq)-max_num_tokens, 0, -sliding_step))
                beg_idx.append(0)
                all_documents[user] = [item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]]

    instances = []
    if force_last:
        for user in all_documents:
            instances.extend(
                create_instances_from_document_test(
                    all_documents, user, max_seq_length))
        print("num of instance:{}".format(len(instances)))
        write_sample_data(vocab, instances, max_seq_length, max_predictions_per_seq, output_filenamem+ "-test" + ".txt")
    else:
        start_time = time.clock()
        pool = multiprocessing.Pool(processes=pool_size)
        instances = []
        print("document num: {}".format(len(all_documents)))

        def log_result(result):
            print("callback function result type: {}, size: {} ".format(type(result), len(result)))
            #instances.extend(result)

        for step in range(dupe_factor):
            pool.apply_async(
                create_instances_threading, args=(
                    all_documents, user, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab, random.Random(random.randint(1,10000)),
                    mask_prob, step, output_filenamem), callback=log_result)
        pool.close()
        pool.join()
        
        for user in all_documents:
            instances.extend(
                mask_last(
                    all_documents, user, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab, rng, output_filenamem))

        print("num of instance:{}; time:{}".format(len(instances), time.clock() - start_time))
    
    rng.shuffle(instances)
    return instances


def create_instances_threading(all_documents, user, max_seq_length, short_seq_prob,
                               masked_lm_prob, max_predictions_per_seq, vocab, rng,
                               mask_prob, step, output_filename):
    cnt = 0
    start_time = time.clock()
    instances = []
    for user in all_documents:
        cnt += 1
        if cnt % 1000 == 0:
            # print("step: {}, name: {}, step: {}, time: {}".format(step, multiprocessing.current_process().name, cnt, time.clock()-start_time))
            start_time = time.clock()
        instances.extend(create_instances_from_document_train(
            all_documents, user, max_seq_length, short_seq_prob,
            masked_lm_prob, max_predictions_per_seq, vocab, rng,
            mask_prob))
    write_sample_data(vocab, instances, max_seq_length, max_predictions_per_seq, output_filename + "_train_" +str(step) + ".txt")
    return instances


def mask_last(
        all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, rng, output_filename):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length
    
    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens
        
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
    write_sample_data(vocab, instances, max_seq_length, max_predictions_per_seq, output_filename + "_train_" +str(args.dupe_factor) + ".txt")
    return instances


def create_instances_from_document_test(all_documents, user, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length
    
    assert len(document) == 1 and len(document[0]) <= max_num_tokens
    
    tokens = document[0]
    assert len(tokens) >= 1

    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)

    info = [int(user.split("_")[1])]
    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)

    return [instance]


def create_instances_from_document_train(
        all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, rng, mask_prob):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]

    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens
        
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq,
             vocab_items, rng, mask_prob)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions_force_last(tokens):
    """Creates the predictions for the masked LM objective."""

    last_index = -1
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[PAD]" or token == '[NO_USE]':
            continue
        last_index = i

    assert last_index > 0

    output_tokens = list(tokens)
    output_tokens[last_index] = "[MASK]"

    masked_lm_positions = [last_index]
    masked_lm_labels = [tokens[last_index]]

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng,
                                 mask_prob):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token not in vocab_words:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < mask_prob:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                # masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                masked_token = rng.choice(vocab_words)  

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def gen_samples(data,
                output_filename,
                rng,
                vocab,
                max_seq_length,
                dupe_factor,
                short_seq_prob,
                mask_prob,
                masked_lm_prob,
                max_predictions_per_seq,
                prop_sliding_window,
                pool_size,
                force_last=False):
    # create train
    instances = create_training_instances(output_filename,
        data, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, rng, vocab, mask_prob, prop_sliding_window,
        pool_size, force_last)

    print("*** Writing to output files ***")
    print("  %s", output_filename)


def write_sample_data(vocab, instances, max_seq_length, max_predictions_per_seq, output_file):
    fw = open(output_file, "w")
    for (inst_index, instance) in enumerate(instances):
        try:
            input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        except:
            print(instance.tokens)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        input_pos = list(range(len(input_ids)))

        input_ids += [0] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_mask))
        input_pos += [0] * (max_seq_length - len(input_pos))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(input_pos) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        #masked_lm_weights = [1.0] * len(masked_lm_ids)

        #masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        #masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        #masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))
        
        fw.writelines(str(instance.info[0]) + ";")
        fw.writelines(str(input_ids).replace('[',' ').replace(']', '') + ";")
        fw.writelines(str(input_mask).replace('[',' ').replace(']', '') + ";")
        fw.writelines(str(input_pos).replace('[',' ').replace(']', '') + ";")
        fw.writelines(str(masked_lm_positions).replace('[',' ').replace(']', '') + ";")
        fw.writelines(str(masked_lm_ids).replace('[',' ').replace(']', '') + "\n")

    fw.close()

def main():
    max_seq_length = args.max_seq_length
    max_predictions_per_seq = args.max_predictions_per_seq
    masked_lm_prob = args.masked_lm_prob
    mask_prob = args.mask_prob
    dupe_factor = args.dupe_factor
    prop_sliding_window = args.prop_sliding_window
    pool_size = args.pool_size

    output_dir = args.data_dir
    dataset_name = args.dataset_name
    version_id = "2.0.2"
    print(version_id)

    if not os.path.isdir(output_dir):
        print(output_dir + ' is not exist')
        print(os.getcwd())
        exit(1)

    dataset = data_partition(output_dir+dataset_name+'.txt')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    cc = 0.0
    max_len = 0
    min_len = 100000
    for u in user_train:
        cc += len(user_train[u])
        max_len = max(len(user_train[u]), max_len)
        min_len = min(len(user_train[u]), min_len)

    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('max:{}, min:{}'.format(max_len, min_len))

    print('len_train:{}, len_valid:{}, len_test:{}, usernum:{}, itemnum:{}'.
        format(
        len(user_train),
        len(user_valid), len(user_test), usernum, itemnum))

    for idx, u in enumerate(user_train):
        if idx <= 1:
            print(user_train[u])
            print(user_valid[u])
            print(user_test[u])

    # put validate into train
    for u in user_train:
        if u in user_valid:
            user_train[u].extend(user_valid[u])

    # get the max index of the data
    user_train_data = {
        'user_' + str(k): ['item_' + str(item) for item in v]
        for k, v in user_train.items() if len(v) > 0
    }
    user_test_data = {
        'user_' + str(u):
            ['item_' + str(item) for item in (user_train[u] + user_test[u])]
        for u in user_train if len(user_train[u]) > 0 and len(user_test[u]) > 0
    }
    rng = random.Random(random_seed)

    vocab = FreqVocab(user_test_data)
    user_test_data_output = {
        k: [vocab.convert_tokens_to_ids(v)]
        for k, v in user_test_data.items()
    }

    print('begin to generate train')
    output_filename = output_dir + dataset_name
    gen_samples(
        user_train_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        prop_sliding_window,
        pool_size,
        force_last=False)
    print('train:{}'.format(output_filename))


    print('begin to generate test')
    output_filename = output_dir + dataset_name
    gen_samples(
        user_test_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        -1.0,
        pool_size,
        force_last=True)
    print('test:{}'.format(output_filename))

    train_total = open(output_dir + dataset_name+"-train.txt", 'a')
    # notice that the final train data contain the 10 cloze forms of data and one mask_last form of data
    # while the form of mask_last aims to narrow the gap of training and test
    for i in range(dupe_factor+1):
        f = open(output_dir + dataset_name + "_train_" + str(i) + ".txt")
        buf = f.read()
        f.close()
        os.remove(output_dir + dataset_name + "_train_" + str(i) + ".txt")
        train_total.write(buf)
    train_total.close()

    print('vocab_size:{}, user_size:{}, item_size:{}, item_with_other_size:{}'.
          format(vocab.get_vocab_size(),
                 vocab.get_user_count(),
                 vocab.get_item_count(),
                 vocab.get_item_count() + vocab.get_special_token_count()))
    vocab_file_name = output_dir + dataset_name + version_id + '.vocab'
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pickle.dump(vocab, output_file, protocol=2)

    his_file_name = output_dir + dataset_name + version_id + '.his'
    print('test data pickle file: ' + his_file_name)
    with open(his_file_name, 'wb') as output_file:
        pickle.dump(user_test_data_output, output_file, protocol=2)
    print('done.')


if __name__ == "__main__":
    main()

