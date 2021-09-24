#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT pretraining."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re

import paddle
import paddle.nn as nn
import paddle.fluid as fluid

import pickle
from bert4rec.dataset import DataReader
from bert4rec.bert4rec_ac import BertModel, BertConfig
from evaluate import *
from utils.args import ArgumentGroup, print_arguments

parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("bert_config_path", str, "./bert_train/bert_config_beauty_64.json",
                "Path to the json file for bert model config.")

train_g = ArgumentGroup(parser, "training", "training options")
train_g.add_arg("epoch", int, 300, "Number of epoch for training")
train_g.add_arg("learning_rate", float, 0.0001, "Learning rate")
train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
train_g.add_arg("num_train_steps", int, 800000, "Total steps to perform pretraining.")
train_g.add_arg("warmup_steps", int, 1000, "Total steps to perform warmup when pretraining.")
train_g.add_arg("validation_steps", int, 1000, "The steps interval to evaluate model performance.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_name", str, "beauty", "Path to training data.")
data_g.add_arg("data_dir", str, "./bert_train/data/beauty-train.txt", "Path to training data.")
data_g.add_arg("test_set_dir", str, "./bert_train/data/beauty-test.txt", "Path to test data.")
data_g.add_arg("vocab_path", str, "./bert_train/data/beauty2.0.2.vocab", "Vocabulary path.")
data_g.add_arg("candidate_path", str, "./bert_train/data/beauty.candidate", "candidate path.")
data_g.add_arg("max_seq_len", int, 50, "The maximum length of item sequence")
data_g.add_arg("batch_size", int, 256, "The batch size of data")
args = parser.parse_args()


def main(args):
    print("Start to train Bert4rec")
    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    BertRec = BertModel(config=bert_config)
    data_path = args.data_dir
    train_dataset = DataReader(
        data_path=data_path,
        batch_size=args.batch_size,
        max_len=args.max_seq_len,
    )

    train_loader = train_dataset.get_samples()

    val_dataset = DataReader(
        data_path=args.test_set_dir,
        batch_size=args.batch_size,
        max_len=args.max_seq_len,
    )
    val_loader = val_dataset.get_samples()

    # Loading candidate data
    print('load candidate from :' + args.candidate_path)
    with open(args.candidate_path, 'rb') as input_file:
        candidate = pickle.load(input_file)

    epochs = args.epoch
    total_steps = 0
    def apply_decay_param(param_name):
        for r in ["layer_norm", "b_0"]:
            if re.search(r, param_name) is not None:
                return False
        return True

    for epoch in range(epochs):
        cand_list = candidate
        BertRec.train()
        if total_steps < args.warmup_steps:
            scheduler = paddle.optimizer.lr.LinearWarmup(
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                start_lr=0,
                end_lr=args.learning_rate,
                last_epoch=total_steps,
                verbose=False)
        else:
            scheduler = paddle.optimizer.lr.PolynomialDecay(
                learning_rate=args.learning_rate,
                decay_steps=args.num_train_steps,
                end_lr=0,
                last_epoch=total_steps - args.warmup_steps,
                verbose=False)
        optim = paddle.optimizer.AdamW(
            learning_rate=scheduler,
            weight_decay=args.weight_decay,
            apply_decay_param_fun=apply_decay_param,
            grad_clip=nn.ClipGradByGlobalNorm(clip_norm=5.0),
            parameters=BertRec.parameters()
        )
        total_loss = 0
        batch_id = 0
        for batch_id, data in enumerate(
                train_loader()):
            src_ids, pos_ids, input_mask, mask_pos, mask_label = data
            src_ids = paddle.to_tensor(src_ids, dtype='int32')
            pos_ids = paddle.to_tensor(pos_ids, dtype='int32')
            input_mask = paddle.to_tensor(input_mask, dtype='int32')
            mask_pos = paddle.to_tensor(mask_pos, dtype='int32')
            mask_label = paddle.to_tensor(mask_label, dtype='int64')
            sent_ids = paddle.zeros(shape=[args.batch_size, args.max_seq_len], dtype='int32')

            fc_out = BertRec(src_ids, pos_ids, sent_ids, input_mask, mask_pos)
            mask_lm_loss, lm_softmax = nn.functional.softmax_with_cross_entropy(
                logits=fc_out, label=mask_label, return_softmax=True)
            mean_mask_lm_loss = paddle.mean(mask_lm_loss)

            loss = mean_mask_lm_loss
            total_loss += loss.numpy()

            loss.backward()
            optim.step()
            optim.clear_grad()
            scheduler.step()
            total_steps += 1
        print("epoch: {}, aver loss is: {}".format(epoch, total_loss / (1 + batch_id)))

        BertRec.eval()
        results = [0., 0., 0.]
        num_user = 0
        for batch_id, data in enumerate(val_loader()):
            pred_ratings = []
            src_ids, pos_ids, input_mask, mask_pos, mask_label = data
            src_ids = paddle.to_tensor(src_ids, dtype='int32')
            pos_ids = paddle.to_tensor(pos_ids, dtype='int32')
            input_mask = paddle.to_tensor(input_mask, dtype='int32')
            mask_pos = paddle.to_tensor(mask_pos, dtype='int32')
            sent_ids = paddle.zeros(shape=[args.batch_size, args.max_seq_len], dtype='int32')
            fc_out = BertRec(src_ids, pos_ids, sent_ids, input_mask, mask_pos)

            for i in range(args.batch_size):
                pred_ratings.append(fluid.layers.gather(fc_out[i], paddle.to_tensor(cand_list[i])).numpy())
            cand_list = cand_list[args.batch_size:]
            evaluate_rec_ndcg_mrr_batch(pred_ratings, results, top_k=10, row_target_position=0)
            num_user += args.batch_size

        rec, ndcg, mrr = results[0] / num_user, results[1] / num_user, results[2] / num_user
        print("epoch: %4d, HR@10: %.6f, NDCG@10: %.6f, MRR: %.6f" % (
            epoch, rec, ndcg, mrr), end='\n')


if __name__ == '__main__':
    print_arguments(args)
    main(args)
