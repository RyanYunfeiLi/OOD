# -*- coding: utf-8 -*-
# @Time : 2022/3/28 11:38
# @Author : Ryan Li
# @Func : Evaluate test dataset
# @File : eval.py
# @Software: PyCharm

import os
import json
import math
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from itertools import chain
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from nltk.corpus import wordnet
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
import corenlp
from torchnlp.encoders import LabelEncoder

sys.path.append("../")
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer
from data_utils import MyDataset
from loss_func import CrossEntropy
import torch.nn.functional as F

MAX_LENGTH = 512
# trans_dict = {"[positive]": "[negative]", "[negative]": "[positive]",
#               "[entailment]": "[contradiction]", "[contradiction]": "[entailment]",
#               "[0]": "[1]", "[1]": "[0]"}

trans_label = {
    "SST2": {
        "1": "positive",
        "0": "negative"
    },
    "CR": {
        "1": "positive",
        "0": "negative"
    },
    "TREC": {
        "0": "description abstract concepts",
        "1": "entity",
        "2": "abbreviation",
        "3": "human",
        "4": "location",
        "5": "numeric"
    },
    "SUBJ": {
        "0": "subjective",
        "1": "objective"
    },
    "procon": {
        "positive": "positive",
        "negative": "negative"
    }
}


def collate_fn(batch, label_length, label_start, label_end, LABEL_CLASS):
    tot_label_length = sum(label_length)
    out = [[] for _ in range(len(batch[0]) + 1)]
    for row in batch:
        inputs_id = row[0]  # (bs, sl)
        keywords_labels = row[3]
        label_mask = row[5]

        SL = len(inputs_id)
        cur_index = [i for i in range(1, 1 + LABEL_CLASS)]
        random.shuffle(cur_index)
        cur_index_expand = []
        for idx in cur_index:
            cur_index_expand.extend(list(range(label_start[idx - 1] + 1, label_end[idx - 1] + 1)))
        assert (len(cur_index_expand) == tot_label_length)
        cur_all_index = [0] + cur_index_expand + [i for i in range(1 + tot_label_length, SL)]
        inputs_id = np.take(inputs_id, cur_all_index)  # torch.gather
        keywords_labels = np.take(keywords_labels, cur_all_index)
        label_mask = np.take(label_mask, cur_all_index)
        row[0] = inputs_id
        row[3] = keywords_labels
        row[5] = label_mask

        for idx in range(len(batch[0])):
            out[idx].append(row[idx])
        cur_all_order = np.argsort(np.asarray(cur_all_index))
        out[-1].append(cur_all_order)
    return [torch.as_tensor(e) for e in out]


class Evaluation(object):
    """ Model evaluation """

    def __init__(self, option):  # prepare for training the model
        test_data = self.load_data(dataset=option.dataset, directory=option.directory, test_file=option.test_file)
        self._initialization(option, test_data)
        # self._print_args()

    @staticmethod
    def load_data(dataset, directory, test=True, test_file='Test.json'):
        datasets = [
            'SST2',
            'CR',
            'procon',
            'SUBJ',
            'TREC'
        ]
        if dataset not in datasets:
            raise ValueError('dataset: {} not in support list!'.format(dataset))

        ret = []
        # splits = ['SST2_Test.json']
        splits = [
            '_'.join([dataset, fn_]) for (requested, fn_) in [(test, test_file), ]
            if requested
        ]
        for split_file in splits:
            full_filename = os.path.join(directory, split_file)
            examples = []

            with open(full_filename, 'r', encoding="utf-8") as f:
                tmp = f.readlines()
                for idx, j in enumerate(tmp):
                    N = len(tmp)
                    a_data = json.loads(j)
                    sent = a_data["sentence"].lower()
                    a_data["sentence"] = " ".join(sent.split(" ")[:MAX_LENGTH - 4])
                    a_data["polarity"] = str(a_data["polarity"])
                    examples.append(a_data)
            ret.append(examples)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def _initialization(self, option, test_data):
        # don't need sentiment label encoder
        # senti_label_corpus = [trans_label[option.dataset][row["polarity"]] for row in test_data]
        # senti_label_encoder = LabelEncoder(senti_label_corpus, reserved_labels=[], unknown_index=None)

        senti_label_corpus = [trans_label[option.dataset][key] for key in trans_label[option.dataset]]
        senti_label_encoder = LabelEncoder(senti_label_corpus, reserved_labels=[], unknown_index=None)

        label_token2idx = {}
        for k, v in trans_label[option.dataset].items():
            label_token2idx[v] = int(k)
        option.label_class = len(label_token2idx)  # label_class=2

        # load model
        model = torch.load(option.trained_model_path, map_location='cuda:0')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model.eval()
        max_length = 0

        list_labels = list(label_token2idx.keys())
        if option.class_use_bert_embedding == 1:
            pass
        else:
            label_token2idx = {f"[{idx}]": v for idx, v in enumerate(label_token2idx.values())}
            list_labels = list(label_token2idx.keys())
            tokenizer.add_tokens(list_labels, special_tokens=True)
            model.resize_vocab(tokenizer)

        # list_labels: ['negative', 'positive']
        label_length = self.get_label_length(tokenizer, list_labels)  # label_length: [1, 1]
        label_start = [0]
        label_end = [label_length[0]]
        for idx in range(len(label_length) - 1):
            label_start.append(label_start[-1] + label_length[idx])
            label_end.append(label_end[-1] + label_length[idx + 1])
        tot_label_length = sum(label_length)  # tot_label_length = 2

        for idx, a_data in enumerate(chain(test_data)):
            label = a_data["polarity"]
            sent = a_data["sentence"]
            sent_list = sent.split(" ")
            if option.model_type == "bert":
                sent_list = ["[CLS]"] + ["[PAD]"] * option.label_class + ["[SEP]"] + sent_list + ["[SEP]"]
            elif option.model_type == "roberta":
                sent_list = ["<s>"] + ["<pad>"] * option.label_class + ["</s>", "</s>"] + sent_list + ["</s>"]
            else:
                raise ValueError("wrong model type!")
            for k, v in label_token2idx.items():
                sent_list[v + 1] = k
            tokens_list = tokenizer.tokenize(" ".join(sent_list))
            tokens_length = len(tokens_list)
            max_length = max(max_length, tokens_length)
            a_data["tokens_length"] = tokens_length
            a_data["tokens_list"] = tokens_list

        for idx, a_data in enumerate(chain(test_data)):
            tokens_list = a_data["tokens_list"]
            tokens_length = a_data["tokens_length"]
            label = a_data["polarity"]

            inputs_id = tokenizer.convert_tokens_to_ids(tokens_list)
            inputs_id = inputs_id + [0] * (max_length - tokens_length)
            attention_mask = [1] * tokens_length + [0] * (max_length - tokens_length)

            label = senti_label_encoder.encode(trans_label[option.dataset][label])
            inputs_id = np.asarray(inputs_id)
            attention_mask = np.asarray(attention_mask)
            label = np.asarray(label)
            a_data = [inputs_id, attention_mask, label]
            test_data[idx] = a_data

        test_set = MyDataset(test_data)
        test_dataloader = DataLoaderX(dataset=test_set, batch_size=64, shuffle=False,
                                      num_workers=4, pin_memory=True)  # test dataloader
        self.opt = option
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataloader = test_dataloader
        self.label_token2idx = label_token2idx
        self.label_length = label_length
        self.label_start = label_start
        self.label_end = label_end

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        if self.opt.device == 'cuda':
            print(f"cuda memory allocated: {torch.cuda.memory_allocated(self.opt.device.index)}")
        print(f"n_trainable_params: {int(n_trainable_params)}, n_nontrainable_params: {int(n_nontrainable_params)}")
        print('training arguments:')
        for arg in vars(self.opt):
            print(f">>> {arg}: {getattr(self.opt, arg)}")

    @staticmethod
    def get_label_length(tokenizer, labels):
        if opt.model_type == "bert":
            label_length = [len(tokenizer.tokenize(label)) for label in labels]  # [['negative'], ['positive']]
        elif opt.model_type == "roberta":
            label_tokenized = tokenizer.tokenize(" " + " ".join(labels))
            label_length = [0] * len(labels)
            idx_label = 0
            cur_label = ""
            for idx_sub, sub_word in enumerate(label_tokenized):
                if sub_word.startswith("\u0120"):
                    sub_word = sub_word.replace("\u0120", "")
                    if not sub_word:
                        continue
                    cur_label += (" " + sub_word)
                else:
                    cur_label += sub_word
                if cur_label.strip() == labels[idx_label]:
                    label_length[idx_label] = idx_sub + 1
                    idx_label += 1
                    cur_label = ""
            label_length = [label_length[0]] + [l1 - l0 for l1, l0 in zip(label_length[1:], label_length[:-1])]
        return label_length

    def _join_label_feature(self, label_feature, label_length, LABEL_CLASS):
        BS, _, HS = label_feature.shape
        out = torch.zeros((BS, LABEL_CLASS, HS)).to(self.opt.device)
        start, end, idx_label = 0, label_length[0], 0
        for idx in range(LABEL_CLASS):
            out[:, idx, :] = torch.mean(label_feature[:, start: end, :], dim=1)
            if idx != LABEL_CLASS - 1:
                start += label_length[idx]
                end += label_length[idx + 1]
        return out

    def test(self):
        criterion = CrossEntropy(self.opt)
        LABEL_CLASS = self.opt.label_class
        SENTENCE_BEGIN = sum(self.label_length) + 2
        test_loss, n_correct, n_test = 0, 0, 0  # reset counters
        labels_all, predicts_all = None, None  # initialize variables

        self.model.eval()  # switch model to training mode
        with torch.no_grad():
            test_loss = 0
            for sample_batched in self.test_dataloader:  # mini-batch optimization
                if self.opt.device == "cuda":
                    inputs = list(map(lambda x: x.cuda(non_blocking=True), sample_batched))
                else:
                    inputs = list(sample_batched)
                inputs_id, attention_mask, labels = inputs
                # cross entropy
                outputs = self.model([inputs_id, attention_mask])  # compute outputs
                word_feature, cls_feature = outputs.last_hidden_state, outputs.pooler_output
                BS, SL, HS = word_feature.shape
                # label feature
                label_feature = word_feature[:, 1: SENTENCE_BEGIN - 1, :]
                label_feature = self._join_label_feature(label_feature, self.label_length,
                                                         LABEL_CLASS)  # (bs, label_class, 768)
                label_feature = self.model.label_dropout(
                    self.model.label_activation(self.model.label_trans(label_feature)))
                # cls feature
                cls_feature = self.model.cls_dropout(
                    self.model.cls_activation(self.model.cls_trans(cls_feature)))
                predicts = torch.bmm(label_feature, self.model.fc_dropout(cls_feature.unsqueeze(-1))).squeeze(-1)
                ce_loss = criterion([predicts, None, None], labels)  # compute batch loss

                test_loss += ce_loss.item() * len(labels)
                n_correct += (torch.argmax(predicts, -1) == labels).sum().item()
                n_test += len(labels)
                labels_all = torch.cat((labels_all, labels), dim=0) if labels_all is not None else labels
                predicts_all = torch.cat((predicts_all, predicts), dim=0) if predicts_all is not None else predicts
        macro_f1 = metrics.f1_score(labels_all.detach().cpu(), torch.argmax(predicts_all, -1).detach().cpu(),
                                    average='macro')  # compute f1 score
        precision = metrics.precision_score(labels_all.detach().cpu(), torch.argmax(predicts_all, -1).detach().cpu(),
                                            average='macro')
        recall = metrics.recall_score(labels_all.detach().cpu(), torch.argmax(predicts_all, -1).detach().cpu(),
                                      average='macro')
        return test_loss / n_test, n_correct / n_test, macro_f1, precision, recall


def _main(option):
    test_files = ['test.json', 'dev.json']
    for file in test_files:
        option.directory = './datasets_processed'
        option.test_file = file
        evaluation = Evaluation(option)
        print(option.dataset + '_' + file, evaluation.test()[1])
    return


if __name__ == "__main__":
    ''' hyper_parameters '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SST2', type=str, help='Restaurants, Laptops, SST2, CR, TREC, IMDB, '
                                                                    'snli_1.0, yahoo, agnews')
    parser.add_argument('--directory', default='./datasets_manual', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--percentage', default=1, type=float)
    parser.add_argument('--num_epoch', default=5, type=int)
    parser.add_argument('--warm_up_epoch', default=0, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--word_dim', default=768, type=int)
    parser.add_argument('--fc_dropout', default=0.1, type=float)
    parser.add_argument('--eps', default=1e-2, type=float)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--run_times', default=1, type=int)
    parser.add_argument('--cuda_device', default=0, type=int, help='0, 1, 2, 3')

    parser.add_argument('--sentence_mode', default="cls", type=str, help='mean, cls')
    parser.add_argument('--alpha1', default=0.01, type=float)  # mlm loss
    parser.add_argument('--alpha2', default=0.01, type=float)  # contrast loss
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--model_type', default="bert", type=str, help='bert, roberta')
    parser.add_argument('--class_use_bert_embedding', default=1, type=int, help='fake bool')
    parser.add_argument('--trained_model_path', default='results_polarity/SST2/model_5.pkl', type=str)

    opt = parser.parse_args()
    assert (opt.word_dim == 768)


    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())


    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if opt.device == "cuda":
        ''' if you are using cudnn '''
        torch.backends.cudnn.deterministic = True  # Deterministic mode can have a performance impact
        torch.backends.cudnn.benchmark = False
    _main(opt)

