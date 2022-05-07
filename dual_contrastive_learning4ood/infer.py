# -*- coding: utf-8 -*-
# @Time : 2022/3/28 11:38
# @Author : Ryan Li
# @Func :
# @File : infer.py
# @Software: PyCharm

import time
import argparse
import numpy as np
import sys

import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

# from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append("../")
from transformers import BertTokenizer
from loss_func import CrossEntropy

MAX_LENGTH = 512

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


class Inference(object):
    """ Model inference """
    _model_infer = None
    _first_init = True

    def __new__(cls, *args, **kwargs):  # load model to cache
        if cls._model_infer is None and cls._first_init:
            cls._model_infer = object.__new__(cls)
        return cls._model_infer

    def __init__(self, option):  # prepare for training the model
        # load model
        self.model = torch.load(option.trained_model_path, map_location='cuda:0')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.label_token2idx = {}
        for k, v in trans_label[option.dataset].items():
            self.label_token2idx[v] = int(k)
        option.label_class = len(self.label_token2idx)  # label_class=2
        self.criterion = CrossEntropy(option)

        list_labels = list(self.label_token2idx.keys())  # list_labels: ['negative', 'positive']
        label_length = self.get_label_length(self.tokenizer, list_labels)  # label_length=[1, 1]
        label_start = [0]
        label_end = [label_length[0]]
        for idx in range(len(label_length) - 1):
            label_start.append(label_start[-1] + label_length[idx])
            label_end.append(label_end[-1] + label_length[idx + 1])

        self.opt = option
        self.label_length = label_length
        self._print_args()
        self.__class__._first_init = False

    def encode(self, query):  # query encode
        max_length = 0
        a_data = {'sentence': query}
        sent_list = query.split(" ")
        if self.opt.model_type == "bert":
            sent_list = ["[CLS]"] + ["[PAD]"] * self.opt.label_class + ["[SEP]"] + sent_list + ["[SEP]"]
        elif self.opt.model_type == "roberta":
            sent_list = ["<s>"] + ["<pad>"] * self.opt.label_class + ["</s>", "</s>"] + sent_list + ["</s>"]
        else:
            raise ValueError("wrong model type!")
        for k, v in self.label_token2idx.items():
            sent_list[int(v) + 1] = k
        tokens_list = self.tokenizer.tokenize(" ".join(sent_list))
        tokens_length = len(tokens_list)
        max_length = max(max_length, tokens_length)
        a_data["tokens_length"] = tokens_length
        a_data["tokens_list"] = tokens_list

        inputs_id = self.tokenizer.convert_tokens_to_ids(tokens_list)
        inputs_id = inputs_id + [0] * (max_length - tokens_length)
        attention_mask = [1] * tokens_length + [0] * (max_length - tokens_length)

        inputs_id = np.asarray(inputs_id)
        attention_mask = np.asarray(attention_mask)
        a_data = [[inputs_id], [attention_mask]]

        return torch.tensor(a_data)

    def infer(self, query):  # model infer
        # encode
        inputs = self.encode(query)
        # load model
        self.model.eval()  # switch model to training mode

        LABEL_CLASS = self.opt.label_class
        SENTENCE_BEGIN = sum(self.label_length) + 2

        if self.opt.device == "cuda":
            inputs = list(map(lambda x: x.cuda(non_blocking=True), inputs))
        else:
            inputs = list(inputs)

        inputs_id, attention_mask = inputs

        # cross entropy
        outputs = self.model([inputs_id, attention_mask])  # compute outputs
        word_feature, cls_feature = outputs.last_hidden_state, outputs.pooler_output

        # label feature
        label_feature = word_feature[:, 1: SENTENCE_BEGIN - 1, :]
        # (bs, label_class, 768)
        label_feature = self._join_label_feature(label_feature, self.label_length, LABEL_CLASS)
        label_feature = self.model.label_dropout(self.model.label_activation(self.model.label_trans(label_feature)))

        # cls feature
        cls_feature = self.model.cls_dropout(self.model.cls_activation(self.model.cls_trans(cls_feature)))
        predicts = torch.bmm(label_feature, self.model.fc_dropout(cls_feature.unsqueeze(-1))).squeeze(-1)
        predict_label = torch.argmax(predicts, -1)
        return int(predict_label)

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


def _main(option):
    infer = Inference(option)
    # test examples
    # ig. one long string of cliches .
    #     it 's played in the most straight-faced fashion , with little humor to lighten things up .
    #     despite its title , punch-drunk love is never heavy-handed .
    while True:
        start_time = time.time()
        test_query = input('query: ')
        label = infer.infer(query=test_query)
        print(f'predict_label: {label}, predict_label_name: {trans_label[option.dataset][str(label)]}, '
              f'time_consumed: {time.time()-start_time: .4}ms')
    return


if __name__ == "__main__":
    ''' hyper_parameters '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SST2', type=str, help='Restaurants, Laptops, SST2, CR'
                                                                    'TREC, IMDB, snli_1.0, yahoo, agnews')
    parser.add_argument('--directory', default='./datasets_manual', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--percentage', default=0.1, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
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
    parser.add_argument('--trained_model_path', default='results_polarity/SST2/model_2.pkl', type=str)

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
