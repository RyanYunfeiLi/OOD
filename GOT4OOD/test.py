# -*- coding: utf-8 -*-
# @Time : 2022/3/21 10:03
# @Author : Ryan Li
# @Func : 
# @File : eval.py
# @Software: PyCharm

import os
import torch
import time
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from model.Classifier import Classifier
from util.utils import load_args, set_seed


class infer(object):
    def __init__(self):
        pass

    def run(self):
        pass


def energy(input_vec):
    input_vec = torch.exp(input_vec / args.T)
    energy_output = - torch.log(torch.sum(input_vec, dim=-1)) * args.T
    energy_output = energy_output.tolist()
    _, predicted = torch.max(input_vec.data, 1)
    max_pred = _.tolist()
    pred_index = predicted.tolist()
    return energy_output, max_pred, pred_index


def msp(input_vec):
    s_output = softmax(input_vec)
    max_pred, predicted = torch.max(s_output.data, 1)
    max_pred = max_pred.tolist()
    pred_index = predicted.tolist()
    return max_pred, pred_index


def evaluate(inf: str, ouf: str, msp_th: int = 0.75, e_th: int = -5.0):
    """
    @file_name: file name
    @msp_th: msp probability threshold
    @e_th: energy threshold
    """
    with open(inf, 'r') as f, open(ouf, 'w') as t:
        lines = f.readlines()
        t.write('query\tintent\tpredict_intent\n')
        for line in tqdm(lines[1:]):
            splits = line.strip().split('\t')
            utt, label = splits[1], splits[0]
            _, output = mdl(utt)
            msp_output = msp(output)
            e_output = energy(output)
            if msp_output[0][0] < msp_th and e_output[0][0] > e_th:
                pred_label = -1
            else:
                pred_label = msp_output[1][0]
            t.write(f'{utt}\t{label}\t{label_idx_map[str(pred_label)]}\n')


if __name__ == '__main__':
    args = load_args('configs/infer.yaml')

    rep = torch.tensor([])

    model_param = torch.load('output/params/navi_intent/nan/params.pkl', map_location='cuda:{}'.format(args.gpu) \
        if args.gpu != -1 else 'cpu')

    mdl = Classifier(args).to(args.device)
    mdl.eval()
    mdl.load_state_dict(model_param)
    softmax = nn.Softmax(dim=-1)

    # label和index映射
    label_idx_map = {
        '0': 'navigation_close',
        '1': 'navigation_display_off',
        '2': 'navigation_display_on',
        '3': 'navigation_inquire_where',
        '4': 'navigation_navigate_destination',
        '5': 'navigation_navigate_waypoints',
        '6': 'navigation_open',
        '7': 'navigation_query_distance',
        '8': 'navigation_query_eta',
        '9': 'navigation_search',
        '10': 'navigation_set',
        '11': 'navigation_start',
        '12': 'navigation_stop',
        '13': 'navigation_zoom_in',
        '14': 'navigation_zoom_out',
        '15': 'navigation_save',
        '16': 'navigation_unsave',
        '-1': 'navigation_unknown'
    }

    # inference and predict ind
    abs_path = '/diskb/liyunfei/jupyter/data/'
    out_path = abs_path + 'navigation_eval/'

    comp_path = abs_path + 'navigation_test/'
    # comp_dir = os.listdir(comp_path)
    # for item in comp_dir:
    #     input_file = comp_path + item + '/system_test.tsv'
    #     out_file = out_path + item + '/test_result.tsv'
    #     evaluate(input_file, out_file)

    # inference and predict ood
    input_file = comp_path + 'p2_tmp' + '/system_test.tsv'
    out_file = out_path + 'p2_tmp' + '/test_result.tsv'
    evaluate(input_file, out_file)
