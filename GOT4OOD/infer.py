# -*- coding: utf-8 -*-
# @Time : 2022/3/17 10:09
# @Author : Ryan Li
# @Func : 
# @File : infer.py
# @Software: PyCharm

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


def evaluate(file_name: str, msp_th: int = 0.75, e_th: int = -5.0):
    """
    @file_name: file name
    @msp_th: msp probability threshold
    @e_th: energy threshold
    """
    with open(f'data/navi_intent/{file_name}', 'r') as f, \
            open(f'/diskb/liyunfei/jupyter/data/eval_{file_name}', 'w') as t:
        # with open(f'{abs_path + file_name}.tsv', 'r') as f, \
        #         open(f'{abs_path}eval_{file_name}.csv', 'w') as t:
        lines = f.readlines()
        t.write('query,intent,predict_intent\n')
        for line in tqdm(lines[1:]):
            splits = line.strip().split(',')
            utt, label = splits[1], splits[-1]
            _, output = mdl(utt)
            msp_output = msp(output)
            e_output = energy(output)
            if msp_output[0][0] < msp_th and e_output[0][0] > e_th:
                pred_label = -1
            else:
                pred_label = msp_output[1][0]
            t.write(f'{utt}, {label_idx_map[str(label)]}, {label_idx_map[str(pred_label)]}\n')


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
    # abs_path = '/diskb/liyunfei/jupyter/data/'
    file = 'ood_qa.csv'
    evaluate(file)

    # 验证测试集
    # with open('data/navi_intent/test.csv', 'r') as f:
    #     lines = f.readlines()
    #
    # with open('data/navi_intent/tmp.csv', 'w') as t:
    #     for line in tqdm(lines[1:]):
    #         splits = line.strip().split(',')
    #         utt, label = splits[1], splits[-1]
    #         _, output = mdl(utt)
    #         msp_output = msp(output)
    #         e_output = energy(output)
    #         if msp_output[0][0] < 0.75 and e_output[0][0] > - 6.0:
    #             pred_label = -1
    #         else:
    #             pred_label = msp_output[1][0]
    #         t.write(f'{utt}, {label_idx_map[str(label)]}, {label_idx_map[str(pred_label)]}\n')

    # 验证ood_valid.csv
    # with open('data/navi_intent/ood_other.csv', 'r') as f:
    #     lines = f.readlines()
    #
    # with open('data/navi_intent/tmp_ood_other.csv', 'w') as t:
    #     t.write('query,intent,predict_intent\n')
    #     for line in tqdm(lines[1:]):
    #         splits = line.strip().split(',')
    #         utt, label = splits[1], splits[-1]
    #         _, output = mdl(utt)
    #         msp_output = msp(output)
    #         e_output = energy(output)
    #         if msp_output[0][0] < 0.75 and e_output[0][0] > - 6.0:
    #             pred_label = -1
    #         else:
    #             pred_label = msp_output[1][0]
    #         t.write(f'{utt}, {label_idx_map[str(label)]}, {label_idx_map[str(pred_label)]}\n')

    # 单query请求
    # queries = [
    #     'I need to go wash my car',
    #     'I wonder if they have baseball cards in Japan',
    #     'Mcdonalds added drive thrus to accommodate soldiers who were not allowed to leave their cars',
    #     'Arizona is such an awesome place I want to go back soon',
    #     'but I would go see a broadway show if I did visit',
    #     'Id would definitely recommend South Park',
    #     'michigan state library has the largest public collection of comic books in the world',
    #     'Look at big pharma',
    #     'when I have the time',
    #     "maybe I wouldn't have to work again",
    #     'Do you use an Amazon Alexa device'
    # ]
    # for x in queries:
    #     start_time = time.time()
    #     _, output = mdl(x)
    #     msp_output = msp(output)
    #     e_output = energy(output)
    #     print(f'msp_output: {msp_output[0][0]}, e_output: {e_output[0][0]}')
    #     print(f'consumed time: {time.time() - start_time} s')
    #     print('=' * 120)

    # 单query请求时间
    # for x in queries:
    #     start_time = time.time()
    #     _, output = mdl(x)
    #     msp_output = msp(output)
    #     e_output = energy(output)
    #     if msp_output[0][0] < 0.75 and e_output[0][0] > - 6.0:
    #         label = -1
    #     else:
    #         label = msp_output[1][0]
    #     print(f'query: {x}, predict label: {label}')
    #     print(f'consumed time: {time.time() - start_time} s')
    #     print('=' * 120)
