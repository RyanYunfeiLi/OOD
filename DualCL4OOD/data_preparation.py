# -*- coding: utf-8 -*-
# @Time : 2022/3/30 10:46
# @Author : Ryan Li
# @Func : transfer .tsv to .json
# @File : data_preparation.py
# @Software: PyCharm

import json


def convert_tsv2json(source_path, target_path):
    converted_lines = []
    with open(source_path, 'r') as s, open(target_path, 'w', encoding='utf-8') as t:
        for line in s.readlines():
            parts = line.split('\t')
            converted = {'polarity': parts[0].strip(), 'sentence': parts[1].strip()}
            converted_lines.append(json.dumps(converted, indent=None))
        t.writelines('\n'.join(converted_lines))
    return


if __name__ == '__main__':
    source = 'datasets_origin/SST-2/dev.tsv'
    target = 'datasets_manual/SST-2_Dev.json'
    convert_tsv2json(source, target)
    print('hello world.')
    source = 'datasets_origin/SST-2/train.tsv'
    target = 'datasets_manual/SST-2_Train.json'
    convert_tsv2json(source, target)
    print('hello world.')
    source = 'datasets_origin/SST-2/Test.tsv'
    target = 'datasets_manual/SST-2_Test.json'
    convert_tsv2json(source, target)
    print('hello world.')
