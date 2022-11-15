import json
import os
import argparse

import numpy as np

from data import DATA_DIR


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model',  default='t5-base')
    parser.add_argument('--level', default='l1')
    parser.add_argument('--subset', default='predict')
    config = parser.parse_args()

    print('-' * 100)
    print(config.dataset.upper())

    for mode in [('standard', 'ENC-HEAD'), ('lwan', 'LWAN'), ('seq2seq', 'SEQ2SEQ'), ('enc2dec', 'ENC-DEC-1'), ('t5enc-multi', 'ENC-DEC-N')]:
        for dataset in ['uklex', 'eurlex', 'mimic', 'bioasq']:
            dataset_line = f'{mode[1]:>10}'
            BASE_DIR = f'{DATA_DIR}/logs/adafactor/{dataset}-{config.level}/{config.model}-{mode[0]}/fp32'
            scores = {'eval_micro-f1': [], 'eval_macro-f1': [], 'predict_micro-f1': [], 'predict_macro-f1': []}
            for seed in [21, 32, 42, 84]:
                seed = f'seed_{seed}'
                dataset_line += ' & '
                try:
                    with open(os.path.join(BASE_DIR, seed, 'all_results.json')) as json_file:
                        json_data = json.load(json_file)
                        dev_micro_f1 = float(json_data['eval_micro-f1'])
                        scores['eval_micro-f1'].append(dev_micro_f1)
                        dev_macro_f1 = float(json_data['eval_macro-f1'])
                        scores['eval_macro-f1'].append(dev_macro_f1)
                        test_micro_f1 = float(json_data['predict_micro-f1'])
                        scores['predict_micro-f1'].append(test_micro_f1)
                        test_macro_f1 = float(json_data['predict_macro-f1'])
                        scores['predict_macro-f1'].append(test_macro_f1)
                except:
                    continue
                dataset_line += f'{np.mean(scores[f"{config.subset}_micro-f1"]):.2f} $\pm$ {np.std(scores[f"{config.subset}_micro-f1"]):.2f}'
            dataset_line += f' \\'
            print(dataset_line)


if __name__ == '__main__':
    main()
