import json
import os
import argparse
from data import DATA_DIR
import numpy as np


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model',  default='t5-base')
    parser.add_argument('--dataset', default='uklex-l1')
    parser.add_argument('--optimizer', default='adafactor')
    config = parser.parse_args()

    print('-' * 100)
    print(config.dataset.upper())

    for label_desc_type in ['original', 'numbers', 'simplified']:
        BASE_DIR = f'{DATA_DIR}/logs/{config.optimizer}/{config.dataset}/{config.model}-seq2seq-{label_desc_type}/fp16'
        print('-' * 100)
        print(f'{f"{label_desc_type}":<15} | {"VALIDATION":<40} | {"TEST":<40}')
        print('-' * 100)
        scores = {'eval_micro-f1': [], 'eval_macro-f1': [], 'predict_micro-f1': [], 'predict_macro-f1': []}
        for seed in [21, 32, 42, 84]:
            seed = f'seed_{seed}'
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
                    epoch = float(json_data['epoch'])
                report_line = f'EPOCH: {epoch: 2.1f} | '
                report_line += f'MICRO-F1: {dev_micro_f1 * 100:.1f}\t'
                report_line += f'MACRO-F1: {dev_macro_f1 * 100:.1f}\t'
                report_line += ' | '
                report_line += f'MICRO-F1: {test_micro_f1 * 100:.1f}\t'
                report_line += f'MACRO-F1: {test_macro_f1 * 100:.1f}\t'
                print(report_line)
            except:
                continue
        print('-' * 100)
        report_line = f'EPOCH: {"N/A"} | '
        report_line += f'MICRO-F1: {np.mean(scores["eval_micro-f1"]) * 100:.1f}\t'
        report_line += f'MACRO-F1: {np.mean(scores["eval_macro-f1"]) * 100:.1f}\t'
        report_line += ' | '
        report_line += f'MICRO-F1: {np.mean(scores["predict_micro-f1"]) * 100:.1f}\t'
        report_line += f'MACRO-F1: {np.mean(scores["predict_macro-f1"]) * 100:.1f}\t'
        print(report_line)

if __name__ == '__main__':
    main()
