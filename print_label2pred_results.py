import pickle
import os
import argparse
from data import DATA_DIR


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model',  default='t5-base')
    parser.add_argument('--dataset', default='eurlex-l1')
    config = parser.parse_args()

    print('-' * 100)
    print(config.dataset.upper())

    for mode in ['standard', 'lwan', 'seq2seq']:
        BASE_DIR = f'{DATA_DIR}/logs/{config.dataset}/{config.model}-{mode}'
        print('-' * 100)
        print(f'{mode.upper():<10}   | {"VALIDATION":<40} | {"TEST":<40}')
        print('-' * 100)
        for seed in [21, 32, 42, 84]:
            seed = f'seed_{seed}'
            try:
                with open(os.path.join(BASE_DIR, seed, 'test_predictions.pkl'), 'rb') as pred_file:
                    preds = pickle.load(pred_file)
                with open(os.path.join(BASE_DIR, seed, 'test_labels.pkl'), 'rb') as pred_file:
                    labels = pickle.load(pred_file)
                for doc_preds, doc_labels in zip(preds, labels):
                    print(f'Predictions: {doc_preds}')
                    print(f'Labels: {doc_labels}')
                    print(f'Correct: {"NO" if doc_preds != doc_labels else "YES"}')
                    print('--------------------------')

            except:
                continue


if __name__ == '__main__':
    main()
