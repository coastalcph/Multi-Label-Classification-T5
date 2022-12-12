import json
import os
import argparse
import pickle

import numpy as np

from data import DATA_DIR
from datasets import load_dataset
from data.multilabel_bench.label_descriptors import *
from sklearn.metrics import f1_score

def fix_predictions(predictions, label_descs):
    # Replace corrupted predictions with the first label that fits the starting characters

    label_descs = [label_desc[0].replace(',', '').lower() for label_desc in label_descs]

    for i in range(len(predictions)):
        preds_list = predictions[i]
        for j in range(len(preds_list)):
            if preds_list[j] in label_descs:
                continue
            if not len(preds_list[j].strip()):
                continue
            for label in label_descs:
                if label.startswith(preds_list[j]):
                    print('replacing {} with {}'.format(predictions[i][j], label))
                    predictions[i][j] = label

    return predictions

def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model',  default='t5-base')
    parser.add_argument('--dataset', default='mimic-l2')
    parser.add_argument('--subset', default='predict')
    config = parser.parse_args()

    # Load dataset
    predict_dataset = load_dataset(os.path.join(DATA_DIR, 'multilabel_bench'), config.dataset, split="test")
    # Labels
    labels_codes = predict_dataset.features['concepts'].feature.names

    # Load label descriptors
    if 'eurlex' in config.dataset:
        label_descriptors = EUROVOC_CONCEPTS[f'level_{config.dataset.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code] for label_code in labels_codes]
    elif 'bioasq' in config.dataset:
        label_descriptors = MESH_CONCEPTS[f'level_{config.dataset.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code] for label_code in labels_codes]
    elif 'mimic' in config.dataset:
        label_descriptors = ICD9_CONCEPTS[f'level_{config.dataset.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code] for label_code in labels_codes]
    elif 'uklex' in config.dataset:
        label_descriptors = UKLEX_CONCEPTS[f'level_{config.dataset.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code] for label_code in labels_codes]
    else:
        raise Exception(f'Dataset {config.dataset} is not supported!')

    # Create label dicts
    label_desc2id = {label_desc[0].replace(',', '').lower(): idx for idx, label_desc in enumerate(label_descs)}
    label_id2desc = {idx: label_desc[0].replace(',', '').lower() for idx, label_desc in enumerate(label_descs)}

    # Test set gold label
    labels = []
    for label_list in predict_dataset['concepts']:
        labels.append([label_id2desc[label_id] for label_id in label_list])

    discrete_labels = np.zeros((len(labels), len(label_desc2id)), dtype=np.int32)
    for idx, label_list in enumerate(labels):
        for label in label_list:
            if label.strip().lower() in label_desc2id:
                discrete_labels[idx][label_desc2id[label.strip().lower()]] = 1

    macro_f1s = []
    micro_f1s = []
    for seed in [21, 32, 42, 84]:
        seed = f'seed_{seed}'
        # Read predictions
        with open(os.path.join(f'{DATA_DIR}/predictions/{config.dataset}/seq2seq-original', seed, 'test_predictions.pkl'), 'rb') as pkl_file:
            predictions = pickle.load(pkl_file)
            predictions = [[pred.strip() for pred in predictions_list.split(',')] for predictions_list in predictions]
            predictions = fix_predictions(predictions, label_descs)
            discrete_predictions = np.zeros((len(predictions), len(label_desc2id)), dtype=np.int32)
            # Text predictions to binary
            for idx, pred_list in enumerate(predictions):
                for p in pred_list:
                    if p.strip().lower() in label_desc2id:
                        discrete_predictions[idx][label_desc2id[p.strip().lower()]] = 1
        # Compute scores
        macro_f1 = f1_score(y_true=discrete_labels, y_pred=discrete_predictions, average='macro', zero_division=0)
        macro_f1s.append(macro_f1)
        micro_f1 = f1_score(y_true=discrete_labels, y_pred=discrete_predictions, average='micro', zero_division=0)
        micro_f1s.append(micro_f1)

    # Print averaged scores
    print(f'MICRO-F1: {np.mean(micro_f1s)*100:.1f} +/- {np.std(micro_f1s)*100:.1f}\t'
          f'MACRO-F1: {np.mean(macro_f1s)*100:.1f} +/- {np.std(macro_f1s)*100:.1f}')


if __name__ == '__main__':
    main()
