from datasets import load_dataset
import os
from data import DATA_DIR
import numpy as np
from data.multilabel_bench.label_descriptors import *
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('t5-small')

for dataset_name in ['uklex-l1', 'eurlex-l1', 'mimic-l1', 'bioasq-l1', 'uklex-l2', 'eurlex-l2', 'mimic-l2', 'bioasq-l2']:
    train_dataset = load_dataset(os.path.join(DATA_DIR, 'multilabel_bench'), dataset_name, split="train")
    label_counts = [len(labels) for labels in train_dataset['concepts']]
    max_labels = max(label_counts)
    mean_labels = int(sum(label_counts) / len(label_counts))
    # doc_lengths = [len(tokenizer.tokenize(document)) for document in train_dataset['text']]

    labels_codes = train_dataset.features['concepts'].feature.names
    if 'eurlex' in dataset_name:
        label_descriptors = EUROVOC_CONCEPTS[f'level_{dataset_name.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code][0].replace(',', '').lower() for label_code in labels_codes]
    elif 'bioasq' in dataset_name:
        label_descriptors = MESH_CONCEPTS[f'level_{dataset_name.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code][0].replace(',', '').lower() for label_code in labels_codes]
    elif 'mimic' in dataset_name:
        label_descriptors = ICD9_CONCEPTS[f'level_{dataset_name.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code][0].replace(',', '').lower() for label_code in labels_codes]
    elif 'uklex' in dataset_name:
        label_descriptors = UKLEX_CONCEPTS[f'level_{dataset_name.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code][0].replace(',', '').lower() for label_code in labels_codes]

    label_descs_tokens = [len(tokenizer.tokenize(label)) for label in label_descs]

    print(f'{dataset_name} LABELS: MEAN_ASSIGNMENT: {mean_labels} MAX_ASSIGNMENT: {max_labels} MEAN_TOKENS: {int(np.mean(label_descs_tokens))} MAX_TOKENS: {int(max(label_descs_tokens))}') # DOC_LENGTH: {int(np.mean(doc_lengths))}')


