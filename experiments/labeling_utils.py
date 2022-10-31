import numpy as np


def fix_generated_scores(tokenizer, predictions, labels, label2id):
    # Get generated label descriptor predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Text to Label IDs
    discrete_predictions = np.zeros((len(decoded_preds), len(label2id)), dtype=np.int)
    print('Predictions: ' + str(decoded_preds))
    for idx, pred in enumerate(decoded_preds):
        for p in pred.split(','):
            if p.strip().lower() in label2id:
                discrete_predictions[idx][label2id[p.lower()]] = 1

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Get gold label descriptors
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Text to Label IDs
    discrete_labels = np.zeros((len(decoded_labels), len(label2id)), dtype=np.int)
    print('Labels: ' + str(decoded_labels))
    for idx, label in enumerate(decoded_labels):
        for l in label.split(', '):
            if l.lower() in label2id:
                discrete_labels[idx][label2id[l.lower()]] = 1

    return discrete_predictions, discrete_labels, decoded_preds, decoded_labels


if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    label2id = {'agriculture': 0, 'children': 1, 'crime': 2, 'education': 3, 'environment': 4, 'european': 5,
                'finance': 6, 'health': 7, 'housing': 8, 'immigration': 9, 'local': 10, 'planning': 11, 'politics': 12,
                'public': 13, 'social': 14, 'taxation': 15, 'telecommunications': 16, 'transportation': 17}

    labels = [['agriculture, children']]
    predictions = [['agriculture, child']]
    labels = tokenizer.encode(labels)
    predictions = tokenizer.encode(predictions)
    fix_generated_scores(tokenizer, predictions, labels, label2id)
