import numpy as np


def fix_generated_scores(tokenizer, predictions, labels, label2id):
    # Get generated label descriptor predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Text to Label IDs
    discrete_predictions = np.zeros((len(decoded_preds), len(label2id)), dtype=np.int)
    print('Predictions: ' + str(decoded_preds))
    for idx, pred in enumerate(decoded_preds):
        for p in pred.split(', '):
            if p.lower() in label2id:
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
