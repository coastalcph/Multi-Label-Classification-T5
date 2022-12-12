import json
import os
import argparse
from data import DATA_DIR


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--dataset', default='uklex')
    config = parser.parse_args()
    results = {}

    for model in ['small', 'base', 'large']:
        results[f't5-{model}'] = {}
        for mode in ['seq2seq-original', 'lwan', 't5enc-multi']:
            results[f't5-{model}'][mode.replace('-original', '')] = {}
            BASE_DIR = f'{DATA_DIR}/logs/adafactor/{config.dataset}-l2/t5-{model}-{mode}/fp32'
            for seed in [21, 32, 42, 84]:
                results[f't5-{model}'][mode.replace('-original', '')][seed] = {}
                try:
                    print(os.path.join(BASE_DIR, f'seed_{seed}', 'all_results.json'))
                    with open(os.path.join(BASE_DIR, f'seed_{seed}', 'all_results.json')) as json_file:
                        json_data = json.load(json_file)
                        for metric in ['eval_micro-f1', 'eval_macro-f1', 'predict_micro-f1', 'predict_macro-f1']:
                            try:
                                results[f't5-{model}'][mode.replace('-original', '')][seed][metric] = float(json_data[metric])
                            except:
                                continue
                except:
                    continue

    with open(os.path.join(DATA_DIR, f'{config.dataset}_results.json'), 'w') as out_file:
        json.dump(results, out_file, indent=1)


if __name__ == '__main__':
    main()
