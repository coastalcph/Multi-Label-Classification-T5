#!/usr/bin/env python
# coding=utf-8
""" Finetuning T5-based Multi-Label Classifiers."""

import logging
import os
import pickle
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
from scipy.special import expit
import glob
import shutil

import transformers
from transformers import (
    Trainer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from data import AUTH_KEY, DATA_DIR
from data_collator import DataCollatorForMultiLabelClassification
from models.t5_classifier import T5ForSequenceClassification
from experiments.trainer_seq2seq import Seq2SeqTrainer
from data.multilabel_bench.label_descriptors import EUROVOC_CONCEPTS, ICD9_CONCEPTS, MESH_CONCEPTS, UKLEX_CONCEPTS, ECTHR_ARTICLES
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.20.0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

from tokenizers.normalizers import NFKD
from tokenizers.pre_tokenizers import WhitespaceSplit

normalizer = NFKD()
pre_tokenizer = WhitespaceSplit()

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="eurlex-l1",
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    generation_max_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The minimum length of the label sequence to be generated."
        },
    )
    generation_min_length: Optional[int] = field(
        default=1,
        metadata={
            "help": "The minimum total label sequence length after tokenization. Sequences shorter "
                    "than are not plausible."
        },
    )
    label_descriptors_mode: Optional[str] = field(
        default="simplified",
        metadata={
            "help": "The mode of label descriptors (original, simplified, numbers)."
        },
    )
    word_constraints: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to force word constraints"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    seq2seq: bool = field(
        default=False,
        metadata={"help": "Whether the model is a seq2seq generator model."},
    )
    t5_enc2dec: bool = field(
        default=False,
        metadata={"help": "Whether the model is a seq2seq classification model, similar to T5Enc Liu et al. (2022)."},
    )
    n_dec_layers: int = field(
        default=-1,
        metadata={"help": "Number of decoder layers for T5Enc."},
    )
    t5_enc2dec_mode: str = field(
        default="single-step",
        metadata={"help": "Mode for T5Enc (single-step, or multi-step)."},
    )
    causal_masking: bool = field(
        default=True,
        metadata={"help": "Whether to use causal masking or not for T5 decoder."},
    )
    decoder_attention: bool = field(
        default=True,
        metadata={"help": "Whether to not attend other decoder steps on T5 decoder."},
    )
    use_lwan: bool = field(
        default=False,
        metadata={"help": "Whether the model is a Label-Wise Attention Network (LWAN)."},
    )
    lwan_version: int = field(
        default=3,
        metadata={"help": "Whether the model is a Label-Wise Attention Network (LWAN)."},
    )
    lwan_heads: int = field(
        default=1,
        metadata={"help": "Number of Label-Wise Attention Heads."},
    )
    use_lwan_advanced: bool = field(
        default=False,
        metadata={"help": "Whether the model is a Label-Wise Attention Network (LWAN) with GAT."},
    )
    n_beams: int = field(
        default=4,
        metadata={"help": "Number of beams for seq2seq."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=True,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Define se2seq training configuration parameters
    if model_args.seq2seq:
        training_args.generation_max_length = data_args.generation_max_length
        training_args.generation_min_length = data_args.generation_min_length
        training_args.word_constraints = data_args.word_constraints
        training_args.generation_num_beams = model_args.n_beams
        training_args.predict_with_generate = True

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    label_list = None

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading eurlex dataset from the hub.
    if training_args.do_train:
        train_dataset = load_dataset(os.path.join(DATA_DIR, 'multilabel_bench'), data_args.dataset_name, split="train",
                                     cache_dir=model_args.cache_dir, use_auth_token=AUTH_KEY)
        # Labels
        label_list = list(
            range(train_dataset.features['concepts'].feature.num_classes))
        labels_codes = train_dataset.features['concepts'].feature.names
        num_labels = len(label_list)

    if training_args.do_eval:
        eval_dataset = load_dataset(os.path.join(DATA_DIR, 'multilabel_bench'), data_args.dataset_name, split="validation",
                                    cache_dir=model_args.cache_dir, use_auth_token=AUTH_KEY)
        if label_list is None:
            # Labels
            label_list = list(
                range(eval_dataset.features['concepts'].feature.num_classes))
            labels_codes = eval_dataset.features['concepts'].feature.names
            num_labels = len(label_list)

    if training_args.do_predict:
        predict_dataset = load_dataset(os.path.join(DATA_DIR, 'multilabel_bench'), data_args.dataset_name, split="test",
                                       cache_dir=model_args.cache_dir, use_auth_token=AUTH_KEY)
        if label_list is None:
            # Labels
            label_list = list(
                range(predict_dataset.features['concepts'].feature.num_classes))
            labels_codes = predict_dataset.features['concepts'].feature.names
            num_labels = len(label_list)

    # Load label descriptors
    if 'eurlex' in data_args.dataset_name:
        label_descriptors = EUROVOC_CONCEPTS[f'level_{data_args.dataset_name.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code] for label_code in labels_codes]
    elif 'bioasq' in data_args.dataset_name:
        label_descriptors = MESH_CONCEPTS[f'level_{data_args.dataset_name.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code] for label_code in labels_codes]
    elif 'mimic' in data_args.dataset_name:
        label_descriptors = ICD9_CONCEPTS[f'level_{data_args.dataset_name.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code] for label_code in labels_codes]
    elif 'uklex' in data_args.dataset_name:
        label_descriptors = UKLEX_CONCEPTS[f'level_{data_args.dataset_name.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code] for label_code in labels_codes]
    elif 'ecthr' in data_args.dataset_name:
        label_descriptors = ECTHR_ARTICLES[f'level_{data_args.dataset_name.split("-")[-1][-1]}']
        label_descs = [label_descriptors[label_code] for label_code in labels_codes]
    else:
        raise Exception(f'Dataset {data_args.dataset_name} is not supported!')

    # Label descriptors mode
    if model_args.seq2seq or model_args.t5_enc2dec or model_args.use_lwan_advanced:
        # Use original descriptors, e.g., EUROVOC 100153 ->  `employment and working conditions`
        if data_args.label_descriptors_mode == 'original':
            label_desc2id = {label_desc[0].replace(',', '').lower(): idx for idx, label_desc in enumerate(label_descs)}
            label_id2desc = {idx: label_desc[0].replace(',', '').lower() for idx, label_desc in enumerate(label_descs)}
        # Use simplified descriptors, e.g., EUROVOC 100153 ->  `employment`
        elif data_args.label_descriptors_mode == 'simplified':
            label_desc2id = {label_desc[1]: idx for idx, label_desc in enumerate(label_descs)}
            label_id2desc = {idx: label_desc[1] for idx, label_desc in enumerate(label_descs)}
        # Use pseudo number descriptors, e.g., EUROVOC 100153 ->  `<extra_id_11>`
        elif data_args.label_descriptors_mode == 'numbers':
            label_desc2id = {f'<extra_id_{idx}>': idx for idx in range(num_labels)}
            label_id2desc = {idx: f'<extra_id_{idx}>' for idx in range(num_labels)}
    else:
        # Use original descriptors, e.g., EUROVOC 100153 ->  `employment and working conditions`
        label_desc2id = {label_desc[0]: idx for idx, label_desc in enumerate(label_descs)}
        label_id2desc = {idx: label_desc[0] for idx, label_desc in enumerate(label_descs)}

    print(f'LabelDesc2Id: {label_desc2id}')

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id={l[0]: i for i, l in enumerate(label_descs)},
        id2label={i: l[0] for i, l in enumerate(label_descs)},
        finetuning_task=data_args.dataset_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    if data_args.label_descriptors_mode == 'numbers':
        new_tokens = [f'<extra_id_{idx+100}>' for idx in range(max(0, num_labels - 100))]
        tokenizer.add_tokens(new_tokens)

    if model_args.seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
    elif config.model_type in ['bert', 'roberta']:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
    else:
        # Register pooling method to config
        config.use_lwan = model_args.use_lwan
        config.lwan_version = model_args.lwan_version
        config.use_lwan_advanced = model_args.use_lwan_advanced
        config.t5_enc2dec = model_args.t5_enc2dec
        config.t5_enc2dec_mode = model_args.t5_enc2dec_mode
        config.causal_masking = model_args.causal_masking
        config.decoder_attention = model_args.decoder_attention
        config.lwan_heads = model_args.lwan_heads if model_args.lwan_heads > 0 else config.num_heads
        config.n_dec_layers = model_args.n_dec_layers if model_args.lwan_heads > 0 else config.num_decoder_layers
        model = T5ForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
        if data_args.label_descriptors_mode == 'numbers':
            model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        if 'mimic' in data_args.dataset_name:
            for idx, text in enumerate(examples['text']):
                if 'Service:' in text:
                    text = 'Service:' + re.split('Service:', text, maxsplit=1)[1]
                elif 'Sex:' in text:
                    text = re.split('\n', re.split('Sex:', text, maxsplit=1)[1], maxsplit=1)[1]
                text = normalizer.normalize_str(text)
                text = ' '.join([token[0] for token in pre_tokenizer.pre_tokenize_str(text)])
                text = re.sub('[^a-z ]{2,}', ' ', text, flags=re.IGNORECASE)
                text = re.sub(' +', ' ', text, flags=re.IGNORECASE)
                examples['text'][idx] = text

        # Tokenize the texts
        batch = tokenizer(
            examples["text"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
            add_special_tokens=True if (model_args.seq2seq or model_args.use_lwan
                                        or model_args.t5_enc2dec or
                                        config.model_type not in ['bert', 'roberta']) else False
        )

        if model_args.seq2seq:
            label_batch = tokenizer(
                [', '.join(sorted([label_id2desc[label] for label in labels])) if len(labels) else 'none' for labels in
                 examples["concepts"]],
                padding=False,
                max_length=data_args.generation_max_length,
                truncation=True,
            )
            batch['labels'] = label_batch['input_ids']
        else:
            if model_args.t5_enc2dec and model_args.t5_enc2dec_mode == 'single-step':
                decoder_inputs = tokenizer(
                    ['label' for _ in examples["text"]],
                    padding=False,
                    max_length=1,
                    add_special_tokens=False,
                    truncation=True,
                )
                batch['decoder_input_ids'] = decoder_inputs['input_ids']
                batch['decoder_attention_mask'] = decoder_inputs['attention_mask']
            elif model_args.t5_enc2dec_mode == 'multi-step' or model_args.use_lwan_advanced:
                decoder_inputs = tokenizer(
                    [' '.join([label_id2desc[label] for label in label_id2desc]) for _ in examples['text']],
                    padding=False,
                    max_length=len(label_id2desc),
                    truncation=True,
                    add_special_tokens=False
                )
                batch['decoder_input_ids'] = decoder_inputs['input_ids']
                batch['decoder_attention_mask'] = decoder_inputs['attention_mask']

            if not model_args.use_lwan and not model_args.t5_enc2dec and config.model_type not in ['bert', 'roberta']:
                for idx, _ in enumerate(batch['input_ids']):
                    batch['input_ids'][idx][-1] = tokenizer.eos_token_id
                    batch['attention_mask'][idx][-1] = 1

            batch["label_ids"] = [[1.0 if label in labels else 0.0 for label in label_list] for labels in
                                  examples["concepts"]]
            batch['labels'] = batch['label_ids']

        return batch

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['concepts', 'text'],
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['concepts', 'text'],
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['concepts', 'text'],
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        if model_args.seq2seq:
            from labeling_utils import fix_generated_scores
            preds, p.label_ids, _, _ = fix_generated_scores(tokenizer, p.predictions, p.label_ids,
                                                            label2id=label_desc2id)
        else:
            logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = (expit(logits) > 0.5).astype('int32')
        macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
        return {'macro-f1': macro_f1, 'micro-f1': micro_f1}

    trainer_class = Seq2SeqTrainer if model_args.seq2seq else Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                           pad_to_multiple_of=data_args.generation_max_length) \
        if model_args.seq2seq else DataCollatorForMultiLabelClassification(tokenizer)

    # Initialize our Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        if model_args.seq2seq:
            from labeling_utils import fix_generated_scores
            hard_predictions, labels, text_preds, text_labels = fix_generated_scores(tokenizer, predictions, labels,
                                                                                     label2id=label_desc2id)
        else:
            hard_predictions = (expit(predictions) > 0.5).astype('int32')
            text_preds = [', '.join(sorted([label_id2desc[idx] for idx, val in enumerate(doc_predictions) if val == 1]))
                          for doc_predictions in hard_predictions]
            text_labels = [', '.join(sorted([label_id2desc[idx] for idx, val in enumerate(doc_labels) if val == 1]))
                           for doc_labels in labels]

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        report_predict_file = os.path.join(training_args.output_dir, "test_classification_report.txt")
        predictions_file = os.path.join(training_args.output_dir, "test_predictions.pkl")
        labels_file = os.path.join(training_args.output_dir, "test_labels.pkl")
        if trainer.is_world_process_zero():
            cls_report = classification_report(y_true=labels, y_pred=hard_predictions,
                                               target_names=list(config.label2id.keys()),
                                               zero_division=0)
            with open(report_predict_file, "w") as writer:
                writer.write(cls_report)
            with open(predictions_file, 'wb') as writer:
                pickle.dump(text_preds, writer, protocol=pickle.HIGHEST_PROTOCOL)
            with open(labels_file, 'wb') as writer:
                pickle.dump(text_labels, writer, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(cls_report)

    # Clean up checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


if __name__ == "__main__":
    main()
