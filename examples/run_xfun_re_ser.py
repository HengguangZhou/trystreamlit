#!/usr/bin/env python
# coding=utf-8
with open('tag.txt', 'w') as tagf:
    tagf.write('multilingual')
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric
import torch

import LiLTfinetune.data.datasets.xfun
import transformers
from LiLTfinetune import AutoModelForRelationExtraction
from LiLTfinetune.evaluation import re_score
from LiLTfinetune.data.data_collator import DataCollatorForKeyValueExtraction
from LiLTfinetune.data import DataCollatorForKeyValueExtraction
from LiLTfinetune.data.data_args import XFUNDataTrainingArguments
from LiLTfinetune.models.model_args import ModelArguments
from LiLTfinetune.reser.reser_args import ReSerArguments
from LiLTfinetune.trainers import XfunSerTrainer, XfunReTrainer
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((ReSerArguments, ModelArguments, XFUNDataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        reser_args, model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        reser_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # fix each model on only 1 gpu
    # training_args.n_gpu = 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)
    datasets = load_dataset(
        os.path.abspath(LiLTfinetune.data.datasets.xfun.__file__),
        f"xfun.{data_args.lang}",
        additional_langs=data_args.additional_langs,
        keep_in_memory=True,
    )

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    # =================================Ser Module=================================
    torch.cuda.set_device(1)
    training_args.per_device_train_batch_size = 16
    training_args._n_gpu = 1

    ser_last_checkpoint = None
    if os.path.isdir(reser_args.ser_output_dir) and reser_args.ser_do_train and not reser_args.overwrite_ser_output_dir:
        ser_last_checkpoint = get_last_checkpoint(reser_args.ser_output_dir)
        if ser_last_checkpoint is None and len(os.listdir(reser_args.ser_output_dir)) > 0:
            raise ValueError(
                f"Output directory ({reser_args.ser_output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif ser_last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {ser_last_checkpoint}. To avoid this behavior, change "
                "the `--ser_output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    if reser_args.ser_do_train:
        ser_column_names = datasets["train"].column_names
        ser_features = datasets["train"].features
    else:
        ser_column_names = datasets["validation"].column_names
        ser_features = datasets["validation"].features
    ser_text_column_name = "input_ids"
    ser_label_column_name = "labels"

    ser_remove_columns = ser_column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(ser_features[ser_label_column_name].feature, ClassLabel):
        ser_label_list = ser_features[ser_label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(ser_label_list))}
    else:
        ser_label_list = get_label_list(datasets["train"][ser_label_column_name])
        label_to_id = {l: i for i, l in enumerate(ser_label_list)}
    ser_num_labels = len(ser_label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    ser_config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else reser_args.ser_model_name_or_path,
        num_labels=ser_num_labels,
        finetuning_task="ner",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    ser_tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if reser_args.ser_tokenizer_name else reser_args.ser_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    ser_model = AutoModelForTokenClassification.from_pretrained(
        reser_args.ser_model_name_or_path,
        from_tf=bool(".ckpt" in reser_args.ser_model_name_or_path),
        config=ser_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(ser_tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False # True

    if reser_args.ser_do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if reser_args.ser_do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if reser_args.ser_do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        ser_tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    # Metrics
    ser_metric = load_metric("seqeval")

    def compute_ser_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [ser_label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [ser_label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = ser_metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    training_args.max_steps = reser_args.ser_max_steps
    # Initialize our Trainer
    ser_trainer = XfunSerTrainer(
        model=ser_model,
        args=training_args,
        train_dataset=train_dataset if reser_args.ser_do_train else None,
        eval_dataset=eval_dataset if reser_args.ser_do_eval else None,
        tokenizer=ser_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ser_metrics,
    )

    # Training
    if reser_args.ser_do_train:
        ser_checkpoint = ser_last_checkpoint if ser_last_checkpoint else None
        ser_train_result = ser_trainer.train(resume_from_checkpoint=ser_checkpoint)
        ser_metrics = ser_train_result.metrics
        ser_trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        ser_metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        ser_trainer.log_metrics("train", ser_metrics)
        ser_trainer.save_metrics("train", ser_metrics)
        ser_trainer.save_state()

    # Evaluation
    if reser_args.ser_do_eval:
        logger.info("*** Evaluate ***")

        ser_metrics = ser_trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        ser_metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        ser_trainer.log_metrics("eval", ser_metrics)
        ser_trainer.save_metrics("eval", ser_metrics)

    # Predict
    if reser_args.ser_do_predict:
        logger.info("*** Predict ***")

        ser_predictions, ser_labels, ser_metrics = ser_trainer.predict(test_dataset)
        ser_predictions = np.argmax(ser_predictions, axis=2)

        # Remove ignored index (special tokens)
        ser_true_predictions = [
            [ser_label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(ser_predictions, ser_labels)
        ]

        ser_trainer.log_metrics("test", ser_metrics)
        ser_trainer.save_metrics("test", ser_metrics)

        # Save predictions
        ser_output_test_predictions_file = os.path.join(reser_args.ser_output_dir, "ser_test_predictions.txt")
        if ser_trainer.is_world_process_zero():
            with open(ser_output_test_predictions_file, "w") as writer:
                for prediction in ser_true_predictions:
                    writer.write(" ".join(prediction) + "\n")    

    

    # =================================Re Module=================================

    training_args._n_gpu = 1
    torch.cuda.set_device(0)
    training_args.per_device_train_batch_size = 8

    re_last_checkpoint = None
    if os.path.isdir(reser_args.re_output_dir) and reser_args.re_do_train and not reser_args.overwrite_re_output_dir:
        re_last_checkpoint = get_last_checkpoint(reser_args.re_output_dir)
        if re_last_checkpoint is None and len(os.listdir(reser_args.re_output_dir)) > 0:
            raise ValueError(
                f"Output directory ({reser_args.re_output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif re_last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {re_last_checkpoint}. To avoid this behavior, change "
                "the `--re_output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if reser_args.re_do_train:
        re_column_names = datasets["train"].column_names
        re_features = datasets["train"].features
    else:
        re_column_names = datasets["validation"].column_names
        re_features = datasets["validation"].features
    re_text_column_name = "input_ids"
    re_label_column_name = "labels"

    re_remove_columns = re_column_names


    if isinstance(re_features[re_label_column_name].feature, ClassLabel):
        re_label_list = re_features[re_label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(re_label_list))}
    else:
        re_label_list = get_label_list(datasets["train"][re_label_column_name])
        label_to_id = {l: i for i, l in enumerate(re_label_list)}
    re_num_labels = len(re_label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    re_config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else reser_args.re_model_name_or_path,
        num_labels=re_num_labels,
        finetuning_task="ner",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    re_tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if reser_args.re_tokenizer_name else reser_args.re_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    re_model = AutoModelForRelationExtraction.from_pretrained(
        reser_args.re_model_name_or_path,
        from_tf=bool(".ckpt" in reser_args.re_model_name_or_path),
        config=re_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(re_tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if reser_args.re_do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if reser_args.re_do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if reser_args.re_do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        re_tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    def compute_re_metrics(p):
        pred_relations, gt_relations = p
        score = re_score(pred_relations, gt_relations, mode="boundaries")
        return score

    training_args.max_steps = reser_args.re_max_steps
    # training_args.device = reser_args.re_device
    # Initialize our Trainer
    re_trainer = XfunReTrainer(
        model=re_model,
        args=training_args,
        train_dataset=train_dataset if reser_args.re_do_train else None,
        eval_dataset=eval_dataset if reser_args.re_do_eval else None,
        tokenizer=re_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_re_metrics,
    )

    # Training
    if reser_args.re_do_train:
        re_checkpoint = re_last_checkpoint if re_last_checkpoint else None
        re_train_result = re_trainer.train(resume_from_checkpoint=re_checkpoint)
        re_metrics = re_train_result.metrics
        re_trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        re_metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        re_trainer.log_metrics("train", re_metrics)
        re_trainer.save_metrics("train", re_metrics)
        re_trainer.save_state()

    # Evaluation
    if reser_args.re_do_eval:
        logger.info("*** Evaluate ***")

        re_metrics = re_trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        re_metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        re_trainer.log_metrics("eval", re_metrics)
        re_trainer.save_metrics("eval", re_metrics)

    # =================================Modules finished=================================

    re_predictions, re_labels, re_metrics = re_trainer.predict(eval_dataset)
    ser_predictions, ser_labels, ser_metrics = ser_trainer.predict(eval_dataset)
    print("yo")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()
