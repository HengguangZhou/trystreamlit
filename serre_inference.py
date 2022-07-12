import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data.dataloader import DataLoader

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric
import torch
from typing import Any, Dict, Union

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

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

import argparse

def prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], device) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if hasattr(v, "to") and hasattr(v, "device"):
            inputs[k] = v.to(device)

    return inputs

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ser_model_name_or_path', type=str, required=True)
    parser.add_argument('--re_model_name_or_path', type=str, required=True)
    parser.add_argument('--tokenizer', type=str, required=True)

    datasets = load_dataset(
        os.path.abspath(LiLTfinetune.data.datasets.xfun.__file__),
        f"xfun.en",
        keep_in_memory=True,
    )

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    args = parser.parse_args()
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if True else None,
        padding="max_length",
        max_length=512,
    )
    eval_dataset = datasets["validation"]
    eval_dataset = eval_dataset.select(range(0, 1))
    dataloader = DataLoader(
            eval_dataset,
            batch_size=1,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
    )
    # =================================Ser Module=================================
    # ser_config = AutoConfig.from_pretrained('lilt-infoxlm-base')
    # re_config = AutoConfig.from_pretrained('lilt-infoxlm-base')
    device0 = torch.device("cpu")
    device1 = torch.device("cpu")

    ser_model = AutoModelForTokenClassification.from_pretrained(
        args.ser_model_name_or_path,
        # config=ser_config
        ).to(device0)
    re_model = AutoModelForRelationExtraction.from_pretrained(
        args.ser_model_name_or_path,
        # config=re_config
        ).to(device1)

    for step, inputs in enumerate(dataloader):

        ser_inputs = {
            'bbox': inputs['bbox'],
            'input_ids': inputs["input_ids"],
            "labels": inputs["labels"],
            "attention_mask": inputs["attention_mask"]}
        prepare_inputs(ser_inputs,ser_model.device)
        ser_out = ser_model(**ser_inputs)
        ser_pred = ser_out.logits.argmax(-1)

        re_inputs = {
            'bbox': inputs['bbox'],
            'input_ids': inputs["input_ids"],
            'entities': inputs['entities'],
            "labels": inputs["labels"],
            'relations': inputs['relations'],
            "attention_mask": inputs["attention_mask"]}
        prepare_inputs(re_inputs, re_model.device)
        re_out = re_model(**re_inputs)

        torch.save(ser_pred, "./ser_pred.pt")
        torch.save(re_out, "./re_out.pt")
        torch.save(inputs, "./inputs.pt")

        plt.imshow(inputs["image"][0].permute(1, 2, 0))
        plt.savefig("./ori.png")
        print("yyo")
    print("yo")
    # =================================Re Module==================================

def get_midpoint(x1, x2, y1, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    inference()
    
    ser_pred = torch.load("./ser_pred.pt")
    re_out = torch.load("./re_out.pt")
    inputs = torch.load("./inputs.pt")

    im_path = "./xfund&funsd/en/86236474_6476.png"
    im = Image.open(im_path).convert('RGB')
    plt.figure()
    fig, ax = plt.subplots()
    ax.imshow(im)

    bboxes = inputs['bbox'][0]
    bboxes[:, 0] = bboxes[:, 0] / 1000 * im.size[0]
    bboxes[:, 2] = bboxes[:, 2] / 1000 * im.size[0]
    bboxes[:, 1] = bboxes[:, 1] / 1000 * im.size[1]
    bboxes[:, 3] = bboxes[:, 3] / 1000 * im.size[1]

    relations = inputs['relations'][0]
    pred_rels = re_out['pred_relations'][0]
    pred = ser_pred[0].cpu() * inputs['attention_mask'][0]
    color = {1: 'r', 2: 'b', 4: 'b', 5: 'r'}
    for i in range(len(pred)):
        
        if pred[i] in [1, 2, 4, 5]:
            # print(i)
            bbox = bboxes[i].tolist()
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
            linewidth=0.5, edgecolor=color[pred[i].item()], 
            facecolor='none')
            
            ax.add_patch(rect)
    
    fk = len(relations["head"])
    print(f"len: {fk}")
    names = names=["O", "B-QUESTION", "B-ANSWER", "B-HEADER", "I-ANSWER", "I-QUESTION", "I-HEADER"]
    for i in range(len(relations["head"])):
        # head = pred_rels[i]['head_id']
        # tail = pred_rels[i]['tail_id']
        head = relations["head"][i]
        tail = relations["tail"][i]
        head_box = bboxes[head]
        tail_box = bboxes[tail]
        head_label = names[inputs["labels"][0][head]]
        tail_label = names[inputs["labels"][0][tail]]
        print(f"head: {head}, tail: {tail}, head_label: {head_label}, tail_label: {tail_label}")
        start = get_midpoint(head_box[0], head_box[2], head_box[1], head_box[3])
        end = get_midpoint(tail_box[0], tail_box[2], tail_box[1], tail_box[3])
        plt.arrow(x=start[0], y=start[1], dx=end[0]- start[0], dy= end[1] - start[1], width=0.0001)
        
    plt.savefig("./test.png")