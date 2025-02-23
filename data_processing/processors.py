import sys
import csv
import glob
import json
import logging
import random
import os
from enum import Enum
from typing import List, Optional, Union
import numpy as np
import argparse
import pprint

from tqdm import tqdm, trange

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
from torch.utils.data import DataLoader, SequentialSampler

# Processors.
from .dummy_data import DummyDataProcessor
from .com2sense_data import Com2SenseDataProcessor
from .semeval_data import SemEvalDataProcessor
from transformers import (
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Dummy Dataset."""

    def __init__(self, examples, tokenizer,
                 max_seq_length=None,
                 seed=None, args=None):
        """
        Args:
            examples (list): input examples of type `DummyExample`.
            tokenizer (huggingface.tokenizer): tokenizer in used.
            max_seq_length (int): maximum length to truncate the input ids.
            seed (int): random seed.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            self.seed = seed

        self.args = args
        self.examples = examples

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.cls_id = self.tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.pad_id = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.sep_id = self.tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

        self.args = args

    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):

        example = self.examples[idx]
        guid = example.guid
        text = example.text
        label = example.label

        batch_encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        input_ids = torch.Tensor(batch_encoding["input_ids"]).long()
        attention_mask = torch.Tensor(batch_encoding["attention_mask"]).long()
        if "token_type_ids" not in batch_encoding:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = torch.Tensor(batch_encoding["token_type_ids"]).long()

        labels = torch.Tensor([label]).long()[0]

        if not self.args.do_train:
            return input_ids, attention_mask, token_type_ids, labels, guid

        return input_ids, attention_mask, token_type_ids, labels


class SemEvalDataset(Dataset):
    """Sem-Eval 2020 Task 4 Dataset."""

    def __init__(self, examples, tokenizer,
                 max_seq_length=None,
                 seed=None, args=None):
        """
        Args:
            examples (list): input examples of type `DummyExample`.
            tokenizer (huggingface.tokenizer): tokenizer in used.
            max_seq_length (int): maximum length to truncate the input ids.
            seed (int): random seed.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            self.seed = seed

        self.args = args
        self.examples = examples

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.cls_id = self.tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.pad_id = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.sep_id = self.tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

        self.args = args

    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):

        ##################################################
        # TODO: (Optional) Please finish this function (refer DummyDataset __getitem__)
        # Note that `token_type_ids` may not exist from
        # the outputs of tokenizer for certain types of
        # models (e.g. RoBERTa), please take special care
        # of it with an if-else statement.
        example = self.examples[idx]
        guid = example.guid
        text = example.text
        label = example.label

        batch_encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = torch.Tensor(batch_encoding["input_ids"]).long()
        attention_mask = torch.Tensor(batch_encoding["attention_mask"]).long()
        if "token_type_ids" not in batch_encoding:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = torch.Tensor(batch_encoding["token_type_ids"]).long()
        # End of TODO.
        ##################################################

        label = example.label
        if label is not None:
            labels = torch.Tensor([label]).long()[0]

        if not self.args.do_train:
            if label is None:
                return input_ids, attention_mask, token_type_ids, guid
            return input_ids, attention_mask, token_type_ids, labels, guid

        return input_ids, attention_mask, token_type_ids, labels


class Com2SenseDataset(Dataset):
    """Com2Sense Dataset."""

    def __init__(self, examples, tokenizer,
                 max_seq_length=None,
                 seed=None, args=None):
        """
        Args:
            examples (list): input examples of type `DummyExample`.
            tokenizer (huggingface.tokenizer): tokenizer in used.
            max_seq_length (int): maximum length to truncate the input ids.
            seed (int): random seed.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            self.seed = seed

        self.args = args
        self.examples = examples

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.cls_id = self.tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.pad_id = self.tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.sep_id = self.tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

        self.args = args

    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):

        ##################################################
        # TODO: Please finish this function (refer DummyDataset __getitem__)
        # Note that `token_type_ids` may not exist from
        # the outputs of tokenizer for certain types of
        # models (e.g. RoBERTa), please take special care
        # of it with an if-else statement.
        example = self.examples[idx]
        guid = example.guid
        text = example.text
        label = example.label
        domain = example.domain
        scenario = example.scenario
        numeracy = example.numeracy

        batch_encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = torch.Tensor(batch_encoding["input_ids"]).long()
        attention_mask = torch.Tensor(batch_encoding["attention_mask"]).long()
        if "token_type_ids" not in batch_encoding:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = torch.Tensor(batch_encoding["token_type_ids"]).long()

        # End of TODO.
        ##################################################

        label = example.label
        if label is not None:
            labels = torch.Tensor([label]).long()[0]

        if not self.args.do_train:
            if label is None:
                return input_ids, attention_mask, token_type_ids, guid
            return input_ids, attention_mask, token_type_ids, labels, guid

        return input_ids, attention_mask, token_type_ids, labels, guid


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Basic args.
    parser.add_argument(
        "-d", "--dataset",
        default="dummy",
        type=str,
        help=("dataset name"),
    )
    data_args = parser.parse_args()

    class dummy_args(object):
        def __init__(self):
            self.model_type = "bert"
            self.cls_ignore_index = -100
            self.do_train = True
            self.max_seq_length = 32

    args = dummy_args()
    args.dataset = data_args.dataset

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    if args.dataset == "dummy":
        processor = DummyDataProcessor(data_dir="datasets/dummies", args=args)
        examples = processor.get_dev_examples()
        dataset = DummyDataset(examples, tokenizer,
                            max_seq_length=args.max_seq_length,
                            args=args)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=2)

        for step, batch in enumerate(dataloader):
            for each in batch:
                assert each.size()[0] == 2, "Batch not loading correctly! Some error!"
            break
        print ("Dummy Dataset loading correctly.")
    
    elif args.dataset == "com2sense":
        processor = Com2SenseDataProcessor(data_dir="datasets/com2sense", args=args)
        examples = processor.get_dev_examples()
        dataset = Com2SenseDataset(examples, tokenizer,
                            max_seq_length=args.max_seq_length,
                            args=args)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=2)

        for step, batch in enumerate(dataloader):
            for each in batch:
                assert each.size()[0] == 2, "Batch not loading correctly! Some error!"
            break
        print ("Com2Sense Dataset loading correctly.")
    
    elif args.dataset == "sem-eval":
        processor = SemEvalDataProcessor(data_dir="datasets/semeval_2020_task4", args=args)
        examples = processor.get_dev_examples()
        dataset = SemEvalDataset(examples, tokenizer,
                            max_seq_length=args.max_seq_length,
                            args=args)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=2)

        for step, batch in enumerate(dataloader):
            for each in batch:
                assert each.size()[0] == 2, "Batch not loading correctly! Some error!"
            break
        print ("SemEval Dataset loading correctly.")
