import csv
import glob
import json
import random
import logging
import os
from enum import Enum
from typing import List, Optional, Union

import tqdm
import numpy as np

import torch
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

def mask_tokens(inputs, tokenizer, args, special_tokens_mask=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK,
    10% random, 10% original.
    inputs should be tokenized token ids with size: (batch size X input length).
    """
    # print("original inputs size",inputs.size())
    # The eventual labels will have the same size of the inputs,
    # with the masked parts the same as the input ids but the rest as
    # args.mlm_ignore_index, so that the cross entropy loss will ignore it.
    labels = inputs.clone()
    # return inputs,labels

    # Constructs the special token masks.
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    ##################################################
    # Optional TODO: only needed to be completed if you are doing MLM training

    # First sample a few tokens in each sequence for the MLM, with probability
    # `args.mlm_probability`.
    # Hint: you may find these functions handy: `torch.full`, Tensor's built-in
    # function `masked_fill_`, and `torch.bernoulli`.
    # Check the inputs to the bernoulli function and use other hinted functions
    # to construct such inputs.
    is_on_gpu = args.device == "gpu"
    probability_matrix = torch.full(labels.size(), args.mlm_probability)
    if is_on_gpu:
        probability_matrix = probability_matrix.cuda()
        special_tokens_mask = special_tokens_mask.cuda()
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # The "non-masked" parts in labels should be filled with ignore index (args.mlm_ignore_index).
    labels.masked_fill_((masked_indices == False),value = args.mlm_ignore_index)
    # raise NotImplementedError("Please finish the TODO!")

    # For 80% of the time, we will replace masked input tokens with  the
    # tokenizer.mask_token (e.g. for BERT it is [MASK] for for RoBERTa it is
    # <mask>, check tokenizer documentation for more details)
    # multiply by the masked_indices to only mask the selected indices
    rand_probs = torch.rand((masked_indices.size()))
    if is_on_gpu:
        rand_probs = rand_probs.cuda()
    masked_probs = (rand_probs * masked_indices.long() )
    if is_on_gpu:
        masked_probs = masked_probs.cuda()
    indices_replaced = ( masked_probs <= 0.8)
    # mask is last token
    # mask_token_index = (inputs == tokenizer.mask_token)[0].nonzero(as_tuple=True)[0]
    # print(" token id ",tokenizer.mask_token_id)
    probability_matrix.masked_fill_(indices_replaced, value=tokenizer.mask_token_id ) # value = tokenizer.mask_token??
    # raise NotImplementedError("Please finish the TODO!")

    # For 10% of the time, we replace masked input tokens with random word.
    # Hint: you may find function `torch.randint` handy.
    # Hint: make sure that the random word replaced positions are not overlapping
    # with those of the masked positions, i.e. "~indices_replaced".
    random_word_indices = ((masked_probs >0.8) <=0.9)
    if is_on_gpu:
        random_word_indices = random_word_indices.cuda()
    if is_on_gpu:
        random_words = np.random.choice(inputs.cpu().flatten(),probability_matrix.shape)
    else:
        random_words = np.random.choice(inputs.flatten(),probability_matrix.shape)
    if is_on_gpu:
        probability_matrix.masked_scatter_(random_word_indices, torch.from_numpy(random_words).float().cuda())
    else:
        probability_matrix.masked_scatter_(random_word_indices, torch.from_numpy(random_words).float())

    # raise NotImplementedError("Please finish the TODO!")
    inputs = probability_matrix.long()
    labels = labels
    # End of TODO
    ##################################################

    # For the rest of the time (10% of the time) we will keep the masked input
    # tokens unchanged
    pass  # Do nothing.


    # print("new inputs size",inputs.size())
    # print("new label size",labels.size())


    return inputs, labels


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    pass


if __name__ == "__main__":

    class mlm_args(object):
        def __init__(self):
            self.mlm_probability = 0.4
            self.mlm_ignore_index = -100
            self.device = "cpu"
            self.seed = 42
            self.n_gpu = 0

    args = mlm_args()
    set_seed(args)

    # Unit-testing the MLM function.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    input_sentence = "I am a good student and I love NLP."
    input_ids = tokenizer.encode(input_sentence)
    input_ids = torch.Tensor(input_ids).long().unsqueeze(0)
    
    inputs, labels = mask_tokens(input_ids, tokenizer, args,
                                 special_tokens_mask=None)
    inputs, labels = list(inputs.numpy()[0]), list(labels.numpy()[0])
    ans_inputs = [101, 146, 103, 170, 103, 2377, 103, 146, 1567, 103, 2101, 119, 102]
    ans_labels = [-100, -100, 1821, -100, 1363, -100, 1105, -100, -100, 21239, -100, -100, -100]
    
    if inputs == ans_inputs and labels == ans_labels:
        print("Your `mask_tokens` function is correct!")
    else:
        raise NotImplementedError("Your `mask_tokens` function is INCORRECT!")
