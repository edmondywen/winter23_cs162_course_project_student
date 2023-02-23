import os
import sys
import json
import csv
import glob
import pprint
import numpy as np
import random
import argparse
import pandas as pd
from tqdm import tqdm
from .utils import DataProcessor
from .utils import SemEvalSingleSentenceExample
from transformers import (
    AutoTokenizer,
)


class SemEvalDataProcessor(DataProcessor):
    """Processor for Sem-Eval 2020 Task 4 Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        examples = []   # Store your examples in this list

        ##################################################
        # TODO: (Optional) 
        # Some instructions for reading data:
        # 1. Use csv.DictReader or pd.read_csv to load the csv file 
        # and process the data properly.
        # 2. Use the provided class `SemEvalSingleSentenceExample` 
        # in `utils.py` for creating examples
        # 3. Store the two complementary statements as two 
        # individual examples 
        # e.g. example_1 = ...
        #      example_2 = ...
        #      examples.append(example_1)
        #      examples.append(example_2)
        # 4. Make sure that the order is maintained.
        # i.e. sent_1 in the data is stored/appended first and
        # sent_2 in the data is stored/appened after it.
        # 5. For the guid, simply use the row number (0-
        # indexed) for each data instance.
        # Use the same guid for statements from the same complementary pair.
        
        csv_path = os.path.join(data_dir, split+".csv")
        i = 0

        with open(csv_path, newline='\n') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                new_example_correct = SemEvalSingleSentenceExample(
                    guid = i,
                    text = row['Correct Statement'],
                    label = 1,
                    right_reason1 = row['Right Reason1'],
                    right_reason2 = row['Right Reason1'],
                    right_reason3 = row['Right Reason1'],
                    confusing_reason1 = row['Confusing Reason1'],
                    confusing_reason2 = row['Confusing Reason1'], 
                )
                new_example_incorrect = SemEvalSingleSentenceExample(
                    guid = i,
                    text = row['Incorrect Statement'],
                    label = 0,
                    right_reason1 = row['Right Reason1'],
                    right_reason2 = row['Right Reason1'],
                    right_reason3 = row['Right Reason1'],
                    confusing_reason1 = row['Confusing Reason1'],
                    confusing_reason2 = row['Confusing Reason1'], 
                )
                examples.append(new_example_correct)
                examples.append(new_example_incorrect)
                i+=1
            

            
            
            
        # End of TODO.
        ##################################################

        return examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="train")

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="dev")

    def get_test_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="test")


if __name__ == "__main__":

    # Test loading data.
    proc = SemEvalDataProcessor(data_dir="datasets/semeval_2020_task4")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print()
    for i in range(3):
        print(test_examples[i])
    print()
