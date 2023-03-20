import os
import sys
import json
import csv
import glob
import pprint
import numpy as np
import random
import argparse
from tqdm import tqdm
from .utils import DataProcessor
from .utils import Coms2SenseSingleSentenceExample
from transformers import (
    AutoTokenizer,
)


class Com2SenseDataProcessor(DataProcessor):
    """Processor for Com2Sense Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

        # TODO: Label to Int mapping, dict type.
        self.label2int = {"True": 1, "False": 0}

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        examples = []   # Store your examples in this list

        ##################################################
        # TODO:
        # Some instructions for reading data:
        # 1. Use json python package to load the data properly.
        # 2. Use the provided class `Coms2SenseSingleSentenceExample` 
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
        # 6. Make sure to handle if data do not have labels field.
        # This is useful for loading test data
        json_path = os.path.join(data_dir, split+".json")
        data = json.load(open(json_path, "r"))
        
        for i in range(len(data)):
            datum = data[i]
            guid = i
            label1 = None
            label2 = None
            if split != "test":
                label1 = 1 if datum['label_1'] == "True" else 0
                label2 = 1 if datum['label_2'] == "True" else 0
            example1 = Coms2SenseSingleSentenceExample(
                guid=guid,
                text=datum['sent_1'],
                label=label1,
                domain=datum['domain'],
                scenario=datum['scenario'],
                numeracy=datum['numeracy']
            )

            example2 = Coms2SenseSingleSentenceExample(
                guid=guid,
                text=datum['sent_2'],
                label=label2,
                domain=datum['domain'],
                scenario=datum['scenario'],
                numeracy=datum['numeracy']
            )
            
            examples.append(example1)
            examples.append(example2)
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
    proc = Com2SenseDataProcessor(data_dir="datasets/com2sense")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    def get_domains(examples):
        all_examples = map(lambda x: x.domain,examples)
        return set(all_examples)
        
    def get_scenarios(examples):
        all_scenarios = map(lambda x: x.scenario, examples)
        return set(all_scenarios)
    val_domain = get_domains(val_examples)
    val_scenario = get_scenarios(val_examples)
    with open("./datasets/com2sense/dev.json","r") as infile:
        json_stuff = infile.read()
        examples = json.loads(json_stuff)
    # print(val_domain)
        for domain in val_domain:
            print(domain)
            # print(examples)
            filtered_domain_examples = list(filter(lambda x: x["domain"] == domain,examples))
            # print(filtered_val_examples)
            with open(f"./datasets/com2sense/domain/{domain}/dev.json", "w") as outfile:
                filtered_json = json.dump(filtered_domain_examples, outfile)
    
        for scenario in val_scenario:
            print(scenario)
            filtered_scenario_examples = list(filter(lambda x: x["scenario"] == scenario,examples))
            with open(f"./datasets/com2sense/scenario/{scenario}/dev.json", "w") as outfile:
                filtered_json = json.dump(filtered_scenario_examples, outfile)
            
