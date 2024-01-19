
import logging
import os
import sys
import glob
import pandas as pd
from statistics import mean 

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


def read_dataset(dataset_name):
    logger.info("*** Using dataset: %s", dataset_name)
    dataset_dir = ROOT_DIR + '/datasets/SemEval2024-Task8/SubtaskA/' + dataset_name
    dataset = pd.read_json(path_or_buf=dataset_dir, lines=True)
    return dataset


def explore_dataset(dataset):
    print("-"*50, " Dataset Info")
    print(dataset.info())
    print("-"*50, " Freq model")
    print(dataset['model'].value_counts())
    print("-"*50, " Freq label")
    print(dataset['label'].value_counts())
    print("-"*50, " Text len info")
    texts_len = [len(str.split()) for str in dataset['text']]
    print("avg:", mean(texts_len))
    print("max:", max(texts_len))
    print("min:", min(texts_len))
 

def main():
    dataset_name = 'subtaskA_train_monolingual.jsonl'
    train_data = read_dataset(dataset_name)
    explore_dataset(train_data)


if __name__ == '__main__':
    main()