import os
import sys
import logging

#************************************* CONFIGS
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s; - %(levelname)s; - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#************************************* CONSTANTS

LANGUAGE = 'en' #es, en, fr
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR_PATH = os.path.join(ROOT_DIR, 'data/')
DATASETS_DIR_PATH = "C:/Users/anvaldez/Documents/Docto/Projects/SemEval/datasets/SemEval2024-Task8/"
CUT_PERCENTAGE_DATASET = 1
