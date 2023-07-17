"""
This examples trains a CrossEncoder for the Quora Duplicate Questions Detection task. 
A CrossEncoder takes a sentence pair as input and outputs a label. 
Here, it output a continious labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python cross_encoder_training_qqp_from_bert_model.py

or 

python cross_encoder_training_qqp_from_bert_model.py \
    --model distilroberta-base \
    --batch_size 16 \
    --epoch 4 \
    --model_save_path   
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
import gzip
import csv
from zipfile import ZipFile
from datetime import datetime as ddt
from pytz import timezone

import argparse

# -------------------- PARSER --------------------
parser = argparse.ArgumentParser(description='Training information')
parser.add_argument('--model', dest='model', type=str, default='distilroberta-base')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--epoch', dest='epoch', type=int, default=1)
parser.add_argument('--model_save_path', dest='model_save_path', type=str)
args = parser.parse_args()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout

# -------------------- Model Config --------------------
model_name = args.model
train_batch_size = args.batch_size
num_epochs = args.epoch

time_str = ddt.now(timezone('Asia/Taipei')).strftime("%Y%m%d%H%M%S")
model_save_path = args.model_save_path \
    if args.model_save_path \
    else f'output/cross_encoder_training_qqp-{model_name}-{time_str}'

#We use distilroberta-base with a single label, i.e., it will output a value between 0 and 1 indicating the similarity of the two questions
model = CrossEncoder(model_name, num_labels=1)

# -------------------- Read Train Dataset --------------------
# Check if dataset exsist. If not, download and extract  it
dataset_path = 'data/quora-dataset/'

if not os.path.exists(dataset_path):
    logger.info("Dataset not found. Download")
    zip_save_path = 'data/quora-IR-dataset.zip'
    util.http_get(url='https://sbert.net/datasets/quora-IR-dataset.zip', path=zip_save_path)
    with ZipFile(zip_save_path, 'r') as zip:
        zip.extractall(dataset_path)

# Read the quora dataset split for classification
logger.info("Read train dataset")
train_samples = []
with open(os.path.join(dataset_path, 'classification', 'train_pairs.tsv'), 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        train_samples.append(InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate'])))
        train_samples.append(InputExample(texts=[row['question2'], row['question1']], label=int(row['is_duplicate'])))

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# -------------------- Evaluator --------------------
logger.info("Read dev dataset")
dev_samples = []
with open(os.path.join(dataset_path, 'classification', 'dev_pairs.tsv'), 'r', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_samples.append(InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate'])))
        
# We add an evaluator, which evaluates the performance during training
evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='Quora-dev')

# -------------------- Train --------------------
# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


