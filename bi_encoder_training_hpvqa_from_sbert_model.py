"""
This example loads the pre-trained SentenceTransformer model 'distiluse-base-multilingual-cased-v1' from the server.
It then fine-tunes this model for some epochs on the HPV QA dataset.

Usage:
python bi_encoder_training_hpvqa_from_sbert_model.py

OR

python bi_encoder_training_hpvqa_from_sbert_model.py \
--model distiluse-base-multilingual-cased-v1 \
--batch_size 64 \
--epoch 10
"""

from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample, util, SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime as ddt
from pytz import timezone
import json
import pandas as pd
import os
import gzip
import csv

import argparse

# -------------------- PARSER --------------------
parser = argparse.ArgumentParser(description='Training information')
parser.add_argument('--model', dest='model', type=str, help='Name of the sentence embeding model', default='distiluse-base-multilingual-cased-v1')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--epoch', dest='epoch', type=int, default=6)
parser.add_argument('--model_save_path', dest='model_save_path', type=str)
args = parser.parse_args()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# -------------------- Model Config --------------------
model_name = args.model
# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

train_batch_size = args.batch_size
num_epochs = args.epoch

time_str = ddt.now(timezone('Asia/Taipei')).strftime("%Y%m%d%H%M%S")
model_save_path = args.model_save_path \
    if args.model_save_path \
    else f'output/training_hpvqa-{model_name.split("/")[-1]}-{time_str}'

# -------------------- Read Train Dataset --------------------
logging.info("Read HPV QA Train Dataset")

# Check if dataset exsist. If not, download and extract  it
hpv_qa_dataset_path = 'data/HPV_QA.xlsx'
if not os.path.exists(hpv_qa_dataset_path):
    raise Exception('HPV QA dataset was not found.')

hpv_qa = pd.read_excel(hpv_qa_dataset_path)

train_samples = []
for idx, row in hpv_qa.iterrows():
    sent1 = row['question']
    sent2 = row['answer']
    train_samples.append(InputExample(texts=[sent1, sent2]))

train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
logging.info(f"Train DataLoader:{len(train_dataloader)}")

# -------------------- Loss --------------------
train_loss = losses.MultipleNegativesRankingLoss(model)

# -------------------- Evaluator --------------------
logging.info("Read STSBenchmark Dev Dataset")

sts_dataset_path = 'data/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# -------------------- Train --------------------

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
