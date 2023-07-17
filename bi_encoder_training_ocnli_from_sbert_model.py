"""
This example loads the pre-trained SentenceTransformer model 'distiluse-base-multilingual-cased-v1' from the server.
It then fine-tunes this model for some epochs on the OCNLI dataset.

OCNLI Dataset: https://github.com/CLUEbenchmark/OCNLI

Usage:
python bi_encoder_training_ocnli_from_sbert_model.py

OR

python bi_encoder_training_ocnli_from_sbert_model.py \
    --model distiluse-base-multilingual-cased-v1 \
    --batch_size 32 \
    --epoch 4 \
    --model_save_path 
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample, datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
from datetime import datetime as ddt
from pytz import timezone
import json
import base64
import requests

import argparse

# -------------------- PARSER --------------------
parser = argparse.ArgumentParser(description='Training information')
parser.add_argument('--model', dest='model', type=str, help='Name of the sentence embeding model', default='distiluse-base-multilingual-cased-v1')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--epoch', dest='epoch', type=int, default=1)
parser.add_argument('--model_save_path', dest='model_save_path', type=str)
args = parser.parse_args()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


# -------------------- Model Config --------------------
model_name = args.model
# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

train_batch_size = args.batch_size
num_epochs = args.epoch

time_str = ddt.now(timezone('Asia/Taipei')).strftime("%Y%m%d%H%M%S")
model_save_path = args.model_save_path \
    if args.model_save_path \
    else f'output/bi_encoder_training_ocnli-{model_name}-{time_str}'


# -------------------- Read Train/Dev Dataset --------------------
# Check if dataset exsist. If not, download it
ocnli_train_set_path = 'data/ocnli/train.3k.json'

if not os.path.exists(ocnli_train_set_path):
    
    if os.path.dirname(ocnli_train_set_path) != '':
        os.makedirs(os.path.dirname(ocnli_train_set_path), exist_ok=True)

    user = 'CLUEbenchmark'
    repo_name = 'OCNLI'
    path_to_file = 'data/ocnli/train.3k.json'

    url = f'https://api.github.com/repos/{user}/{repo_name}/contents/{path_to_file}'
    req = requests.get(url)
    if req.status_code == requests.codes.ok:
        req = req.json()  # the response is a JSON
        # req is now a dict with keys: name, encoding, url, size ...
        # and content. But it is encoded with base64.
        content = base64.b64decode(req['content']).decode("utf-8") 
        with open(ocnli_train_set_path, 'w') as f:
            f.writelines(content.splitlines())
    else:
        raise Exception('OCNLI train set was not found.')


# Convert the dataset to a DataLoader ready for training
logging.info("Read OCNLI train dataset")

# training set
train_set = []
with open(ocnli_train_set_path, 'r') as f:
  for line in f:
    train_set.append(json.loads(line))

logging.info(f"Train Set:{len(train_set)}")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}

train_samples = []
for row in train_set:
    sent1 = row['sentence1'].strip()
    sent2 = row['sentence2'].strip()
    label = row['label']
    if (label not in ['contradiction','entailment','neutral']) or (not sent1) or (not sent2):
        continue
    label_id = label2int[row['label']]
    train_samples.append(InputExample(texts=[sent1, sent2], label=label_id))

# Special data loader that avoid duplicates within a batch
train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
logging.info(f"Train DataLoader:{len(train_dataloader)}")


# -------------------- Loss --------------------
# Our training loss
train_loss = losses.SoftmaxLoss(
    model, 
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(), 
    num_labels=len(label2int)
    )

# -------------------- Evaluator --------------------
# Check if dataset exsist. If not, download it
ocnli_dev_set_path = 'data/ocnli/dev.json'

if not os.path.exists(ocnli_dev_set_path):
    
    if os.path.dirname(ocnli_dev_set_path) != '':
        os.makedirs(os.path.dirname(ocnli_dev_set_path), exist_ok=True)

    user = 'CLUEbenchmark'
    repo_name = 'OCNLI'
    path_to_file = 'data/ocnli/dev.json'

    url = f'https://api.github.com/repos/{user}/{repo_name}/contents/{path_to_file}'
    req = requests.get(url)
    if req.status_code == requests.codes.ok:
        req = req.json() 
        content = base64.b64decode(req['content']).decode("utf-8") 
        with open(ocnli_dev_set_path, 'w') as f:
            f.writelines(content.splitlines())
    else:
        raise Exception('OCNLI dev set was not found.') 

# dev set
dev_set = []
with open(ocnli_dev_set_path, 'r') as f:
  for line in f:
    dev_set.append(json.loads(line))

logging.info(f"Dev Set:{len(dev_set)}")

dev_samples = []
for row in dev_set:
    sent1 = row['sentence1'].strip()
    sent2 = row['sentence2'].strip()
    label = row['label']
    if (label not in ['contradiction','entailment','neutral']) or (not sent1) or (not sent2):
        continue
    label_id = label2int[row['label']]
    dev_samples.append(InputExample(texts=[sent1, sent2], label=label_id))

# Special data loader that avoid duplicates within a batch
dev_dataloader = datasets.NoDuplicatesDataLoader(dev_samples, batch_size=train_batch_size)
logging.info(f"Dev DataLoader:{len(dev_dataloader)}")

# Development set: Measure correlation between cosine score and gold labels
evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss, name='ocnli-dev')


# -------------------- Train --------------------
# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
