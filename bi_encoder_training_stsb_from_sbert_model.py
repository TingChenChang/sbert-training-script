"""
This example loads the pre-trained SentenceTransformer model 'nli-distilroberta-base-v2' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

Usage:
python bi_encoder_training_stsb_from_sbert_model.py

OR

python bi_encoder_training_stsb_from_sbert_model.py \
    --model nli-distilroberta-base-v2 \
    --batch_size 32 \
    --epoch 4 \
    --model_save_path 
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
from datetime import datetime as ddt
from pytz import timezone

import argparse

# -------------------- PARSER --------------------
parser = argparse.ArgumentParser(description='Training information')
parser.add_argument('--model', dest='model', type=str, help='Name of the sentence embeding model', default='nli-distilroberta-base-v2')
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
    else f'output/bi_encoder_training_stsb-{model_name}-{time_str}'

# -------------------- Read Train/Dev/Test Dataset --------------------
# Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'data/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# -------------------- Loss --------------------
train_loss = losses.CosineSimilarityLoss(model=model)

# -------------------- Evaluator --------------------
# Development set: Measure correlation between cosine score and gold labels
logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

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


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)