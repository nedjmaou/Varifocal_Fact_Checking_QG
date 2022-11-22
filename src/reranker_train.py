import torch
import sys

SEED = 0

""" GPU setup """
if torch.cuda.is_available():
    device = torch.device('cuda')

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

from datasets import load_dataset

data_type = 'question'
model_name = sys.argv[1]

dataset = load_dataset('json', data_files={'train': [f'data/reranker/{model_name}_train_{data_type}.json'],
                                          'validation': [f'data/reranker/{model_name}_valid_{data_type}.json']})

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_batch(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return tokenizer(batch["text"], padding='max_length', truncation=True, max_length=64)

# Encode the input data
dataset = dataset.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
dataset.rename_column_("score", "labels")
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

config = BertConfig.from_pretrained(
    "bert-base-uncased",
    num_labels=1,
)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    config=config,
)

model.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import numpy as np
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

training_args = TrainingArguments(
    num_train_epochs=50,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    output_dir=f"training_output/reranker_{model_name}_{data_type}",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_total_limit=10,
    load_best_model_at_end=True,
)

early_stopper = EarlyStoppingCallback(early_stopping_patience=10)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    callbacks=[early_stopper],
)

print('training starts:')
trainer.train()
print('training ends:')
print('evaluation starts:')
trainer.evaluate()
print('evaluation ends:')