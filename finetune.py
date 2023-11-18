import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import load_dataset, load_metric
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
from datasets import ClassLabel
import pandas as pd
from IPython.display import display, HTML

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda")
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union

@dataclass
class DataCollatorForEntailment:
    """
    Data collator that will dynamically pad the inputs for entailment data.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):

        labels = [feature.pop('labels') for feature in features]
        batch_size = len(features)

        # Assume a fixed number of choices (2 in this case)
        num_choices = 2

        # Flatten
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        # Apply Padding
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1).to(device) for k, v in batch.items()}

        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64).to(device)
        return batch

def load_data(tokenizer, params):
    def filter_function(example):
        return len(example['dialog_speaker_list']) == 2

    def tokenize_function(examples):
        dialog_texts = examples['dialog_text_list']
        labels = examples['entailment']

        tokenized_examples = tokenizer(
            dialog_texts,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )

        tokenized_examples["labels"] = torch.tensor(labels, dtype=torch.int64).to(device)
        return tokenized_examples

    dataset = load_dataset(params.dataset)
    dataset = dataset.filter(filter_function)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    accepted_keys = ["input_ids", "attention_mask", "labels"]

    # for key in tokenized_datasets['train'].features.keys():
    #     if key not in accepted_keys:
    #         tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(key)

    for key in tokenized_datasets['validation'].features.keys():
        if key not in accepted_keys:
            tokenized_datasets['validation'] = tokenized_datasets['validation'].remove_columns(key)

    for key in tokenized_datasets['test'].features.keys():
        if key not in accepted_keys:
            tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(key)

    tokenized_datasets.set_format("torch")
    data_collator = DataCollatorForEntailment(tokenizer)

    # train_dataset = tokenized_datasets["train"].shuffle(seed=SEED)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=params.batch_size, collate_fn=data_collator)

    eval_dataset = tokenized_datasets["validation"]
    eval_dataloader = DataLoader(eval_dataset, batch_size=params.batch_size, collate_fn=data_collator)

    test_dataset = tokenized_datasets["test"]
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, collate_fn=data_collator)

    return eval_dataloader, test_dataloader

def finetune(model, eval_dataloader, params):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    # test hyperparameters: lr = 1e-4, num_epochs = 10, batch size, num_warmup_steps = 0 -- maybe weight decay and grad clipping?
    num_training_steps = params.num_epochs * len(eval_dataloader)
    num_warmup_steps = 0
    lr = 5e-5
    optimizer = AdamW(model.parameters(), lr=lr)
    
    progress_bar = tqdm(range(num_training_steps))
    metric = evaluate.load("accuracy")
    
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    for epoch in range(params.num_epochs):

        # model.train()
        # for batch in train_dataloader:
        #     batch = {k: v.to(device) for k, v in batch.items()}
        #     outputs = model(**batch)
        #     loss = outputs.loss
        #     loss.backward()

        #     optimizer.step()
        #     lr_scheduler.step()
        #     optimizer.zero_grad()
        #     progress_bar.update(1)
    
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        
        score = metric.compute()
        print('Validation Accuracy:', score['accuracy'])

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    return model


def test(model, test_dataloader, prediction_save='predictions.torch'):

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #
    #                   TODO: Implementation                      #
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

    metric = load_metric('accuracy')
    model.eval()
    all_predictions = []

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(list(predictions))
        metric.add_batch(predictions=predictions, references=batch["labels"])

    score = metric.compute()
    print('Test Accuracy:', score)
    torch.save(all_predictions, prediction_save)

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #


def main(params):

    tokenizer = AutoTokenizer.from_pretrained(params.model)
    eval_dataloader, test_dataloader = load_data(tokenizer, params)

    model = AutoModelForMultipleChoice.from_pretrained(params.model)
    model.to(device)
    model = finetune(model, eval_dataloader, params)

    test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--dataset", type=str, default="sled-umich/Conversation-Entailment")
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)

    params, unknown = parser.parse_known_args()
    main(params)