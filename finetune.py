import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from datasets import load_dataset, load_metric
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda")

class SimpleDataCollator:
    def __init__(self, tokenizer, max_length=None, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        labels = [feature['labels'] for feature in features]
        batch_size = len(features)

        inputs = {
            'input_ids': torch.stack([ex['input_ids'] for ex in features]),
            'attention_mask': torch.stack([ex['attention_mask'] for ex in features]),
            'labels': torch.tensor(labels, dtype=torch.int64).to(device),
        }

        return inputs

def load_data(tokenizer, params):
    def tokenize_function(data):
        dialog_text_list = data['dialog_text_list']
        h = data['h']
        labels = data['entailment']
        input_texts = [f"{dialog} {h}" for dialog in dialog_text_list]  # Adjust as needed

        tokenized_example = tokenizer(
            input_texts,
            truncation=True,
            padding=True,
            return_tensors='pt',
        )

        return {
            'input_ids': tokenized_example['input_ids'],
            'attention_mask': tokenized_example['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.int64).to(device),
        }
    dataset = load_dataset("sled-umich/Conversation-Entailment")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    accepted_keys = ["input_ids", "attention_mask", "labels"]

    for key in tokenized_datasets['validation'].features.keys():
        if key not in accepted_keys:
            tokenized_datasets['validation'] = tokenized_datasets['validation'].remove_columns(key)

    for key in tokenized_datasets['test'].features.keys():
        if key not in accepted_keys:
            tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(key)

    tokenized_datasets.set_format("torch")
    data_collator = SimpleDataCollator(tokenizer)

    eval_dataset = tokenized_datasets["validation"]
    eval_dataloader = DataLoader(eval_dataset, batch_size=params.batch_size, collate_fn=data_collator)
    
    test_dataset = tokenized_datasets["test"]
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, collate_fn=data_collator)

    return eval_dataloader, test_dataloader

def finetune(model, dataloader, params):
    num_training_steps = params.num_epochs * len(dataloader)
    num_warmup_steps = 0
    lr = 9e-2
    optimizer = AdamW(model.parameters(), lr=lr)
    
    progress_bar = tqdm(range(num_training_steps))
    metric = load_metric('accuracy')
    
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps
    )

    for epoch in range(params.num_epochs):
        model.train()
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
        model.eval()
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        
        score = metric.compute()
        print('Validation Accuracy:', score['accuracy'])

    return model

def test(model, test_dataloader, prediction_save='predictions.torch'):
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

def main(params):
    tokenizer = AutoTokenizer.from_pretrained(params.model)
    validation_dataloader, test_dataloader = load_data(tokenizer, params)

    model = AutoModelForSequenceClassification.from_pretrained(params.model)
    model.to(device)

    model = finetune(model, validation_dataloader, params)

    test(model, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--pad_to_multiple_of", type=int, default=None)

    params, unknown = parser.parse_known_args()
    main(params)